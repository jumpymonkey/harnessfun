import json
import shlex
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from harnessfun import __version__
from harnessfun.config import (
    AuthError,
    SecurityValidationError,
    load_config,
    verify_gcp_adc,
)
from harnessfun.harness import UniversalHarness
from harnessfun.mcp import MCPClientManager
from harnessfun.providers import (
    BaseLLMProvider,
    GCPGeminiProvider,
    VertexAnthropicProvider,
    get_provider,
    is_anthropic_model,
)
from harnessfun.tools import default_registry

console = Console()


def stream_turn_to_console(harness: UniversalHarness, prompt: str) -> None:
    """Executes a harness turn while streaming events to the rich console."""
    model_name = harness.config.active_model

    for event in harness.run_turn_stream(prompt):
        if event.type == "step_start":
            step_num = event.data.get("step", 1)
            max_steps = event.data.get("max_steps", 10)
            if step_num > 1:
                console.print(f"[dim]── Step {step_num}/{max_steps} ──[/dim]")

        elif event.type == "model_thought":
            thought_text = event.data.get("thought", "").strip()
            if thought_text:
                console.print(
                    Panel(
                        thought_text,
                        title="[dim italic]Model Reasoning[/dim italic]",
                        border_style="dim",
                        expand=False,
                    )
                )

        elif event.type == "tool_call":
            tool_name = event.data.get("tool", "unknown")
            args = event.data.get("args", {})
            args_str = json.dumps(args)
            console.print(
                f" [bold cyan]⚡ Tool Call:[/bold cyan] [bold yellow]{tool_name}[/bold yellow] [dim]args={args_str}[/dim]"
            )

        elif event.type == "tool_result":
            tool_name = event.data.get("tool", "unknown")
            output = event.data.get("output", {})
            is_error = event.data.get("is_error", False)
            output_str = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
            if is_error:
                console.print(
                    f" [bold red]❌ Tool Error ({tool_name}):[/bold red] {output_str}"
                )
            else:
                console.print(
                    f" [bold green]✓ Tool Output ({tool_name}):[/bold green] [dim]{output_str}[/dim]"
                )

        elif event.type == "turn_complete":
            final_content = event.data.get("content", "")
            console.print(
                Panel(
                    final_content,
                    title=f"harnessfun [{model_name}]",
                    border_style="blue",
                )
            )

        elif event.type == "error":
            err_msg = event.data.get("error", "An unknown error occurred.")
            console.print(
                Panel(
                    f"[bold red]{err_msg}[/bold red]",
                    title="Execution Error",
                    border_style="red",
                )
            )


def _show_mcp_help() -> None:
    """Displays interactive help for /mcp subcommands."""
    table = Table(title="Interactive MCP (Model Context Protocol) Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    table.add_row("/mcp list", "List all configured MCP servers and their connection statuses.")
    table.add_row("/mcp connect <name> <cmd> [args...]", "Connect to a stdio MCP server (e.g. /mcp connect sqlite uvx mcp-server-sqlite --db-path ./test.db).")
    table.add_row("/mcp connect-url <name> <url> [H=V...]", "Connect to an HTTP/SSE MCP server (e.g. /mcp connect-url bigquery https://bigquery.googleapis.com/mcp).")
    table.add_row("/mcp disconnect <name>", "Disconnect an MCP server and unregister all its tools.")
    table.add_row("/mcp tools [server]", "List all tools discovered from active MCP servers.")
    table.add_row("/mcp load <filepath>", "Load and connect servers declared in a JSON or YAML config file.")
    table.add_row("/mcp help", "Display this MCP command guide.")
    console.print(table)


def _handle_mcp_command(arg: str, harness: UniversalHarness, mcp_manager: MCPClientManager) -> None:
    """Processes interactive /mcp slash commands."""
    if not arg:
        _show_mcp_help()
        return

    try:
        parts = shlex.split(arg)
    except ValueError as e:
        console.print(f"[bold red]Command parsing error:[/bold red] {e}")
        return

    subcmd = parts[0].lower()

    if subcmd in ["help", "-h", "--help"]:
        _show_mcp_help()

    elif subcmd == "list":
        servers = mcp_manager.list_servers()
        if not servers:
            console.print("[dim]No MCP servers registered yet. Use '/mcp connect' or '/mcp load' to register.[/dim]")
            return

        table = Table(title="Configured MCP Servers")
        table.add_column("Server", style="cyan")
        table.add_column("Transport", style="dim")
        table.add_column("Target")
        table.add_column("Status")
        table.add_column("Tools", style="yellow")

        for s in servers:
            status_style = "bold green" if s["status"] == "connected" else ("bold red" if s["status"] == "error" else "dim")
            table.add_row(
                s["name"],
                s["transport"],
                s["target"],
                f"[{status_style}]{s['status']}[/{status_style}]",
                f"{s['tool_count']} ({', '.join(s['tools']) if s['tools'] else 'none'})",
            )
        console.print(table)

    elif subcmd == "connect":
        if len(parts) < 3:
            console.print("[bold red]Usage:[/bold red] /mcp connect <name> <command> [args...]")
            console.print("[dim]Example: /mcp connect sqlite uvx mcp-server-sqlite --db-path ./mydb.db[/dim]")
            return

        name = parts[1]
        command = parts[2]
        cmd_args = parts[3:]

        console.print(f"[dim]Connecting to MCP server '{name}' via stdio ({command} {' '.join(cmd_args)})...[/dim]")
        try:
            tools = mcp_manager.connect_stdio(name=name, command=command, args=cmd_args)
            for t in tools:
                harness.registry.register_tool_definition(t)
            console.print(f"[bold green]✓ Successfully connected to '{name}' ({len(tools)} tools registered):[/bold green]")
            for t in tools:
                console.print(f"  • [bold cyan]{t.name}[/bold cyan]: [dim]{t.description}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Failed to connect to MCP server '{name}':[/bold red] {e}")

    elif subcmd == "connect-url":
        if len(parts) < 3:
            console.print("[bold red]Usage:[/bold red] /mcp connect-url <name> <url> [Header=Value ...]")
            console.print("[dim]Example: /mcp connect-url bigquery https://bigquery.googleapis.com/mcp[/dim]")
            return

        name = parts[1]
        url = parts[2]
        headers: Dict[str, str] = {}
        for p in parts[3:]:
            if ":" in p:
                k, v = p.split(":", 1)
                headers[k.strip()] = v.strip()
            elif "=" in p:
                k, v = p.split("=", 1)
                headers[k.strip()] = v.strip()

        console.print(f"[dim]Connecting to MCP server '{name}' ({url})...[/dim]")
        try:
            tools = mcp_manager.connect_url(name=name, url=url, headers=headers)
            for t in tools:
                harness.registry.register_tool_definition(t)
            console.print(f"[bold green]✓ Successfully connected to '{name}' ({len(tools)} tools registered):[/bold green]")
            for t in tools:
                console.print(f"  • [bold cyan]{t.name}[/bold cyan]: [dim]{t.description}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Failed to connect to MCP server '{name}':[/bold red] {e}")

    elif subcmd == "disconnect":
        if len(parts) < 2:
            console.print("[bold red]Usage:[/bold red] /mcp disconnect <name>")
            return

        name = parts[1]
        removed_count = harness.registry.unregister_server(name)
        try:
            disconnected = mcp_manager.disconnect(name)
            if disconnected or removed_count > 0:
                console.print(f"[bold green]✓ Disconnected MCP server '{name}' and unregistered {removed_count} tools.[/bold green]")
            else:
                console.print(f"[bold yellow]MCP server '{name}' not found or already disconnected.[/bold yellow]")
        except Exception as e:
            console.print(f"[bold red]Error disconnecting '{name}':[/bold red] {e}")

    elif subcmd == "tools":
        server_filter = parts[1] if len(parts) > 1 else None
        tools = mcp_manager.get_server_tools(server_filter)
        if not tools:
            console.print(f"[dim]No tools found for server '{server_filter}'.[/dim]" if server_filter else "[dim]No active MCP tools registered.[/dim]")
            return

        table = Table(title=f"MCP Tools{' (' + server_filter + ')' if server_filter else ''}")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Server", style="magenta")
        table.add_column("Description")
        for t in tools:
            table.add_row(t.name, t.server_name or "local", t.description)
        console.print(table)

    elif subcmd == "load":
        if len(parts) < 2:
            console.print("[bold red]Usage:[/bold red] /mcp load <filepath.json|.yaml>")
            return

        filepath = parts[1]
        try:
            console.print(f"[dim]Loading MCP servers from {filepath}...[/dim]")
            results = mcp_manager.load_config_file(filepath)
            total_tools = 0
            for srv_name, srv_tools in results.items():
                for t in srv_tools:
                    harness.registry.register_tool_definition(t)
                total_tools += len(srv_tools)
                console.print(f"  • [bold green]✓[/bold green] Server '{srv_name}': {len(srv_tools)} tools")
            console.print(f"[bold green]✓ Successfully loaded {len(results)} servers ({total_tools} total tools).[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to load MCP configuration:[/bold red] {e}")

    else:
        console.print(f"[bold red]Unknown MCP subcommand:[/bold red] '{subcmd}'. Type '/mcp help' for options.")


@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, help="Show harnessfun version.")
@click.pass_context
def cli(ctx, version):
    """harnessfun: Provider-configurable LLM Execution Harness for GCP."""
    if version:
        console.print(f"[bold green]harnessfun v{__version__}[/bold green]")
        sys.exit(0)
    if ctx.invoked_subcommand is None:
        # Default to interactive chat mode if no subcommand provided
        ctx.invoke(chat)


@cli.command()
def auth_check():
    """Verify GCP Application Default Credentials and security configuration."""
    console.print("[bold blue]Checking GCP Authentication & Security Policies...[/bold blue]")
    try:
        credentials, project_id = verify_gcp_adc()
        console.print(
            Panel.fit(
                f"[bold green]✓ Authentication Successful[/bold green]\n"
                f"[bold]Active GCP Project:[/bold] {project_id}\n"
                f"[bold]Auth Method:[/bold] Google Cloud ADC (No API Keys Used)",
                title="GCP Status",
                border_style="green"
            )
        )
    except (SecurityValidationError, AuthError) as e:
        console.print(f"[bold red]❌ Authentication Verification Failed:[/bold red]\n{e}")
        sys.exit(1)


@cli.command()
@click.option("--project", "-p", help="GCP Project ID.")
@click.option("--location", "-l", help="GCP Location/Region.")
@click.option("--model-location", help="Location used to reach models (default: global for Gemini, us-east5 for Claude).")
def models_list(project, location, model_location):
    """List available Gemini and Anthropic Claude models in Vertex AI."""
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location or location
        )
        gemini_provider = GCPGeminiProvider(project_id=cfg.project_id, location=cfg.model_location)
        anthropic_provider = VertexAnthropicProvider(project_id=cfg.project_id, location=cfg.model_location)
        
        console.print(f"[bold blue]Querying Vertex AI models for project '{cfg.project_id}'...[/bold blue]")
        
        table = Table(title=f"Available Vertex AI Models ({cfg.project_id})")
        table.add_column("Provider", style="cyan")
        table.add_column("Model Identifier", style="magenta")
        table.add_column("Status", style="green")
        
        # List Gemini models
        try:
            for m in gemini_provider.list_models():
                table.add_row("Google Gemini", m, "Available")
        except Exception as e:
            table.add_row("Google Gemini", f"[dim]Error: {e}[/dim]", "Unavailable")
            
        # List Anthropic Claude models
        for m in anthropic_provider.list_models():
            table.add_row("Anthropic Claude", m, "Available (Vertex AI)")
            
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to fetch models:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("prompt")
@click.option("--model", "-m", help="Model ID to use (e.g. gemini-2.5-flash, claude-3-5-sonnet).")
@click.option("--project", "-p", help="GCP Project ID.")
@click.option("--location", "-l", help="GCP Location/Region.")
@click.option("--model-location", help="Location used to reach models (default: global for Gemini, us-east5 for Claude).")
@click.option("--trace", help="Optional filepath to export execution trajectory JSONL log.")
@click.option("--mcp-config", help="Optional filepath to JSON or YAML MCP configuration file.")
def run(prompt, model, project, location, model_location, trace, mcp_config):
    """Execute a single one-shot prompt."""
    mcp_manager = MCPClientManager()
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location,
            model=model
        )
        provider = get_provider(cfg.active_model, cfg.project_id, cfg.model_location)
        harness = UniversalHarness(provider=provider, config=cfg, registry=default_registry, mcp_manager=mcp_manager)

        if mcp_config:
            loaded = mcp_manager.load_config_file(mcp_config)
            for srv_name, srv_tools in loaded.items():
                for t in srv_tools:
                    harness.registry.register_tool_definition(t)

        provider_name = "Anthropic Claude (Vertex AI)" if is_anthropic_model(cfg.active_model) else "Google Gemini"
        console.print(f"[dim]Running model '{cfg.active_model}' [{provider_name}] on project '{cfg.project_id}'...[/dim]\n")
        stream_turn_to_console(harness, prompt)
        if trace:
            harness.export_trajectory_jsonl(trace)
            console.print(f"\n[bold green]✓ Trajectory log saved to:[/bold green] {trace}")
    except Exception as e:
        console.print(f"[bold red]Execution Error:[/bold red] {e}")
        sys.exit(1)
    finally:
        mcp_manager.close()


@cli.command()
@click.option("--model", "-m", help="Initial model ID to use (Gemini or Claude).")
@click.option("--project", "-p", help="GCP Project ID.")
@click.option("--location", "-l", help="GCP Location/Region.")
@click.option("--model-location", help="Location used to reach models (default: global for Gemini, us-east5 for Claude).")
@click.option("--trace", help="Optional filepath to export execution trajectory JSONL log on exit.")
@click.option("--mcp-config", help="Optional filepath to JSON or YAML MCP configuration file.")
def chat(model, project, location, model_location, trace, mcp_config):
    """Start an interactive REPL session with on-the-fly model switching."""
    mcp_manager = MCPClientManager()
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location,
            model=model
        )
        provider = get_provider(cfg.active_model, cfg.project_id, cfg.model_location)
        harness = UniversalHarness(provider=provider, config=cfg, registry=default_registry, mcp_manager=mcp_manager)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize harness session:[/bold red] {e}")
        mcp_manager.close()
        sys.exit(1)

    if mcp_config:
        try:
            loaded = mcp_manager.load_config_file(mcp_config)
            for srv_name, srv_tools in loaded.items():
                for t in srv_tools:
                    harness.registry.register_tool_definition(t)
            console.print(f"[bold green]✓ Preloaded MCP configuration from '{mcp_config}'[/bold green]")
        except Exception as ex:
            console.print(f"[bold red]Failed to load MCP configuration '{mcp_config}':[/bold red] {ex}")

    provider_name = "Anthropic Claude (Vertex AI)" if is_anthropic_model(cfg.active_model) else "Google Gemini"

    # REPL Welcome Banner
    banner = (
        f"[bold green]harnessfun Interactive REPL v{__version__}[/bold green]\n"
        f"[bold]GCP Project:[/bold] {cfg.project_id} | [bold]Region:[/bold] {cfg.location}\n"
        f"[bold]Active Model:[/bold] {cfg.active_model} [dim]({provider_name})[/dim]\n"
        f"[dim]Type [bold]/help[/bold] for available commands or [bold]/exit[/bold] to quit.[/dim]"
    )
    console.print(Panel(banner, title="Session Started", border_style="cyan"))

    def _cleanup_and_export():
        if trace:
            try:
                harness.export_trajectory_jsonl(trace)
                console.print(f"[bold green]✓ Session trajectory log saved to:[/bold green] {trace}")
            except Exception as ex:
                console.print(f"[bold red]Failed to save trajectory log:[/bold red] {ex}")
        harness.close()

    # Interactive Loop
    while True:
        try:
            prompt_str = f"[bold cyan]harnessfun [{harness.config.active_model}]>[/bold cyan] "
            user_input = console.input(prompt_str).strip()

            if not user_input:
                continue

            # Handle Slash Commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                command = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if command in ["/exit", "/quit"]:
                    console.print("[dim]Exiting interactive session. Goodbye![/dim]")
                    _cleanup_and_export()
                    break

                elif command == "/help":
                    help_table = Table(title="Interactive Slash Commands")
                    help_table.add_column("Command", style="cyan")
                    help_table.add_column("Description")
                    help_table.add_row("/model <id>", "View or switch active model (Gemini or Claude) on the fly.")
                    help_table.add_row("/models", "List available Gemini and Anthropic Claude models.")
                    help_table.add_row("/clear", "Reset conversation history and recorded events.")
                    help_table.add_row("/history", "View message history turn count.")
                    help_table.add_row("/events, /trajectory", "View step-by-step trajectory events for current session.")
                    help_table.add_row("/export <file>", "Export session trajectory as a JSONL log file.")
                    help_table.add_row("/system <prompt>", "View or update system instructions.")
                    help_table.add_row("/tools", "List registered local and MCP tools.")
                    help_table.add_row("/mcp <subcommand>", "Manage MCP servers interactively (connect, disconnect, list, tools, load).")
                    help_table.add_row("/info", "Display active GCP session & provider information.")
                    help_table.add_row("/help", "Show this help table.")
                    help_table.add_row("/exit, /quit", "Exit REPL session.")
                    console.print(help_table)

                elif command == "/model":
                    if arg:
                        harness.set_model(arg)
                        harness.provider = get_provider(arg, cfg.project_id, cfg.model_location)
                        p_name = "Anthropic Claude (Vertex AI)" if is_anthropic_model(arg) else "Google Gemini"
                        console.print(f"[bold green]✓ Active model switched to:[/bold green] {arg} [dim]({p_name})[/dim]")
                    else:
                        p_name = "Anthropic Claude (Vertex AI)" if is_anthropic_model(harness.config.active_model) else "Google Gemini"
                        console.print(f"Active Model: [bold cyan]{harness.config.active_model}[/bold cyan] [dim]({p_name})[/dim]")

                elif command == "/models":
                    console.print("[dim]Fetching available models from GCP Vertex AI...[/dim]")
                    models_table = Table(title=f"Available Models ({cfg.project_id})")
                    models_table.add_column("Provider", style="cyan")
                    models_table.add_column("Model Identifier", style="magenta")

                    try:
                        gp = GCPGeminiProvider(project_id=cfg.project_id, location=cfg.model_location)
                        for m in gp.list_models():
                            models_table.add_row("Google Gemini", m)
                    except Exception as e:
                        models_table.add_row("Google Gemini", f"[dim]Error: {e}[/dim]")

                    ap = VertexAnthropicProvider(project_id=cfg.project_id, location=cfg.model_location)
                    for m in ap.list_models():
                        models_table.add_row("Anthropic Claude", m)

                    console.print(models_table)

                elif command == "/clear":
                    harness.clear()
                    console.print("[bold green]✓ Conversation history and trajectory cleared.[/bold green]")

                elif command == "/history":
                    turn_count = len([m for m in harness.messages if m.role == "user"])
                    console.print(f"Conversation Turns: [bold]{turn_count}[/bold] | Total Messages: [bold]{len(harness.messages)}[/bold]")

                elif command in ["/events", "/trajectory"]:
                    events = harness.events
                    if not events:
                        console.print("[dim]No events recorded in this session yet.[/dim]")
                    else:
                        event_table = Table(title=f"Session Trajectory Events ({len(events)} total)")
                        event_table.add_column("#", style="dim", width=4)
                        event_table.add_column("Step", style="cyan", width=6)
                        event_table.add_column("Type", style="magenta", width=14)
                        event_table.add_column("Details", style="green")
                        for idx, ev in enumerate(events, 1):
                            summary = ""
                            if ev.type == "step_start":
                                summary = f"Starting step {ev.data.get('step', '?')}/{ev.data.get('max_steps', '?')}"
                            elif ev.type == "model_thought":
                                thought = ev.data.get("thought", "").replace("\n", " ")
                                summary = thought[:80] + "..." if len(thought) > 80 else thought
                            elif ev.type == "tool_call":
                                args = json.dumps(ev.data.get("args", {}))
                                summary = f"{ev.data.get('tool', 'tool')}({args})"
                            elif ev.type == "tool_result":
                                out = json.dumps(ev.data.get("output", {}))
                                summary = f"{ev.data.get('tool', 'tool')} -> {out[:80]}"
                            elif ev.type == "turn_complete":
                                content = ev.data.get("content", "").replace("\n", " ")
                                summary = content[:80] + "..." if len(content) > 80 else content
                            elif ev.type == "error":
                                summary = f"[red]{ev.data.get('error', '')}[/red]"
                            event_table.add_row(str(idx), str(ev.step + 1), ev.type, summary)
                        console.print(event_table)

                elif command == "/export":
                    if not arg:
                        console.print("[bold red]Usage:[/bold red] /export <filepath.jsonl>")
                    else:
                        try:
                            harness.export_trajectory_jsonl(arg)
                            console.print(f"[bold green]✓ Trajectory successfully exported to:[/bold green] {arg}")
                        except Exception as e:
                            console.print(f"[bold red]Failed to export trajectory:[/bold red] {e}")

                elif command == "/system":
                    if arg:
                        harness.set_system_instruction(arg)
                        console.print(f"[bold green]✓ System instruction updated:[/bold green] '{arg}'")
                    else:
                        console.print(f"[bold]System Instruction:[/bold] {harness.config.system_instruction}")

                elif command == "/tools":
                    tools_info = harness.registry.get_info()
                    tool_table = Table(title="Registered Tools")
                    tool_table.add_column("Tool Name", style="cyan")
                    tool_table.add_column("Origin", style="magenta")
                    tool_table.add_column("Description")
                    for t in tools_info:
                        tool_table.add_row(t["name"], t.get("server") or "local", t["description"])
                    console.print(tool_table)

                elif command == "/mcp":
                    _handle_mcp_command(arg, harness, mcp_manager)

                elif command == "/info":
                    p_name = "Anthropic Claude (Vertex AI)" if is_anthropic_model(harness.config.active_model) else "Google Gemini"
                    provider_loc = getattr(harness.provider, "location", cfg.model_location)
                    console.print(
                        f"[bold]Project ID:[/bold] {cfg.project_id}\n"
                        f"[bold]Location:[/bold] {cfg.location}\n"
                        f"[bold]Model Endpoint Region:[/bold] {provider_loc}\n"
                        f"[bold]Active Model:[/bold] {harness.config.active_model}\n"
                        f"[bold]Active Provider:[/bold] {p_name}\n"
                        f"[bold]Max Steps:[/bold] {cfg.max_steps}"
                    )

                else:
                    console.print(f"[bold red]Unknown command:[/bold red] '{command}'. Type /help for available options.")

            else:
                # Regular prompt turn with live event streaming
                stream_turn_to_console(harness, user_input)

        except KeyboardInterrupt:
            console.print("\n[dim]Session interrupted. Exiting...[/dim]")
            _cleanup_and_export()
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")


def main():
    cli()


if __name__ == "__main__":
    main()
