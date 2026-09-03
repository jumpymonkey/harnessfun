import json
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
def run(prompt, model, project, location, model_location, trace):
    """Execute a single one-shot prompt."""
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location,
            model=model
        )
        provider = get_provider(cfg.active_model, cfg.project_id, cfg.model_location)
        harness = UniversalHarness(provider=provider, config=cfg)
        
        provider_name = "Anthropic Claude (Vertex AI)" if is_anthropic_model(cfg.active_model) else "Google Gemini"
        console.print(f"[dim]Running model '{cfg.active_model}' [{provider_name}] on project '{cfg.project_id}'...[/dim]\n")
        stream_turn_to_console(harness, prompt)
        if trace:
            harness.export_trajectory_jsonl(trace)
            console.print(f"\n[bold green]✓ Trajectory log saved to:[/bold green] {trace}")
    except Exception as e:
        console.print(f"[bold red]Execution Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", help="Initial model ID to use (Gemini or Claude).")
@click.option("--project", "-p", help="GCP Project ID.")
@click.option("--location", "-l", help="GCP Location/Region.")
@click.option("--model-location", help="Location used to reach models (default: global for Gemini, us-east5 for Claude).")
@click.option("--trace", help="Optional filepath to export execution trajectory JSONL log on exit.")
def chat(model, project, location, model_location, trace):
    """Start an interactive REPL session with on-the-fly model switching."""
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location,
            model=model
        )
        provider = get_provider(cfg.active_model, cfg.project_id, cfg.model_location)
        harness = UniversalHarness(provider=provider, config=cfg, registry=default_registry)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize harness session:[/bold red] {e}")
        sys.exit(1)

    provider_name = "Anthropic Claude (Vertex AI)" if is_anthropic_model(cfg.active_model) else "Google Gemini"

    # REPL Welcome Banner
    banner = (
        f"[bold green]harnessfun Interactive REPL v{__version__}[/bold green]\n"
        f"[bold]GCP Project:[/bold] {cfg.project_id} | [bold]Region:[/bold] {cfg.location}\n"
        f"[bold]Active Model:[/bold] {cfg.active_model} [dim]({provider_name})[/dim]\n"
        f"[dim]Type [bold]/help[/bold] for available commands or [bold]/exit[/bold] to quit.[/dim]"
    )
    console.print(Panel(banner, title="Session Started", border_style="cyan"))

    def _export_trace_if_requested():
        if trace:
            try:
                harness.export_trajectory_jsonl(trace)
                console.print(f"[bold green]✓ Session trajectory log saved to:[/bold green] {trace}")
            except Exception as ex:
                console.print(f"[bold red]Failed to save trajectory log:[/bold red] {ex}")

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
                    _export_trace_if_requested()
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
                    help_table.add_row("/tools", "List registered local Python tools.")
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
                    tool_table.add_column("Description")
                    for t in tools_info:
                        tool_table.add_row(t["name"], t["description"])
                    console.print(tool_table)

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
            console.print("\n[dim]Session interrupted. Type /exit to quit.[/dim]")
            _export_trace_if_requested()
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")


def main():
    cli()


if __name__ == "__main__":
    main()
