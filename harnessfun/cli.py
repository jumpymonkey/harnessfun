"""Rich & Click Powered CLI & Interactive REPL for harnessfun."""

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
from harnessfun.providers.gcp_gemini import GCPGeminiProvider
from harnessfun.tools import default_registry

console = Console()


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
@click.option("--model-location", help="Location used to reach Gemini models (default: global).")
def models_list(project, location, model_location):
    """List all available Gemini models in the active GCP project/region."""
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location or location
        )
        provider = GCPGeminiProvider(project_id=cfg.project_id, location=cfg.model_location)
        
        console.print(f"[bold blue]Querying Vertex AI models for project '{cfg.project_id}' ({cfg.model_location})...[/bold blue]")
        models = provider.list_models()
        
        table = Table(title=f"Available Gemini Models ({cfg.project_id} / {cfg.model_location})")
        table.add_column("Model Identifier", style="cyan")
        table.add_column("Status", style="green")
        
        for m in models:
            table.add_row(m, "Available")
            
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to fetch models:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("prompt")
@click.option("--model", "-m", help="Gemini model ID to use.")
@click.option("--project", "-p", help="GCP Project ID.")
@click.option("--location", "-l", help="GCP Location/Region.")
@click.option("--model-location", help="Location used to reach Gemini models (default: global).")
def run(prompt, model, project, location, model_location):
    """Execute a single one-shot prompt."""
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location,
            model=model
        )
        provider = GCPGeminiProvider(project_id=cfg.project_id, location=cfg.model_location)
        harness = UniversalHarness(provider=provider, config=cfg)
        
        console.print(f"[dim]Running model '{cfg.active_model}' on project '{cfg.project_id}'...[/dim]\n")
        response = harness.run_turn(prompt)
        console.print(Panel(response, title=f"harnessfun [{cfg.active_model}]", border_style="blue"))
    except Exception as e:
        console.print(f"[bold red]Execution Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", help="Initial Gemini model ID to use.")
@click.option("--project", "-p", help="GCP Project ID.")
@click.option("--location", "-l", help="GCP Location/Region.")
@click.option("--model-location", help="Location used to reach Gemini models (default: global).")
def chat(model, project, location, model_location):
    """Start an interactive REPL session with on-the-fly model switching."""
    try:
        cfg = load_config(
            project_id=project,
            location=location,
            model_location=model_location,
            model=model
        )
        provider = GCPGeminiProvider(project_id=cfg.project_id, location=cfg.model_location)
        harness = UniversalHarness(provider=provider, config=cfg, registry=default_registry)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize harness session:[/bold red] {e}")
        sys.exit(1)

    # REPL Welcome Banner
    banner = (
        f"[bold green]harnessfun Interactive REPL v{__version__}[/bold green]\n"
        f"[bold]GCP Project:[/bold] {cfg.project_id} | [bold]Region:[/bold] {cfg.location}\n"
        f"[bold]Active Model:[/bold] {cfg.active_model}\n"
        f"[dim]Type [bold]/help[/bold] for available commands or [bold]/exit[/bold] to quit.[/dim]"
    )
    console.print(Panel(banner, title="Session Started", border_style="cyan"))

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
                    break

                elif command == "/help":
                    help_table = Table(title="Interactive Slash Commands")
                    help_table.add_column("Command", style="cyan")
                    help_table.add_column("Description")
                    help_table.add_row("/model <id>", "View or switch active Gemini model on the fly.")
                    help_table.add_row("/models", "List available Gemini models from GCP API.")
                    help_table.add_row("/clear", "Reset conversation history.")
                    help_table.add_row("/history", "View message history turn count.")
                    help_table.add_row("/system <prompt>", "View or update system instructions.")
                    help_table.add_row("/tools", "List registered local Python tools.")
                    help_table.add_row("/info", "Display active GCP session information.")
                    help_table.add_row("/help", "Show this help table.")
                    help_table.add_row("/exit, /quit", "Exit REPL session.")
                    console.print(help_table)

                elif command == "/model":
                    if arg:
                        harness.set_model(arg)
                        console.print(f"[bold green]✓ Active model switched to:[/bold green] {arg}")
                    else:
                        console.print(f"Active Model: [bold cyan]{harness.config.active_model}[/bold cyan]")

                elif command == "/models":
                    console.print("[dim]Fetching available models from GCP Vertex AI...[/dim]")
                    models = provider.list_models()
                    console.print("[bold]Available Models:[/bold] " + ", ".join(models))

                elif command == "/clear":
                    harness.clear()
                    console.print("[bold green]✓ Conversation history cleared.[/bold green]")

                elif command == "/history":
                    turn_count = len([m for m in harness.messages if m.role == "user"])
                    console.print(f"Conversation Turns: [bold]{turn_count}[/bold] | Total Messages: [bold]{len(harness.messages)}[/bold]")

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
                    console.print(
                        f"[bold]Project ID:[/bold] {cfg.project_id}\n"
                        f"[bold]Location:[/bold] {cfg.location}\n"
                        f"[bold]Model Location:[/bold] {cfg.model_location}\n"
                        f"[bold]Active Model:[/bold] {harness.config.active_model}\n"
                        f"[bold]Max Steps:[/bold] {cfg.max_steps}"
                    )

                else:
                    console.print(f"[bold red]Unknown command:[/bold red] '{command}'. Type /help for available options.")

            else:
                # Regular prompt turn
                with console.status(f"[dim]Executing turn with {harness.config.active_model}...[/dim]"):
                    response = harness.run_turn(user_input)
                console.print(Panel(response, title=f"harnessfun [{harness.config.active_model}]", border_style="blue"))

        except KeyboardInterrupt:
            console.print("\n[dim]Session interrupted. Type /exit to quit.[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")


def main():
    cli()


if __name__ == "__main__":
    main()
