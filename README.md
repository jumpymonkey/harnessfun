# harnessfun

`harnessfun` is a provider-configurable LLM Execution Harness and interactive REPL CLI designed to run multi-step agentic workflows and tool execution natively powered by **Google Cloud Platform (GCP)**, **Google Gemini**, and **Anthropic Claude** models via Vertex AI and Application Default Credentials (ADC).

> ⚠️ **Security & Authentication Note:** `harnessfun` strictly requires a Google Cloud Project with Application Default Credentials (`gcloud auth application-default login`) or Service Account credentials. Static API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) are intentionally prohibited.

---

## Key Features

- **GCP Native Identity & Governance:** Uses GCP ADC and Project IAM roles rather than static API keys.
- **Multi-Model Provider Architecture:** Seamlessly switch between **Google Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-pro`, etc.) and **Anthropic Claude** (`claude-3-7-sonnet`, `claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-3-opus`, etc.) hosted in Google Cloud Vertex AI.
- **Zero-Key Anthropic Access via Vertex AI:** Connect to Anthropic Claude models using Google Cloud ADC OAuth tokens directly via Vertex AI's `publishers/anthropic` endpoints or the official SDK.
- **Interactive REPL Session:** Chat interactively in a persistent terminal session (`harnessfun chat`) with on-the-fly model switching (`/model`), history management (`/clear`, `/history`), tool inspection (`/tools`), and custom system prompts (`/system`).
- **Event-Driven Streaming Pipeline:** Granular step lifecycle events (`step_start`, `model_thought`, `tool_call`, `tool_result`, `turn_complete`, `error`) streamed in real-time.
- **Model Thought & Rationale Preservation:** Captures and displays chain-of-thought and reasoning rationale generated alongside tool calls.
- **Trajectory Export & Auditing:** Export full session traces to JSONL (`--trace <path>`, `/export <path>`, or `.export_trajectory_jsonl()`) for debugging, evaluation benchmarks, and deterministic replays.
- **Tool Registry & Execution Loop:** Decorator-based tool registration (`@registry.register`) with automatic multi-step tool calling loop management across both Gemini and Claude models.
- **Provider-Agnostic Architecture:** Built with a generic `BaseLLMProvider` interface to support multi-provider adapters.

---

## Installation & Setup


### 1. Prerequisites
- Python 3.10+
- Google Cloud SDK (`gcloud`) CLI installed

### 2. Authenticate with Google Cloud
Ensure you are logged in and have Application Default Credentials configured:
```bash
# Log in to gcloud CLI
gcloud auth login

# Log in for Application Default Credentials (ADC)
gcloud auth application-default login

# Set your default active GCP Project ID
gcloud config set project YOUR_GCP_PROJECT_ID
```

### 3. Install `harnessfun`
Clone the repository and install dependencies:
```bash
git clone https://github.com/jumpymonkey/harnessfun.git
cd harnessfun

# Install in editable mode
pip install -e .

# Or using uv:
uv pip install -e .
```

---

## Configuration

Set your GCP Project ID and Region via environment variables or a configuration file:

### Option A: Environment Variables
```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### Option B: `config.yaml` Configuration File
Create a `config.yaml` file in your working directory or project root:
```yaml
gcp:
  project_id: "your-gcp-project-id"
  location: "us-central1"

harness:
  default_model: "gemini-2.5-flash"
  model_location: "global"
  system_instruction: "You are a helpful assistant with access to local tools."
  max_steps: 10
```

---

## CLI Usage Examples

### 1. Verify GCP Authentication & Security Policies
Check that your GCP ADC credentials and Project ID are properly recognized:
```bash
harnessfun auth-check
```
**Output Example:**
```text
✓ Authentication Successful
Active GCP Project: my-gcp-project-123
Auth Method: Google Cloud ADC (No API Keys Used)
```

---

### 2. List Available Gemini & Anthropic Models
Query the GCP Vertex AI API directly to list active models in your project and region:
```bash
harnessfun models-list
```

---

### 3. Run a One-Shot Prompt
Execute a single query with the default Gemini model, a specific Gemini model, or an Anthropic Claude model:
```bash
# Using default model (gemini-2.5-flash)
harnessfun run "What is the weather in Tokyo?"

# Specifying a Gemini model
harnessfun run "Analyze this dataset." --model gemini-2.5-pro

# Specifying an Anthropic Claude model on Vertex AI
harnessfun run "Write a concise summary of distributed consensus." --model claude-3-5-sonnet

# Export trajectory trace to JSONL
harnessfun run "What is 15 * 87?" --trace trace.jsonl
```

> **Note on Anthropic Models:** Anthropic Claude models on Vertex AI run in regional endpoints (defaults to `us-east5`). Ensure the model is enabled in your Google Cloud Console Model Garden. You can customize the region with `--model-location europe-west1` or in `config.yaml`.

---

### 4. Interactive REPL Chat Session
Launch an interactive session (`harnessfun chat` or `harnessfun`):
```bash
# Launch with Gemini
harnessfun chat --model gemini-2.5-flash

# Or launch directly with Claude
harnessfun chat --model claude-3-5-sonnet

# Launch with session trajectory recording on exit
harnessfun chat --trace session_trace.jsonl
```

**Session Transcript Example:**
```text
╭─────────────────────────────────────────────────────────────────╮
│ harnessfun Interactive REPL v0.1.0                              │
│ GCP Project: my-gcp-project-123 | Region: us-central1           │
│ Active Model: gemini-2.5-flash                                  │
│ Type /help for available commands or /exit to quit.            │
╰─────────────────────────────────────────────────────────────────╯

harnessfun [gemini-2.5-flash]> What's the capital of France?
Paris is the capital of France.

harnessfun [gemini-2.5-flash]> /model gemini-2.5-pro
✓ Active model switched to: gemini-2.5-pro

harnessfun [gemini-2.5-pro]> What is the current weather in Tokyo and what is 22C in Fahrenheit?
── Step 1/10 ──
 ⚡ Tool Call: get_weather args={"location": "Tokyo"}
 ✓ Tool Output (get_weather): {"temperature": 22, "condition": "Sunny"}
── Step 2/10 ──
 ⚡ Tool Call: calculate args={"expression": "(22 * 9/5) + 32"}
 ✓ Tool Output (calculate): {"result": 71.6}
╭─ harnessfun [gemini-2.5-pro] ───────────────────────────────────╮
│ The current weather in Tokyo is 22°C and Sunny. 22°C converted   │
│ to Fahrenheit is 71.6°F.                                        │
╰─────────────────────────────────────────────────────────────────╯

harnessfun [gemini-2.5-pro]> /events
╭─ Session Trajectory Events (5 total) ───────────────────────────╮
│ #  │ Step │ Type          │ Details                             │
│ 1  │ 1    │ step_start    │ Starting step 1/10                  │
│ 2  │ 1    │ tool_call     │ get_weather({"location": "Tokyo"})  │
│ 3  │ 1    │ tool_result   │ get_weather -> {"temperature": 22}  │
│ 4  │ 2    │ tool_call     │ calculate({"expression": "..."})    │
│ 5  │ 2    │ turn_complete │ The current weather in Tokyo...     │
╰─────────────────────────────────────────────────────────────────╯

harnessfun [gemini-2.5-pro]> /export my_trajectory.jsonl
✓ Trajectory successfully exported to: my_trajectory.jsonl

harnessfun [gemini-2.5-pro]> /clear
✓ Conversation history and trajectory cleared.

harnessfun [gemini-2.5-pro]> /exit
Exiting interactive session. Goodbye!
```

---

### Interactive REPL Slash Commands

Inside `harnessfun chat`, the following slash commands are available:

| Command | Usage Example | Description |
| :--- | :--- | :--- |
| `/model [id]` | `/model gemini-2.5-pro` | View current model or switch active Gemini or Claude models on the fly. |
| `/models` | `/models` | Query Vertex AI API and list available Gemini and Claude model IDs. |
| `/system [prompt]` | `/system You are a Senior DevOps Engineer` | View or update system instructions dynamically. |
| `/clear` | `/clear` | Reset message history and recorded trajectory events. |
| `/history` | `/history` | Display current turn history count and message statistics. |
| `/events` / `/trajectory` | `/events` | Inspect granular step-by-step trajectory events recorded in the session. |
| `/export <path>` | `/export log.jsonl` | Export session trajectory events to a JSONL log file. |
| `/tools` | `/tools` | List currently registered Python tools and their signatures. |
| `/info` | `/info` | Display active GCP Project ID, location, model, and session settings. |
| `/help` | `/help` | Display interactive commands help table. |
| `/exit` / `/quit` | `/exit` | Exit the interactive session (exports trajectory if `--trace` was provided). |


---

## Adding Custom Python Tools

You can easily register custom Python functions as tools using `@registry.register`:

```python
from harnessfun.harness import UniversalHarness
from harnessfun.providers.gcp_gemini import GCPGeminiProvider
from harnessfun.config import load_config
from harnessfun.tools import default_registry

# Register a custom tool
@default_registry.register
def query_database(user_id: str) -> dict:
    """Queries user record from local database."""
    return {"user_id": user_id, "status": "active", "tier": "premium"}

# Initialize harness
cfg = load_config(project_id="my-gcp-project")
provider = GCPGeminiProvider(project_id=cfg.project_id)
harness = UniversalHarness(provider=provider, config=cfg, registry=default_registry)

# Run query requiring tool call
response = harness.run_turn("Check the database status for user ID 1042.")
print(response)
```

---

## Running Unit Tests

Run the test suite using `pytest`:
```bash
pytest
```

---

## Documentation

- **[PRD.md](file:///home/jeffleinen/jeffdev/harnessfun/PRD.md)**: Product Requirement Document
- **[ARCHITECTURE.md](file:///home/jeffleinen/jeffdev/harnessfun/ARCHITECTURE.md)**: System Architecture Design Document
