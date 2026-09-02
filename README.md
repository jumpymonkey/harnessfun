# harnessfun

`harnessfun` is a provider-configurable LLM Execution Harness and interactive REPL CLI designed to run multi-step agentic workflows and tool execution natively powered by **Google Cloud Platform (GCP)** and **Google Gemini** models via Vertex AI and Application Default Credentials (ADC).

> ⚠️ **Security & Authentication Note:** `harnessfun` strictly requires a Google Cloud Project with Application Default Credentials (`gcloud auth application-default login`) or Service Account credentials. Plain API keys (`GEMINI_API_KEY`) are intentionally prohibited.

---

## Key Features

- **GCP Native Identity & Governance:** Uses GCP ADC and Project IAM roles rather than static API keys.
- **Dynamic Model Support & Discovery:** Specify any available Gemini model ID dynamically (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-1.5-pro`, etc.) with built-in API model discovery (`client.models.list()`).
- **Interactive REPL Session:** Chat interactively in a persistent terminal session (`harnessfun chat`) with on-the-fly model switching (`/model`), history management (`/clear`, `/history`), tool inspection (`/tools`), and custom system prompts (`/system`).
- **Tool Registry & Execution Loop:** Decorator-based tool registration (`@registry.register`) with automatic multi-step tool calling loop management.
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

### 2. List Available Gemini Models
Query the GCP Vertex AI API directly to list active models in your project and region:
```bash
harnessfun models-list
```

---

### 3. Run a One-Shot Prompt
Execute a single query with the default model (`gemini-2.5-flash`) or specify a model:
```bash
# Using default model
harnessfun run "What is the weather in Tokyo?"

# Specifying a specific Gemini model
harnessfun run "Analyze this dataset and calculate the summary." --model gemini-2.5-pro
```

---

### 4. Interactive REPL Chat Session
Launch an interactive session (`harnessfun chat` or `harnessfun`):
```bash
harnessfun chat --model gemini-2.5-flash
```

**Session Transcript Example:**
```text
╭─────────────────────────────────────────────────────────────────╮
│ harnessfun Interactive REPL v0.1.0                              │
│ GCP Project: my-gcp-project-123 | Region: us-central1          │
│ Active Model: gemini-2.5-flash                                  │
│ Type /help for available commands or /exit to quit.            │
╰─────────────────────────────────────────────────────────────────╯

harnessfun [gemini-2.5-flash]> What's the capital of France?
Paris is the capital of France.

harnessfun [gemini-2.5-flash]> /model gemini-2.5-pro
✓ Active model switched to: gemini-2.5-pro

harnessfun [gemini-2.5-pro]> What is the current weather in Tokyo and what is 22C in Fahrenheit?
Thinking...
  -> Executing tool: get_weather({'location': 'Tokyo'})
  -> Executing tool: calculate({'expression': '(22 * 9/5) + 32'})

The current weather in Tokyo is 22°C and Sunny. 22°C converted to Fahrenheit is 71.6°F.

harnessfun [gemini-2.5-pro]> /clear
✓ Conversation history cleared.

harnessfun [gemini-2.5-pro]> /exit
Exiting interactive session. Goodbye!
```

---

### Interactive REPL Slash Commands

Inside `harnessfun chat`, the following slash commands are available:

| Command | Usage Example | Description |
| :--- | :--- | :--- |
| `/model [id]` | `/model gemini-2.5-pro` | View current model or switch active Gemini models on the fly. |
| `/models` | `/models` | Query Vertex AI API and list available Gemini model IDs. |
| `/system [prompt]` | `/system You are a Senior DevOps Engineer` | View or update system instructions dynamically. |
| `/clear` | `/clear` | Reset message history while keeping tools and model settings intact. |
| `/history` | `/history` | Display current turn history count and message statistics. |
| `/tools` | `/tools` | List currently registered Python tools and their signatures. |
| `/info` | `/info` | Display active GCP Project ID, location, model, and session settings. |
| `/help` | `/help` | Display interactive commands help table. |
| `/exit` / `/quit` | `/exit` | Exit the interactive session. |

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
