# harnessfun

`harnessfun` is a provider-configurable LLM Execution Harness designed to run multi-step agentic workflows and tool execution, powered natively by **Google Cloud Platform (GCP)** and **Google Gemini** models via Vertex AI / Application Default Credentials (ADC).

> ⚠️ **Authentication Note:** `harnessfun` strictly requires a Google Cloud Project with Application Default Credentials (`gcloud auth application-default login`) or Service Account credentials. Plain API keys are intentionally disallowed.

---

## Features

- **Google Cloud Native Authentication:** Uses GCP ADC and Project IAM roles rather than static API keys.
- **Dynamic Model Support & Discovery:** Specify any available Gemini model ID dynamically (including experimental, preview, or fine-tuned endpoints) with built-in API model discovery (`client.models.list()`).
- **Interactive REPL Session:** Chat interactively in a persistent terminal session (`harnessfun chat`) with on-the-fly model switching (`/model`), history management (`/clear`, `/history`), tool inspection (`/tools`), and custom system prompts (`/system`).
- **Tool Registry & Execution Loop:** Decorator-based tool registration (`@registry.register`) with automatic tool calling loop management.
- **Provider-Agnostic Architecture:** Built with a generic `BaseLLMProvider` interface to support multi-provider adapters in the future.

---

## Quick Setup

### 1. Authenticate with Google Cloud
Ensure you have the `gcloud` CLI installed and authenticated:
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
```

### 2. Set Environment Variables
```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### 3. Documentation
See [PRD.md](file:///home/jeffleinen/jeffdev/harnessfun/PRD.md) for full project specifications and architecture details.
