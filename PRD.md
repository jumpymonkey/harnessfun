# Product Requirement Document (PRD): `harnessfun`

**Project Name:** `harnessfun`  
**Version:** 0.1.0  
**Status:** Draft / Approved for Phase 1  
**Author:** Pair Programming Agent  
**Location:** `/home/jeffleinen/jeffdev/harnessfun`  

---

## 1. Executive Summary & Goals

`harnessfun` is a modular, provider-configurable LLM Execution Harness engineered to run multi-step agentic tasks, function calling, state management, and tool execution. 

The primary objective of this project is to provide a clean, production-ready execution runtime that natively integrates with **Google Cloud Platform (GCP)** infrastructure. It allows users to seamlessly switch between various **Google Gemini models** while strictly enforcing **Google Cloud authentication (Application Default Credentials / Vertex AI IAM)**—prohibiting raw API key usage.

---

## 2. Core Requirements & Constraints

### 2.1 Authentication & Authorization (Mandatory Rule)
* **GCP Credentials Only:** All access to Gemini models MUST be authenticated using **Google Cloud Project credentials**.
* **Supported Auth Methods:**
  * Application Default Credentials (ADC) via `gcloud auth application-default login`
  * Google Cloud Service Account Keys (`GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`)
  * Workload Identity / GCP Managed Service Identities (Compute Engine, GKE, Cloud Run)
* **Prohibited Auth:** **API Keys (`GEMINI_API_KEY`) are strictly prohibited** and disallowed in configuration validation.
* **GCP Project Configuration:** Must require `GOOGLE_CLOUD_PROJECT` (GCP Project ID) and `GOOGLE_CLOUD_LOCATION` (e.g., `us-central1`).

### 2.2 Model Configuration & Flexibility
* **Dynamic Model Support:** The harness accepts **any valid Google Gemini model identifier** string (e.g., `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`, fine-tuned endpoints, experimental models, or preview models) without restricting users to a hardcoded whitelist.
* **Dynamic Model Discovery:** The harness provider layer supports dynamically querying and listing all available models directly from the GCP Vertex AI / Google GenAI API (`client.models.list()`), allowing users to discover and inspect supported Gemini models in their active GCP project/location.
* **Configurable Default Model:** If no model is specified by the user, the harness falls back to a user-configurable default model (e.g., `gemini-2.5-flash`).
* **Model Parameters:** Configurable `temperature`, `top_p`, `top_k`, and `max_output_tokens` per model invocation.

---

## 3. Architecture & Functional Components

```
                          ┌───────────────────────────────┐
                          │   harnessfun CLI / Driver     │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  Universal Execution Harness  │
                          │   (State, Memory & Loop)     │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │      GCP Gemini Adapter       │
                          │  (Vertex AI / ADC Auth Only)  │
                          └───────────────┬───────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
     ┌───────────────────────┐                         ┌───────────────────────┐
     │  gcloud ADC / IAM     │                         │  Google Cloud Project │
     │  (No API Keys Allowed)│                         │  (Project ID & Region)│
     └───────────────────────┘                         └───────────────────────┘
```

### 3.1 Component Breakdown

#### A. Configuration & Auth Validator (`harnessfun.config`)
* Loads configuration from YAML config file or environment variables.
* Validates that a valid GCP Project ID is set (`GOOGLE_CLOUD_PROJECT`).
* Verifies ADC credentials exist or prompts the user with `gcloud` setup instructions if missing.
* Rejects any attempt to pass or load plain API keys.

#### B. Provider Adapter Layer (`harnessfun.providers`)
* Implements a generic `BaseLLMProvider` interface to ensure future expansion capabilities.
* **`GCPGeminiProvider`:** Uses the official `google-genai` SDK initialized in Vertex AI / GCP mode (`vertexai=True`, `project=...`, `location=...`).

#### C. Tool Registry (`harnessfun.tools`)
* Decorator-based system (`@registry.register`) to turn standard Python functions into Gemini-compatible tools.
* Automatically handles tool execution and returns formatted responses back to the harness loop.

#### D. Execution Loop Controller (`harnessfun.harness`)
* Manages conversation state (`user`, `assistant`, `tool` messages).
* Controls multi-step execution loops (`while not done`).
* Implements safety limits (`max_steps=10`) to prevent infinite recursion.

---

## 4. User Interface & CLI Specifications

`harnessfun` provides both a one-shot execution mode and a feature-rich **Interactive REPL (Read-Eval-Print Loop) Session**.

### 4.1 One-Shot CLI Commands
```bash
# Verify GCP ADC authentication and project configuration
harnessfun auth check

# Query available models in the active GCP project/region
harnessfun models list

# One-shot prompt execution
harnessfun run "What is the weather in Tokyo?" --model gemini-2.5-flash
```

### 4.2 Interactive REPL Mode (`harnessfun chat` / `harnessfun interactive`)
Launching `harnessfun chat` enters an interactive terminal session where conversation history, model configuration, and tools persist across turns.

```text
$ harnessfun chat --model gemini-2.5-flash

╭─────────────────────────────────────────────────────────────╮
│ harnessfun interactive session (GCP Project: my-project)    │
│ Active Model: gemini-2.5-flash                              │
│ Type /help for available commands or /exit to quit.        │
╰─────────────────────────────────────────────────────────────╯

harnessfun [gemini-2.5-flash] > What's the capital of France?
Paris

harnessfun [gemini-2.5-flash] > /model gemini-2.5-pro
✓ Active model switched to: gemini-2.5-pro

harnessfun [gemini-2.5-pro] > Write a complex Python script...
```

### 4.3 REPL Slash Commands

The interactive session supports the following on-the-fly slash commands:

| Command | Usage | Description |
| :--- | :--- | :--- |
| `/model [model_id]` | `/model gemini-2.5-pro` | View current model or switch models on the fly without losing conversation context. |
| `/models` | `/models` | Query GCP Vertex AI API (`client.models.list()`) and display available models. |
| `/clear` | `/clear` | Reset session message history while retaining tool registration and settings. |
| `/history` | `/history` | Display current turn history and token usage summary. |
| `/system [prompt]`| `/system You are a Senior DevOps Engineer` | View or update system instructions for subsequent turns. |
| `/tools` | `/tools` | List currently registered tools available to the harness. |
| `/info` | `/info` | Display active GCP Project ID, Location, Auth type, and Session status. |
| `/help` | `/help` | Display list of interactive commands and shortcuts. |
| `/exit` / `/quit` | `/exit` | Terminate the interactive REPL session cleanly. |

---

## 5. Directory Structure Plan

```text
/home/jeffleinen/jeffdev/harnessfun/
├── PRD.md                       # Product Requirement Document
├── README.md                    # Setup and usage guide
├── pyproject.toml               # Poetry/pip dependencies
├── config.example.yaml          # Sample configuration file
├── harnessfun/
│   ├── __init__.py
│   ├── cli.py                   # Command Line Interface
│   ├── config.py                # GCP Auth & Settings Loader
│   ├── harness.py               # Core Execution Loop
│   ├── tools.py                 # Tool Registry & Execution
│   ├── models.py                # Normalized Data Models
│   └── providers/
│       ├── __init__.py
│       ├── base.py              # Base Provider Interface
│       └── gcp_gemini.py        # GCP / Vertex AI Gemini Adapter
└── tests/
    └── test_harness.py          # Unit & Integration Tests
```

---

## 6. Success Criteria & Milestones

| Milestone | Deliverable | Target |
| :--- | :--- | :--- |
| **M1: PRD & Setup** | Folder structure, `PRD.md`, and Python dependencies setup. | Completed |
| **M2: GCP Auth & Config** | Configuration module validating ADC/GCP project without API keys. | Pending |
| **M3: GCP Gemini Adapter** | Vertex AI / ADC `GCPGeminiProvider` supporting dynamic discovery and execution of any available Gemini model ID. | Pending |
| **M4: Tool Registry & Loop** | Multi-step tool execution loop with recursion protection. | Pending |
| **M5: CLI & Testing** | Interactive `harnessfun` CLI tool verified against GCP environment. | Pending |
