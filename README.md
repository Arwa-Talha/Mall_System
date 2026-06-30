# Mall_System
Here is a highly polished, professional, and production-ready `README.md` for your **Smart Mall Management & Monitoring System** repository, tailored to reflect advanced multi-agent orchestration and scalable AI architecture.

---

# Smart Mall Management & Monitoring System (v2.0)

An enterprise-grade, multi-agent computer vision ingestion and automated orchestration platform designed for real-time surveillance across smart retail environments. The system continuously processes distributed edge telemetry across three core operational sectors—**Crowd Dynamics**, **Smart Parking**, and **Security/Violence Alerts**—ingesting event logs through a high-performance Unified API Gateway and driving an asynchronous, multi-level incident escalation matrix via n8n and Telegram.

---

## 🏗️ Core Architecture Overview

The system architecture decouples analytical edge modules from orchestration logic to ensure maximum throughput and sub-second alert dispatching:

```
┌────────────────┐     ┌────────────────┐     ┌─────────────────┐
│ Parking Module │     │  Crowd Module  │     │ Violence Module │
└───────┬────────┘     └───────┬────────┘     └────────┬────────┘
        │                      │                       │
        └───────────────┐      │      ┌────────────────┘
                        ▼      ▼      ▼
              ┌─────────────────────────────────┐
              │ Unified API Gateway (FastAPI)   │ ──► [Relational SQLite DB]
              └────────────────┬────────────────┘
                               │ (Resilient Webhook Forwarding)
                               ▼
              ┌─────────────────────────────────┐
              │  n8n Multi-Agent Workflow       │
              └────────────────┬────────────────┘
                               │ (Dynamic Escalation Grid)
                               ▼
              ┌─────────────────────────────────┐
              │     Targeted Telegram Alerts    │
              └─────────────────────────────────┘

```

1. **AI Edge Analytics Layer (CV Modules):** Independent edge microservices optimizing advanced architectures (such as custom U-Net networks and YOLO deep architectures) to infer macro telemetry variables from real-time stream buffers.
2. **Unified API Gateway (`full_api.py`):** An asynchronous FastAPI integration layer managing highly concurrent thread pools utilizing SQLite WAL (Write-Ahead Logging) persistence. The gateway autonomously scales real-time data into unified, flat schema properties, classifies dynamic risk weights, and tracks alerts end-to-end.


3. **Orchestration Matrix Engine (`n8n Workflow`):** A centralized, rule-based automation engine mapping multi-branch decision pipelines. It handles instant delivery routing to the physically nearest active responder, evaluates acknowledgement status timeouts, and cascades back-off escalation notifications if a priority alert remains unaddressed.



---

## 🛠️ Technology Stack & Dependencies

* **Inference Ingestion & Backend API:** FastAPI, Uvicorn, Pydantic v2


* **Orchestration Server Engine:** n8n Workflow Automation Platform


* **Relational Storage Engine:** SQLite3 (Configured with WAL Mode concurrency)


* **Communication Interface:** Telegram Bot API Engine (Markdown Parsing)


* **Development Stack Ecosystem:** PyTorch Framework, VS Code IDE Environment

---

## 🚀 Key Enterprise Features

### ⚡ Zero-Block Asynchronous Ingestion & Persistence

The entry route (`/api/v1/events`) acts as a non-blocking consumer channel. Incoming frames pass telemetry configurations immediately to Python `BackgroundTasks`, keeping downstream processing times constant regardless of payload size:

* **Dual relational storage tracking:** Events are appended instantly to logs while active alerts populate a state-machine tracking database (`incidents` table).


* **Resilient Forwarding Queue:** Utilizes a dedicated background tracking thread (`queue.Queue`) coupled with exponential back-off retries to buffer n8n downstream connections against high network latency. Failed pushes are dumped into a crash-safe local audit log (`failed_forwards.jsonl`).



### ⚖️ Automated Incident Severity Classification

The core gateway engine systematically parses parameters via an analytical logic matrix to isolate environmental anomalies on the fly:

* **Violence Vectors:** Maps events directly to `CRITICAL` or `HIGH` whenever inference engine reliability meets a high metric probability combined with strict sequence-window validation bounds.


* **Crowd Congestion:** Translates spatial density metrics directly into macro control alerts to execute environmental optimizations.


* **Parking Logistics:** Dynamically adjusts metrics between `HIGH`, `MEDIUM`, and `LOW` based on inverse linear lot capacities.



### 🚨 Smart Escalation & Acknowledgement Matrix

During a security emergency, the workflow triggers a priority escalation route over a structured security shift roster:

* **Direct Guard Target Dispatching:** Employs explicit parameter variables to inject metadata (`incident_id`) directly into the target guard's session.


* **Time-Out Polling Verification:** n8n execution blocks for exactly 30 seconds inside a polling loop to verify the incident state change via gateway check routes.


* **Dynamic Cascade Escalation:** If an incident fails to change state within the timeout boundary, n8n invokes `/api/v1/guards/next` to systematically alert the next available tier on the team roster, ensuring immediate accountability.



---

## 📂 Production Code Architecture

```text
├── full_api.py                 # Production Unified Ingestion Gateway Engine (FastAPI & DB Logic)
├── gateway.py                  # Legacy Log Storage Interface (v1.0 Flat CSV System Architecture)
├── mall_events.db              # High-concurrency relational data warehouse storage file (SQLite WAL)
├── failed_forwards.jsonl       # High-availability network failover transaction log
└── Smart Mall Surveillance v2.json # Complex JSON Multi-Agent Orchestration n8n Blueprint Template

```

---

## 💻 System Ingestion Endpoints (`full_api.py`)

| HTTP Method | API Ingestion Route | Purpose | Access Category |
| --- | --- | --- | --- |
| **POST** | `/api/v1/events` | Ingests real-time computer vision telemetry payloads.

 | Edge Analytics Modules |
| **POST** | `/api/v1/acknowledge` | Processes guard "ACCEPT" response payloads coming from Telegram.

 | n8n Hook Interface |
| **GET** | `/api/v1/incidents/{incident_id}` | Returns verification status state-changes for an active tracking hash.

 | n8n Intercept Node |
| **GET** | `/api/v1/guards/next` | Fetches the next available security agent in the escalation path on timeout.

 | n8n Cascade Node |
| **GET** | `/health` | Evaluates system status parameters, database weights, and pipeline sizing.

 | Monitoring Console |

---

## 🔧 Installation & Operational Deployment

### 1. Webhook Topology Setup

Update your internal endpoint references within `full_api.py` to route safely to your active cloud network architecture:

```python
N8N_WEBHOOK_URL = "https://your-domain.n8n.cloud/webhook/your-unique-endpoint-id"

```

### 2. Environment Infrastructure Installation

Construct a localized python isolation container environment to isolate your system dependencies:

```bash
pip install fastapi uvicorn pydantic requests

```

### 3. Initialize the Gateway Environment

Fire up your ASGI worker process on the preferred target loopback address profile:

```bash
python full_api.py

```

Note: Bootstrapping automatically builds and applies underlying relational table systems and initiates asynchronous worker-queue threads.

### 4. Import the Orchestration Blueprint

1. Copy the raw layout configuration schema inside `Smart Mall Surveillance v2.json`.


2. Access your active **n8n orchestration canvas panel**, open the dropdown menu, and select **Import from JSON**.


3. Bind your active **Telegram Bot Token** string to clear the authorization nodes across the workflow paths.
