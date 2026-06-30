# gateway.py
import csv
import datetime
import os
import threading
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import requests
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_csv_log()
    print(f"[Gateway] CSV log ready at {CSV_LOG_PATH}")
    yield


app = FastAPI(title="Smart Mall Unified API Gateway", version="1.0", lifespan=lifespan)


class EventData(BaseModel):
    # parking fields
    occupied_spots: Optional[List[int]] = None
    occupied_count: Optional[int] = None
    total_spots: Optional[int] = None
    free_count: Optional[int] = None
    newly_occupied: Optional[List[int]] = None
    newly_vacated: Optional[List[int]] = None
    # crowd fields
    count: Optional[int] = None
    level: Optional[str] = None
    lighting: Optional[str] = None
    hvac: Optional[str] = None
    # suspicious fields
    label: Optional[str] = None
    confidence: Optional[float] = None
    consecutive_windows: Optional[int] = None


class EventPayload(BaseModel):
    source: str
    event_type: str
    data: EventData


# ── YOUR LIVE N8N WEBHOOK URL ──
N8N_WEBHOOK_URL = ""

# ── PERSISTENT EVENT LOG ──
CSV_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mall_events_2.csv")

CSV_FIELDS = [
    "timestamp",
    "source",
    "event_type",
    # parking fields
    "occupied_count",
    "total_spots",
    "free_count",
    "occupied_spots",
    "newly_occupied",
    "newly_vacated",
    # crowd fields
    "crowd_count",
    "crowd_level",
    "lighting",
    "hvac",
    # suspicious activity fields
    "suspicious_label",
    "suspicious_confidence",
    "suspicious_consecutive_windows",
]

_csv_lock = threading.Lock()


def _init_csv_log():
    """Create the CSV with a header row if it doesn't exist yet."""
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_to_csv(payload: dict):
    """Background task: flatten the event payload and append it as a CSV row."""
    try:
        data = payload.get("data", {})
        event_type = payload.get("event_type", "")

        row = {field: "" for field in CSV_FIELDS}
        row["timestamp"] = payload.get("timestamp")
        row["source"] = payload.get("source")
        row["event_type"] = event_type

        if event_type in ("state_change", "parking"):
            row["occupied_count"] = data.get("occupied_count")
            row["total_spots"] = data.get("total_spots")
            row["free_count"] = data.get("free_count")
            row["occupied_spots"] = ";".join(map(str, data.get("occupied_spots") or []))
            row["newly_occupied"] = ";".join(map(str, data.get("newly_occupied") or []))
            row["newly_vacated"] = ";".join(map(str, data.get("newly_vacated") or []))
        elif event_type == "heartbeat":
            row["crowd_count"] = data.get("count")
            row["crowd_level"] = data.get("level")
            row["lighting"] = data.get("lighting")
            row["hvac"] = data.get("hvac")
        elif event_type == "violence_alert":
            row["suspicious_label"] = data.get("label")
            row["suspicious_confidence"] = data.get("confidence")
            row["suspicious_consecutive_windows"] = data.get("consecutive_windows")

        with _csv_lock:
            with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(row)
        print(f"[CSV Log] Row appended -> {CSV_LOG_PATH}")
    except Exception as e:
        print(f"[CSV Log] Failed to write row: {e}")


def forward_to_n8n(payload: dict):
    """Background task to push data to n8n without blocking FastAPI's responses."""
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
        print(f"[n8n Forward] Status: {response.status_code}")
    except Exception as e:
        print(f"[n8n Forward] Failed to connect: {e}")


@app.post("/api/v1/events")
async def receive_event(event: EventPayload, background_tasks: BackgroundTasks):
    full_payload = event.model_dump()

    full_payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    event_type = full_payload["event_type"]
    data = full_payload["data"]

    print(f"\n[Gateway] Received '{event_type}' from '{full_payload['source']}'")

    if event_type == "violence_alert":
        print(f"          ALERT: {data['label']} (confidence={data['confidence']:.2f}, "
              f"consecutive_windows={data['consecutive_windows']})")
    elif event_type == "heartbeat":
        print(f"          Crowd: {data['count']} people | Level: {data['level']}")
    else:
        print(f"          Occupied: {data['occupied_count']} | Free: {data['free_count']}")

    background_tasks.add_task(append_to_csv, full_payload)
    background_tasks.add_task(forward_to_n8n, full_payload)

    return {"status": "accepted", "timestamp": full_payload["timestamp"]}


if __name__ == "__main__":
    uvicorn.run("full_api:app", host="127.0.0.1", port=8000, reload=True)