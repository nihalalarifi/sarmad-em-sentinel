"""
EM-Sentinel backend (Laptop 2).

    uvicorn server:app --host 0.0.0.0 --port 8000

  Simulator (Laptop 1) --I/Q--> ws://<laptop2>:8000/ws/sim
  Dashboard            <-JSON-- ws://<laptop2>:8000/ws
  REST: /api/health  /api/baseline  /api/metrics
  Pages: /  (dashboard)    /simulator  (signal simulator, open it on Laptop 1)
         /lab  (3D devices-and-signals lab; its mode buttons drive the simulator)

The backend is the only component that analyses signals. The simulator sends
raw samples only (no labels); the dashboard only displays what the backend says.
"""
import asyncio
import json
import os
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sarmad.engine import Engine
from sarmad.model import Detector

ROOT = Path(__file__).parent
MODEL = os.environ.get("SARMAD_MODEL", str(ROOT / "model" / "model.joblib"))

app = FastAPI(title="EM-Sentinel backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = Engine(Detector.load(MODEL), log_dir=str(ROOT / "logs"), metrics_path=str(ROOT / "eval" / "metrics.json"))
dashboards: set = set()
simulators: set = set()
stats = {"windows": 0, "sim_connected": 0}


def parse_frame(buf: bytes):
    """[uint32 LE header length][JSON header][pad to 4 bytes][float32 I,Q,I,Q,...]"""
    hl = struct.unpack_from("<I", buf, 0)[0]
    header = json.loads(buf[4:4 + hl].decode("utf-8"))
    off = 4 + hl
    off += (-off) % 4
    f = np.frombuffer(buf, dtype="<f4", offset=off)
    iq = f[0::2] + 1j * f[1::2]
    return header, iq.astype(np.complex64)


async def broadcast(msgs, targets=None):
    targets = dashboards if targets is None else targets
    dead = []
    for ws in list(targets):
        for m in msgs:
            try:
                await ws.send_text(json.dumps(m))
            except Exception:
                dead.append(ws)
                break
    for ws in dead:
        targets.discard(ws)


@app.get("/")
def dashboard_page():
    return FileResponse(ROOT / "dashboard" / "index.html")


@app.get("/simulator")
def simulator_page():
    return FileResponse(ROOT / "simulator" / "index.html")


@app.get("/lab")
def lab_page():
    return FileResponse(ROOT / "lab" / "index.html")


app.mount("/vendor", StaticFiles(directory=str(ROOT / "vendor")), name="vendor")


@app.get("/api/health")
def health():
    return {"ok": True, "windows_processed": stats["windows"], "simulators": len(simulators),
            "dashboards": len(dashboards), "threshold": engine.threshold, "persist": engine.persist,
            "decision_log": str(engine.log_path)}


@app.get("/api/baseline")
def baseline():
    return JSONResponse(engine.baseline_json())


@app.get("/api/metrics")
def metrics():
    return JSONResponse(engine.metrics_json())


@app.websocket("/ws/sim")
async def ws_sim(ws: WebSocket):
    await ws.accept()
    simulators.add(ws)
    await ws.send_text(json.dumps({"type": "config", "probe": engine.probe}))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                header, iq = parse_frame(msg["bytes"])
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") != "samples":
                    continue
                v = np.asarray(data["iq"], dtype=np.float32)
                header, iq = data, (v[0::2] + 1j * v[1::2]).astype(np.complex64)
            else:
                continue
            out = await asyncio.to_thread(engine.process, iq, header)
            stats["windows"] += 1
            await broadcast(out)
    except WebSocketDisconnect:
        pass
    finally:
        simulators.discard(ws)


@app.websocket("/ws")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    dashboards.add(ws)
    await ws.send_text(json.dumps({"type": "baseline", "data": engine.baseline_json()}))
    await ws.send_text(json.dumps({"type": "metrics", "data": engine.metrics_json()}))
    for a in reversed(engine.alerts[:50]):          # alerts raised before this dashboard connected
        await ws.send_text(json.dumps(a))
    try:
        while True:
            data = json.loads(await ws.receive_text())
            t = data.get("type")
            if t == "config":
                engine.configure(data)
                if "probe" in data:                      # tell the simulator which probe point is active
                    await broadcast([{"type": "config", "probe": data["probe"]}], simulators)
            elif t == "scenario":                         # sent by the 3D lab: switch the simulator
                await broadcast([{"type": "scenario", "scenario": str(data.get("scenario", "idle"))}], simulators)
            elif t == "ack_all":
                await broadcast(engine.ack_all())
            elif t == "recalibrate":
                engine.start_recalibration(data.get("state", "idle"), data.get("duration_s", 60))
    except (WebSocketDisconnect, json.JSONDecodeError):
        pass
    finally:
        dashboards.discard(ws)
