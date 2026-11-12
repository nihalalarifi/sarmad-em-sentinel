"""
Command-line signal simulator (backup for the browser simulator, same framing).

    python sim_client.py --url ws://192.168.1.15:8000/ws/sim --mode demo
    python sim_client.py --mode random --minutes 10        # evaluation run

Sends raw I/Q only. Ground truth is written locally to logs/truth-*.csv
so that score_run.py can measure the live run afterwards.
"""
import argparse
import asyncio
import csv
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import websockets

from sarmad import signals as S

DEMO = [("idle", 15), ("active", 15), ("covert_fast", 15), ("idle", 15)]


def frame(header, iq):
    h = json.dumps(header).encode()
    pad = (-(4 + len(h))) % 4
    data = np.empty(2 * len(iq), dtype="<f4")
    data[0::2], data[1::2] = iq.real, iq.imag
    return struct.pack("<I", len(h)) + h + b" " * pad + data.tobytes()


def schedule(mode, rng):
    if mode == "demo":
        while True:
            for sc, d in DEMO:
                yield sc, d
    elif mode == "random":
        while True:
            sc = rng.choice(S.ALL_SCENARIOS, p=[.16, .16, .14, .08, .14, .16, .16])
            yield sc, int(rng.integers(10, 21))
    else:
        while True:
            yield mode, 10**9


async def run(a):
    rng = np.random.default_rng(a.seed)
    Path("logs").mkdir(exist_ok=True)
    truth_path = Path("logs") / f"truth-{datetime.now():%Y%m%d-%H%M%S}.csv"
    probe = {"v": "CPU"}
    override = {"v": None}     # scenario chosen from the 3D lab
    async with websockets.connect(a.url, max_size=None) as ws:
        async def listen():
            async for m in ws:
                d = json.loads(m)
                if d.get("type") == "config" and "probe" in d:
                    probe["v"] = d["probe"]
                    print(f"  probe -> {probe['v']}")
                if d.get("type") == "scenario" and d.get("scenario") in S.ALL_SCENARIOS:
                    override["v"] = d["scenario"]
                    print(f"  scenario set by the lab -> {override['v']}")
        lt = asyncio.create_task(listen())
        end = time.time() + a.minutes * 60 if a.minutes else None
        seq = 0
        with open(truth_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seq", "ts", "scenario", "label", "snr_db", "probe"])
            for sc, dur in schedule(a.mode, rng):
                print(f"[{datetime.now():%H:%M:%S}] scenario {sc} for {dur}s")
                for _ in range(int(dur * a.rate)):
                    if end and time.time() > end:
                        lt.cancel()
                        print(f"done. ground truth: {truth_path}")
                        return
                    t0 = time.time()
                    seq += 1
                    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                    sc_now = override["v"] or sc
                    iq = S.window(sc_now, rng, a.snr, probe["v"])
                    hdr = {"type": "samples", "seq": seq, "ts": ts, "fs": S.FS, "center_mhz": S.CENTER_MHZ,
                           "n": len(iq), "probe": probe["v"], "format": "f32_iq"}
                    await ws.send(frame(hdr, iq))
                    w.writerow([seq, ts, sc_now, "anomaly" if S.is_anomaly(sc_now) else "normal", a.snr, probe["v"]])
                    fh.flush()
                    await asyncio.sleep(max(0, 1 / a.rate - (time.time() - t0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8000/ws/sim")
    p.add_argument("--mode", default="demo", help="demo | random | " + " | ".join(S.ALL_SCENARIOS))
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--rate", type=float, default=1.0, help="windows per second")
    p.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run forever)")
    p.add_argument("--seed", type=int, default=2026)
    asyncio.run(run(p.parse_args()))
