"""
Live processing engine: one call per I/Q window from the simulator (or a real SDR).

Produces the messages the dashboard expects (see dashboard contract):
  {"type":"window", ...}   every window
  {"type":"alert", ...}    when a deviation persists for `persist` windows
  {"type":"baseline", ...} after a recalibration
"""
import csv
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .features import extract, model_vector, resample_trace
from .signals import FS, N, CENTER_MHZ


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Engine:
    def __init__(self, detector, threshold=0.70, persist=3, log_dir="logs", metrics_path="eval/metrics.json"):
        self.det = detector
        self.threshold = float(threshold)
        self.persist = int(persist)
        self.probe = "CPU"
        self.count = 0
        self.alert_seq = 0
        self.active_alert = None      # one alert per incident
        self.normal_streak = 0
        self.cooldown = 10            # normal windows needed before a new incident can start
        self.alerts = []
        self.scores = deque(maxlen=600)
        self.recal = None
        self.last_calibrated = datetime.now().strftime("%H:%M")
        self.metrics_path = Path(metrics_path)
        self.lock = threading.Lock()
        Path(log_dir).mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = Path(log_dir) / f"decisions-{stamp}.csv"
        self._log = open(self.log_path, "w", newline="")
        self._csv = csv.writer(self._log)
        self._csv.writerow(["seq", "ts", "score", "status", "state", "alert_id"])

    # ---------------------------------------------------------------- config
    def configure(self, msg):
        with self.lock:
            if "threshold" in msg:
                self.threshold = float(msg["threshold"])
            if "persist" in msg:
                self.persist = max(1, int(msg["persist"]))
            if "probe" in msg:
                self.probe = str(msg["probe"])

    def ack_all(self):
        with self.lock:
            changed = [a for a in self.alerts if a["status"] == "open"]
            for a in changed:
                a["status"] = "ackd"
            return [dict(a) for a in changed]

    def start_recalibration(self, state, duration_s):
        with self.lock:
            self.recal = {"state": state, "need": max(5, int(duration_s)), "X": [], "feats": [], "fft": [], "env": []}

    # --------------------------------------------------------------- process
    def process(self, iq, header):
        with self.lock:
            return self._process(iq, header)

    def _process(self, iq, header):
        out = []
        seq = header.get("seq")
        ts = header.get("ts") or _now_iso()
        fs = float(header.get("fs", FS))
        feats, disp = extract(np.asarray(iq, dtype=np.complex64), self.det.track_hz)
        vec = model_vector(feats)
        score = self.det.score(vec)
        state = self.det.state(vec)
        self.scores.append(score)

        # recalibration capture (operator guarantees the device is in that normal state)
        if self.recal is not None:
            r = self.recal
            r["X"].append(vec); r["feats"].append(feats); r["fft"].append(disp["fft_db"]); r["env"].append(resample_trace(disp["envelope"]).astype("float32"))
            if len(r["X"]) >= r["need"]:
                self.det.recalibrate(r["state"], np.array(r["X"]), r["feats"], np.array(r["fft"]), r["env"])
                self.recal = None
                self.last_calibrated = datetime.now().strftime("%H:%M")
                out.append({"type": "baseline", "data": self.baseline_json()})

        # persistence: alert only when the deviation lasts `persist` consecutive windows
        self.count = self.count + 1 if score > self.threshold else 0
        status = "alert" if self.count >= self.persist else ("warning" if self.count > 0 else "normal")
        features, reason = self.det.explain(feats, state)
        st = self.det.state_stats.get(state) or next(iter(self.det.state_stats.values()))

        env = np.clip(disp["envelope"], 0, 1.2)
        out.insert(0, {
            "type": "window", "seq": seq, "ts": ts, "device": header.get("device", "WS-LAB-03"),
            "probe": header.get("probe", self.probe), "state": state, "status": status,
            "score": round(score, 3), "threshold": self.threshold,
            "persist": {"count": min(self.count, self.persist), "required": self.persist},
            "center_mhz": float(header.get("center_mhz", CENTER_MHZ)), "fs_msps": fs / 1e6,
            "rms_dbm": round(feats["rms_dbm"], 1), "features": features,
            "waveform": {"span_ms": round(len(iq) / fs * 1000, 2), "ymin": 0, "ymax": 1.2,
                         "ytitle": "Carrier envelope (norm.)",
                         "cur": np.round(resample_trace(env), 3).tolist(),
                         "base": np.round(np.clip(st["envelope"], 0, 1.2), 3).tolist()},
            "fft": {"span_khz": fs / 1e3, "cur": np.round(disp["fft_db"], 1).tolist(),
                    "base": np.round(st["fft_db"], 1).tolist()},
            "spec_column": np.round(disp["spec_db"], 1).tolist(),
            "reason": reason if status != "normal" else None,
            "symbol_rate_hz": round(feats["symbol_rate_hz"], 1),
        })

        alert_id = ""
        self.normal_streak = self.normal_streak + 1 if status == "normal" else 0
        if self.active_alert is not None and self.normal_streak >= self.cooldown:
            self.active_alert = None          # incident over
        if self.count == self.persist and self.active_alert is not None:
            self.active_alert["score"] = max(self.active_alert["score"], round(score, 3))
        elif self.count == self.persist:
            self.alert_seq += 1
            alert_id = f"A-{self.alert_seq:04d}"
            alert = {"type": "alert", "id": alert_id, "ts": ts, "device": header.get("device", "WS-LAB-03"),
                     "severity": "crit" if score >= 0.9 else "warn", "score": round(score, 3),
                     "alert_type": "spectralAnomaly", "reason": reason, "status": "open", "seq": seq}
            self.active_alert = alert
            self.alerts.insert(0, alert)
            del self.alerts[200:]
            out.append(alert)

        self._csv.writerow([seq, ts, round(score, 4), status, state, alert_id])
        self._log.flush()
        return out

    # ------------------------------------------------------------- REST data
    def baseline_json(self):
        states = []
        for key in ["idle", "active", "boot", "background"]:
            s = self.det.state_stats.get(key)
            if not s:
                continue
            states.append({"key": key, "windows": s["windows"], "rms_dbm": round(s["mean"]["rms_dbm"], 1),
                           "spark": s["spark"],
                           "means": {"peak_offset_khz": round(s["mean"]["peak_offset_khz"], 2),
                                     "band_energy": round(s["mean"]["band_energy"], 3),
                                     "periodicity": round(s["mean"]["periodicity"], 3)}})
        return {"center_mhz": CENTER_MHZ, "fs_msps": FS / 1e6, "last_calibrated": self.last_calibrated,
                "band": self.det.band, "tracked_carrier_khz": round(self.det.track_hz / 1e3, 1),
                "states": states, "recalibrating": self.recal is not None}

    def metrics_json(self):
        if self.metrics_path.exists():
            m = json.loads(self.metrics_path.read_text())["dashboard"]
        else:
            m = {"detection_rate": None, "false_positive_rate": None, "latency_s": None,
                 "is_placeholder": True, "score_history": []}
        if len(self.scores) >= 20:   # show the live session if there is one
            m = dict(m)
            m["score_history"] = [round(float(v), 3) for v in list(self.scores)[-70:]]
        return m
