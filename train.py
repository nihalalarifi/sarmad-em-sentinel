"""
Train the EM-Sentinel detector on NORMAL behaviour only.

    python train.py            -> model/model.joblib
"""
import time
from pathlib import Path

from sarmad.dataset import build, find_tracked_carrier
from sarmad.model import Detector

PER_SCENARIO = 400


def main():
    t0 = time.time()
    track = find_tracked_carrier()
    print(f"tracked clock carrier: {track/1e3:+.1f} kHz from center")
    counts = {sc: PER_SCENARIO for sc in ["idle", "active", "background", "boot", "interference"]}
    ds = build(counts, track, seed=1, probes=("CPU", "CPU", "CPU", "RAM", "VRM", "Cables"))
    print(f"generated {len(ds['X'])} normal windows (no attacks) in {time.time()-t0:.1f}s")
    det = Detector(track_hz=track).fit(ds["X"], ds["state"], ds["feats"], ds["fft"], ds["env"])
    Path("model").mkdir(exist_ok=True)
    det.save("model/model.joblib")
    print(f"normal score band: {det.band['lo']:.2f} - {det.band['hi']:.2f}")
    for st, s in det.state_stats.items():
        print(f"  state {st:10s} windows={s['windows']:4d}  periodicity={s['mean']['periodicity']:.2f}")
    print(f"saved model/model.joblib  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
