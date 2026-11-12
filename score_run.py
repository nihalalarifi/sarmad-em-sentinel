"""
Measure a LIVE run: join the simulator's ground truth with the backend's decisions.

    python score_run.py logs/truth-XXXX.csv logs/decisions-YYYY.csv

(The browser simulator's "Download truth CSV" file works too.)
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict


def load(path, key="seq"):
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r[key]): r for r in csv.DictReader(fh) if r.get(key)}


def main(truth_path, dec_path):
    truth, dec = load(truth_path), load(dec_path)
    seqs = sorted(set(truth) & set(dec))
    if not seqs:
        sys.exit("no matching sequence numbers between the two files")
    per = defaultdict(lambda: [0, 0])
    tp = fp = fn = tn = 0
    for s in seqs:
        anom = truth[s]["label"] == "anomaly"
        flagged = dec[s]["status"] in ("warning", "alert")
        per[truth[s]["scenario"]][0] += flagged
        per[truth[s]["scenario"]][1] += 1
        tp += anom and flagged; fn += anom and not flagged
        fp += (not anom) and flagged; tn += (not anom) and not flagged

    # event level: each continuous attack segment -> caught? latency?
    events, latencies, false_alerts = 0, [], 0
    i = 0
    while i < len(seqs):
        s = seqs[i]
        if truth[s]["label"] == "anomaly":
            j = i
            while j < len(seqs) and truth[seqs[j]]["label"] == "anomaly":
                j += 1
            events += 1
            for k in range(i, j):
                if dec[seqs[k]]["alert_id"]:
                    latencies.append(k - i + 1)
                    break
            i = j
        else:
            false_alerts += bool(dec[s]["alert_id"])
            i += 1

    print(f"windows matched: {len(seqs)}")
    print(f"window detection rate : {tp / max(1, tp + fn):.1%}")
    print(f"window false positives: {fp / max(1, fp + tn):.2%}")
    print(f"attack events caught  : {len(latencies)}/{events}")
    if latencies:
        latencies.sort()
        print(f"median latency        : {latencies[len(latencies)//2]} windows (= seconds at 1 window/s)")
    print(f"false alerts          : {false_alerts}")
    Path("eval").mkdir(exist_ok=True)
    Path("eval/live_run.json").write_text(json.dumps({
        "windows": len(seqs), "window_detection_rate": tp / max(1, tp + fn),
        "window_false_positive_rate": fp / max(1, fp + tn), "events": events,
        "events_caught": len(latencies), "latency_median_windows": latencies[len(latencies)//2] if latencies else None,
        "false_alerts": false_alerts, "per_scenario": {k: [f, n] for k, (f, n) in per.items()},
        "truth_file": truth_path, "decisions_file": dec_path}, indent=2))
    print("per scenario (flagged / total):")
    for k, (f, n) in sorted(per.items()):
        print(f"  {k:15s} {f:4d}/{n:<4d} {f / n:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
