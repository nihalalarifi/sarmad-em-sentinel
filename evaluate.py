"""
Evaluate EM-Sentinel on an INDEPENDENT test set (new random seed, several noise levels).

    python evaluate.py        -> eval/metrics.json + eval/*.png + console report

Compares the AI detector with a classic energy-threshold detector to show
why learning the normal behaviour matters.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from sarmad.dataset import build
from sarmad.model import Detector
from sarmad.signals import NORMAL_SCENARIOS, ANOMALY_SCENARIOS, ALL_SCENARIOS

THRESHOLD = 0.70
PERSIST = 3
SNRS = [0, 5, 10, 15, 20]
PER = 300          # windows per scenario per SNR
OUT = Path("eval")


def wilson(k, n, z=1.96):
    """95% confidence interval for a rate k/n (Wilson score)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def energy_scores(ds):
    return np.array([f["rms_dbm"] for f in ds["feats"]])


def event_level(pool, rng, runs=300, pre=20, post=20, persist=PERSIST, thr=THRESHOLD):
    """Streams of normal activity followed by an attack, scored with persistence."""
    caught, lat, false_alerts = 0, [], 0
    for _ in range(runs):
        normal = [rng.choice(pool[rng.choice(NORMAL_SCENARIOS)]) for _ in range(pre)]
        atk = pool[rng.choice(ANOMALY_SCENARIOS)]
        attack = [rng.choice(atk) for _ in range(post)]
        c, alerted = 0, False
        for s in normal:                       # count false alerts before the attack
            c = c + 1 if s > thr else 0
            if c == persist:
                false_alerts += 1
        c = 0
        for i, s in enumerate(attack):
            c = c + 1 if s > thr else 0
            if c >= persist:
                caught += 1
                lat.append(i + 1)               # windows = seconds at 1 window/s
                alerted = True
                break
    hours = runs * pre / 3600
    return {"event_detection_rate": caught / runs,
            "latency_median_s": float(np.median(lat)) if lat else None,
            "latency_mean_s": float(np.mean(lat)) if lat else None,
            "false_alerts_per_hour": false_alerts / hours}


def main():
    OUT.mkdir(exist_ok=True)
    det = Detector.load("model/model.joblib")
    print(f"tracked carrier {det.track_hz/1e3:+.1f} kHz | threshold {THRESHOLD} | persistence {PERSIST}")

    sets = {}
    for snr in SNRS:
        sets[snr] = build({sc: PER for sc in ALL_SCENARIOS}, det.track_hz, seed=1000 + snr, snr_fixed=snr)
        sets[snr]["score"] = det.score_many(sets[snr]["X"])
        print(f"  test set SNR {snr:2d} dB: {len(sets[snr]['X'])} windows")

    # energy detector thresholds, calibrated on a separate normal set
    cal = build({sc: 150 for sc in NORMAL_SCENARIOS}, det.track_hz, seed=77)
    e_cal = energy_scores(cal)
    thr_naive = np.percentile(e_cal[cal["state"] == "idle"], 99.5)   # idle-only baseline
    thr_fair = np.percentile(e_cal, 99.5)                              # all normal activity

    rep = {"threshold": THRESHOLD, "persistence": PERSIST, "per_snr": {}, "per_scenario_10db": {}}
    for snr, ds in sets.items():
        a, s, sc = ds["anom"], ds["score"], ds["scenario"]
        e = energy_scores(ds)
        rep["per_snr"][snr] = {
            "auc": float(roc_auc_score(a, s)),
            "dr_fast": float((s[sc == "covert_fast"] > THRESHOLD).mean()),
            "dr_stealth": float((s[sc == "covert_stealth"] > THRESHOLD).mean()),
            "fpr": float((s[~a] > THRESHOLD).mean()),
            "n_normal": int((~a).sum()), "n_attack": int(a.sum()),
            "fp_count": int((s[~a] > THRESHOLD).sum()), "tp_count": int((s[a] > THRESHOLD).sum()),
            "fpr_ci95": wilson(int((s[~a] > THRESHOLD).sum()), int((~a).sum())),
            "dr_ci95": wilson(int((s[a] > THRESHOLD).sum()), int(a.sum())),
            "energy_naive": {"dr": float((e[a] > thr_naive).mean()), "fpr": float((e[~a] > thr_naive).mean())},
            "energy_fair": {"dr": float((e[a] > thr_fair).mean()), "fpr": float((e[~a] > thr_fair).mean())},
            "energy_auc": float(roc_auc_score(a, e)),
            "iforest_auc": float(roc_auc_score(a, det.iforest_raw(ds["X"]))),
        }
    ds10 = sets[10]
    for scn in ALL_SCENARIOS:
        m = ds10["scenario"] == scn
        rep["per_scenario_10db"][scn] = {"mean_score": float(ds10["score"][m].mean()),
                                         "flagged": float((ds10["score"][m] > THRESHOLD).mean())}

    rng = np.random.default_rng(5)
    pool = {scn: ds10["score"][ds10["scenario"] == scn] for scn in ALL_SCENARIOS}
    rep["event_10db"] = event_level(pool, rng, runs=500)

    r10 = rep["per_snr"][10]
    ev = rep["event_10db"]
    dash = {"detection_rate": ev["event_detection_rate"], "false_positive_rate": r10["fpr"],
            "latency_s": ev["latency_median_s"], "is_placeholder": False,
            "score_history": [round(float(v), 3) for v in
                              np.concatenate([pool["idle"][:30], pool["active"][:20], pool["covert_fast"][:5], pool["idle"][30:45]])]}
    rep["dashboard"] = dash
    (OUT / "metrics.json").write_text(json.dumps(rep, indent=2))

    # ---------------- plots ----------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 5))
    a, s, e = ds10["anom"], ds10["score"], energy_scores(ds10)
    ifr = det.iforest_raw(ds10["X"])
    for v, lbl in [(s, f"EM-Sentinel k-NN novelty (AUC {roc_auc_score(a, s):.3f})"),
                   (ifr, f"Isolation Forest (AUC {roc_auc_score(a, ifr):.3f})"),
                   (e, f"Energy threshold (AUC {roc_auc_score(a, e):.3f})")]:
        fpr, tpr, _ = roc_curve(a, v)
        ax.plot(fpr, tpr, label=lbl)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="Detection rate", title="ROC at SNR 10 dB")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "roc.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SNRS, [rep["per_snr"][x]["dr_fast"] for x in SNRS], "o-", label="Covert fast (~2 kbit/s)")
    ax.plot(SNRS, [rep["per_snr"][x]["dr_stealth"] for x in SNRS], "o-", label="Covert stealth (~1 kbit/s, shallow)")
    ax.plot(SNRS, [rep["per_snr"][x]["fpr"] for x in SNRS], "o--", label="False positive rate")
    ax.set(xlabel="SNR (dB)", ylabel="Rate (per window)", title="Detection vs noise level", ylim=(-0.02, 1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "detection_vs_snr.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([pool[x] for x in ALL_SCENARIOS], showfliers=False)
    ax.set_xticks(range(1, len(ALL_SCENARIOS) + 1), ALL_SCENARIOS, rotation=20)
    ax.axhline(THRESHOLD, color="red", ls="--", label="threshold")
    ax.set(ylabel="Anomaly score", title="Score per scenario (SNR 10 dB)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "scores_by_scenario.png", dpi=140); plt.close(fig)

    # ---------------- report ----------------
    print("\n=== Window level (SNR 10 dB) ===")
    print(f"AI      : detection fast {r10['dr_fast']:.1%} | stealth {r10['dr_stealth']:.1%} | false positives {r10['fpr']:.2%} | AUC {r10['auc']:.3f}")
    print(f"Isolation Forest (comparison): AUC {r10['iforest_auc']:.3f}")
    en, ef = r10["energy_naive"], r10["energy_fair"]
    print(f"Energy (idle-only baseline) : detection {en['dr']:.1%} | false positives {en['fpr']:.1%}")
    print(f"Energy (all-normal baseline): detection {ef['dr']:.1%} | false positives {ef['fpr']:.1%}")
    print("\n=== Per scenario (SNR 10 dB) ===")
    for k, v in rep["per_scenario_10db"].items():
        print(f"  {k:15s} mean score {v['mean_score']:.2f}  flagged {v['flagged']:.1%}")
    print("\n=== Detection vs SNR ===")
    for x in SNRS:
        r = rep["per_snr"][x]
        print(f"  {x:2d} dB  fast {r['dr_fast']:.1%}  stealth {r['dr_stealth']:.1%}  FPR {r['fpr']:.2%}  AUC {r['auc']:.3f}")
    print(f"\n=== Event level (SNR 10 dB, persistence {PERSIST}) ===")
    print(f"  attacks caught {ev['event_detection_rate']:.1%} | median latency {ev['latency_median_s']} s | "
          f"false alerts {ev['false_alerts_per_hour']:.1f} per hour")


if __name__ == "__main__":
    main()
