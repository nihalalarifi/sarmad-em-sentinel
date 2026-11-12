"""Build labelled datasets from the signal model (training = normal only)."""
import numpy as np

from . import signals as S
from .features import extract, model_vector, psd, resample_trace


def find_tracked_carrier(seed=123, per_state=12):
    """
    Calibration step: which spectral line follows the device's activity?
    Record each normal state, average its spectrum, and pick the line whose
    power changes the most between states (largest variance, linear power).
    """
    rng = np.random.default_rng(seed)
    means = []
    for sc in ["idle", "active", "boot", "background"]:
        ps = [psd(S.window(sc, rng, 15.0))[1] for _ in range(per_state)]
        means.append(np.mean(ps, axis=0))
    f = psd(S.window("idle", rng))[0]
    v = np.var(np.array(means), axis=0)
    v[np.abs(f) < 5e3] = 0
    return float(f[int(np.argmax(v))])


def build(scenario_counts, track_hz, seed, snr_range=(5, 20), probes=("CPU",), snr_fixed=None):
    rng = np.random.default_rng(seed)
    X, y_anom, scen, states, feats, ffts, envs, snrs = [], [], [], [], [], [], [], []
    for sc, n in scenario_counts.items():
        for _ in range(n):
            snr = snr_fixed if snr_fixed is not None else rng.uniform(*snr_range)
            probe = probes[int(rng.integers(0, len(probes)))]
            f, d = extract(S.window(sc, rng, snr, probe), track_hz)
            X.append(model_vector(f)); feats.append(f); ffts.append(d["fft_db"].astype("float32")); envs.append(resample_trace(d["envelope"]).astype("float32"))
            y_anom.append(S.is_anomaly(sc)); scen.append(sc); snrs.append(snr)
            states.append(S.SCENARIO_STATE.get(sc, "attack"))
    return {"X": np.array(X), "anom": np.array(y_anom), "scenario": np.array(scen),
            "state": np.array(states), "feats": feats, "fft": np.array(ffts),
            "env": envs, "snr": np.array(snrs)}
