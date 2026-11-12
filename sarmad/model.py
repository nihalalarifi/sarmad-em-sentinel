"""
EM-Sentinel detector.

  * k-NN novelty detection, trained on NORMAL windows only (idle, active use,
    boot, background services, environmental interference). The anomaly
    evidence of a window is its distance to the learned normal behaviour
    (distance to its k-th nearest normal neighbour, in standardised feature
    space). The model never sees an attack during training, so it can flag
    patterns it has never seen before.
  * The distance is turned into a calibrated score with a conformal p-value
    measured on held-out normal windows (exponential tail beyond the 90th
    percentile):  score = -log10(p) / 4   ->  0.5 = 1 in 100 normal windows,
    0.70 = about 1 in 600, 1.0 = 1 in 10,000 or rarer.
  * An Isolation Forest is trained on the same data and kept for comparison
    in evaluate.py (it separates the classes, AUC ~0.99, but its scores
    saturate, which makes thresholds fragile).
  * A small k-NN classifier recognises the current operating state
    (idle / active / boot / background) so deviations are explained against
    the right baseline.
  * Per-state feature statistics give a human-readable reason for each alert.
"""
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .features import MODEL_KEYS, DISPLAY_KEYS, resample_trace

STATES = ["idle", "active", "boot", "background"]
TIMING_KEYS = {"periodicity", "onoff_regularity"}
ENERGY_KEYS = {"rms_dbm", "band_energy"}


class Detector:
    K = 10

    def __init__(self, track_hz=150e3, seed=0):
        self.track_hz = float(track_hz)
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, X, states, feats, fft_db, envs, calib_frac=0.25):
        """
        X       : (n, len(MODEL_KEYS)) model vectors of NORMAL windows
        states  : operating state label of each window (from calibration)
        feats   : list of feature dicts (for per-state statistics)
        fft_db  : (n, 256) display spectra   envs: list of envelope arrays
        """
        rng = np.random.default_rng(self.seed)
        n = len(X)
        idx = rng.permutation(n)
        n_cal = int(n * calib_frac)
        cal, tr = idx[:n_cal], idx[n_cal:]
        states = np.asarray(states)

        self.scaler = StandardScaler().fit(X[tr])
        Xs = self.scaler.transform(X)
        self.nn = NearestNeighbors(n_neighbors=self.K).fit(Xs[tr])
        self.iforest = IsolationForest(n_estimators=300, random_state=self.seed).fit(Xs[tr])
        self.state_clf = KNeighborsClassifier(n_neighbors=15).fit(Xs[tr], states[tr])

        # conformal calibration on held-out normal windows
        raw = self._raw(Xs[cal])
        self._cal = np.sort(raw)
        self._u = float(np.percentile(raw, 90))
        self._beta = float(max(np.mean(raw[raw > self._u] - self._u), 1e-6))
        cal_scores = self._to_score(raw)
        self.band = {"lo": float(np.percentile(cal_scores, 5)),
                     "hi": float(np.percentile(cal_scores, 99))}

        # per-state statistics for explanations and dashboard baselines
        all_scores = self._to_score(self._raw(Xs))
        self.state_stats = {}
        for st in STATES:
            m = states == st
            if not m.any():
                continue
            rows = [feats[i] for i in np.flatnonzero(m)]
            mean = {k: float(np.mean([r[k] for r in rows])) for k in DISPLAY_KEYS + MODEL_KEYS}
            std = {k: float(max(np.std([r[k] for r in rows]), 1e-3, 0.02 * abs(mean[k])))
                   for k in DISPLAY_KEYS + MODEL_KEYS}
            sc = all_scores[m]
            chunks = np.array_split(sc, 9)
            ex = np.flatnonzero(m)[int(rng.integers(0, m.sum()))]
            self.state_stats[st] = {
                "windows": int(m.sum()), "mean": mean, "std": std,
                "fft_db": fft_db[m].mean(axis=0).tolist(),
                "envelope": resample_trace(envs[ex]).tolist(),
                "spark": [round(float(c.mean()), 3) for c in chunks],
            }
        self._train = (X, states, feats, fft_db, envs)
        return self

    # --------------------------------------------------------------- scoring
    def _raw(self, Xs):
        """Distance to the K-th nearest normal training window."""
        return self.nn.kneighbors(Xs)[0][:, -1]

    def p_value(self, raw):
        raw = np.asarray(raw, dtype=float)
        n = len(self._cal)
        emp = (n - np.searchsorted(self._cal, raw, side="left") + 1) / (n + 1)
        tail = 0.10 * np.exp(-(raw - self._u) / self._beta)
        return np.where(raw > self._u, np.minimum(emp, tail), emp)

    def _to_score(self, raw):
        return np.clip(-np.log10(self.p_value(raw)) / 4.0, 0.0, 1.0)

    def score(self, x_vec):
        xs = self.scaler.transform(np.atleast_2d(x_vec))
        return float(self._to_score(self._raw(xs))[0])

    def state(self, x_vec):
        xs = self.scaler.transform(np.atleast_2d(x_vec))
        return str(self.state_clf.predict(xs)[0])

    def score_many(self, X):
        return self._to_score(self._raw(self.scaler.transform(X)))

    def iforest_raw(self, X):
        return -self.iforest.score_samples(self.scaler.transform(X))

    def explain(self, feats, state):
        """Deviation of each displayed feature from the detected state's baseline."""
        st = self.state_stats.get(state) or next(iter(self.state_stats.values()))
        out, z = [], {}
        for k in DISPLAY_KEYS:
            zk = (feats[k] - st["mean"][k]) / st["std"][k]
            z[k] = zk
            out.append({"key": k, "value": round(float(feats[k]), 3),
                        "baseline": round(float(st["mean"][k]), 3),
                        "dev": round(float(min(1.0, abs(zk) / 6.0)), 2)})
        ranked = sorted((k for k in DISPLAY_KEYS if k != "peak_offset_khz"),
                        key=lambda k: -abs(z[k]))
        top = ranked[:2]
        if TIMING_KEYS & set(top):
            code = "periodicBurst"
        elif ENERGY_KEYS & set(top):
            code = "bandEnergyRise"
        else:
            code = "spectralAnomaly"
        return out, {"code": code, "top_features": top}

    # ------------------------------------------------------- recalibration
    def recalibrate(self, state, new_X, new_feats, new_fft, new_envs):
        """Replace the stored windows of one state with a fresh capture and refit."""
        X, states, feats, fft_db, envs = self._train
        keep = states != state
        X2 = np.vstack([X[keep], new_X])
        st2 = np.concatenate([states[keep], [state] * len(new_X)])
        f2 = [f for f, k in zip(feats, keep) if k] + list(new_feats)
        fft2 = np.vstack([fft_db[keep], new_fft])
        e2 = [e for e, k in zip(envs, keep) if k] + list(new_envs)
        return self.fit(X2, st2, f2, fft2, e2)

    # --------------------------------------------------------------- io
    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        return joblib.load(path)
