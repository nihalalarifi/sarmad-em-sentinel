"""
DSP + feature extraction for one window of complex I/Q.

Pipeline per window:
  1. Power spectrum (Welch)                 -> spectrum chart, spectrogram column,
                                               band energy, bandwidth, flatness
  2. Lock onto the tracked clock carrier     -> dominant frequency (offset)
  3. Demodulate the carrier's envelope       -> activity envelope a(t)
     (mix down, low-pass, decimate, |.|)
  4. Timing analysis of the envelope         -> periodicity (symbol-clock line)
                                               on/off regularity, bimodality
"""
import numpy as np
from scipy import signal as sps

from .signals import FS

PSD_NPERSEG = 2048            # 1 kHz resolution at 2.048 MS/s
DISPLAY_BINS = 256            # spectrum sent to the dashboard
SPEC_BINS = 64                # spectrogram column sent to the dashboard
DECIM = 32                    # envelope sample rate = 64 kHz
ENV_FS = FS / DECIM
DB_OFFSET = -35.0             # calibrates PSD to the dashboard's dB scale
DBM_OFFSET = -42.0            # calibrates window power to an indicative dBm value
SEARCH_HZ = 20e3              # carrier search half-width around the tracked line

# features shown on the dashboard (order matters for the table)
DISPLAY_KEYS = ["rms_dbm", "peak_offset_khz", "bandwidth_khz", "band_energy",
                "periodicity", "onoff_regularity", "spectral_flatness"]
# features the AI model uses (peak offset is excluded: it is a tuning value, not behaviour)
MODEL_KEYS = ["rms_dbm", "band_energy", "bandwidth_khz", "spectral_flatness",
              "env_cv", "bimodality", "periodicity", "onoff_regularity"]


def psd(x):
    f, p = sps.welch(x, fs=FS, nperseg=PSD_NPERSEG, return_onesided=False,
                     scaling="spectrum", detrend=False)
    order = np.argsort(f)
    return f[order], p[order]


def pool_db(p_lin, bins):
    """Max-pool a linear spectrum to `bins` points and convert to dB (keeps narrow lines visible)."""
    g = p_lin.reshape(bins, -1).max(axis=1)
    return 10 * np.log10(g + 1e-20) + DB_OFFSET


def _otsu_eta(v):
    """Otsu separability (0..1): ~1 for a clean two-level signal."""
    v = np.asarray(v)
    tot = v.var()
    if tot < 1e-12:
        return 0.0
    s = np.sort(v)
    best = 0.0
    for q in np.linspace(0.1, 0.9, 33):
        thr = s[int(q * (len(s) - 1))]
        a, b = v[v <= thr], v[v > thr]
        if len(a) < 5 or len(b) < 5:
            continue
        w0, w1 = len(a) / len(v), len(b) / len(v)
        best = max(best, w0 * w1 * (a.mean() - b.mean()) ** 2 / tot)
    return float(best)


def envelope(x, f_carrier):
    n = len(x)
    t = np.arange(n) / FS
    y = x * np.exp(-2j * np.pi * f_carrier * t)
    y = sps.resample_poly(y, 1, DECIM)          # anti-alias low-pass + decimate
    return np.abs(y)


def edge_times(env):
    """Switching instants of the envelope (Schmitt trigger, so noise does not chatter)."""
    e = np.convolve(env, np.ones(3) / 3, mode="same")
    lo, hi = np.percentile(e, 10), np.percentile(e, 90)
    mid, hyst = (lo + hi) / 2, 0.15 * (hi - lo)
    state = e[0] > mid
    edges = []
    for i, v in enumerate(e):
        if state and v < mid - hyst:
            state = False
            edges.append(i)
        elif not state and v > mid + hyst:
            state = True
            edges.append(i)
    return np.asarray(edges, dtype=float), (hi - lo) / (hi + 1e-9)


FREQS = np.arange(300.0, 4000.0, 10.0)


def timing_features(env):
    """
    Periodicity: Rayleigh phase-coherence of the switching instants over candidate
    symbol rates. If every edge falls on a common clock grid (covert keying) the
    phases line up and R -> 1. Natural activity switches at random instants (R small).
    On/off regularity: how close the on/off run lengths are to integer multiples
    of that clock period.
    """
    t_idx, contrast = edge_times(env)
    K = len(t_idx)
    if K < 6 or contrast < 0.15:
        return 0.0, 0.0, 0.0, K
    t = t_idx / ENV_FS
    ph = np.exp(2j * np.pi * np.outer(FREQS, t))
    R = np.abs(ph.mean(axis=1))
    k = int(np.argmax(R))
    r_max, f_best = float(R[k]), float(FREQS[k])

    runs = np.diff(t_idx) / (ENV_FS / f_best)
    m = np.round(runs)
    ok = m >= 1
    reg = 0.0
    if ok.sum() >= 4:
        fit = 1 - 2 * np.abs(runs[ok] - m[ok])
        reg = float(np.clip((fit.mean() - 0.5) / 0.5, 0, 1))
    return r_max, f_best, reg, K


def extract(x, track_hz):
    """All features + display products for one window."""
    x = x - x.mean()
    f, p = psd(x)
    total = p.sum() + 1e-20

    # lock onto the tracked carrier
    m = np.abs(f - track_hz) <= SEARCH_HZ
    idx = np.flatnonzero(m)
    k = idx[int(np.argmax(p[idx]))]
    if 0 < k < len(p) - 1:   # parabolic interpolation in dB
        a, b, c = 10 * np.log10(p[k - 1:k + 2] + 1e-20)
        delta = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    else:
        delta = 0.0
    df = f[1] - f[0]
    f_peak = float(f[k] + delta * df)

    band = np.abs(f - f_peak) <= 50e3
    band_energy = float(p[band].sum() / total)

    pk_db = 10 * np.log10(p[k] + 1e-20)
    above = (10 * np.log10(p + 1e-20) >= pk_db - 20) & (np.abs(f - f_peak) <= 50e3)
    lo_i = k
    while lo_i > 0 and above[lo_i - 1]:
        lo_i -= 1
    hi_i = k
    while hi_i < len(p) - 1 and above[hi_i + 1]:
        hi_i += 1
    bandwidth_khz = float((hi_i - lo_i + 1) * df / 1e3)

    flat = float(np.exp(np.mean(np.log(p + 1e-20))) / np.mean(p))
    rms_dbm = float(10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20) + DBM_OFFSET)

    env = envelope(x, f_peak)
    env_cv = float(env.std() / (env.mean() + 1e-12))
    bimod = _otsu_eta(env)
    periodicity, f_line, reg, n_edges = timing_features(env)

    feats = {
        "rms_dbm": rms_dbm, "peak_offset_khz": f_peak / 1e3, "bandwidth_khz": bandwidth_khz,
        "band_energy": band_energy, "spectral_flatness": flat, "env_cv": env_cv,
        "bimodality": bimod, "periodicity": periodicity, "onoff_regularity": reg,
        "symbol_rate_hz": f_line,
    }
    display = {
        "fft_db": pool_db(p, DISPLAY_BINS),
        "spec_db": pool_db(p, SPEC_BINS),
        "envelope": env,
    }
    return feats, display


def model_vector(feats):
    return np.array([feats[k] for k in MODEL_KEYS], dtype=float)


def resample_trace(v, n=320):
    v = np.asarray(v, dtype=float)
    return np.interp(np.linspace(0, len(v) - 1, n), np.arange(len(v)), v)
