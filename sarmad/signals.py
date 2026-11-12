"""
Signal model for EM-Sentinel (Sarmad).

Simulates what an SDR tuned near a computer's clock harmonic would capture:
complex baseband I/Q at FS samples/s around CENTER_MHZ.

Physics (simplified but grounded in the published side-channel literature):
  * The CPU/RAM clock leaks narrow carriers. Their amplitude follows the
    activity of the processor/memory bus  ->  "activity envelope" a(t).
  * Normal activity (idle, user work, boot, background services) changes a(t)
    at IRREGULAR times.
  * A covert channel (RAMBO-style on/off keying) switches a(t) between two
    levels on a FIXED symbol clock, because the receiver must decode it.
    That regular timing is what EM-Sentinel looks for.

The JavaScript simulator (simulator/index.html) implements the same model;
this file is the reference used to build training and test datasets.
"""
import numpy as np

FS = 2_048_000          # samples per second (RTL-SDR friendly)
N = 32_768              # samples per analysis window  (= 16 ms)
CENTER_MHZ = 125.0      # SDR tuning frequency (display only)

F_MAIN = 150e3          # main clock harmonic offset (Hz), activity-modulated
F_OTHER = -600e3        # unrelated, constant clock line
F_HARM = 450e3          # weaker harmonic, also activity-modulated

NORMAL_SCENARIOS = ["idle", "active", "background", "boot", "interference"]
ANOMALY_SCENARIOS = ["covert_fast", "covert_stealth"]
ALL_SCENARIOS = NORMAL_SCENARIOS + ANOMALY_SCENARIOS

# operating state each normal scenario corresponds to (used for per-state baselines)
SCENARIO_STATE = {"idle": "idle", "active": "active", "background": "background",
                  "boot": "boot", "interference": "idle"}

PROBE_GAIN = {"CPU": 1.0, "RAM": 0.8, "VRM": 0.6, "Cables": 0.4}


def _smooth(x, k):
    """Short moving average = finite rise/fall time of real switching."""
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _segments(rng, n, mean_len, lo, hi, min_len=50):
    """Piecewise-constant levels with random (exponential) durations -> irregular timing."""
    a = np.empty(n)
    i = 0
    while i < n:
        L = max(min_len, int(rng.exponential(mean_len)))
        a[i:i + L] = rng.uniform(lo, hi)
        i += L
    return a


def activity(scenario, rng, n=N, fs=FS):
    """Activity envelope a(t) in [0, 1] for one window."""
    ms = fs / 1000.0
    wander = _smooth(rng.normal(0, 1, n), int(0.3 * ms))  # slow natural wander
    wander /= (np.std(wander) + 1e-12)

    if scenario in ("idle", "interference"):
        a = 0.22 + 0.025 * wander
        # rare tiny housekeeping blip
        if rng.random() < 0.3:
            s = rng.integers(0, n - int(0.4 * ms))
            a[s:s + int(rng.uniform(0.05, 0.4) * ms)] += rng.uniform(0.05, 0.15)

    elif scenario == "active":
        a = _segments(rng, n, mean_len=0.6 * ms, lo=0.30, hi=1.0)
        a += 0.04 * wander

    elif scenario == "background":
        a = 0.24 + 0.03 * wander
        for _ in range(rng.poisson(6)):            # sparse random service spikes
            L = int(rng.uniform(0.05, 0.6) * ms)
            s = rng.integers(0, max(1, n - L))
            a[s:s + L] += rng.uniform(0.2, 0.55)

    elif scenario == "boot":
        a = _segments(rng, n, mean_len=0.25 * ms, lo=0.55, hi=1.0)
        a += 0.05 * wander

    elif scenario in ("covert_fast", "covert_stealth"):
        if scenario == "covert_fast":       # ~2 kbit/s, strong on/off depth
            Ts, lo, hi = 0.5e-3, 0.22, 0.90
        else:                                # ~1 kbit/s, shallow depth (harder)
            Ts, lo, hi = 1.0e-3, 0.22, 0.45
        sym = int(Ts * fs)
        offset = rng.integers(0, sym)
        nsym = (n + offset) // sym + 2
        bits = rng.integers(0, 2, nsym)
        a = np.repeat(np.where(bits == 1, hi, lo), sym)[offset:offset + n]
        a = a + 0.02 * wander
    else:
        raise ValueError(f"unknown scenario {scenario}")

    a = _smooth(a, int(0.02 * ms))          # ~20 us switching edges
    return np.clip(a, 0.0, 1.2)


def window(scenario, rng, snr_db=10.0, probe="CPU", n=N, fs=FS, return_activity=False):
    """One window of complex baseband I/Q (complex64)."""
    t = np.arange(n) / fs
    a = activity(scenario, rng, n, fs)
    g = PROBE_GAIN.get(probe, 1.0)
    ph = rng.uniform(0, 2 * np.pi, 3)
    df = rng.normal(0, 300, 3)               # small clock drift (Hz)

    x = (g * a) * np.exp(1j * (2 * np.pi * (F_MAIN + df[0]) * t + ph[0]))
    x += 0.35 * np.exp(1j * (2 * np.pi * (F_OTHER + df[1]) * t + ph[1]))
    x += (0.15 * g * a) * np.exp(1j * (2 * np.pi * (F_HARM + df[2]) * t + ph[2]))

    if scenario == "interference":           # phone / Wi-Fi style bursts nearby
        f_i = rng.choice([-1, 1]) * rng.uniform(220e3, 900e3)
        gate = _segments(rng, n, mean_len=1.5e-3 * fs, lo=0, hi=1) > 0.5
        x += rng.uniform(0.3, 0.8) * gate * np.exp(1j * (2 * np.pi * f_i * t + rng.uniform(0, 6.3)))

    noise_p = 1.0 / (10 ** (snr_db / 10))     # relative to a full-scale main carrier
    x += np.sqrt(noise_p / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    x = x.astype(np.complex64)
    return (x, a) if return_activity else x


def is_anomaly(scenario):
    return scenario in ANOMALY_SCENARIOS
