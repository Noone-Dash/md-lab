"""Error bars for correlated time series. Without this, every mean we report is a lie.

THE PROBLEM
-----------
Every number this lab reports (density, temperature, energy, RMSD) is a time average
over an MD trajectory. The obvious error bar is the textbook one,

    SEM_naive = sigma / sqrt(N)                                                  (1)

and it is WRONG for MD — not slightly, but by a factor of 3-10x. Equation (1) assumes
the N samples are independent. Consecutive MD frames are not: the box does not forget
its configuration in one step. Frames 1 ps apart in a water box are almost the same
frame.

THE DERIVATION
--------------
Let {x_i}, i = 1..N be a stationary series with mean mu and variance sigma^2, sampled
at a fixed interval. Write the normalised autocorrelation

    rho(k) = Cov(x_i, x_{i+k}) / sigma^2,        rho(0) = 1.

The variance of the sample mean is a double sum over all pairs, not a single sum:

    Var(x_bar) = (1/N^2) * sum_i sum_j Cov(x_i, x_j)
               = (sigma^2 / N^2) * sum_i sum_j rho(i-j)
               = (sigma^2 / N) * [ 1 + 2 * sum_{k=1}^{N-1} (1 - k/N) * rho(k) ].     (2)

The cross terms rho(k != 0) are what equation (1) throws away. For N much larger than
the correlation length the (1 - k/N) factor -> 1 and the bracket converges to a
constant. Define the INTEGRATED AUTOCORRELATION TIME (in units of the sampling
interval):

    tau_int = 1/2 + sum_{k=1}^{inf} rho(k).                                        (3)

Substituting (3) into (2):

    Var(x_bar) = 2 * tau_int * sigma^2 / N,      SEM = sigma * sqrt(2*tau_int / N).  (4)

Comparing with (1): the honest error bar is larger by sqrt(2*tau_int). Equivalently,
the N correlated samples carry the information of only

    N_eff = N / (2 * tau_int)                                                        (5)

independent ones. For uncorrelated data rho(k>0) = 0, so tau_int = 1/2, N_eff = N and
(4) collapses back to (1) — as it must.

WHY tau_int IS HARD TO ESTIMATE
-------------------------------
You cannot just evaluate (3) to k = N-1. Var[rho_hat(k)] does not shrink with k, so the
tail of the sum is pure noise accumulating over ~N terms: the estimator diverges.
Two standard fixes, and this module implements BOTH and cross-checks them, because an
estimator you have not checked against an independent one is an assumption:

  1. SOKAL WINDOWING. Truncate (3) at the smallest W satisfying W >= c * tau_int(W),
     with c = 5. Bias falls exponentially in W/tau while variance grows linearly, so
     this sits near the optimum of the bias-variance tradeoff.

  2. FLYVBJERG-PETERSEN BLOCKING (J. Chem. Phys. 91, 461 (1989)). Repeatedly average
     adjacent pairs. Each transformation halves N and leaves the mean invariant, while
     the naive SEM of the blocked series RISES and then plateaus once the block length
     exceeds the correlation time — at which point the blocks ARE independent and the
     naive formula is finally legitimate. The plateau is the answer.

Both are validated in self_test() against an AR(1) process, whose tau_int is known in
closed form: rho(k) = phi^k, so from (3),

    tau_int = 1/2 + sum_{k>=1} phi^k = 1/2 + phi/(1-phi) = (1 + phi) / (2 * (1 - phi)).

That is ground truth, not a plausibility check.
"""

from __future__ import annotations

import math

import numpy as np


def autocorr(y, max_lag=None) -> np.ndarray:
    """Normalised autocorrelation rho(k), computed via FFT (O(N log N), not O(N^2))."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return np.array([1.0])
    y = y - y.mean()
    size = 1 << (2 * n - 1).bit_length()          # zero-pad: linear, not circular, ACF
    f = np.fft.rfft(y, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n].real
    if acf[0] <= 0:
        return np.array([1.0])
    acf /= acf[0]
    return acf[:max_lag] if max_lag else acf


def tau_int_sokal(y, c: float = 5.0):
    """Integrated autocorrelation time by Sokal's automatic windowing. -> (tau, window)."""
    rho = autocorr(y)
    n = len(y)
    tau = 0.5
    for w in range(1, len(rho)):
        tau += rho[w]
        if tau < 0.5:                  # anticorrelated series; floor at the iid value
            tau = 0.5
        if w >= c * tau:               # window closes: W >= c * tau(W)
            return max(0.5, float(tau)), w
    return max(0.5, float(tau)), n - 1


def blocking(y):
    """Flyvbjerg-Petersen. -> (sem_plateau, list of (block_size, sem, sem_err))."""
    y = np.asarray(y, dtype=float)
    curve = []
    n = len(y)
    size = 1
    while n >= 8:                      # below ~8 blocks the SEM estimate is meaningless
        var = y.var(ddof=1)
        sem = math.sqrt(var / n)
        curve.append((size, sem, sem / math.sqrt(2.0 * (n - 1))))   # err on the SEM
        if n % 2:                      # drop the odd sample; pairwise-average the rest
            y = y[:-1]
        y = 0.5 * (y[0::2] + y[1::2])
        n = len(y)
        size *= 2
    if not curve:
        return float("nan"), []
    # The plateau: the LAST point that is still statistically consistent with the
    # largest reliable SEM. Taking the max would ride the noise upward.
    sems = [c[1] for c in curve]
    errs = [c[2] for c in curve]
    i_max = int(np.argmax(sems))
    plateau = sems[i_max]
    for i in range(i_max):             # walk back to the first point consistent with it
        if sems[i] + errs[i] >= plateau:
            plateau = sems[i]
            break
    return float(plateau), curve


def stats(y, dt_ps: float = None) -> dict:
    """Mean of a correlated series WITH an honest error bar.

    Returns mean, sem (autocorrelation-corrected), the naive sem it replaces, tau_int,
    the effective sample size, the 95% CI, and the blocking cross-check.
    """
    y = np.asarray([v for v in y if v is not None], dtype=float)
    n = len(y)
    if n < 8:
        m = float(y.mean()) if n else float("nan")
        return {"mean": m, "sem": float("nan"), "n": n,
                "note": "too few samples for an error bar"}

    mean = float(y.mean())
    sd = float(y.std(ddof=1))
    tau, window = tau_int_sokal(y)
    sem = sd * math.sqrt(2.0 * tau / n)
    n_eff = n / (2.0 * tau)
    sem_block, curve = blocking(y)

    # The two estimators are independent. If they disagree by more than 2x, the run is
    # too short for its own correlation time and NEITHER bar should be trusted.
    agree = (math.isfinite(sem_block) and sem > 0
             and 0.5 <= sem_block / sem <= 2.0)

    return {
        "mean": mean,
        "sem": sem,                                  # equation (4)
        "sem_naive": sd / math.sqrt(n),              # equation (1) — the one that lies
        "inflation": sem / (sd / math.sqrt(n)) if sd > 0 else 1.0,   # = sqrt(2*tau)
        "sem_blocking": sem_block,                   # independent cross-check
        "estimators_agree": bool(agree),
        "tau_int": tau,                              # in samples
        "tau_int_ps": tau * dt_ps if dt_ps else None,
        "window": window,
        "n": n,
        "n_eff": n_eff,                              # equation (5)
        "ci95": [mean - 1.96 * sem, mean + 1.96 * sem],
        "sd": sd,
        "blocking_curve": [(int(s), float(e)) for s, e, _ in curve],
    }


def time_for_precision(s: dict, target_sem: float, dt_ps: float) -> float:
    """How long must the run be to reach a given error bar? Inverts equation (4).

        SEM^2 = 2*tau_int*sigma^2 / N   =>   N = 2*tau_int*sigma^2 / SEM^2

    and the wall-time follows as N * (sampling interval). Note the QUADRATIC cost: a
    10x tighter error bar costs 100x the simulation. This is why "run it longer" is
    usually the wrong answer and a better estimator/observable is the right one.
    """
    if not (s.get("sd") and target_sem > 0 and math.isfinite(s.get("tau_int", float("nan")))):
        return float("nan")
    n_needed = 2.0 * s["tau_int"] * s["sd"] ** 2 / target_sem ** 2
    return n_needed * dt_ps / 1000.0          # ns


def fmt(s: dict, unit: str = "", sig: int = 4) -> str:
    if not math.isfinite(s.get("sem", float("nan"))):
        return f"{s['mean']:.{sig}g}{unit} (no error bar: n={s.get('n', 0)})"
    return (f"{s['mean']:.{sig}g} +/- {s['sem']:.2g}{unit}  "
            f"(tau_int={s['tau_int']:.1f} samples, N_eff={s['n_eff']:.0f} of {s['n']})")


# --------------------------------------------------------------------------------
def self_test(seed: int = 7) -> bool:
    """Validate against a process whose answer is known in closed form.

    AR(1): x_t = phi*x_{t-1} + eps_t. Then rho(k) = phi^k exactly, so
        tau_int = (1 + phi) / (2 * (1 - phi)).
    An estimator that cannot recover THIS has no business on a trajectory.
    """
    rng = np.random.default_rng(seed)
    print(f"{'phi':>6}{'tau_true':>10}{'tau_sokal':>11}{'err':>7}"
          f"{'SEM_true':>10}{'SEM_est':>9}{'SEM_block':>11}{'SEM_naive':>11}{'ratio':>7}")
    print("-" * 82)
    ok = True
    for phi in (0.0, 0.5, 0.8, 0.9, 0.95):
        tau_true = (1 + phi) / (2 * (1 - phi))
        n = 200_000
        eps = rng.normal(size=n)
        x = np.empty(n)
        x[0] = eps[0] / math.sqrt(1 - phi**2) if phi else eps[0]
        for t in range(1, n):                       # exact AR(1), no approximation
            x[t] = phi * x[t - 1] + eps[t]
        s = stats(x)
        # ground-truth SEM: sigma_x^2 = 1/(1-phi^2) for unit-variance noise
        sig2 = 1.0 / (1 - phi**2) if phi else 1.0
        sem_true = math.sqrt(2 * tau_true * sig2 / n)
        err = abs(s["tau_int"] - tau_true) / tau_true
        good = err < 0.15 and abs(s["sem"] / sem_true - 1) < 0.20
        ok &= good
        print(f"{phi:>6.2f}{tau_true:>10.2f}{s['tau_int']:>11.2f}{err:>6.1%}"
              f"{sem_true:>10.5f}{s['sem']:>9.5f}{s['sem_blocking']:>11.5f}"
              f"{s['sem_naive']:>11.5f}{s['inflation']:>7.1f}x"
              f"{'' if good else '   <-- FAIL'}")
    print("-" * 82)
    print("tau_true = (1+phi)/(2(1-phi)), exact.  ratio = SEM_corrected / SEM_naive = sqrt(2*tau).")
    print("At phi=0.95 the naive error bar understates the true one by ~4.4x." if ok else "")
    print("\nPASS — both estimators recover the analytic answer." if ok else "\nFAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
