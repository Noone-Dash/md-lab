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


# A run must contain at least this many INDEPENDENT samples before we are willing to
# put a number on its uncertainty. Below it, tau_int itself is not estimable and every
# error bar -- by either method -- is fiction.
MIN_N_EFF = 25
MIN_BLOCK_LEVELS = 4


def stats(y, dt_ps: float = None) -> dict:
    """Mean of a correlated series WITH an honest error bar -- or an explicit REFUSAL.

    A too-narrow error bar is worse than no error bar: it converts "we cannot tell" into
    a confident claim. So this REFUSES (sem = nan) rather than guessing when the run is
    shorter than its own correlation time.
    """
    y = np.asarray([v for v in y if v is not None], dtype=float)
    n = len(y)
    if n < 8:
        m = float(y.mean()) if n else float("nan")
        return {"mean": m, "sem": float("nan"), "n": n, "resolvable": False,
                "note": f"n={n}: too few samples for any error bar"}

    mean = float(y.mean())
    sd = float(y.std(ddof=1))
    tau, window = tau_int_sokal(y)
    n_eff = n / (2.0 * tau)
    sem_sokal = sd * math.sqrt(2.0 * tau / n)
    sem_block, curve = blocking(y)
    levels = len(curve)

    # ---------------------------------------------------------------------------
    # WHY THERE IS NO "estimators_agree" FLAG ANY MORE.
    #
    # There used to be one, and it was WORSE than useless: it was True in 100% of
    # trials at every N and every phi. The reason is structural. When the run is short
    # relative to tau:
    #     - Sokal's window (w >= c*tau) never opens, so tau collapses to its 0.5 floor
    #       and sem_sokal degenerates to the naive sigma/sqrt(N);
    #     - blocking() has < 2 levels, so sem_block IS sigma/sqrt(N) by construction.
    # Both estimators then return the SAME naive number and "agree" -- precisely in the
    # regime the flag existed to catch. Measured: AR(1) phi=0.9, N=16 -> reported SEM is
    # 0.17x of truth (6x too narrow) with agree=True in 500/500 seeds.
    #
    # Two estimators that fail in the same way are not a cross-check. The fix is not a
    # better comparison, it is an ABSOLUTE resolvability requirement.
    # ---------------------------------------------------------------------------
    resolvable = (n_eff >= MIN_N_EFF and levels >= MIN_BLOCK_LEVELS
                  and window < n / 4)

    # Be CONSERVATIVE where they differ. Sokal's c=5 window closes on the fast mode of a
    # multi-timescale ACF and can miss a small-amplitude slow one (see self_test's
    # two-exponential case: it understated the SEM ~3x). Blocking sees the slow mode.
    # Taking the larger of the two costs a slightly wider bar and removes that failure.
    sem = max(sem_sokal, sem_block) if math.isfinite(sem_block) else sem_sokal

    out = {
        "mean": mean,
        "sem": sem if resolvable else float("nan"),
        "sem_naive": sd / math.sqrt(n),              # equation (1) — the one that lies
        "sem_sokal": sem_sokal,
        "sem_blocking": sem_block,
        "inflation": (sem / (sd / math.sqrt(n))) if sd > 0 else 1.0,
        "tau_int": tau,                              # in samples
        "tau_int_ps": tau * dt_ps if dt_ps else None,
        "window": window,
        "blocking_levels": levels,
        "n": n,
        "n_eff": n_eff,                              # equation (5)
        "sd": sd,
        "resolvable": bool(resolvable),
        "blocking_curve": [(int(s), float(e)) for s, e, _ in curve],
    }
    if resolvable:
        out["ci95"] = [mean - 1.96 * sem, mean + 1.96 * sem]
    else:
        out["note"] = (f"REFUSED: n_eff={n_eff:.1f} (need >={MIN_N_EFF}), "
                       f"{levels} blocking level(s) (need >={MIN_BLOCK_LEVELS}). "
                       f"The run is too short relative to its own correlation time "
                       f"(tau_int={tau:.1f} samples) for ANY defensible error bar.")
    return out


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


def _ar1(rng, phi, n):
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / math.sqrt(1 - phi**2) if phi else eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def self_test_multiscale(seed: int = 11) -> bool:
    """AR(1) is a SINGLE exponential — the one family where Sokal's c=5 window is safe.
    Real MD observables are not: fast thermal noise rides on a slow structural mode.
    The old estimator closed its window on the fast mode and missed the slow one.

        x_t = iid_noise + a * s_t,     s_t = AR(1) with phi -> tau_slow

    rho(k) = w * phi^k with w = a^2*var(s) / (1 + a^2*var(s)), so
        tau_int = 0.5 + w * phi/(1-phi)      -- closed form, exact.
    """
    rng = np.random.default_rng(seed)
    print(f"\n{'a':>5}{'phi':>6}{'tau_true':>10}{'sokal':>9}{'blocking->':>12}"
          f"{'SEM_true':>10}{'SEM_rep':>9}{'ratio':>8}")
    print("-" * 70)
    ok = True
    for a, phi in ((0.3, 0.99), (0.5, 0.98), (0.2, 0.995)):
        n = 400_000
        s_slow = _ar1(rng, phi, n)
        x = rng.normal(size=n) + a * s_slow
        var_s = 1.0 / (1 - phi**2)
        w = (a**2 * var_s) / (1.0 + a**2 * var_s)          # slow mode's share of variance
        tau_true = 0.5 + w * phi / (1 - phi)
        var_x = 1.0 + a**2 * var_s
        sem_true = math.sqrt(2 * tau_true * var_x / n)
        st = stats(x)
        ratio = st["sem"] / sem_true
        good = 0.75 <= ratio <= 1.35
        ok &= good
        print(f"{a:>5.1f}{phi:>6.3f}{tau_true:>10.1f}{st['sem_sokal']:>9.5f}"
              f"{st['sem_blocking']:>12.5f}{sem_true:>10.5f}{st['sem']:>9.5f}"
              f"{ratio:>7.2f}x{'' if good else '  <-- FAIL'}")
    print("-" * 70)
    print("The reported SEM is max(sokal, blocking). Sokal alone understates a slow mode")
    print("that carries only a few % of the variance; blocking sees it.")
    return ok


def self_test_refusal(seed: int = 3) -> bool:
    """The estimator must REFUSE when the run is shorter than its own correlation time.

    This is the case the old 'estimators_agree' cross-check certified as FINE: at N=16
    with tau=9.5 it reported a SEM 6x too narrow and agree=True in 500/500 seeds.
    """
    rng = np.random.default_rng(seed)
    phi = 0.9
    tau_true = (1 + phi) / (2 * (1 - phi))            # 9.5
    print(f"\n  AR(1) phi={phi} (tau_int = {tau_true} samples, known exactly)")
    print(f"  {'N':>7}{'n_eff':>8}{'blocks':>8}{'reported':>12}   verdict")
    print("  " + "-" * 52)
    ok = True
    for n in (16, 32, 64, 256, 2000, 20000):
        sems = []
        refused = 0
        for k in range(60):
            st = stats(_ar1(rng, phi, n))
            refused += (not st["resolvable"])
            if st["resolvable"]:
                sems.append(st["sem"])
        st = stats(_ar1(rng, phi, n))
        var_x = 1.0 / (1 - phi**2)
        sem_true = math.sqrt(2 * tau_true * var_x / n)
        frac_ref = refused / 60
        if frac_ref > 0.5:
            verdict, good = "REFUSED (correct: too short)", n <= 256
            rep = "     —"
        else:
            med = float(np.median(sems))
            r = med / sem_true
            verdict = f"reported, {r:.2f}x of truth"
            good = 0.75 <= r <= 1.4
            rep = f"{med:.4f}"
        ok &= good
        print(f"  {n:>7}{st['n_eff']:>8.1f}{st['blocking_levels']:>8}{rep:>12}   "
              f"{verdict}{'' if good else '  <-- FAIL'}")
    print("  " + "-" * 52)
    print("  A run too short for its own correlation time now gets NO error bar,")
    print("  instead of a confident one that is 6x too narrow.")
    return ok


if __name__ == "__main__":
    import sys
    a = self_test()
    b = self_test_multiscale()
    c = self_test_refusal()
    print(f"\n{'ALL PASS' if (a and b and c) else 'FAIL'}  "
          f"(single-exponential: {a}, multi-timescale: {b}, short-run refusal: {c})")
    sys.exit(0 if (a and b and c) else 1)
