"""Unpenalized calibration slope, with finite-MLE and convergence checks.

The objective is the SUM of Bernoulli negative log likelihoods, not its mean.
Both convergence tolerances apply in the original (intercept, logit(p)) basis.
Invalid or non-estimable inputs raise CalibrationError with a specific reason.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

TOL = 1e-10
MAX_ITER = 100


class CalibrationError(ValueError):
    """A finite, identifiable, converged calibration slope is unavailable."""


def _prepare(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not y.size:
        raise CalibrationError("y and p must be nonempty, aligned one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise CalibrationError("non-finite input")
    if not np.isin(y, [0, 1]).all():
        raise CalibrationError("labels must be binary 0/1")
    if y.min() == y.max():
        raise CalibrationError("single class: finite intercept MLE does not exist")
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(pc / (1 - pc))
    if x.min() == x.max():
        raise CalibrationError("constant predictions after clipping: slope is not identifiable")
    x0, x1 = x[y == 0], x[y == 1]
    # In one dimension plus an intercept these are exactly the two possible
    # separating directions, including a shared endpoint (quasi-separation).
    for left, right in ((x0, x1), (x1, x0)):
        if left.max() <= right.min():
            kind = "complete" if left.max() < right.min() else "quasi"
            raise CalibrationError(f"{kind} separation: unbounded likelihood direction")
    center, scale = float(x.mean()), float(x.std())
    return y, (x - center) / scale, center, scale


def _terms(beta, z, y):
    eta = beta[0] + beta[1] * z
    # Signed softplus and residuals avoid cancellation for near-boundary risks.
    signed = (1 - 2 * y) * eta
    loss = float(np.sum(np.logaddexp(0, signed), dtype=np.longdouble))
    residual = np.where(y == 0, expit(eta), -expit(-eta))
    weight = expit(eta) * expit(-eta)
    total = lambda a: float(np.sum(a, dtype=np.longdouble))
    gradient = np.array([total(residual), total(residual * z)])
    hessian = np.array([[total(weight), total(weight * z)],
                        [total(weight * z), total(weight * z * z)]])
    return loss, gradient, hessian


def _diagnostics(beta, z, y, center, scale):
    loss, g, h = _terms(beta, z, y)
    if not np.isfinite(beta).all() or not np.isfinite(h).all():
        raise np.linalg.LinAlgError("non-finite iterate or information")
    np.linalg.cholesky(h)
    step = np.linalg.solve(h, g)
    raw_gradient = np.array([g[0], center * g[0] + scale * g[1]])
    raw_step = np.array([step[0] - center * step[1] / scale, step[1] / scale])
    return {"nll": loss, "gradient_norm": float(np.linalg.norm(raw_gradient)),
            "newton_step_norm": float(np.linalg.norm(raw_step))}, step


def _converged(info):
    return info["gradient_norm"] < TOL and info["newton_step_norm"] < TOL


def _newton(beta, z, y, center, scale):
    for _ in range(MAX_ITER):
        info, step = _diagnostics(beta, z, y, center, scale)
        if _converged(info):
            return beta
        loss, g, _ = _terms(beta, z, y)
        damping = 1.0
        # At floating-point likelihood resolution, a full Newton step can still
        # reduce the score. Never mistake a damped step for convergence.
        slack = 8 * np.finfo(float).eps * max(1.0, abs(loss))
        for _ in range(50):
            proposal = beta - damping * step
            new_loss = _terms(proposal, z, y)[0]
            if np.isfinite(new_loss) and new_loss <= loss - 1e-4 * damping * float(g @ step) + slack:
                beta = proposal
                break
            damping *= 0.5
        else:
            break
    return beta


def fit_calibration(y, p):
    """Return slope and aggregate numerical diagnostics (no fitted predictions)."""
    y, z, center, scale = _prepare(y, p)
    initial = np.array([np.log(y.mean() / (1 - y.mean())), 0.0])
    beta, failures = initial.copy(), []
    try:
        beta = _newton(beta, z, y, center, scale)
        info, _ = _diagnostics(beta, z, y, center, scale)
        if _converged(info):
            return {**info, "intercept": float(beta[0] - center * beta[1] / scale),
                    "slope": float(beta[1] / scale), "method": "Newton/IRLS"}
        failures.append(f"Newton residuals {info}")
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        failures.append(f"Newton: {exc}")
        beta = initial.copy()
    try:
        def objective(b):
            loss, gradient, _ = _terms(b, z, y)
            return loss, gradient

        result = minimize(objective, beta, jac=True, method="BFGS",
                          options={"gtol": 1e-12, "maxiter": 2000})
        info, _ = _diagnostics(result.x, z, y, center, scale)
        # Neither success=True nor success=False overrides these two checks.
        if _converged(info):
            return {**info, "intercept": float(result.x[0] - center * result.x[1] / scale),
                    "slope": float(result.x[1] / scale), "method": "BFGS"}
        failures.append(f"BFGS residuals {info}; {result.message}")
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError) as exc:
        failures.append(f"BFGS: {exc}")
    raise CalibrationError("both solvers failed convergence: " + "; ".join(failures))


def calib_slope_mle(y, p):
    """Slope of y ~ intercept + logit(clip(p, 1e-6, 1-1e-6)); raises on failure."""
    return fit_calibration(y, p)["slope"]
