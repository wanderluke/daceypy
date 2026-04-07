from daceypy import DA, array, ADS
from pathlib import Path
import numpy as np
import time
import shutil
from sklearn.mixture import GaussianMixture

from main_model import (
    advanced_characteristics_ADS,
    eFunGMM,
)
from Export_v3_r_u import export_ads_v3_r_u


# =========================================================
# IMPORTANT:
# If your local DACEyPy build uses a different point-eval call,
# adapt ONLY this function.
# =========================================================
def _eval_da_1d(da_obj, xi):
    val = da_obj.eval([xi])   # <-- adapt only this line if needed
    try:
        return float(val.cons())
    except AttributeError:
        return float(val)


def _invert_r_da_to_local(r_da, r_target, tol=1e-10, itmax=15):
    # Newton on manifold radius map r(xi), xi in [-1, 1].
    xi = 0.0
    dr_dxi_da = r_da.deriv(1)
    for _ in range(itmax):
        r_x = _eval_da_1d(r_da, xi)
        err = r_x - r_target
        if abs(err) <= tol:
            return max(-1.0, min(1.0, xi))

        dr_dxi = _eval_da_1d(dr_dxi_da, xi)
        if abs(dr_dxi) < 1e-12:
            break

        xi_new = xi - err / dr_dxi
        xi_new = max(-1.0, min(1.0, xi_new))
        if abs(xi_new - xi) <= 1e-12:
            return xi_new
        xi = xi_new

    return max(-1.0, min(1.0, xi))


# =========================================================
# Evaluate final_list directly on a uniform altitude grid.
# Assumes final patch altitude boxes do NOT overlap.
# =========================================================
def evaluate_final_list_on_uniform_grid(final_list, h_grid, rE):
    h_grid = np.asarray(h_grid, dtype=float)
    w_grid = np.zeros_like(h_grid, dtype=float)

    if len(final_list) == 0:
        return w_grid

    patch_boxes = []
    for dom in final_list:
        # Use propagated manifold radius bounds for physical patch selection.
        r_da = dom.manifold[0]
        r_lb, r_ub = np.asarray(r_da.bound(), dtype=float)
        h_lb = r_lb - rE
        h_ub = r_ub - rE
        if h_lb > h_ub:
            h_lb, h_ub = h_ub, h_lb
        patch_boxes.append((h_lb, h_ub, dom))

    patch_boxes.sort(key=lambda x: x[0])

    for k, h in enumerate(h_grid):
        r_target = float(h) + rE
        for h_lb, h_ub, dom in patch_boxes:
            if h_lb <= h <= h_ub:
                dr = _invert_r_da_to_local(dom.manifold[0], r_target)
                w_grid[k] = _eval_da_1d(dom.manifold[1], dr)
                break

    return w_grid


# =========================================================
# Source evaluated on the same uniform altitude grid
# Returns Sw(h) [objects/(km sr s)]
# =========================================================
def source_w_grid(h_grid):
    h_grid = np.asarray(h_grid, dtype=float)

    Lambda0_year = 3000.0
    Lambda0 = Lambda0_year / (365.25 * 24.0 * 3600.0)   # objects/s

    mu_h = np.array([550.0, 1200.0], dtype=float)
    sigma_h = np.array([40.0, 80.0], dtype=float)
    weight = np.array([0.7, 0.3], dtype=float)
    weight = weight / np.sum(weight)

    f = np.zeros_like(h_grid, dtype=float)
    for wj, muj, sj in zip(weight, mu_h, sigma_h):
        if sj <= 0.0:
            continue
        f += (wj / (sj * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((h_grid - muj) / sj) ** 2
        )

    Sw = (Lambda0 / (4.0 * np.pi)) * f
    return Sw


# =========================================================
# Fit one GMM to total profile w(h)
# =========================================================
def fit_gmm_from_profile(
    h_grid,
    w_grid,
    n_samples_fit=20000,
    random_state=0,
    k_mode="bic",
    k_target=12,
    k_min=1,
    k_max=20,
):
    h_grid = np.asarray(h_grid, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float)
    w_grid = np.maximum(w_grid, 0.0)

    # MATLAB-like settings on the already-binned altitude grid (5 km spacing).
    binw = float(np.median(np.diff(h_grid)))
    sigma_min_km = 5.0
    reg_val = sigma_min_km ** 2
    k_target = int(k_target)
    k_min = int(k_min)
    k_max = int(k_max)

    # n(h) = 4*pi*w(h), so total objects:
    NN_tot = 4.0 * np.pi * np.trapezoid(w_grid, h_grid)

    # Build weighted expanded sample directly from h_grid bins (already 5 km grouped).
    centers = h_grid.copy()
    counts_float = np.maximum(w_grid, 0.0) * binw

    counts_sum = float(np.sum(counts_float))
    if counts_sum <= 0.0:
        raise ValueError("Histogram mass is non-positive. Cannot fit GMM.")

    counts_scaled = np.maximum(0, np.rint((counts_float / counts_sum) * n_samples_fit).astype(int))
    mask = counts_scaled > 0
    if not np.any(mask):
        raise ValueError("Expanded sample is empty. Cannot fit GMM.")

    h_samples = np.repeat(centers[mask], counts_scaled[mask]).reshape(-1, 1)
    n_cap = min(len(h_samples), int(np.count_nonzero(mask)), k_max)

    if n_cap < 1:
        raise ValueError("No valid number of GMM components available.")

    if k_mode == "fixed":
        n_components_eff = min(max(1, k_target), n_cap)
        gm = GaussianMixture(
            n_components=n_components_eff,
            covariance_type="full",
            reg_covar=reg_val,
            n_init=50,
            random_state=random_state
        )
        gm.fit(h_samples)
    else:
        k_start = min(max(1, k_min), n_cap)
        best_bic = np.inf
        best_gm = None
        for k in range(k_start, n_cap + 1):
            gm_k = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=reg_val,
                n_init=20,
                random_state=random_state
            )
            gm_k.fit(h_samples)
            bic_k = gm_k.bic(h_samples)
            if bic_k < best_bic:
                best_bic = bic_k
                best_gm = gm_k

        if best_gm is None:
            raise RuntimeError("BIC model selection failed to produce a valid GMM.")

        gm = best_gm

    mu_fit = np.asarray(gm.means_, dtype=float).ravel()
    sigma_fit = np.sqrt(np.asarray(gm.covariances_, dtype=float).reshape(-1))
    weight_fit = np.asarray(gm.weights_, dtype=float).ravel()

    idx = np.argsort(mu_fit)
    mu_fit = mu_fit[idx]
    sigma_fit = sigma_fit[idx]
    weight_fit = weight_fit[idx]
    sigma_fit = np.maximum(sigma_fit, sigma_min_km)
    weight_fit = weight_fit / np.sum(weight_fit)

    return NN_tot, mu_fit, sigma_fit, weight_fit

def main():

    base_out = Path(r"C:\PhD_Luca\Data Tests") / "exp_012_const_Sw_gmm_discrete_B"
    if base_out.exists():
        shutil.rmtree(base_out)
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = base_out

    DA.init(4, 1)
    DA.setEps(1e-40)

    split_tol = 1e-6
    split_depth = 20

    mu = 3.986004418e5  # km^3/s^2
    rE = 6378.0         # km

    t0 = 0.0
    tf = 3600.0 * 12.0 * 365.25
    Ts = 2
    tgrid = np.linspace(t0, tf, Ts)

    h_min = 200.0
    h_max = 2000.0

    # uniform altitude grid: 200,205,...,2000
    h_grid = np.arange(h_min, h_max + 5.0, 5.0)

    r_mid = rE + (h_min + h_max) / 2.0
    dr = 0.5 * (h_max - h_min)
    r_DA = r_mid + dr * DA(1)

    h_reentry = 200.0
    r_reentry = rE + h_reentry

    # initial population: constant w0 on [h_min, h_max], as in P_004
    NN = 10000.0
    deltah = h_max - h_min
    w_DA = DA((1.0 / (4.0 * np.pi)) * (NN / deltah))

    dom = array([r_DA])
    init_domain = ADS(dom)
    man = array([r_DA, w_DA])
    final_list = [ADS(dom, [], man)]

    # export initial condition
    out_dir0 = out_dir / f"{0:04d}"
    export_ads_v3_r_u(
        final_list=final_list,
        out_dir=str(out_dir0),
        rEarth=rE,
        h_range=(h_min, h_max),
        idx_r0=1,
        file_prefix="dom",
    )

    start_adv = time.time()

    for i in range(1, Ts):

        # 1) propagate current state
        final_list = ADS.eval(
            final_list,
            split_tol,
            split_depth,
            lambda domain: advanced_characteristics_ADS(
                domain,
                t0=float(tgrid[i-1]),
                tf=float(tgrid[i]),
                mu=mu,
                rE=rE,
                idx_r0=1
            )
        )

        # 2) filter reentry
        filtered_list = []
        for ads_el in final_list:
            lb, ub = ads_el.manifold[0].bound()
            if ub >= r_reentry:
                filtered_list.append(ads_el)

        final_list = filtered_list

        # 3) evaluate current final_list on uniform 5-km grid
        w_curr_grid = evaluate_final_list_on_uniform_grid(
            final_list=final_list,
            h_grid=h_grid,
            rE=rE
        )

        # 4) evaluate source on same grid
        dt_sec = float(tgrid[i] - tgrid[i-1])
        w_inj_grid = dt_sec * source_w_grid(h_grid)

        # 5) total profile
        w_tot_grid = w_curr_grid + w_inj_grid

        # 6) fit one GMM to the total profile
        NN_tot, mu_fit, sigma_fit, weight_fit = fit_gmm_from_profile(
            h_grid=h_grid,
            w_grid=w_tot_grid,
            n_samples_fit=30000,
            random_state=i,
            k_mode="bic",
            k_min=1,
            k_max=20,
        )

        # 7) rebuild final_list from fitted GMM
        init_list = ADS.eval(
            [init_domain],
            split_tol,
            split_depth,
            lambda d: eFunGMM(d, float(NN_tot), mu_fit, sigma_fit, weight_fit, rE)
        )

        final_list = [
            ADS(dom.box, dom.nsplit, array([dom.box[0], dom.manifold[0]]))
            for dom in init_list
        ]

        # 8) export
        out_dir_i = out_dir / f"{i:04d}"
        export_ads_v3_r_u(
            final_list=final_list,
            out_dir=str(out_dir_i),
            rEarth=rE,
            h_range=(h_min, h_max),
            idx_r0=1,
            file_prefix="dom",
        )

    print("execution time method 2:", time.time() - start_adv)
    print("Final list length =", len(final_list))

if __name__ == "__main__":
    main()