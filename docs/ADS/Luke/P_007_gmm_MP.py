from daceypy import DA, array, ADS
from pathlib import Path
import numpy as np
import time

from main_model import (
    advanced_characteristics_ADS,
    eFunGMM,
)
from Export_v3_r_u import export_ads_v3_r_u

def main():
    
    base_out = Path(r"C:\PhD_Luca\Data Tests") / "exp_010_gmm_Sw_gmm"
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = base_out

    DA.init(4, 1)
    DA.setEps(1e-40)

    split_tol = 1e-6
    split_depth = 20

    mu = 3.986004418e5  # km^3/s^2
    rE = 6378.0         # km

    t0 = 0.0
    tf = 3600.0 * 24.0 * 365.25  
    Ts = 100
    tgrid = np.linspace(t0, tf, Ts)

    h_min = 200.0
    h_max = 2000.0

    r_mid = rE + (h_min + h_max) / 2.0
    dr = 0.5 * (h_max - h_min)
    r_DA = r_mid + dr * DA(1)
    
    h_reentry = 200.0
    r_reentry = rE + h_reentry

    NN = 14852.0
    mu_h = np.array([361.64, 382.93, 432.55, 441.32, 459.71,
                    487.02, 487.02, 540.45, 557.64, 557.64,
                    634.68, 692.38, 766.99, 782.39, 784.87,
                    821.6, 870.5, 977.63, 1083.0, 1162.2,
                    1205.3, 1451.3, 1457.6, 1474.5, 1555.0], dtype=float)

    sigma_h = np.array([8.9804, 78.406, 74.861, 71.328, 40.4,
                        7.1422, 7.1422, 29.473, 19.368, 19.367,
                        13.106, 28.258, 105.66, 104.37, 17.833,
                        92.797, 55.118, 22.652, 17.85, 19.368,
                        16.337, 161.56, 36.954, 136.47, 186.26], dtype=float)

    weight = np.array([0.055668, 0.03049, 0.022085, 0.016423, 0.01585,
                    0.070333, 0.25794, 0.028313, 0.13124, 0.14168,
                    0.026276, 0.01482, 0.0012648, 0.0078566, 0.020052,
                    0.00012413, 0.018526, 0.021095, 0.014682, 0.0080837,
                    0.046765, 0.007764, 0.041025, 0.0013653, 0.0072966], dtype=float)

    dom = array([r_DA])
    init_domain = ADS(dom)
    init_list = ADS.eval(
        [init_domain],
        split_tol,
        split_depth,
        lambda d: eFunGMM(d, NN, mu_h, sigma_h, weight, rE)
    ) 
    final_list = [
        ADS(dom.box, dom.nsplit, array([dom.box[0], dom.manifold[0]]))
        for dom in init_list
    ]
    
    # ADVANCED ADS application + export at each time step

    out_dir0 = out_dir / f"{0:04d}" # export initial condition at t=t0
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
        final_list = ADS.eval(
            final_list, split_tol, split_depth,
            lambda domain: advanced_characteristics_ADS(
                domain, t0=float(tgrid[i-1]), tf=float(tgrid[i]),
                mu=mu, rE=rE, idx_r0=1
            )
        )

        filtered_list = []
        for ads_el in final_list:
            lb, ub = ads_el.manifold[0].bound()   # first component = radius
            if ub >= r_reentry:
                filtered_list.append(ads_el)

        final_list = filtered_list

        out_dir_i = out_dir / f"{i:04d}"
        export_ads_v3_r_u(
            final_list=final_list,
            out_dir=str(out_dir_i),
            rEarth=rE,
            h_range=(h_min, h_max),
            idx_r0=1,
            file_prefix="dom",
        )
    print("execution time v3 advanced ADS Multi Patch + export:", time.time() - start_adv) # 315 s 
    print("Final list filtered length =", len(final_list))

if __name__ == "__main__":
    main()