from daceypy import DA, array, ADS
from pathlib import Path
import numpy as np
import time

from main_model import (
    compute_u_list,
    propagation,
    naive_characteristics_ADS,
    advanced_characteristics_ADS,
)
from Export_v3_r_u import export_ads_v3_r_u
from Export_v3_r_w import export_ads_v3_r_w

def main():
    
    base_out = Path(r"C:\Users\lgao111\OneDrive - The University of Auckland\Desktop\Data Tests") / "L_v3_1_const"
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir1 = base_out / "r_u_2000_0.5yrs"
    out_dir2 = base_out / "r_w_2000_0.5yrs"

    DA.init(4, 1)
    DA.setEps(1e-40)

    split_tol = 1e-6
    split_depth = 20

    mu = 3.986004418e5  # km^3/s^2
    rE = 6378.0         # km

    t0 = 0.0
    tf = 3600.0 * 12.0 * 365.25  
    Ts = 100
    tgrid = np.linspace(t0, tf, Ts)

    h_min = 200.0
    h_max = 2000.0

    r_mid = rE + (h_min + h_max) / 2.0
    dr = 0.5 * (h_max - h_min)
    r_DA = r_mid + dr * DA(1)

    NN = 10000.0 # w0 costante
    deltah = (h_max - h_min)
    w_DA = DA((1.0 / (4.0 * np.pi)) * (NN / deltah))

    dom = array([r_DA])
    man = array([r_DA, w_DA])
    init_list = [ADS(dom, [], man)]

    # Point 3 - DA + ADS    
    
    # NAIVE ADS application
    # final_lists_rw = []
    # final_lists_u  = []
    # start_basic = time.time()
    # for i in range(1, len(tgrid)):
    #     final_list = ADS.eval(
    #         init_list, split_tol, split_depth,
    #         lambda domain: naive_characteristics_ADS(
    #             domain, t0=t0, tf=float(tgrid[i]),
    #             mu=mu, rE=rE, w0=w_DA, idx_r0=1
    #         )
    #     )
    #     final_lists_rw.append(final_list)
    #     final_lists_u.append(compute_u_list(final_list))
    #     print("time", tgrid[i], "reached!  Ndomains =", len(final_list))
    # print("execution time v3 naive ADS:", time.time() - start_basic)

    # ADVANCED ADS application
    final_lists_rw = []
    final_lists_u  = []
    final_list = init_list.copy()
    final_lists_rw.append(final_list)
    final_lists_u.append(compute_u_list(final_list)) # questo è u a t0

    start_adv = time.time()
    for i in range(Ts - 1):
        final_list = ADS.eval(
            final_list, split_tol, split_depth,
            lambda domain: advanced_characteristics_ADS(
                domain, t0=tgrid[i], tf=tgrid[i+1],
                mu=mu, rE=rE, idx_r0=1
            )
        )
        final_lists_rw.append(final_list)
        final_lists_u.append(compute_u_list(final_list))
        print("time", tgrid[i+1], "reached!")
    print("execution time v3 advanced ADS:", time.time() - start_adv) 
    export_ads_v3_r_w(final_list=final_list, out_dir=str(out_dir2), rEarth=rE, 
                      h_range=(h_min, h_max), idx_r0=1, file_prefix="dom")
    export_ads_v3_r_u(final_list=final_list, out_dir=str(out_dir1), rEarth=rE,
                      h_range=(h_min, h_max), idx_r0=1, file_prefix="dom")

if __name__ == "__main__":
    main()