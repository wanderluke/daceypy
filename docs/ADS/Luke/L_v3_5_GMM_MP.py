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
    
    base_out = Path(r"C:\Users\lgao111\OneDrive - The University of Auckland\Desktop\Data Tests") / "L_v3_5_GMM_MP"
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = base_out / "1000_0.5yrs"

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
    h_max = 1000.0

    r_mid = rE + (h_min + h_max) / 2.0
    dr = 0.5 * (h_max - h_min)
    r_DA = r_mid + dr * DA(1)

    NN = 14918.0
    mu_h = np.array([476.683941294885, 482.342860009956, 490.151665365949, 
                    558.280898165623, 899.417557734721, 1204.31574424304, 
                    1458.28462767136, 1624.40322683476], dtype=float)
    sigma_h = np.array([110.641593027782, 0.0559395368590357, 0.0607795063179836, 
                        13.9506721367245, 174.116652760815, 16.6523167430062, 
                        37.179036553829, 188.203606763823], dtype=float)
    weight = np.array([0.292891771482999, 0.159871844707075, 0.124328202931273, 
                        0.223878903525426, 0.103716700730265, 0.0463041509259556, 
                        0.041988751121499, 0.00701967457550748], dtype=float)
    
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

if __name__ == "__main__":
    main()