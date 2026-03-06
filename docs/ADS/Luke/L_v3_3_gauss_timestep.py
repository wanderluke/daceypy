from daceypy import DA, array, ADS
from pathlib import Path
import numpy as np
import time
from daceypy.op import exp, sqrt, tanh

from main_model import (
    advanced_characteristics_ADS,
)
from Export_v3_r_u import export_ads_v3_r_u

def main():
    
    base_out = Path(r"C:\Users\lgao111\OneDrive - The University of Auckland\Desktop\Data Tests") / "L_v3_3_gauss_timestep"
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

    NN = 10000.0 # w0 gaussiana con 1 patch
    mu_h = 750.0      # km
    sigma_h = 150.0   # km
    h_DA = r_DA - rE  # quota in km (DA)
    n_DA = (NN / (sigma_h * sqrt(2.0*np.pi))) * exp(-0.5 * ((h_DA - mu_h)/sigma_h)**2) # n(h): oggetti/km, integra a NN su (-inf,inf)
    w_DA = n_DA / (4.0*np.pi) # w(h): oggetti/(km·sr), isotropo

    dom = array([r_DA])
    man = array([r_DA, w_DA])
    init_list = [ADS(dom, [], man)]
    final_list = init_list.copy()

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
    print("execution time v3 advanced ADS + export:", time.time() - start_adv)  

if __name__ == "__main__":
    main()