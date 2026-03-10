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
    
    base_out = Path(r"C:\Users\lgao111\OneDrive - The University of Auckland\Desktop\Data Tests") / "L_v3_5_gmm_MP"
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = base_out / "2000_0.5yrs"

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
    
    h_reentry = 180.0
    r_reentry = rE + h_reentry

    NN = 14852.0

    mu_h = np.array([
        361.619896224590,
        385.371745991748,
        457.416476171235,
        487.034101421112,
        548.228425313329,
        557.845773745268,
        635.414874525317,
        776.279793723457,
        776.694061791633,
        803.420523589607,
        976.229821489221,
        1084.91139153144,
        1199.96465619125,
        1457.71936068705,
        1529.68622746543
    ], dtype=float)

    sigma_h = np.array([
        9.12825779573521,
        73.7306198426024,
        39.1007607755134,
        7.14765789502748,
        26.4428488360360,
        18.9889655311130,
        13.6238464580194,
        102.027086403864,
        74.5110665276919,
        93.5619018265373,
        23.1431880268199,
        20.2026477454182,
        21.5661915097850,
        37.0399832423732,
        189.906810227754
    ], dtype=float)

    weight = np.array([
        0.0565622973231700,
        0.0473043762466881,
        0.0319537673580596,
        0.328811808557579,
        0.0595549273826328,
        0.244462496317079,
        0.0259167613382126,
        0.0224367614908243,
        0.0292032040002305,
        0.0123833922320526,
        0.0214768864137829,
        0.0152621593009483,
        0.0541035722918357,
        0.0412066942709584,
        0.00936089547594593
    ], dtype=float)
    
    print("Sum weights =", weight.sum())

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

if __name__ == "__main__":
    main()