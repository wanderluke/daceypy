from daceypy import DA, array, ADS
from pathlib import Path
import numpy as np
import time

from main_model import (
    propagation,
    naive_propagation_ADS,
    advanced_propagation_ADS,
)
from Export_v1_v2 import export_ads

def main():
    
    base_out = Path(r"C:\Users\lgao111\OneDrive - The University of Auckland\Desktop\Data Tests") / "L_v2"
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = str(base_out)

    DA.init(4, 1)
    DA.setEps(1e-40)

    split_tol = 1e-6
    split_depth = 20

    mu = 3.986004418e5  # km^3/s^2
    rE = 6378.0         # km

    t0 = 0.0
    tf = 3600.0 * 24.0 * 30  
    Ts = 100
    tgrid = np.linspace(t0, tf, Ts)

    h_min = 200.0
    h_max = 2000.0

    r_mid = rE + (h_min + h_max) / 2.0
    dr = 0.5 * (h_max - h_min)
    r_DA = r_mid + dr * DA(1)

    dom = array([r_DA])
    init_list = [ADS(dom, [])]
    final_lists = []

    # Point 1 - double
    x0 = array([r_mid])              
    xf = propagation(t0, tf, x0, mu, rE)  
    rf = float(xf[0].cons())          
    print("x0 alt [km] =", r_mid - rE, "xf alt [km] =", rf - rE)

    # Point 2 - DA
    x0_da = array([r_DA])           
    xf_da = propagation(t0, tf, x0_da, mu, rE)  
    print("x0 =\n", r_DA - rE, "\nxf =\n", xf_da[0] - rE)

    # Point 3 - DA + ADS

    # NAIVE ADS application
    # start_basic = time.time()
    # for i in range(1, len(tgrid)):
    #    final_list = ADS.eval(
    #        init_list, split_tol, split_depth,
    #        lambda domain: naive_propagation_ADS(domain, t0, tf=tgrid[i], mu=mu, rE=rE))
    #    final_lists.append(final_list)

    #    print('time ', tgrid[i], ' reached!')
    # print('execution time base ADS: ', time.time() - start_basic)

    # ADVANCED ADS application
    final_list = init_list.copy()
    final_lists.append(final_list) # add also initial domains

    start_advanced = time.time()
    for i in range(Ts - 1):
        final_list = ADS.eval(
            final_list, split_tol, split_depth,
            lambda domain: advanced_propagation_ADS(domain, t0=tgrid[i], tf=tgrid[i+1], mu=mu, rE=rE))
        final_lists.append(final_list)

        print('time ', tgrid[i+1], 'reached!')
    print('execution time advanced ADS: ', time.time() - start_advanced) 
    
    export_ads(final_list, out_dir=out_dir, rEarth=rE, h_range=(h_min, h_max))

if __name__ == "__main__":
    main()