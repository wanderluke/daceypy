from daceypy import DA, array, ADS
from pathlib import Path
from main_model import fnc_drdt, fnc_drdt_ADS
from Export_v1_v2 import export_ads

def main():

    base_out = Path(r"C:\Users\lgao111\OneDrive - The University of Auckland\Desktop\Data Tests") / "L_v1"
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = str(base_out)

    DA.init(4, 1)
    DA.setEps(1e-40)

    split_tol = 1e-6
    split_depth = 20

    mu = 3.986004418e5  # km^3/s^2
    rE = 6378.0         # km

    h_min = 200.0
    h_max = 2000.0

    r_mid = rE + (h_min + h_max) / 2.0
    dr = 0.5 * (h_max - h_min)
    r_DA = r_mid + dr * DA(1)

    dom = array([r_DA])
    init_list = [ADS(dom, [])]

    # Point 1 - double 
    vr = fnc_drdt(r_mid, mu, rE) 
    print(vr) 

    # Point 2 - DA 
    vr_DA = fnc_drdt(r_DA, mu, rE) 
    print(vr_DA) 
    
    # Point 3 - DA + ADS 
    final_list = ADS.eval( 
        init_list, 
        split_tol, 
        split_depth, 
        lambda domain: fnc_drdt_ADS(domain, mu, rE) 
    ) 

    export_ads(final_list, out_dir=out_dir, rEarth=rE, h_range=(h_min, h_max))

if __name__ == "__main__":
    main()
   
   