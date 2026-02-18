import daceypy_import_helper  # noqa: F401
import time
from math import ceil
from time import perf_counter
import numpy as np
from daceypy import DA, array, ADS
from typing import Callable
from daceypy.op import exp, sqrt, tanh
from typing import Union
from provaExport import export_ads
from provaExport_v3_r_w import export_ads_v3_r_w
from provaExport_v3_r_u import export_ads_v3_r_u

def event_reentry_150(X: array, t: float, rE: float) -> bool:
    r = X[0]
    r_cons = float(r.cons()) if hasattr(r, "cons") else float(r) # r puo essere float o DA
    return r_cons <= (rE + 150.0)

def atm_dens(r, rE: float):
    rho_ref = 1e-14 * 1e3                 # [kg/m^3]  
    h_ref = 400.0                         # [km]
    H = 65.0                              # [km]
    h = r - rE                            # [km]
    rho = rho_ref * exp(-(h - h_ref) / H) # [kg/m^3] 
    return rho                            # same type as r

def fnc_drdt(r, mu: float, rE: float):
    rho = atm_dens(r, rE)                      # [kg/m^3]
    BC = 50.0 / (2.2 * 0.5)                    # [kg/m^2]
    v_r = -sqrt(mu * r) * (rho / BC) * 1e3     # [km/s]
    r0 = rE + 200.0                            # [km]
    delta = 20.0                               # [km]
    chi = 0.5 * (1.0 + tanh((r - r0) / delta)) # [-] bridging function around 200 km altitude
    # chi = 1.0
    return v_r * chi                           # same type as r

def fnc_drdt_ADS(domain: ADS, mu: float, rE: float) -> ADS:
    r = domain.box[0]
    vr = fnc_drdt(r, mu, rE)
    out = array([vr])                           # output as 1D vector map
    return ADS(domain.box, domain.nsplit, out)

def characteristicsDynamics(X: array, t: float, mu: float, rE: float) -> array:
    return array([fnc_drdt(X[0], mu, rE)])

def charDensDynamics(X: array, t: float, mu: float, rE: float, idx_r0: int = 1) -> array:
    r = X[0]                        # state X = [r, w], works correctly when r and w are DA objects
    w = X[1]
    v_r = fnc_drdt(r, mu, rE)

    dr_dr0 = r.deriv(idx_r0)
    dv_dr0 = v_r.deriv(idx_r0)
    dv_dr = dv_dr0 / dr_dr0

    drdt = v_r
    dwdt = -dv_dr * w
    return array([drdt, dwdt])

def propagation(t0: float, tf: float, X: array, mu: float, rE: float) -> array:
    x0 = X.copy()
    
    if not isinstance(x0[0], DA):     # ensure DA types
        x0[0] = DA(float(x0[0]))

    xf = RK78(
        x0, t0, tf,
        lambda X, t: characteristicsDynamics(X, t, mu, rE),
        event=lambda X, t: event_reentry_150(X, t, rE)
    )
    return xf
    
def naive_propagation_ADS(domain: ADS, t0: float, tf: float, mu: float, rE: float) -> ADS:

    x0 = domain.box                      
    xf = propagation(t0, tf, x0, mu, rE) # returns DA polynomial

    return ADS(domain.box, domain.nsplit, xf)

def advanced_propagation_ADS(domain: ADS, t0: float, tf: float, mu: float, rE: float) -> ADS:
    
    x0 = domain.manifold                  
    xf = propagation(t0, tf, x0, mu, rE)  
    
    return ADS(domain.box, domain.nsplit, xf)

def propagation_dens(t0: float, tf: float, X: array, mu: float, rE: float, idx_r0: int = 1) -> array:

    x0 = X.copy()                  # X must be array([r, w])

    if not isinstance(x0[0], DA):  # ensure DA types 
        x0[0] = DA(float(x0[0]))
    if not isinstance(x0[1], DA):
        x0[1] = DA(float(x0[1]))

    xf = RK78(
        x0, t0, tf,
        lambda X, t: charDensDynamics(X, t, mu, rE, idx_r0=idx_r0),
        event=lambda X, t: event_reentry_150(X, t, rE)
    )
    return xf                      # array([rf, wf])

def naive_characteristics_ADS(domain: ADS, t0: float, tf: float, mu: float, rE: float, w0: Union[float, DA], idx_r0: int = 1) -> ADS:

    r0 = domain.box[0]         
    w0_DA = w0 if isinstance(w0, DA) else DA(float(w0))

    x0 = array([r0, w0_DA])
    xf = propagation_dens(t0, tf, x0, mu, rE, idx_r0=idx_r0)

    rf, wf = xf[0], xf[1]

    return ADS(domain.box, domain.nsplit, array([rf, wf])) # la manifold finale è un vettore 2D per il rispettivo sotto-dominio

def advanced_characteristics_ADS(domain: ADS, t0: float, tf: float, mu: float, rE: float, idx_r0: int = 1) -> ADS:

    x0 = domain.manifold        
    xf = propagation_dens(t0, tf, x0, mu, rE, idx_r0=idx_r0)

    rf, wf = xf[0], xf[1]

    return ADS(domain.box, domain.nsplit, array([rf, wf]))

def compute_u_list(domains):
    u_domains = []
    for dom in domains:
        r = dom.manifold[0]
        w = dom.manifold[1]
        u = w / (r * r)
        u_domains.append( ADS(dom.box, dom.nsplit, array([u])) )
    return u_domains

def RK78(Y0: array, X0: float, X1: float, f: Callable[[array, float], array], event=None) -> array:
    """
    Propagate using RK78.
    """
    Y0 = Y0.copy()

    N = len(Y0)

    # H0 = 0.001
    # HS = 0.1
    # H1 = 100.0
    H0 = 3600 
    HS = 86400 
    H1 = 86400 * 365.25
    # EPS = 1.e-12
    EPS = 1.e-8
    BS = 20 * EPS

    Z = array.zeros((N, 16))
    Y1 = array.zeros(N)

    VIHMAX = 0.0

    HSQR = 1.0 / 9.0
    A = np.zeros(13)
    B = np.zeros((13, 12))
    C = np.zeros(13)
    D = np.zeros(13)

    A = np.array([
        0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0,
        93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0,
        1201146811.0/1299019798.0, 1.0, 1.0,
    ])

    B[1, 0] = 1.0/18.0
    B[2, 0] = 1.0/48.0
    B[2, 1] = 1.0/16.0
    B[3, 0] = 1.0/32.0
    B[3, 2] = 3.0/32.0
    B[4, 0] = 5.0/16.0
    B[4, 2] = -75.0/64.0
    B[4, 3] = 75.0/64.0
    B[5, 0] = 3.0/80.0
    B[5, 3] = 3.0/16.0
    B[5, 4] = 3.0/20.0
    B[6, 0] = 29443841.0/614563906.0
    B[6, 3] = 77736538.0/692538347.0
    B[6, 4] = -28693883.0/1125000000.0
    B[6, 5] = 23124283.0/1800000000.0
    B[7, 0] = 16016141.0/946692911.0
    B[7, 3] = 61564180.0/158732637.0
    B[7, 4] = 22789713.0/633445777.0
    B[7, 5] = 545815736.0/2771057229.0
    B[7, 6] = -180193667.0/1043307555.0
    B[8, 0] = 39632708.0/573591083.0
    B[8, 3] = -433636366.0/683701615.0
    B[8, 4] = -421739975.0/2616292301.0
    B[8, 5] = 100302831.0/723423059.0
    B[8, 6] = 790204164.0/839813087.0
    B[8, 7] = 800635310.0/3783071287.0
    B[9, 0] = 246121993.0/1340847787.0
    B[9, 3] = -37695042795.0/15268766246.0
    B[9, 4] = -309121744.0/1061227803.0
    B[9, 5] = -12992083.0/490766935.0
    B[9, 6] = 6005943493.0/2108947869.0
    B[9, 7] = 393006217.0/1396673457.0
    B[9, 8] = 123872331.0/1001029789.0
    B[10, 0] = -1028468189.0/846180014.0
    B[10, 3] = 8478235783.0/508512852.0
    B[10, 4] = 1311729495.0/1432422823.0
    B[10, 5] = -10304129995.0/1701304382.0
    B[10, 6] = -48777925059.0/3047939560.0
    B[10, 7] = 15336726248.0/1032824649.0
    B[10, 8] = -45442868181.0/3398467696.0
    B[10, 9] = 3065993473.0/597172653.0
    B[11, 0] = 185892177.0/718116043.0
    B[11, 3] = -3185094517.0/667107341.0
    B[11, 4] = -477755414.0/1098053517.0
    B[11, 5] = -703635378.0/230739211.0
    B[11, 6] = 5731566787.0/1027545527.0
    B[11, 7] = 5232866602.0/850066563.0
    B[11, 8] = -4093664535.0/808688257.0
    B[11, 9] = 3962137247.0/1805957418.0
    B[11, 10] = 65686358.0/487910083.0
    B[12, 0] = 403863854.0/491063109.0
    B[12, 3] = - 5068492393.0/434740067.0
    B[12, 4] = -411421997.0/543043805.0
    B[12, 5] = 652783627.0/914296604.0
    B[12, 6] = 11173962825.0/925320556.0
    B[12, 7] = -13158990841.0/6184727034.0
    B[12, 8] = 3936647629.0/1978049680.0
    B[12, 9] = -160528059.0/685178525.0
    B[12, 10] = 248638103.0/1413531060.0

    C = np.array([
        14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0,
        181606767.0/758867731.0, 561292985.0/797845732.0,
        -1041891430.0/1371343529.0, 760417239.0/1151165299.0,
        118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0,
    ])

    D = np.array([
        13451932.0/455176623.0, 0.0, 0.0, 0.0, 0.0, -808719846.0/976000145.0,
        1757004468.0/5645159321.0, 656045339.0/265891186.0,
        -3867574721.0/1518517206.0, 465885868.0/322736535.0,
        53011238.0/667516719.0, 2.0/45.0, 0.0,
    ])

    Z[:, 0] = Y0

    H = abs(HS)
    HH0 = abs(H0)
    HH1 = abs(H1)
    X = X0
    RFNORM = 0.0
    ERREST = 0.0

    while X != X1:

        # compute new stepsize
        if RFNORM != 0:
            H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
        if abs(H) > abs(HH1):
            H = HH1
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * H > 0:
            H = X1 - X

        for j in range(13):

            for i in range(N):

                Y0[i] = 0.0
                # EVALUATE RHS AT 13 POINTS
                for k in range(j):
                    Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]

                Y0[i] = H * Y0[i] + Z[i, 0]

            Y1 = f(Y0, X + H * A[j])

            for i in range(N):
                Z[i, j + 3] = Y1[i]

        for i in range(N):

            Z[i, 1] = 0.0
            Z[i, 2] = 0.0
            # EXECUTE 7TH,8TH ORDER STEPS
            for j in range(13):
                Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
                Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

            Y1[i] = (Z[i, 2] - Z[i, 1]) * H
            Z[i, 2] = Z[i, 2] * H + Z[i, 0]

        Y1cons = Y1.cons()

        # ESTIMATE ERROR AND DECIDE ABOUT BACKSTEP
        RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
        if RFNORM > BS and abs(H / H0) > 1.2:
            H = H / 3.0
            RFNORM = 0
        else:
            for i in range(N):
                Z[i, 0] = Z[i, 2]
            X = X + H
            VIHMAX = max(VIHMAX, H)
            ERREST = ERREST + RFNORM

            if event is not None and event(Z[:, 0], X):
                break

    Y1 = Z[:, 0]

    return Y1

def main():

    DA.init(4, 1)
    DA.setEps(1e-40)
    
    mu = 3.986004418e5; # km^3/s^2
    rE = 6378.0
    t0 = 0.0
    tf = 3600.0 * 24.0 * 365.25
    Ts = 100
    tgrid = np.linspace(t0, tf, Ts) # from YES 0: HO MODIFICATO IL LOOP TEMPORALE DI NAIVE ADS

    h_min = 200.0
    h_max = 2000.0
    r_mid = rE + (h_min + h_max) / 2.0
    dr = 0.5 * (h_max - h_min)
    r_DA = r_mid + dr * DA(1)

    dom = array([r_DA])
    init_list = [ADS(dom, [])]
    final_lists = []

    split_tol = 1e-6  # tighter => more splits
    split_depth = 15 # max depth

    NN = 10000.0
    deltah = 1800.0
    w_DA = DA((1.0 / (4.0 * np.pi)) * (NN / deltah))

    man = array([r_DA, w_DA])
    init_list_v3 = [ADS(dom, [], man)]

    #################################### test_v0 ####################################
    
    # Point 1 - double
    # vr = fnc_drdt(r_mid, mu, rE)
    # print(vr)

    # Point 2 - DA
    # vr_DA = fnc_drdt(r_DA, mu, rE)
    # print(vr_DA)

    # Point 3 - DA + ADS
    # final_list = ADS.eval(
    #     init_list,
    #     split_tol,
    #     split_depth,
    #     lambda domain: fnc_drdt_ADS(domain, mu, rE)
    # )
    # export_ads(final_list, out_dir="v0", rEarth=rE, h_range=(200.0, 2000.0))
    # print(final_list[0].box[0])  # show the box of the first domain
    # print(final_list[0].manifold[0]) # show the manifold of the first domain
    # print(final_list[1].manifold[0]) # show the manifold of the second domain
    # print(final_list[2].manifold[0]) # show the manifold of the third domain
    # print(final_list[3].manifold[0]) # show the manifold of the fourth domain

    #################################### test_v1 ####################################

    # Point 1 - double
    # x0 = array([r_mid])              
    # xf = propagation(t0, tf, x0, mu, rE)  
    # rf = float(xf[0].cons())          
    # print("x0 alt [km] =", r_mid - rE, "xf alt [km] =", rf - rE)

    # Point 2 - DA
    # x0_da = array([r_DA])           
    # xf_da = propagation(t0, tf, x0_da, mu, rE)  
    # print("x0 =\n", r_DA - rE, "\nxf =\n", xf_da[0] - rE)

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
    # final_list = init_list.copy()
    # final_lists.append(final_list) # add also initial domains

    # start_advanced = time.time()
    # for i in range(Ts - 1):
    #     final_list = ADS.eval(
    #         final_list, split_tol, split_depth,
    #         lambda domain: advanced_propagation_ADS(domain, t0=tgrid[i], tf=tgrid[i+1], mu=mu, rE=rE))
    #     final_lists.append(final_list)

    #     print('time ', tgrid[i+1], 'reached!')
    # print('execution time advanced ADS: ', time.time() - start_advanced) # 259 s
    # export_ads(final_list, out_dir="v1", rEarth=rE, h_range=(200.0, 2000.0))
    #################################### test_v3 ####################################
    
    # Point 3 - DA + ADS    
    
    # NAIVE ADS application
    # final_lists_rw = []
    # final_lists_u  = []
    # start_basic = time.time()
    # for i in range(1, len(tgrid)):
    #     final_list = ADS.eval(
    #         init_list_v3, split_tol, split_depth,
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
    final_list = init_list_v3.copy()
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
    print("execution time v3 advanced ADS:", time.time() - start_adv) # 469 s
    export_ads_v3_r_w(final_list=final_list, out_dir="v3_r_w", rEarth=rE, 
                      h_range=(200.0, 2000.0), idx_r0=1, file_prefix="dom")
    export_ads_v3_r_u(final_list=final_list, out_dir="v3_r_u", rEarth=rE,
                      h_range=(200.0, 2000.0), idx_r0=1, file_prefix="dom")

if __name__ == "__main__":
    main()
