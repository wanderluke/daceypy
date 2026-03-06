# export_v0.py
from pathlib import Path
import numpy as np

def _parse_box_center_halfwidth(box_da_str: str):
    """
    box_da_str è str(final_list[k].box[0]) nel formato:
        I  COEFFICIENT  ORDER EXPONENTS
        1  7.028e+03    0     0
        2  1.500e+02    1     1
        ----
    Estrae:
      center = coeff exp=0
      halfwidth = coeff exp=1
    """
    center = None
    halfwidth = 0.0

    for line in box_da_str.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue

        # parts: [idx, coeff, order, exp]
        coeff = float(parts[1])
        exp   = int(parts[-1])

        if exp == 0 and center is None:
            center = coeff
        elif exp == 1:
            halfwidth = coeff

    if center is None:
        raise RuntimeError("Non trovo il termine exp=0 nel box (center).")

    return float(center), float(halfwidth)

def export_ads(final_list, out_dir, rEarth=6378.0, h_range=(200.0, 2000.0)):
    """
    Crea in out_dir:
      - data.dat con 4 colonne: '2 0 width center' (width,center in xi)
      - dom_k.dat per k=0..nsd-1, uguale a str(final_list[k].manifold[0])

    NOTE:
      - MATLAB stest_v0.m usa xh_w=data(:,3), xh_c=data(:,4)
      - e ricostruisce h = (xh_c + xh_w/2*dr)*dh + h_mid
      => width/center devono essere in xi, non in km.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hmin, hmax = float(h_range[0]), float(h_range[1])
    dh = (hmax - hmin) / 2.0
    h_mid = hmin + dh

    # ---- data.dat ----
    data_lines = []
    for dom in final_list:
        box_str = str(dom.box[0])  # già nel formato COSY
        r_center, r_halfwidth = _parse_box_center_halfwidth(box_str)

        # Nel tuo box: r = r_center + r_halfwidth * dr  (dr in [-1,1])
        # Converti in h:
        h_center = r_center - rEarth
        h_width  = 2.0 * abs(r_halfwidth)

        # Converti in xi globale:
        xi_center = (h_center - h_mid) / dh
        xi_width  = h_width / dh

        data_lines.append(f"2 0 {xi_width:.16g} {xi_center:.16g}")

    (out_dir / "data.dat").write_text("\n".join(data_lines) + "\n")

    # ---- dom_k.dat ----
    for k, dom in enumerate(final_list):
        dom_str = str(dom.manifold[0])  # ESATTAMENTE come vuoi tu
        (out_dir / f"dom_{k}.dat").write_text(dom_str if dom_str.endswith("\n") else dom_str + "\n")

    return out_dir
