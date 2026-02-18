import os
import re
from pathlib import Path
from typing import List, Tuple

# --- parsing helper for DA tables printed as text ---
_line_re = re.compile(
    r"^\s*(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(\d+)\s+(.*)$"
)

def _parse_da_table(da_str: str):
    """Return list of (coeff, exps[]) from DA printed table."""
    out = []
    for raw in da_str.splitlines():
        raw = raw.rstrip()
        if not raw:
            continue
        s = raw.strip()
        if s.startswith("I") or s.startswith("-"):
            continue
        m = _line_re.match(raw)
        if not m:
            continue
        coeff = float(m.group(2))
        exps = [int(x) for x in m.group(4).split()]
        out.append((coeff, exps))
    return out

def _affine_center_halfwidth(box_da) -> Tuple[float, float]:
    """
    Extract center and halfwidth from an affine DA:
      center + halfwidth * DA(1)
    Works when the box variable is 1D in its own DA variable.
    """
    terms = _parse_da_table(str(box_da))
    c0 = None
    c1 = None
    for coeff, exps in terms:
        # exp=0 term
        if len(exps) >= 1 and exps[0] == 0:
            c0 = coeff
        # exp=1 term
        if len(exps) >= 1 and exps[0] == 1:
            c1 = coeff
    if c0 is None or c1 is None:
        raise RuntimeError(
            "Box DA is not affine or cannot find exp=0/exp=1 terms. "
            "Check print(box_da) format."
        )
    return c0, c1  # center, halfwidth

def _write_da_block(f, da_obj):
    """
    Write DA object exactly as printed (COSY-like table).
    LoadCOSY typically tolerates headers; the key is numeric rows.
    """
    s = str(da_obj).strip("\n")
    f.write(s)
    if not s.endswith("\n"):
        f.write("\n")

def export_ads_v3_r_w(
    final_list: List,                 # list of ADS domains at final time
    out_dir: str,
    rEarth: float,
    h_range: Tuple[float, float],
    idx_r0: int = 1,                  # DA variable index (1-based) used for r0
    file_prefix: str = "dom"
):
    """
    Export v3 as:
      out_dir/data.dat
      out_dir/dom_k.dat (k = 0..nsd-1), with m=2 blocks: rf and wf

    data.dat columns: 2 0 xh_w xh_c (same convention as your v1)
    where xh_c/xh_w are normalized in xi coordinates for global h_range scaling.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    hmin, hmax = h_range
    dh = (hmax - hmin) / 2.0
    hmid = hmin + dh

    box_idx = idx_r0 - 1  # convert DA variable index -> python 0-based index for box

    # --- write data.dat
    with (outp / "data.dat").open("w", encoding="utf-8") as f:
        for k, dom in enumerate(final_list):
            # box contains affine mappings for each DA variable
            # pick the one corresponding to r0
            c_r, hw_r = _affine_center_halfwidth(dom.box[box_idx])

            h_c = c_r - rEarth          # [km]
            h_w = 2.0 * hw_r            # full width in km

            xh_c = (h_c - hmid) / dh    # normalized center
            xh_w = h_w / dh             # normalized width

            f.write(f"2 0 {xh_w:.16g} {xh_c:.16g}\n")

    # --- write dom_k.dat
    for k, dom in enumerate(final_list):
        rf = dom.manifold[0]
        wf = dom.manifold[1]

        with (outp / f"{file_prefix}_{k}.dat").open("w", encoding="utf-8") as f:
            _write_da_block(f, rf)
            f.write("------------------------------------------------\n")
            _write_da_block(f, wf)
            f.write("------------------------------------------------\n")

