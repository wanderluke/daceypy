import re
from pathlib import Path
from typing import List, Tuple

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
    Extract center and halfwidth from affine DA of the form:
      center + halfwidth * DA(1)
    """
    terms = _parse_da_table(str(box_da))
    c0 = None
    c1 = None
    for coeff, exps in terms:
        if len(exps) >= 1 and exps[0] == 0:
            c0 = coeff
        if len(exps) >= 1 and exps[0] == 1:
            c1 = coeff
    if c0 is None or c1 is None:
        raise RuntimeError(
            "Box DA is not affine or cannot find exp=0/exp=1 terms."
        )
    return c0, c1

def _write_da_block(f, da_obj):
    """Write DA object in COSY-like format using its string representation."""
    s = str(da_obj).strip("\n")
    f.write(s)
    if not s.endswith("\n"):
        f.write("\n")

def _split_depth(dom) -> int:
    """
    Return split-tree depth for this domain.
    Works whether dom.nsplit is a python list or a daceypy.array-like.
    """
    ns = getattr(dom, "nsplit", None)
    if ns is None:
        return 0
    try:
        return len(ns)
    except TypeError:
        # some array-like types may not expose len; try converting
        try:
            return len(list(ns))
        except Exception:
            return 0

def export_ads_v3_r_u(
    final_list: List,
    out_dir: str,
    rEarth: float,
    h_range: Tuple[float, float],
    idx_r0: int = 1,                 # DA variable index (1-based)
    file_prefix: str = "dom",
):
    """
    Export v3 as:
      out_dir/data.dat   (2 0 xh_w xh_c nsplit)
      out_dir/dom_k.dat  with m=2 blocks: r and u, where u = w / r^2

    Assumes each domain.manifold = [r, w].
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "data.dat").unlink(missing_ok=True) # questo pezzo sovrascrive data.dat
    for p in outp.glob(f"{file_prefix}_*.dat"): # questo pezzo sovrascrive dom_k.dat files e cancella quelli in eccesso
        p.unlink()

    hmin, hmax = h_range
    dh = (hmax - hmin) / 2.0
    hmid = hmin + dh

    box_idx = idx_r0 - 1  # DA index -> python index

    # ---- data.dat
    with (outp / "data.dat").open("w", encoding="utf-8") as f:
        for dom in final_list:
            c_r, hw_r = _affine_center_halfwidth(dom.box[box_idx])

            h_c = c_r - rEarth          # [km]
            h_w = 2.0 * hw_r            # full width [km]

            xh_c = (h_c - hmid) / dh    # normalized center
            xh_w = h_w / dh             # normalized width

            depth = _split_depth(dom)   # len(dom.nsplit)

            f.write(f"2 0 {xh_w:.16g} {xh_c:.16g} {depth:d}\n")

    # ---- dom_k.dat
    for k, dom in enumerate(final_list):
        r_da = dom.manifold[0]
        w_da = dom.manifold[1]

        # u = w / r^2  (DA algebra should support this)
        u_da = w_da / (r_da * r_da)

        with (outp / f"{file_prefix}_{k}.dat").open("w", encoding="utf-8") as f:
            _write_da_block(f, r_da)
            _write_da_block(f, u_da)

