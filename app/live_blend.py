"""Live blend of two projection sources (the own model and a reference list), computed in memory on demand."""
import numpy as np
import pandas as pd

from .own_model import norm_name


def blend_frame(own: pd.DataFrame, ref: pd.DataFrame, w: float, comps: list) -> pd.DataFrame:
    """Each counting column of every player found in both = (1-w) * own + w * ref. Everyone else, and every non-counting
    column (positions, team, injury status), stays as in `own`. w=0 is exactly `own`."""
    if not (isinstance(w, (int, float)) and np.isfinite(w) and 0.0 <= w <= 1.0):
        raise ValueError("The blend weight must be a number from 0 to 1.")
    comps = [c for c in comps if c in own.columns and c in ref.columns]
    out = own.copy()
    out[comps] = out[comps].astype(float)
    if w == 0 or not comps:
        return out
    rmap = {norm_name(n): i for i, n in ref["name"].items()}
    own_ix, ref_ix = [], []
    for i, n in own["name"].items():
        j = rmap.get(norm_name(n))
        if j is not None:
            own_ix.append(i)
            ref_ix.append(j)
    if own_ix:
        o = own.loc[own_ix, comps].astype(float).to_numpy()
        r = ref.loc[ref_ix, comps].astype(float).to_numpy()
        out.loc[own_ix, comps] = (1 - w) * o + w * r
    return out
