import numpy as np
import pandas as pd
from . import contract as c


def _formal_ids(frame, core):
    """Widen only float32 IDs proven to equal unique original raw identifiers.

    The frozen v2.6 backing cache stores these IDs as float32. Merely casting
    would conceal precision loss; fail unless every value exactly matches the
    raw float64 ID, and that raw ID itself has an exact float32 round trip.
    Row identities, stay partitions, labels and folds are never reassigned.
    """
    if frame[c.H].dtype != np.dtype("float32"):
        c.validate_stay_ids(frame[c.H])
        return frame
    raw = pd.read_csv(core.RAW / "IndexSurgHosp.csv", usecols=["hospitalizationidNEW"],
                      dtype=str, encoding="cp1252")["hospitalizationidNEW"]
    raw = pd.to_numeric(raw, errors="raise").astype(np.float64)
    widened = frame[c.H].astype(np.float64)
    if (raw.isna().any() or raw.duplicated().any() or raw.astype(np.float32).duplicated().any() or widened.isna().any()
            or not widened.isin(raw).all()
            or not raw.loc[raw.isin(widened)].eq(raw.loc[raw.isin(widened)].astype(np.float32).astype(np.float64)).all()):
        raise c.ContractError("Formal float32 stay identity is not exactly recoverable from unique raw IDs")
    result = frame.copy()
    result[c.H] = widened
    c.validate_stay_ids(result[c.H])
    return result
