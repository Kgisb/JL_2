def build_compare_delta(dfA, dfB):
    """
    Safely compare KPI rows between Scenario A and B.
    Ensures numeric dtypes and guards against divide-by-zero.
    """
    if dfA.empty or dfB.empty:
        return pd.DataFrame()

    key = ["Scope", "Metric"]

    # Pull, align, and coerce to numeric
    a = dfA[key + ["Value"]].copy().rename(columns={"Value": "A"})
    b = dfB[key + ["Value"]].copy().rename(columns={"Value": "B"})
    a["A"] = pd.to_numeric(a["A"], errors="coerce")
    b["B"] = pd.to_numeric(b["B"], errors="coerce")

    out = pd.merge(a, b, on=key, how="inner")

    # Ensure numeric and compute delta
    out["A"] = pd.to_numeric(out["A"], errors="coerce")
    out["B"] = pd.to_numeric(out["B"], errors="coerce")
    out["Δ"] = pd.to_numeric(out["B"] - out["A"], errors="coerce")

    # Safe percent change: Δ% = (B - A) / A * 100, with A==0 or NaN -> NaN
    denom = out["A"].astype("float")
    zero_or_nan = denom.isna() | (denom == 0)
    denom = denom.where(~zero_or_nan)  # set zeros to NaN -> avoids inf
    out["Δ%"] = ((out["Δ"].astype("float") / denom) * 100).round(1)

    return out
