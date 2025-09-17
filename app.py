# ==== PREDICTABILITY — Cohort (same-month) with EB + Logistic (Source, Country) ====

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

PRED_DATE_COL = "Payment Received Date"

def _ensure_payment_col(df: pd.DataFrame) -> pd.DataFrame:
    # Try to discover a payment received column if exact name not found
    if PRED_DATE_COL in df.columns:
        return df
    for c in df.columns:
        low = c.strip().lower()
        if ("payment" in low and "date" in low) or ("received" in low and "date" in low):
            return df.rename(columns={c: PRED_DATE_COL})
    # If still not found, create empty col so code won't crash (but warn in UI)
    df[PRED_DATE_COL] = pd.NaT
    return df

def _month_period(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M")

def _cohort_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Create_Month, Deal Source, Country):
      N = deals created in that month
      K = those paid within same month
    """
    use = df.copy()
    use = _ensure_payment_col(use)
    use["Create_Month"] = _month_period(use["Create Date"])
    use["Pay_Month"]    = _month_period(use[PRED_DATE_COL])

    grp_cols = ["Create_Month", "JetLearn Deal Source", "Country"]
    base = use.groupby(grp_cols, dropna=False).size().reset_index(name="N")
    same = use[use[PRED_DATE_COL].notna() & (use["Create_Month"] == use["Pay_Month"])]
    same = same.groupby(grp_cols, dropna=False).size().reset_index(name="K")
    out = base.merge(same, on=grp_cols, how="left").fillna({"K": 0})
    out["K"] = out["K"].astype(int)
    return out

def _empirical_bayes_rates(cohort_df: pd.DataFrame, strength:int=20) -> pd.DataFrame:
    """
    EB smoothing at (Deal Source, Country) level over *recent* history.
    Prior Beta(alpha0, beta0) based on global conversion.
    """
    if cohort_df.empty:
        return pd.DataFrame(columns=["JetLearn Deal Source","Country","p_eb"])

    grp = ["JetLearn Deal Source","Country"]
    global_N = cohort_df["N"].sum()
    global_K = cohort_df["K"].sum()
    global_r = (global_K / global_N) if global_N > 0 else 0.0
    alpha0 = max(global_r * strength, 1e-6)
    beta0  = max((1 - global_r) * strength, 1e-6)

    agg = cohort_df.groupby(grp, dropna=False)[["N","K"]].sum().reset_index()
    agg["p_eb"] = (agg["K"] + alpha0) / (agg["N"] + alpha0 + beta0)
    return agg

def _logit_rates(cohort_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Fit logistic regression on group-month rows:
      y ~ Bernoulli with sample_weight=N, features = OHE(Source,Country)
    Returns per-group p_logit.
    """
    if cohort_df.empty:
        return None
    X = cohort_df[["JetLearn Deal Source","Country"]].astype(str)
    y_rate = cohort_df["K"] / cohort_df["N"].replace(0, np.nan)
    good = y_rate.notna() & cohort_df["N"].gt(0)
    if good.sum() < 20:
        return None

    X = X[good]
    y = y_rate[good].clip(0,1)
    w = cohort_df.loc[good, "N"].astype(float)

    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["JetLearn Deal Source","Country"])],
                           remainder="drop")
    logit = SKPipeline(steps=[
        ("prep", ct),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
    ])
    # Convert to “success/failure” counts via sample weights
    # sklearn LogisticRegression supports sample_weight; pass N as weights
    logit.fit(X, y, clf__sample_weight=w)

    # Predict per group (unique pairs)
    uniq = cohort_df[["JetLearn Deal Source","Country"]].drop_duplicates()
    p = logit.predict_proba(uniq.astype(str))[:,1]
    out = uniq.copy()
    out["p_logit"] = np.clip(p, 0, 1)
    return out

def _blend_p(eb_df: pd.DataFrame, logit_df: pd.DataFrame | None, lam: float) -> pd.DataFrame:
    """
    Blend EB and Logistic: p* = lam * p_logit + (1-lam) * p_eb
    """
    if logit_df is None or logit_df.empty or lam <= 0:
        eb_df = eb_df.rename(columns={"p_eb":"p_star"})
        eb_df["p_star"] = eb_df["p_star"].clip(0,1)
        return eb_df
    m = eb_df.merge(logit_df, on=["JetLearn Deal Source","Country"], how="left")
    m["p_logit"] = m["p_logit"].fillna(m["p_eb"])
    m["p_star"] = (lam * m["p_logit"] + (1.0 - lam) * m["p_eb"]).clip(0,1)
    return m[["JetLearn Deal Source","Country","p_star"]]

def _monthly_creates_forecast(cohort_df: pd.DataFrame, target_month: pd.Period, use_median=False) -> pd.DataFrame:
    """
    Forecast N̂ for target month using trailing 3 months mean/median of N per (Source,Country).
    """
    grp = ["JetLearn Deal Source","Country"]
    past = cohort_df[cohort_df["Create_Month"] < target_month]
    if past.empty:
        return pd.DataFrame(columns=grp+["Nhat"])

    # take last 3 months available per group
    # compute avg N
    def last3(s):
        s = s.sort_index()
        vals = s.tail(3)
        return (vals.median() if use_median else vals.mean())

    temp = past.set_index("Create_Month").groupby(grp)["N"].apply(last3).reset_index(name="Nhat")
    temp["Nhat"] = temp["Nhat"].fillna(0.0).clip(lower=0)
    return temp

def _cap_by_history(df_pred: pd.DataFrame, cohort_df: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    """Cap per (Source,Country) by historical 95th pct of K to avoid spikes."""
    grp = ["JetLearn Deal Source","Country"]
    hist = cohort_df.groupby(grp)["K"].apply(lambda s: np.percentile(s, q*100) if len(s)>0 else 0.0).reset_index(name="Kcap")
    out = df_pred.merge(hist, on=grp, how="left")
    out["Kcap"] = out["Kcap"].fillna(out["Khat"].quantile(q) if "Khat" in out.columns and not out["Khat"].empty else 0.0)
    out["Khat"] = np.minimum(out["Khat"], out["Kcap"])
    return out.drop(columns=["Kcap"], errors="ignore")

def _day_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day-of-month weights of payments historically. Returns df with columns: dom, w (sum to 1).
    """
    if PRED_DATE_COL not in df.columns or df[PRED_DATE_COL].notna().sum()==0:
        return pd.DataFrame({"dom": [1], "w": [1.0]})
    tmp = df.dropna(subset=[PRED_DATE_COL]).copy()
    tmp["dom"] = pd.to_datetime(tmp[PRED_DATE_COL]).dt.day
    cnt = tmp.groupby("dom").size().reset_index(name="n")
    cnt["w"] = cnt["n"] / cnt["n"].sum()
    return cnt[["dom","w"]].sort_values("dom")

def _walkforward_backtest(cohort_df: pd.DataFrame, last_k:int=6, lam:float=0.3, use_median=False) -> dict:
    """
    Walk-forward on last_k complete months. Returns MAE/MAPE for month-level K.
    """
    months = sorted(cohort_df["Create_Month"].unique())
    if len(months) < last_k + 4:
        return {}
    months = months[-(last_k+1):]  # we need at least one month to predict and earlier to train
    maes, mapes = [], []
    grp = ["JetLearn Deal Source","Country"]

    for m in months[:-1]:  # predict m using data < m
        past = cohort_df[cohort_df["Create_Month"] < m]
        truth = cohort_df[cohort_df["Create_Month"] == m][grp+["K","N"]]

        eb = _empirical_bayes_rates(past, strength=20)
        logit = _logit_rates(past)
        blend = _blend_p(eb, logit, lam)
        Nhat = _monthly_creates_forecast(past, m, use_median=use_median)
        pred = Nhat.merge(blend, on=grp, how="inner")
        pred["Khat"] = (pred["Nhat"] * pred["p_star"]).clip(lower=0)
        pred = _cap_by_history(pred, past, q=0.95)

        # align with truth (missing groups → 0)
        merged = truth.merge(pred[grp+["Khat"]], on=grp, how="left").fillna({"Khat":0.0})
        mae = mean_absolute_error(merged["K"], merged["Khat"])
        mape = float(np.mean(np.abs(merged["K"] - merged["Khat"]) / np.maximum(1, merged["K"])) * 100)
        maes.append(mae); mapes.append(mape)

    if not maes:
        return {}
    return {"WF_MAE": round(float(np.mean(maes)),2), "WF_MAPE%": round(float(np.mean(mapes)),1)}
