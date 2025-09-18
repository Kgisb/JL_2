def ensure_program_column(df: pd.DataFrame, target="Pipeline") -> pd.DataFrame:
    """
    Make sure a 'Pipeline' column exists:
    - tolerate case/space differences and common synonyms
    - if nothing found, create a fallback 'Unknown' column (so app doesn't break)
    """
    if target in df.columns:
        return df

    # try case/space-insensitive matches
    norm = {c: c.strip().lower() for c in df.columns}
    # common synonyms you might see in different exports
    aliases = ["pipeline", "program", "programme", "course", "track"]
    hit = None
    for col, low in norm.items():
        if low in aliases:
            hit = col
            break

    if hit is not None:
        df = df.rename(columns={hit: target})
        return df

    # last resort: add a placeholder (prevents crashes; still shows a warning)
    df[target] = "Unknown"
    st.warning("‘Pipeline’ column not found; created a fallback ‘Unknown’ Pipeline. "
               "Rename your CSV column to ‘Pipeline’ for best results.")
    return df
