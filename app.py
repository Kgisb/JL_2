# ---- put these helpers near your other utils (keep your existing coerce_list/in_filter if you have them) ----
def coerce_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    if all_checked:
        return pd.Series(True, index=series.index)
    sel = [str(v) for v in coerce_list(selected_values)]
    if not sel:
        return pd.Series(False, index=series.index)
    return series.astype(str).isin(sel)

def _summary(values, all_flag, max_items=2):
    vals = coerce_list(values)
    if all_flag: return "All"
    if not vals: return "None"
    s = ", ".join(map(str, vals[:max_items]))
    if len(vals) > max_items: s += f" +{len(vals)-max_items} more"
    return s

# ---- unified, safe global multiselect with “All” toggle ----
def unified_multifilter(label: str, df: pd.DataFrame, colname: str, key_prefix: str):
    """
    - All toggle: when ON, we treat selection as “all options” (without mutating the multiselect value).
    - Multiselect: native type-to-search; supports 1 or many values.
    - No programmatic writes to widget keys during render (avoids StreamlitAPIException).
    """
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])

    all_key = f"{key_prefix}_all"  # checkbox key
    ms_key  = f"{key_prefix}_ms"   # multiselect widget key

    # Initialize once
    if all_key not in st.session_state:
        st.session_state[all_key] = True
    if ms_key not in st.session_state:
        st.session_state[ms_key] = options.copy()  # prefill with "all" once

    # Snapshot state
    all_flag = bool(st.session_state[all_key])
    selected = coerce_list(st.session_state.get(ms_key, []))

    # Effective list used for filtering
    effective = options if all_flag else selected

    # Compact header + popover to save space
    header = f"{label}: {_summary(effective, all_flag)}"
    ctx = st.popover(header) if hasattr(st, "popover") else st.expander(header, expanded=False)
    with ctx:
        left, right = st.columns([1, 3])
        with left:
            st.checkbox("All", value=all_flag, key=all_key)
        with right:
            disabled = st.session_state[all_key]
            st.multiselect(
                label,
                options=options,
                default=selected,          # only used on first render
                key=ms_key,                # state lives here after first render
                placeholder=f"Type to search {label.lower()}…",
                label_visibility="collapsed",
                disabled=disabled,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Select all", key=f"{key_prefix}_select_all", use_container_width=True):
                    st.session_state[ms_key] = options.copy()
                    st.session_state[all_key] = True
                    st.rerun()
            with c2:
                if st.button("Clear", key=f"{key_prefix}_clear", use_container_width=True):
                    st.session_state[ms_key] = []
                    st.session_state[all_key] = False
                    st.rerun()

    # Recompute effective (don’t mutate widget keys here)
    all_flag = bool(st.session_state[all_key])
    selected = coerce_list(st.session_state.get(ms_key, []))
    effective = options if all_flag else selected

    return all_flag, effective, f"{label}: {_summary(effective, all_flag)}"

# ---- toolbar that uses the unified filter for all 4 globals ----
def filters_toolbar(name: str, df: pd.DataFrame):
    st.markdown('<div class="filters-bar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([2,2,2,2,1])

    with c1:
        pipe_all, pipe_sel, s1 = unified_multifilter("Pipeline", df, "Pipeline", f"{name}_pipe")
    with c2:
        src_all,  src_sel,  s2 = unified_multifilter("Deal Source", df, "JetLearn Deal Source", f"{name}_src")
    with c3:
        ctry_all, ctry_sel, s3 = unified_multifilter("Country", df, "Country", f"{name}_ctry")
    with c4:
        cslr_all, cslr_sel, s4 = unified_multifilter("Counsellor", df, "Student/Academic Counsellor", f"{name}_cslr")

    with c5:
        if st.button("Clear", key=f"{name}_clear", use_container_width=True):
            # Reset all four at once, then rerun
            for prefix, col in [
                (f"{name}_pipe","Pipeline"),
                (f"{name}_src","JetLearn Deal Source"),
                (f"{name}_ctry","Country"),
                (f"{name}_cslr","Student/Academic Counsellor"),
            ]:
                opts = sorted([v for v in df[col].dropna().astype(str).unique()])
                st.session_state[f"{prefix}_all"] = True
                st.session_state[f"{prefix}_ms"]  = opts
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Filters — " + " · ".join([s1, s2, s3, s4]))

    # Build robust mask
    mask = (
        in_filter(df["Pipeline"], pipe_all, pipe_sel) &
        in_filter(df["JetLearn Deal Source"], src_all, src_sel) &
        in_filter(df["Country"], ctry_all, ctry_sel) &
        in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
    )
    base = df[mask].copy()

    # Return filtered df + what was selected (if you need to print in footer)
    return base, dict(
        pipe_all=pipe_all, pipe_sel=pipe_sel,
        src_all=src_all,   src_sel=src_sel,
        ctry_all=ctry_all, ctry_sel=ctry_sel,
        cslr_all=cslr_all, cslr_sel=cslr_sel
    )
