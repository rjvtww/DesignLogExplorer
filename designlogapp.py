import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from html import escape
from typing import Optional
from pathlib import Path

st.set_page_config(layout="wide")
st.title("TDG Biological Simulation Explorer")

uploaded_file = st.file_uploader("Upload a design_log.txt file", type="txt")

default_log_path = Path("design_log (88) 1044859 CAS.txt")

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
elif default_log_path.exists():
    st.caption("Using bundled CAS design log by default.")
    raw_text = default_log_path.read_text(encoding="utf-8", errors="ignore")
else:
    raw_text = None

if raw_text:
    lines = raw_text.splitlines()
    
    # Attempt to extract a project number from the log (fallback to filename)
    def extract_project_number(text: str, filename: Optional[str] = None):
        if not isinstance(text, str):
            return None
        # Try common patterns near the top of the file
        head = "\n".join(text.splitlines()[:20])
        patterns = [
            r"(?im)^\s*project\s*(?:no\.?|number|id)?\s*[:#-]?\s*(\d{4,9})\b",
            r"(?im)^\s*(\d{5,9})\b(?:\s*[A-Za-z]{2,}\b)?",
        ]
        for pat in patterns:
            m = re.search(pat, head)
            if m:
                return m.group(1)
        # Fallback: any 5-9 digit token in the first lines
        m = re.search(r"\b(\d{5,9})\b", head)
        if m:
            return m.group(1)
        # Fallback to filename if present
        if filename:
            m = re.search(r"\b(\d{5,9})\b", filename)
            if m:
                return m.group(1)
        return None

    project_number = extract_project_number(
        raw_text,
        uploaded_file.name if uploaded_file else (default_log_path.name if default_log_path.exists() else None),
    )

    def clean_cell(cell):
        match = re.match(r'^([-+]?[0-9]*\.?[0-9]+)', cell.strip())
        return match.group(1) if match else cell.strip()

    def parse_resilient_table(lines):
        def is_separator_row(cells):
            return all(re.match(r'^_+$', cell.strip()) for cell in cells)
        headers = []
        data_rows = []
        for line in lines:
            if re.match(r'\|[A-Za-z0-9_\-]+.*\|', line):
                cells = [clean_cell(cell) for cell in line.strip().strip('|').split('|')]
                if not headers:
                    headers = cells
                else:
                    if not is_separator_row(cells):
                        if len(cells) < len(headers):
                            cells = cells + [''] * (len(headers) - len(cells))
                        data_rows.append(cells[:len(headers)])
        if headers and data_rows:
            return pd.DataFrame(data_rows, columns=headers)
        return None

    def extract_relevant_text(text):
        def round_decimals_in_line(line):
            return re.sub(r'([-+]?[0-9]*\.[0-9]+)', lambda m: f"{float(m.group()):.2f}", line)

        lines = text.splitlines()
        filtered = []
        for line in lines:
            lower = line.lower()
            if "check" in lower and "compliance: no" in lower:
                filtered.append(round_decimals_in_line(line.strip()))
            elif "action" in lower and not ("setting reactor volumes" in lower or "setting excess sludge" in lower):
                filtered.append(round_decimals_in_line(line.strip()))

        # Join using <br> but add a new <br> before any new "##" line to force wrapping
        processed = []
        for line in filtered:
            if line.startswith("##") and processed:
                processed.append("<br>" + line)
            else:
                processed.append(line)

        return "<br>".join(processed)

    tables_by_iteration = []
    current_block = []
    grouped_tables = []
    iteration_index = 0
    iteration_texts = []
    iteration_raw_notes = []
    text_accumulator = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|'):
            current_block.append(line)
        elif stripped.startswith('##'):
            text_accumulator.append(stripped)
        elif stripped and not re.match(r'^_+$', stripped):
            text_accumulator.append(stripped)
        else:
            if current_block:
                df = parse_resilient_table(current_block)
                if df is not None:
                    grouped_tables.append(df)
                current_block = []
            if len(grouped_tables) == 2:
                for idx, part in enumerate(grouped_tables):
                    part = part.copy()
                    part['subtable'] = 'reactor' if idx == 0 else 'quality'
                    part['iteration'] = iteration_index
                    tables_by_iteration.append(part)
                grouped_tables = []
                raw_notes = '\n'.join(text_accumulator).strip()
                iteration_raw_notes.append((iteration_index, raw_notes))
                processed = extract_relevant_text(raw_notes)
                iteration_texts.append((iteration_index, processed))
                text_accumulator = []
                iteration_index += 1

    if current_block:
        df = parse_resilient_table(current_block)
        if df is not None:
            grouped_tables.append(df)

    if len(grouped_tables) == 2:
        for idx, part in enumerate(grouped_tables):
            part = part.copy()
            part['subtable'] = 'reactor' if idx == 0 else 'quality'
            part['iteration'] = iteration_index
            tables_by_iteration.append(part)
        raw_notes = '\n'.join(text_accumulator).strip()
        iteration_raw_notes.append((iteration_index, raw_notes))
        processed = extract_relevant_text(raw_notes)
        iteration_texts.append((iteration_index, processed))
        grouped_tables = []
        text_accumulator = []
        iteration_index += 1

    parsed_df = pd.concat(tables_by_iteration, ignore_index=True)
    hover_df = pd.DataFrame(iteration_texts, columns=['iteration', 'hover_text'])
    raw_notes_df = pd.DataFrame(iteration_raw_notes, columns=['iteration', 'raw_notes'])
    iteration_raw_map = raw_notes_df.set_index('iteration')['raw_notes']
    parsed_df['iteration_notes'] = ""
    first_iteration_rows = ~parsed_df['iteration'].duplicated()
    parsed_df.loc[first_iteration_rows, 'iteration_notes'] = parsed_df.loc[first_iteration_rows, 'iteration'].map(
        iteration_raw_map
    ).fillna("")

    # Extract MLSS (mg/L) from INFO and Aerobic HRT (d) from ACTION per iteration
    def extract_mlss(text: str):
        if not isinstance(text, str):
            return None
        # Prefer "Current MLSS <num>"; fallback to any MLSS line's first number
        m = re.search(r"(?i)current\s*mlss\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not m:
            m = re.search(r"(?i)mlss[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
        return float(m.group(1)) if m else None

    def extract_aerob_hrt(text: str):
        if not isinstance(text, str):
            return None
        m = re.search(r"(?i)aerob\s*hrt\s*([0-9]+(?:\.[0-9]+)?)\s*d", text)
        return float(m.group(1)) if m else None

    def extract_anoxic_hrt(text: str):
        if not isinstance(text, str):
            return None
        # Match "anox HRT 6.4 h" or "anoxic HRT 0.3 d" (case-insensitive)
        m = re.search(r"(?i)anox(?:ic)?\s*hrt\s*([0-9]+(?:\.[0-9]+)?)\s*([hd])", text)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2).lower()
        # Normalize to days for consistency with Aerobic HRT
        if unit == 'h':
            return val / 24.0
        return val

    def extract_anox_ratio(text: str):
        if not isinstance(text, str):
            return None
        # Match "Anox ratio 0.3" or "Anoxic ratio 0.3"
        m = re.search(r"(?i)anox(?:ic)?\s*ratio\s*([0-9]+(?:\.[0-9]+)?)", text)
        return float(m.group(1)) if m else None

    # Build maps from raw iteration notes
    mlss_map = {}
    hrt_map = {}
    anox_hrt_map = {}
    anox_ratio_map = {}
    srt_current_map = {}
    srt_min_map = {}
    for it, raw in iteration_raw_notes:
        mlss_val = extract_mlss(raw)
        if mlss_val is not None:
            mlss_map[it] = mlss_val
        hrt_val = extract_aerob_hrt(raw)
        if hrt_val is not None:
            hrt_map[it] = hrt_val
        anox_hrt_val = extract_anoxic_hrt(raw)
        if anox_hrt_val is not None:
            anox_hrt_map[it] = anox_hrt_val
        anox_ratio_val = extract_anox_ratio(raw)
        if anox_ratio_val is not None:
            anox_ratio_map[it] = anox_ratio_val
        # Extract SRT values from CHECK text if present
        if isinstance(raw, str):
            m_curr = re.search(r"(?i)current\s*srt\s*([0-9]+(?:\.[0-9]+)?)\s*d", raw)
            m_min = re.search(r"(?i)min(?:imum)?\s*srt\s*([0-9]+(?:\.[0-9]+)?)\s*d", raw)
            if m_curr:
                srt_current_map[it] = float(m_curr.group(1))
            if m_min:
                srt_min_map[it] = float(m_min.group(1))

    # Append MLSS and Aerobic HRT as additional quality rows (one per iteration)
    extra_rows = []
    for it, val in mlss_map.items():
        extra_rows.append({
            'Reactor': 'MLSS',
            'effluent': float(val),
            'limit': -1.00,
            'iteration': it,
            'subtable': 'quality',
        })
    for it, val in hrt_map.items():
        extra_rows.append({
            'Reactor': 'Aerobic HRT',
            'effluent': float(val),
            'limit': -1.00,
            'iteration': it,
            'subtable': 'quality',
        })
    # Append Anoxic HRT (normalized to days)
    for it, val in anox_hrt_map.items():
        extra_rows.append({
            'Reactor': 'Anoxic HRT',
            'effluent': float(val),
            'limit': -1.00,
            'iteration': it,
            'subtable': 'quality',
        })
    # Append Anoxic Ratio (unitless)
    for it, val in anox_ratio_map.items():
        extra_rows.append({
            'Reactor': 'Anoxic Ratio',
            'effluent': float(val),
            'limit': -1.00,
            'iteration': it,
            'subtable': 'quality',
        })
    # Append SRT current/minimum as additional quality rows (one per iteration)
    for it, val in srt_current_map.items():
        extra_rows.append({
            'Reactor': 'SRT current',
            'effluent': float(val),
            'limit': -1.00,
            'iteration': it,
            'subtable': 'quality',
        })
    for it, val in srt_min_map.items():
        extra_rows.append({
            'Reactor': 'SRT minimum',
            'effluent': float(val),
            'limit': -1.00,
            'iteration': it,
            'subtable': 'quality',
        })
    if extra_rows:
        parsed_df = pd.concat([parsed_df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Ensure Arrow-friendly dtypes for Streamlit (avoid mixed object types)
    if 'effluent' in parsed_df.columns:
        parsed_df['effluent'] = pd.to_numeric(parsed_df['effluent'], errors='coerce')
    if 'limit' in parsed_df.columns:
        parsed_df['limit'] = pd.to_numeric(parsed_df['limit'], errors='coerce')
    parsed_df['iteration'] = pd.to_numeric(parsed_df['iteration'], errors='coerce').astype('Int64')

    with st.expander("Parsed Data Table", expanded=False):
        st.dataframe(parsed_df, width='stretch')

    # Build categorized iteration notes table (INFO, CHECK, OTHER, WARNING, ACTION)
    def categorize_notes(raw_text):
        buckets = {k: [] for k in ["INFO", "CHECK", "OTHER", "WARNING", "ACTION"]}
        for ln in (raw_text or "").splitlines():
            s = ln.strip()
            if not s.startswith("##"):
                continue
            m = re.match(r'^##\s*\d+\s+([A-Za-z]+)\s*(.*)$', s)
            if m:
                cat = m.group(1).upper()
                msg = m.group(2).strip()
                if cat not in buckets:
                    cat = "OTHER"
                buckets[cat].append(msg)
            else:
                buckets["OTHER"].append(s.lstrip('#').strip())
        return {k: "\n".join(v) for k, v in buckets.items()}

    cat_rows = []
    for _, r in raw_notes_df.iterrows():
        cats = categorize_notes(r["raw_notes"])
        row = {"iteration": r["iteration"], **cats}
        cat_rows.append(row)
    notes_cat_df = pd.DataFrame(cat_rows)
    notes_cat_df = notes_cat_df[["iteration", "INFO", "CHECK", "OTHER", "WARNING", "ACTION"]].sort_values("iteration")

    # Helper used for both HTML table and hover notes
    def round_numbers_in_text(s: str) -> str:
        # Round standalone numeric tokens to 1 decimal; avoid touching identifiers like V_1 or units like m3
        pattern = re.compile(r'(?<![A-Za-z_])([-+]?\d+(?:\.\d+)?)(?![A-Za-z_])')
        def repl(m):
            try:
                return f"{float(m.group(1)):.1f}"
            except Exception:
                return m.group(1)
        return pattern.sub(repl, s)

    with st.expander("Iteration Notes by Category", expanded=False):
        def to_html_with_breaks(df, cols):
            df2 = df.copy()
            for c in cols:
                if c in df2.columns:
                    df2[c] = df2[c].apply(lambda v: escape(round_numbers_in_text(str(v))).replace('\n', '<br>'))
            return df2.to_html(escape=False, index=False)

        html_table = to_html_with_breaks(
            notes_cat_df,
            ["INFO", "CHECK", "OTHER", "WARNING", "ACTION"],
        )
        st.markdown(html_table, unsafe_allow_html=True)

    # Subheader showing project number between folded tables and plots (if detected)
    if project_number:
        st.subheader(f"Project {project_number}")

    # Build per-iteration hover HTML from categorized notes for use across plots
    def build_hover_notes_html(row):
        parts = []
        for cat in ["INFO", "CHECK", "OTHER", "WARNING", "ACTION"]:
            val = str(row.get(cat, "")).strip()
            if val:
                line_html = []
                for ln in val.splitlines():
                    rounded = round_numbers_in_text(ln)
                    esc = escape(rounded)
                    if "compliance: no" in ln.lower():
                        esc = f"<b>{esc}</b>"
                    line_html.append(esc)
                val_html = "<br>".join(line_html)
                parts.append(f"<b>{cat}</b>:<br>{val_html}")
        return "<br>".join(parts) if parts else "<i>No notes</i>"

    iter_notes_html_map = {
        int(r["iteration"]): build_hover_notes_html(r)
        for _, r in notes_cat_df.iterrows()
    }

    # Shared preprocessing
    reactor_df = parsed_df[parsed_df['subtable'] == 'reactor']
    quality_df = parsed_df[parsed_df['subtable'] == 'quality'].copy()
    quality_df['iteration'] = pd.to_numeric(quality_df['iteration'])

    do_df = reactor_df[reactor_df['Reactor'] == 'DO'].copy()
    vfcr_df = reactor_df[reactor_df['Reactor'] == 'V_fcr'].copy()

    valid_ids = [c for c in ['1', '2', '3', '4', '5', '6', '7'] if c in do_df.columns and c in vfcr_df.columns]
    for col in valid_ids:
        do_df[col] = pd.to_numeric(do_df[col], errors='coerce')
        vfcr_df[col] = pd.to_numeric(vfcr_df[col], errors='coerce')
    do_df['iteration'] = pd.to_numeric(do_df['iteration'])
    vfcr_df['iteration'] = pd.to_numeric(vfcr_df['iteration'])

    zone_data = []
    for i in do_df['iteration'].unique():
        row = {'iteration': i, 'anaerobic': 0, 'anoxic': 0, 'aerobic': 0, 'post-anoxic': 0}
        do_row = do_df[do_df['iteration'] == i].iloc[0]
        v_row = vfcr_df[vfcr_df['iteration'] == i].iloc[0]

        if do_row['2'] and float(do_row['2']) > 0:
            row['anoxic'] += float(v_row['1'])
        else:
            row['anaerobic'] += float(v_row['1'])
            row['anoxic'] += float(v_row['2'])
            if '3' in valid_ids and float(do_row['3']) == 0:
                row['anoxic'] += float(v_row['3'])

        for r in ['2', '3', '4', '5', '6', '7']:
            if r in valid_ids:
                do_val = float(do_row[r])
                if do_val > 0:
                    row['aerobic'] += float(v_row[r])
                elif r in ['6', '7'] and float(do_val) == 0:
                    row['post-anoxic'] += float(v_row[r])

        zone_data.append(row)

    zone_df = pd.DataFrame(zone_data)
    zone_df['total_volume'] = zone_df[['anaerobic', 'anoxic', 'aerobic', 'post-anoxic']].sum(axis=1)
    zone_hover = hover_df.set_index('iteration').reindex(zone_df['iteration']).fillna('').shift(-1).hover_text.tolist()

    tkn_df = quality_df[quality_df['Reactor'] == 'TKN'][['iteration', 'effluent']].rename(columns={'effluent': 'TKN'})
    no3_df = quality_df[quality_df['Reactor'] == 'NO3'][['iteration', 'effluent']].rename(columns={'effluent': 'NO3'})
    tn_df = pd.merge(tkn_df, no3_df, on='iteration', how='inner')
    # Extract TN limit from hover text if available
    tn_limit = None
    for _, row in hover_df.iterrows():
        if isinstance(row['hover_text'], str):
            match = re.search(r'TN limit:\s*(\d+\.?\d*)\s*mg', row['hover_text'])
            if match:
                tn_limit = float(match.group(1))
                break

    tn_df['TN'] = pd.to_numeric(tn_df['TKN'], errors='coerce') + pd.to_numeric(tn_df['NO3'], errors='coerce')
    tn_hover = hover_df.set_index('iteration').reindex(tn_df['iteration']).fillna('').shift(-1).hover_text.tolist()

    effluent_vars_all = quality_df['Reactor'].unique().tolist()
    # Exclude SRT helper variables; they get a combined subplot
    effluent_vars_all = [v for v in effluent_vars_all if v not in ['SRT current', 'SRT minimum']]
    priority_order = ['NH4', 'NO3', 'TKN', 'BOD']
    prioritized = [v for v in priority_order if v in effluent_vars_all]
    remaining = [v for v in effluent_vars_all if v not in priority_order]
    effluent_vars_ordered = prioritized + remaining

    # Optional sections
    r_ir_check = reactor_df[reactor_df['Reactor'] == 'R_FCR_IR']
    has_r_ir = not r_ir_check.empty
    srt_iters = sorted(set(list(srt_current_map.keys()) + list(srt_min_map.keys())))
    has_srt = len(srt_iters) > 0

    subplot_titles = ['Zone Volumes', 'Total Nitrogen']
    if has_r_ir:
        subplot_titles.append('R_FCR_IR')
    if has_srt:
        subplot_titles.append('SRT')
    subplot_titles += [f"{var} - Effluent vs Limit" for var in effluent_vars_ordered]
    num_plots = len(subplot_titles)

    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=subplot_titles)

    # Volume bars
    zone_colors = {'anaerobic': 'darkblue', 'anoxic': 'orange', 'aerobic': 'green', 'post-anoxic': 'purple'}
    zone_hover_text = zone_df['iteration'].map(iter_notes_html_map).fillna("")
    hover_tmpl_basic = (
        "%{fullData.name}: %{y}<extra></extra>"
    )
    hover_tmpl_with_notes = (
        "Iteration: %{customdata}<br>" \
        "%{fullData.name}: %{y}<br><br>%{text}<extra></extra>"
    )
    for zone in ['anaerobic', 'anoxic', 'aerobic', 'post-anoxic']:
        fig.add_trace(go.Bar(
            x=zone_df['iteration'], y=zone_df[zone], name=zone,
            marker_color=zone_colors.get(zone),
            hovertemplate=hover_tmpl_basic, legendgroup='volume', showlegend=True
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=zone_df['iteration'], y=zone_df['total_volume'], name='Total Volume', mode='lines+markers',
        line=dict(color='black', width=2, dash='dash'), text=zone_hover_text, customdata=zone_df['iteration'],
        hovertemplate=hover_tmpl_with_notes, legendgroup='volume'
    ), row=1, col=1)

    # Inline legend for row 1 (Zone Volumes)
    zone_legend_html = (
        f"<span style='color:{zone_colors['anaerobic']}'>■ anaerobic</span><br>"
        f"<span style='color:{zone_colors['anoxic']}'>■ anoxic</span><br>"
        f"<span style='color:{zone_colors['aerobic']}'>■ aerobic</span><br>"
        f"<span style='color:{zone_colors['post-anoxic']}'>■ post-anoxic</span><br>"
        f"<span style='color:black'>— — Total Volume</span>"
    )
    fig.add_annotation(
        row=1, col=1, x=0.99, y=0.99, xref='x domain', yref='y domain', showarrow=False,
        xanchor='right', yanchor='top', align='right',
        text=zone_legend_html, bgcolor='rgba(255,255,255,0.7)'
    )

    # TN plot
    tn_text = tn_df['iteration'].map(iter_notes_html_map).fillna("")
    fig.add_trace(go.Scatter(
        x=tn_df['iteration'], y=tn_df['TN'], mode='lines+markers', name='Total Nitrogen', line=dict(color='darkcyan'),
        text=tn_text, customdata=tn_df['iteration'], hovertemplate=hover_tmpl_with_notes
    ), row=2, col=1)
    if tn_limit is not None:
      fig.add_trace(go.Scatter(
          x=tn_df['iteration'],
          y=[tn_limit] * len(tn_df),
          mode='lines',
          name='TN Limit',
          line=dict(color='red', dash='dash'),
          hovertemplate=hover_tmpl_basic
      ), row=2, col=1)

    # Inline legend for row 2 (Total Nitrogen)
    tn_legend_html = "<span style='color:darkcyan'>— Total Nitrogen</span>"
    if tn_limit is not None:
        tn_legend_html += "<br><span style='color:red'>— — TN Limit</span>"
    fig.add_annotation(
        row=2, col=1, x=0.99, y=0.99, xref='x domain', yref='y domain', showarrow=False,
        xanchor='right', yanchor='top', align='right',
        text=tn_legend_html, bgcolor='rgba(255,255,255,0.7)'
    )

    # Optional subplot rows start after row 2
    current_row = 3

    # R_FCR_IR subplot (if available)
    if has_r_ir:
        r_ir = r_ir_check.copy()
        iter_series = pd.to_numeric(r_ir['iteration'], errors='coerce')
        cols = [c for c in ['1','2','3','4','5','6','7'] if c in r_ir.columns]
        for c in cols:
            r_ir[c] = pd.to_numeric(r_ir[c], errors='coerce')
        if cols:
            r_ir['value'] = r_ir[cols].apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None, axis=1)
            fig.add_trace(go.Scatter(
                x=iter_series, y=r_ir['value'], mode='lines+markers', name='R_FCR_IR',
                text=iter_series.map(iter_notes_html_map).fillna(""), customdata=iter_series,
                hovertemplate=hover_tmpl_with_notes, line=dict(color='teal')
            ), row=current_row, col=1)
            fig.add_annotation(
                row=current_row, col=1, x=0.99, y=0.99, xref='x domain', yref='y domain', showarrow=False,
                xanchor='right', yanchor='top', align='right',
                text="R_FCR_IR", bgcolor='rgba(255,255,255,0.7)'
            )
        current_row += 1

    # SRT combined subplot (if available)
    if has_srt:
        srt_cur_vals = [srt_current_map.get(i, None) for i in srt_iters]
        srt_min_vals = [srt_min_map.get(i, None) for i in srt_iters]
        srt_text = pd.Series(srt_iters).map(iter_notes_html_map).fillna("")
        fig.add_trace(go.Scatter(x=srt_iters, y=srt_cur_vals, mode='lines+markers', name='SRT Current',
                                 text=srt_text, customdata=srt_iters, hovertemplate=hover_tmpl_with_notes,
                                 line=dict(color='darkorange')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=srt_iters, y=srt_min_vals, mode='lines', name='SRT Minimum',
                                 hovertemplate=hover_tmpl_basic, line=dict(color='red', dash='dash')), row=current_row, col=1)
        fig.add_annotation(
            row=current_row, col=1, x=0.99, y=0.99, xref='x domain', yref='y domain', showarrow=False,
            xanchor='right', yanchor='top', align='right',
            text="SRT: Current / Min", bgcolor='rgba(255,255,255,0.7)'
        )
        current_row += 1

    # Effluent plots begin at the current row
    for i, var in enumerate(effluent_vars_ordered):
        sub = quality_df[quality_df['Reactor'] == var].copy()
        if 'effluent' in sub.columns and 'limit' in sub.columns:
            sub['effluent'] = pd.to_numeric(sub['effluent'], errors='coerce')
            sub['limit'] = pd.to_numeric(sub['limit'], errors='coerce')
            sub_text = sub['iteration'].map(iter_notes_html_map).fillna("")
            fig.add_trace(go.Scatter(x=sub['iteration'], y=sub['effluent'], mode='lines+markers', name=f'{var} Effluent', text=sub_text, customdata=sub['iteration'], hovertemplate=hover_tmpl_with_notes), row=current_row+i, col=1)
            has_limit = not all(sub['limit'] == -1.00)
            if has_limit:
                fig.add_trace(go.Scatter(x=sub['iteration'], y=sub['limit'], mode='lines', name=f'{var} Limit', line=dict(color='red', dash='dash'), hovertemplate=hover_tmpl_basic), row=current_row+i, col=1)

            # Inline legend for each effluent subplot
            eff_legend_html = f"{escape(var)}: Effluent"
            if has_limit:
                eff_legend_html += "<br><span style='color:red'>— — Limit</span>"
            fig.add_annotation(
                row=current_row+i, col=1, x=0.99, y=0.99, xref='x domain', yref='y domain', showarrow=False,
                xanchor='right', yanchor='top', align='right',
                text=eff_legend_html, bgcolor='rgba(255,255,255,0.7)'
            )

    fig.update_layout(height=300*num_plots, title_text="Design Log Data", hovermode="x unified",hoversubplots="axis", xaxis_showticklabels=True, showlegend=False)
    # Hide the unified hover header showing the x-value; we render our own Iteration label
    fig.update_xaxes(hoverformat='')
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})

    st.download_button("Download Full Parsed Data (CSV)", parsed_df.to_csv(index=False), file_name="parsed_data.csv")
    st.download_button("Download Hover Notes (CSV)", hover_df.to_csv(index=False), file_name="hover_notes.csv")
else:
    st.warning("Upload a design log to explore the data.")
    cols = [c for c in ['1','2','3','4','5','6','7'] if c in r_ir.columns]
    for c in cols:
        r_ir[c] = pd.to_numeric(r_ir[c], errors='coerce')
    if cols:
        r_ir['value'] = r_ir[cols].bfill(axis=1).iloc[:,0]
        fig.add_trace(go.Scatter(
            x=iter_series, y=r_ir['value'], mode='lines+markers', name='R_FCR_IR',
            text=iter_series.map(iter_notes_html_map).fillna(""), customdata=iter_series,
            hovertemplate=hover_tmpl_with_notes, line=dict(color='teal')
        ), row=3, col=1)
        # Inline legend for R_FCR_IR
        fig.add_annotation(
            row=3, col=1, x=0.99, y=0.99, xref='x domain', yref='y domain', showarrow=False,
            xanchor='right', yanchor='top', align='right',
            text="R_FCR_IR", bgcolor='rgba(255,255,255,0.7)'
        )

    # SRT subplot (row 4) with current vs minimum as dashed target
    srt_iters = sorted(set(list(srt_current_map.keys()) + list(srt_min_map.keys())))
    if srt_iters:
        srt_cur_vals = [srt_current_map.get(i, None) for i in srt_iters]
        srt_min_vals = [srt_min_map.get(i, None) for i in srt_iters]
        srt_text = pd.Series(srt_iters).map(iter_notes_html_map).fillna("")
        fig.add_trace(go.Scatter(x=srt_iters, y=srt_cur_vals, mode='lines+markers', name='SRT Current',
                                 text=srt_text, customdata=srt_iters, hovertemplate=hover_tmpl_with_notes,
                                 line=dict(color='darkorange')), row=4, col=1)
        fig.add_trace(go.Scatter(x=srt_iters, y=srt_min_vals, mode='lines', name='SRT Minimum',
                                 hovertemplate=hover_tmpl_basic, line=dict(color='red', dash='dash')), row=4, col=1)
        fig.add_annotation(
            row=4, col=1, x=0.99, y=0.99, xref='x domain', yref='y domain', showarrow=False,
            xanchor='right', yanchor='top', align='right',
            text="SRT: Current / Min", bgcolor='rgba(255,255,255,0.7)'
        )
