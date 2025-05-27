import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(layout="wide")
st.title("TDG Design Log for babies - by RV")

uploaded_file = st.file_uploader("Upload a design_log.txt file", type="txt")

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()

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
                    if len(cells) >= len(headers) and not is_separator_row(cells):
                        data_rows.append(cells[:len(headers)])
        if headers and data_rows:
            return pd.DataFrame(data_rows, columns=headers)
        return None

    def extract_relevant_text(text):
        lines = text.splitlines()
        filtered = []
        for line in lines:
            lower = line.lower()
            if "check" in lower and "compliance: no" in lower:
                filtered.append(line.strip())
            elif "action" in lower and not ("setting reactor volumes" in lower or "setting excess sludge" in lower):
                filtered.append(line.strip())
        return "\n".join(filtered)

    tables_by_iteration = []
    current_block = []
    grouped_tables = []
    iteration_index = 0
    iteration_texts = []
    text_accumulator = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|'):
            current_block.append(line)
        elif stripped.startswith('##'):
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
                processed = extract_relevant_text('\n'.join(text_accumulator).strip())
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
        processed = extract_relevant_text('\n'.join(text_accumulator).strip())
        iteration_texts.append((iteration_index, processed))
        grouped_tables = []
        iteration_index += 1

    parsed_df = pd.concat(tables_by_iteration, ignore_index=True)
    hover_df = pd.DataFrame(iteration_texts, columns=['iteration', 'hover_text'])

    #st.subheader("Parsed Table Sample")
    #st.dataframe(parsed_df.head(10))

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

    effluent_vars = quality_df['Reactor'].unique().tolist()
    num_plots = 2 + len(effluent_vars)
    subplot_titles = ['Zone Volumes', 'Total Nitrogen'] + [f"{var} - Effluent vs Limit" for var in effluent_vars]

    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=subplot_titles)

    # Volume bars
    zone_colors = {'anaerobic': 'darkblue', 'anoxic': 'orange', 'aerobic': 'green', 'post-anoxic': 'purple'}
    for zone in ['anaerobic', 'anoxic', 'aerobic', 'post-anoxic']:
        fig.add_trace(go.Bar(
            x=zone_df['iteration'], y=zone_df[zone], name=zone,
            marker_color=zone_colors.get(zone), hovertext=zone_hover, hoverinfo='text+y', legendgroup='volume', showlegend=True
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=zone_df['iteration'], y=zone_df['total_volume'], name='Total Volume', mode='lines+markers',
        line=dict(color='black', width=2, dash='dash'), hovertext=zone_hover, hoverinfo='text+y', legendgroup='volume'
    ), row=1, col=1)

    # TN plot
    fig.add_trace(go.Scatter(
        x=tn_df['iteration'], y=tn_df['TN'], mode='lines+markers', name='Total Nitrogen', line=dict(color='darkcyan'),
        hovertext=tn_hover, hoverinfo='text+y'
    ), row=2, col=1)
    if tn_limit is not None:
      fig.add_trace(go.Scatter(
          x=tn_df['iteration'],
          y=[tn_limit] * len(tn_df),
          mode='lines',
          name='TN Limit',
          line=dict(color='red', dash='dash'),
          hovertext=tn_hover,
          hoverinfo='text+y'
      ), row=2, col=1)

    # Effluent plots
    for i, var in enumerate(effluent_vars):
        sub = quality_df[quality_df['Reactor'] == var].copy()
        if 'effluent' in sub.columns and 'limit' in sub.columns:
            sub['effluent'] = pd.to_numeric(sub['effluent'], errors='coerce')
            sub['limit'] = pd.to_numeric(sub['limit'], errors='coerce')
            sub_hover = hover_df.set_index('iteration').reindex(sub['iteration']).fillna('').shift(-1).hover_text.tolist()
            fig.add_trace(go.Scatter(x=sub['iteration'], y=sub['effluent'], mode='lines+markers', name=f'{var} Effluent', hovertext=sub_hover, hoverinfo='text+y'), row=3+i, col=1)
            if not all(sub['limit'] == -1.00):
                fig.add_trace(go.Scatter(x=sub['iteration'], y=sub['limit'], mode='lines', name=f'{var} Limit', line=dict(color='red', dash='dash'), hovertext=sub_hover, hoverinfo='text+y'), row=3+i, col=1)

    fig.update_layout(height=300*num_plots, title_text="Synchronized Plots", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("Download Full Parsed Data (CSV)", parsed_df.to_csv(index=False), file_name="parsed_data.csv")
    st.download_button("Download Hover Notes (CSV)", hover_df.to_csv(index=False), file_name="hover_notes.csv")
