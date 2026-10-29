"""Streamlit page shared by the standalone app and the existing website."""
from pathlib import Path
import json
import re
import sys

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from engine import Predictor, CLASSES, MODELS


@st.cache_resource(show_spinner=False)
def get_predictor():
    return Predictor(HERE)


def probability_chart(result):
    chart_data = result.melt(id_vars=['Model', 'Prediction'], value_vars=CLASSES,
                             var_name='Class', value_name='Probability')
    base = alt.Chart(chart_data).encode(
        x=alt.X('Model:N', sort=list(MODELS), axis=alt.Axis(labelAngle=0), title=None),
        xOffset=alt.XOffset('Class:N', sort=CLASSES),
        y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1.12]),
                axis=alt.Axis(format='.0%', values=[0, .25, .5, .75, 1]), title='Model probability'),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.Color('Class:N', scale=alt.Scale(domain=CLASSES,
                         range=['#93C5FD', '#2563EB', '#0B1F3A']),
                        legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Model:N', 'Class:N', alt.Tooltip('Probability:Q', format='.2%')],
    )
    labels = base.mark_text(dy=-8, color='#173653', fontSize=10).encode(
        text=alt.Text('Probability:Q', format='.1%'))
    return (bars + labels).properties(height=300).configure_view(strokeWidth=0)



# GRAPH_EXPLORER_V1
def graph_data(predictor, node, limit=16):
    import math
    import numpy as np
    edges = predictor.edges.numpy()
    neighbours = set(edges[1, edges[0] == node].tolist())
    neighbours.update(edges[0, edges[1] == node].tolist())
    neighbours.discard(node)
    ordered = sorted(neighbours, key=lambda i: (-int(predictor.degree[i]), i))
    chosen = ordered[:limit]
    positions = {node: (0.0, 0.0)}
    for k, other in enumerate(chosen):
        angle = 2 * math.pi * k / max(len(chosen), 1) - math.pi / 2
        positions[other] = (math.cos(angle), math.sin(angle))
    visible = list(positions)
    mask = np.isin(edges[0], visible) & np.isin(edges[1], visible)
    pairs = sorted({tuple(sorted((int(a), int(b))))
                    for a, b in edges[:, mask].T if a != b})
    node_rows = []
    for other, (x, y) in positions.items():
        address = str(predictor.addresses[other])
        node_rows.append({'Node': other, 'Address': address, 'x': x, 'y': y,
                         'Label': 'Analysed address' if other == node else address[:8] + '...',
                         'Role': 'Analysed address' if other == node else 'Neighbour',
                         'Endpoint count': int(predictor.degree[other]),
                         'Split': 'Test' if other in predictor.test_set else 'Train / validation'})
    links = [{'x': positions[a][0], 'y': positions[a][1],
              'x2': positions[b][0], 'y2': positions[b][1]} for a, b in pairs]
    return pd.DataFrame(node_rows), pd.DataFrame(links), len(neighbours)


def address_network_chart(nodes, links):
    axes = dict(axis=None, scale=alt.Scale(domain=[-1.4, 1.4]))
    point_base = alt.Chart(nodes).encode(
        x=alt.X('x:Q', **axes), y=alt.Y('y:Q', **axes))
    points = point_base.mark_circle(stroke='white', strokeWidth=2).encode(
        size=alt.condition(alt.datum.Role == 'Analysed address', alt.value(950), alt.value(260)),
        color=alt.Color('Role:N', scale=alt.Scale(
            domain=['Analysed address', 'Neighbour'], range=['#1D4ED8', '#93C5FD']),
            legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Address:N', 'Role:N', 'Split:N', 'Endpoint count:Q'])
    labels = point_base.mark_text(dy=23, fontSize=10, color='#173653').encode(text='Label:N')
    layers = []
    if not links.empty:
        layers.append(alt.Chart(links).mark_rule(color='#B4C9E4', opacity=.7, strokeWidth=1.5)
                      .encode(x=alt.X('x:Q', **axes), y=alt.Y('y:Q', **axes),
                              x2='x2:Q', y2='y2:Q'))
    return alt.layer(*layers, points, labels).properties(height=440).interactive().configure_view(strokeWidth=0)


def render_graph_explorer(predictor, node):
    with st.expander('Explore this address in the benchmark graph', expanded=True):
        limit = st.slider('Maximum neighbours to display', 4, 40, 16,
                          key=f'graph_limit_{node}')
        nodes, links, total = graph_data(predictor, node, limit)
        a, b, c = st.columns(3)
        a.metric('Unique direct neighbours', total)
        b.metric('Neighbours displayed', len(nodes) - 1)
        c.metric('Links displayed', len(links))
        if total == 0:
            st.info('No links to other addresses are present for this address in the released graph. '
                    'This does not establish that it has no blockchain transactions or that it is safe. '
                    'Select Random benchmark example and enable the graph connections filter to explore a linked address.')
        elif total > limit:
            st.caption(f'Showing {limit} of {total} neighbours, ordered by endpoint count. '
                       'The display limit does not change the model input or prediction.')
        st.altair_chart(address_network_chart(nodes, links), width='stretch')
        st.caption('Hover over a node for its full address. Scroll to zoom and drag to pan. '
                   'Lines show adjacency in the released benchmark graph, without assuming '
                   'transaction direction, amount or time. Reciprocal and repeated edges are '
                   'shown as one link; self-loops are omitted. Blue tones identify node roles, not fraud classes.')
        st.dataframe(nodes.drop(columns=['x', 'y', 'Node', 'Label']),
                     hide_index=True, width='stretch')
        st.caption('Neighbours can belong to train, validation or test splits. '
                   'Only test addresses can be submitted to this demo. '
                   'Endpoint count follows the model graph, counting both endpoints of each edge. '
                   'Graph proximity alone is not evidence of fraud.')


def render_prediction_summary(table):
    lead = table[table.Model == 'X-MoE'].iloc[0] if 'X-MoE' in table.Model.tolist() else table.iloc[0]
    prediction = lead['Prediction']
    agreeing = int((table.Prediction == prediction).sum())
    a, b, c = st.columns(3)
    a.metric(f"{lead['Model']} prediction", prediction)
    b.metric('Model probability', f'{lead[prediction]:.2%}')
    c.metric('Models with same prediction', f'{agreeing} / {len(table)}')
    st.caption('Agreement compares the selected models with the displayed model. '
               'Probability and agreement are not accuracy or a guarantee of safety.')



# RESULT_EXPLANATION_V1
def prediction_explanation(table):
    lead = table[table.Model == 'X-MoE'].iloc[0] if 'X-MoE' in table.Model.tolist() else table.iloc[0]
    label = str(lead['Prediction'])
    model = str(lead['Model'])
    probability = float(lead[label])
    runner = max((c for c in CLASSES if c != label), key=lambda c: float(lead[c]))
    margin = 100 * (probability - float(lead[runner]))
    paragraphs = [
        f"{model} predicts **{label}** for the analysed address, with a model probability "
        f"of **{probability:.2%}**. Its next-highest class is **{runner}** "
        f"({float(lead[runner]):.2%}), a gap of **{margin:.2f} percentage points**."
    ]
    if len(table) == 1:
        paragraphs.append('Only one model was selected. Select additional models to compare their predictions.')
    else:
        count = int((table.Prediction == label).sum())
        paragraphs.append(
            f'**{count} of {len(table)} selected models** predict {label}. '
            'Agreement describes these model outputs; it does not establish correctness.'
        )
        different = table[table.Prediction != label]
        if not different.empty:
            paragraphs.append('Different predictions: ' + '; '.join(
                f"{row['Model']} predicts {row['Prediction']} "
                f"({float(row[row['Prediction']]):.2%})"
                for _, row in different.iterrows()
            ) + '. The models reach different classifications for this address.')
    meanings = {
        'Normal': 'Normal is the model\'s predicted benchmark class. It does not verify that an address is safe.',
        'Phishing': 'Phishing is the model\'s predicted benchmark class. This output alone does not confirm a phishing incident.',
        'Scamming': 'Scamming is the model\'s predicted benchmark class. This output alone does not confirm a scam incident.',
    }
    paragraphs.append(meanings[label])
    paragraphs.append(
        'The percentage is a model probability across Normal, Phishing and Scamming. '
        'It is not the accuracy of this individual prediction or a calibrated guarantee. '
        'Displayed values are rounded; 100.00% does not mean absolute certainty. '
        'Analysis uses frozen benchmark features and graph context, not current blockchain activity.'
    )
    return paragraphs


def render_prediction_explanation(table):
    with st.container(border=True):
        st.markdown('#### Understanding this result')
        for paragraph in prediction_explanation(table):
            st.write(paragraph)
        st.caption('This explanation summarises model outputs. It does not identify which '
                   'transaction features caused the prediction. Dataset-label correctness, '
                   'when revealed below, describes this example rather than overall model accuracy.')



# DATA_SCOPE_V1
def demo_scope_tables(predictor):
    edges = predictor.edges.numpy()
    linked = set(edges[:, edges[0] != edges[1]].flatten().tolist())
    test = predictor.test_indices
    full_total = len(predictor.labels)
    distribution, coverage = [], []
    for label_id, label in enumerate(CLASSES):
        full_count = int((predictor.labels == label_id).sum())
        selected = test[predictor.labels[test] == label_id]
        connected = sum(int(i) in linked for i in selected)
        distribution.append({'Class': label, 'Full dataset': full_count,
                             'Full dataset (%)': 100 * full_count / full_total,
                             'Demo test set': len(selected)})
        coverage.append({'Class': label, 'Test addresses with links': connected,
                         'Test addresses without links': len(selected) - connected})
    return pd.DataFrame(distribution), pd.DataFrame(coverage)


def render_demo_scope(predictor):
    distribution, coverage = demo_scope_tables(predictor)
    full_total = len(predictor.labels)
    test_total = len(predictor.test_indices)
    linked_total = int(coverage['Test addresses with links'].sum())
    a, b, c = st.columns(3)
    a.metric('Full dataset addresses', f'{full_total:,}')
    b.metric('Demo test addresses', f'{test_total:,}')
    c.metric('Demo addresses with graph links', f'{linked_total:,}')
    st.info(f'The full dataset contains {full_total:,} addresses. This demo accepts only '
            f'{test_total:,} held-out test addresses from the seed-100 split. '
            'Training and validation addresses are not available for prediction requests.')
    with st.expander('Full Dataset vs Demo Test Set', expanded=True):
        st.markdown('**Class distribution**')
        st.dataframe(distribution.style.format({
            'Full dataset': '{:,.0f}', 'Demo test set': '{:,.0f}',
            'Full dataset (%)': '{:.2f}%'}), hide_index=True, width='stretch')
        st.caption(f'Totals: {full_total:,} full-dataset addresses; '
                   f'{test_total:,} demo test addresses. The demo test set is a subset '
                   'of the full dataset, not a different dataset.')
        st.markdown('**Graph Coverage Within the Demo Test Set**')
        st.dataframe(coverage.style.format({
            'Test addresses with links': '{:,.0f}',
            'Test addresses without links': '{:,.0f}'}), hide_index=True, width='stretch')
        st.caption(f'{linked_total:,} demo test addresses have links to other nodes; '
                   f'{test_total - linked_total:,} do not in the released graph. '
                   'A link means recorded benchmark adjacency, not a confirmed fraud transaction. '
                   'No recorded links does not establish safety or absence of real-world transactions.')
        st.write('**Model context:** Prediction requests are restricted to test addresses, '
                 'while inference uses the full released graph and its node features. '
                 'Training and validation nodes can therefore appear as graph neighbours. '
                 'Test labels are not inputs to the model forward pass.')
    st.caption('Frozen seed 100 | clean protocol | 4 trained models | 3 output classes. '
               'Scores are model probabilities, not accuracy or calibrated certainty. '
               'No live blockchain activity is retrieved.')



# BATCH_REPORTS_V1
def batch_summary_tables(result):
    predicted = result[result['Status'].eq('Predicted')].copy()
    if predicted.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    counts = pd.crosstab(predicted['Model'], predicted['Prediction']).reindex(
        index=[m for m in MODELS if m in predicted.Model.unique()],
        columns=CLASSES, fill_value=0).fillna(0).astype(int)
    counts['Addresses analysed'] = counts.sum(axis=1)
    counts = counts.rename_axis('Model').reset_index()
    comparison = predicted.pivot(index='Address', columns='Model', values='Prediction')
    comparison = comparison.reindex(columns=[m for m in MODELS if m in comparison.columns])
    if len(comparison.columns) > 1:
        comparison['Agreement'] = comparison.nunique(axis=1).map(
            lambda n: 'All selected models agree' if n == 1 else 'Models disagree')
    else:
        comparison['Agreement'] = 'Not assessed: one model selected'
    comparison = comparison.reset_index()
    disagreements = comparison[comparison.Agreement.eq('Models disagree')].copy()
    return counts, comparison, disagreements


def render_batch_summary(result):
    counts, comparison, disagreements = batch_summary_tables(result)
    st.subheader('Batch summary')
    a, b, c = st.columns(3)
    a.metric('Addresses analysed', len(comparison))
    b.metric('Addresses skipped', result.loc[result.Status.ne('Predicted'), 'Address'].nunique())
    if len(counts) > 1:
        c.metric('Addresses with model disagreement', len(disagreements))
    else:
        c.metric('Model disagreement', 'N/A')
    st.dataframe(counts, hide_index=True, width='stretch')
    st.caption('Counts describe predicted classes for this batch, not dataset labels or accuracy. '
               'Each analysed address is counted once per model. Model agreement does not establish correctness.')
    if not counts.empty:
        chart_data = counts.melt(id_vars='Model', value_vars=CLASSES,
                                 var_name='Class', value_name='Addresses')
        chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3,
                                              cornerRadiusTopRight=3).encode(
            x=alt.X('Model:N', sort=list(MODELS), axis=alt.Axis(labelAngle=0), title=None),
            xOffset=alt.XOffset('Class:N', sort=CLASSES),
            y=alt.Y('Addresses:Q', axis=alt.Axis(tickMinStep=1), title='Predicted address count'),
            color=alt.Color('Class:N', scale=alt.Scale(domain=CLASSES,
                           range=['#93C5FD', '#2563EB', '#0B1F3A']),
                           legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Model:N', 'Class:N', 'Addresses:Q'])
        st.altair_chart(chart.properties(height=260).configure_view(strokeWidth=0), width='stretch')
    with st.expander('Compare model decisions for each address', expanded=not disagreements.empty):
        if len(counts) <= 1:
            st.info('Select at least two models to assess disagreement.')
        elif disagreements.empty:
            st.info('All selected models predict the same class for every analysed address in this batch.')
        else:
            st.markdown('**Addresses with different model predictions**')
            st.dataframe(disagreements, hide_index=True, width='stretch')
            st.download_button('Download model disagreements CSV', disagreements.to_csv(index=False),
                               file_name='model_disagreements.csv', mime='text/csv', key='disagreement_csv')
        st.markdown('**All analysed addresses**')
        st.dataframe(comparison, hide_index=True, width='stretch')


def report_table(frame, probabilities=False):
    frame = frame.copy()
    if probabilities:
        for label in CLASSES:
            if label in frame:
                frame[label] = frame[label].map(lambda v: '' if pd.isna(v) else f'{float(v):.2%}')
    return frame.to_html(index=False, escape=True, border=0, na_rep='')


def report_graph_svg(predictor, node, limit):
    from html import escape
    nodes, links, total = graph_data(predictor, node, limit)
    if total == 0:
        return '<p>No links to other addresses are recorded for this node in the benchmark graph. This does not establish safety.</p>'
    def xy(x, y):
        return 350 + 205 * x, 250 - 185 * y
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 500" role="img" aria-label="Benchmark address neighbourhood">']
    for row in links.itertuples():
        x, y = xy(row.x, row.y); x2, y2 = xy(row.x2, row.y2)
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="#B4C9E4"/>')
    for row in nodes.itertuples():
        x, y = xy(row.x, row.y)
        selected = row.Role == 'Analysed address'
        colour = '#1D4ED8' if selected else '#93C5FD'
        radius = 13 if selected else 8
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{colour}" stroke="white"><title>{escape(row.Address)}</title></circle>')
        parts.append(f'<text x="{x}" y="{y+25}" text-anchor="middle" font-size="10" fill="#173653">{escape(row.Label)}</text>')
    parts.append('</svg>')
    parts.append(f'<p>{len(nodes)-1} of {total} direct neighbours shown; {len(links)} unique undirected display links. '
                 'Layout is schematic. Repeated or reciprocal edges are collapsed and self-loops omitted. '
                 'Blue shades indicate node roles, not fraud classes. Links show benchmark adjacency, '
                 'not verified transaction direction, value or time.</p>')
    parts.append(report_table(nodes.drop(columns=['Node', 'x', 'y', 'Label'])))
    return ''.join(parts)


def build_analysis_report(predictor, result, node=None, graph_limit=16):
    from datetime import datetime, timezone
    from html import escape
    distribution, coverage = demo_scope_tables(predictor)
    title = 'Address analysis report' if node is not None else 'Batch analysis report'
    created = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    models = ', '.join(dict.fromkeys(result['Model'].dropna().astype(str)))
    parts = [f'<h1>X-MoE | {title}</h1><p>Report generated: {created}</p>',
             f'<p><strong>Models:</strong> {escape(models)}</p>',
             '<h2>Scope and interpretation</h2>',
             f'<p>Full dataset: {len(predictor.labels):,} addresses. Demo test set: {len(predictor.test_indices):,} addresses. '
             'Frozen seed 100, clean protocol, full released graph context. No training or live blockchain retrieval is performed by this report.</p>',
             '<p>Scores are model probabilities, not accuracy or calibrated certainty. Values are rounded. '
             'Normal does not verify safety, and a fraud-class prediction alone does not confirm an incident. '
             'Benchmark graph topology is label-correlated; these results do not establish performance on new addresses. '
             'Dataset labels are not supplied to the forward pass.</p>',
             report_table(distribution), report_table(coverage)]
    if node is not None:
        parts.extend(['<h2>Analysed address</h2>', f'<p class="address">{escape(str(predictor.addresses[node]))}</p>',
                      '<h2>Understanding this result</h2>'])
        for paragraph in prediction_explanation(result):
            parts.append('<p>' + escape(paragraph).replace('**', '') + '</p>')
        parts.extend(['<h2>Model predictions</h2>', report_table(result, True),
                      '<h2>Benchmark graph</h2>', report_graph_svg(predictor, node, graph_limit),
                      '<h2>Historical address features</h2>',
                      report_table(pd.DataFrame({'Feature': predictor.feature_names,
                                                 'Raw value': predictor.raw_features[node]})),
                      '<p>Graph neighbours and features are benchmark context; this report does not establish causal feature attributions.</p>'])
    else:
        counts, comparison, disagreements = batch_summary_tables(result)
        skipped = int(result.loc[result.Status.ne('Predicted'), 'Address'].nunique())
        parts.extend(['<h2>Batch summary</h2>',
                      f'<p>{len(comparison)} addresses analysed; {skipped} skipped. Counts below are model predictions, not true-label counts or accuracy.</p>',
                      report_table(counts), '<h2>Model disagreement</h2>'])
        if len(counts) <= 1:
            parts.append('<p>Not assessed: only one model was selected.</p>')
        elif disagreements.empty:
            parts.append('<p>All selected models agree on each analysed address. Agreement does not establish correctness.</p>')
        else:
            parts.append(f'<p>{len(disagreements)} addresses have different model predictions.</p>')
            parts.append(report_table(disagreements))
        parts.extend(['<h2>All address decisions</h2>', report_table(comparison),
                      '<h2>Detailed results and skipped inputs</h2>', report_table(result, True)])
    parts.append('<p>Research prototype. Source dataset: salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network.</p>')
    style = 'body{font:15px Arial,sans-serif;color:#173653;max-width:1100px;margin:32px auto;padding:20px;line-height:1.6}h1,h2{color:#153F75}table{border-collapse:collapse;width:100%;font-size:12px;margin:16px 0}th,td{border:1px solid #DAE5F2;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#EEF5FF}svg{width:100%;max-height:500px}.address{overflow-wrap:anywhere;background:#EEF5FF;padding:12px}@media print{body{margin:0;padding:0}table,svg{break-inside:avoid}}'
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + escape(title) + '</title><style>' + style + '</style></head><body>' + ''.join(parts) + '</body></html>'


def render_prediction_page():
    st.html('''<style>
    .prediction-hero {background:linear-gradient(115deg,#0B1F3A,#164E8B);padding:1.7rem 2rem;
      border-radius:18px;margin-bottom:1.3rem;color:white;}
    .prediction-hero h1 {color:white!important;margin:0;font-size:2.2rem;}
    .prediction-hero p {color:#DBEAFE;margin:.5rem 0 0;}
    </style><div class="prediction-hero"><h1>Try a Prediction</h1>
    <p>Ethereum fraud detection · Compare GCN, GAT, GraphSAGE and X-MoE</p></div>''')
    st.info('This demo analyses held-out addresses from the BLTE benchmark. '
            'It does not retrieve current blockchain activity or assess arbitrary new addresses.')
    with st.spinner('Loading trained models and benchmark inputs...'):
        try:
            predictor = get_predictor()
        except Exception as exc:
            st.error(f'Unable to load inference package: {exc}')
            st.stop()
    render_demo_scope(predictor)
    names = st.multiselect('Models to compare', list(MODELS), default=list(MODELS), key='prediction_models')
    single, batch = st.tabs(['Single address', 'Batch comparison'])
    with single:
        # UNIFIED_ADDRESS_ANALYSIS_V1
        supported = [str(predictor.addresses[i]) for i in predictor.test_indices]
        st.caption(f'{len(supported):,} held-out benchmark addresses are supported. '
                   'This demo does not retrieve live blockchain activity.')

        def clear_analysis():
            st.session_state.pop('prediction_result', None)

        mode = st.radio('Choose an address source',
                        ['Search or paste an address', 'Random benchmark example'],
                        horizontal=True, key='prediction_input_mode',
                        on_change=clear_analysis)
        address = ''
        candidates = []
        if mode == 'Search or paste an address':
            def choose_supported_address():
                chosen = st.session_state.get('prediction_search')
                if chosen:
                    st.session_state['prediction_address'] = chosen
                    st.session_state.pop('prediction_result', None)
            st.selectbox('Search available test addresses', supported, index=None,
                         placeholder='Type part of an address to search...',
                         key='prediction_search', on_change=choose_supported_address)
            address = st.text_input('Ethereum address', placeholder='0x...',
                                    key='prediction_address', on_change=clear_analysis)
        else:
            category = st.radio('Example category (dataset label)', CLASSES,
                                horizontal=True, key='prediction_random_category',
                                on_change=clear_analysis)
            connected_only = st.checkbox('Only examples with graph connections', value=True,
                                         key='prediction_connected_only',
                                         on_change=clear_analysis)
            edges = predictor.edges.numpy()
            linked_nodes = set(edges[:, edges[0] != edges[1]].flatten().tolist())
            candidates = [int(i) for i in predictor.test_indices
                          if CLASSES[int(predictor.labels[i])] == category
                          and (not connected_only or int(i) in linked_nodes)]
            st.caption(f'{len(candidates):,} matching examples. Each click selects and analyses '
                       'an example. Categories are dataset labels; predictions may differ.')
            if not candidates:
                st.warning('No matching examples. Change the category or graph filter.')

        st.session_state.pop('graph_example_pending', None)
        if st.button('Analyse Address', type='primary', key='prediction_run',
                     disabled=not names or (mode == 'Random benchmark example' and not candidates)):
            st.session_state.pop('prediction_result', None)
            if mode == 'Random benchmark example':
                import secrets
                previous = st.session_state.get('prediction_last_random_node')
                options = [i for i in candidates if i != previous]
                node = secrets.choice(options or candidates)
                st.session_state['prediction_last_random_node'] = node
                address = str(predictor.addresses[node])
            if not re.fullmatch(r'0x[0-9a-fA-F]{40}', address.strip()):
                st.warning('Enter an Ethereum address: 0x followed by 40 hexadecimal characters.')
            else:
                node = predictor.resolve(address)
                if node is None:
                    st.warning('This address is not in the seed-100 test set. '
                               'Search a supported address or select Random benchmark example. '
                               'No prediction was made.')
                else:
                    with st.spinner('Analysing address and preparing its graph...'):
                        table = predictor.compare(node, names)
                    st.session_state['prediction_result'] = (node, table)
        if not names:
            st.caption('Select at least one model above to enable analysis.')
        if 'prediction_result' in st.session_state:
            node, table = st.session_state['prediction_result']
            st.subheader('Prediction comparison')
            render_prediction_summary(table)
            st.code(str(predictor.addresses[node]), language=None)
            st.caption('Results below belong to the displayed analysed address. '
                       'Click Analyse address again after changing inputs.')
            render_prediction_explanation(table)
            st.dataframe(table.style.format({c: '{:.2%}' for c in CLASSES}),
                         hide_index=True, width='stretch')
            st.altair_chart(probability_chart(table), width='stretch')
            if st.checkbox('Reveal dataset label and prediction correctness', value=True):
                actual = CLASSES[int(predictor.labels[node])]
                st.write(f'**Dataset label:** {actual}')
                for _, row in table.iterrows():
                    st.write(f"**{row['Model']}:** " +
                             ('matches dataset label' if row['Prediction'] == actual else 'differs from dataset label'))
            render_graph_explorer(predictor, node)
            with st.expander('Address features and graph context'):
                st.write(f'Incident edge count in released graph: **{int(predictor.degree[node])}**')
                st.caption('Connectivity uses both endpoints of the released edge set, '
                           'as in the original model. These are historical benchmark features.')
                st.dataframe(pd.DataFrame({'Feature': predictor.feature_names,
                                           'Raw value': predictor.raw_features[node]}), hide_index=True,
                             width='stretch')
            if 'X-MoE' in table.Model.to_list():
                with st.expander('How X-MoE combined its experts', expanded=True):
                    explanation = predictor.explanation
                    expert_rows = []
                    for i, label in enumerate(CLASSES):
                        key = f'{label.lower()}_graphsage_expert_weight'
                        expert_rows.append({'Class': label,
                            'Multitask probability': float(explanation['conflict_aware_probabilities'][node,i]),
                            'GraphSAGE probability': float(explanation['graphsage_probabilities'][node,i]),
                            'GraphSAGE routing weight': float(explanation[key][node]),
                            'Multitask routing weight': 1.0-float(explanation[key][node])})
                    st.dataframe(pd.DataFrame(expert_rows).style.format({
                        k: '{:.2%}' for k in expert_rows[0] if k!='Class'}), hide_index=True,
                        width='stretch')
                    st.caption('Routing weights combine logits separately for each class. '
                               'They are routing diagnostics, not causal feature attributions; '
                               'zero graph weight does not establish that an address is safe.')
            st.download_button('Download address report (HTML)',
                build_analysis_report(predictor, table, node, st.session_state.get(f'graph_limit_{node}', 16)),
                file_name='address_analysis_report.html', mime='text/html', key='address_report_html')
            st.caption('Open the HTML report in a browser. Use Print > Save as PDF for a PDF copy.')
            st.download_button('Download comparison CSV', table.to_csv(index=False),
                               file_name='address_model_comparison.csv', mime='text/csv')
    with batch:
        # BATCH_BROWSER_V1
        st.subheader('Compare a batch of addresses')
        st.caption('Paste addresses, upload a CSV, or start with a sample batch. '
                   'Maximum 50 unique addresses; only seed-100 test addresses can be analysed.')
        sample_nodes = [int(i) for label_id in range(len(CLASSES))
                        for i in predictor.test_indices[
                            predictor.labels[predictor.test_indices] == label_id][:3]]
        sample_addresses = [str(predictor.addresses[i]) for i in sample_nodes]

        def load_sample_batch():
            st.session_state['prediction_batch_text'] = '\n'.join(sample_addresses)
            st.session_state.pop('prediction_batch_result', None)

        def clear_batch_text():
            st.session_state['prediction_batch_text'] = ''
            st.session_state.pop('prediction_batch_result', None)

        left, right = st.columns(2)
        left.button('Load 9 examples (3 per class)', on_click=load_sample_batch,
                    key='prediction_batch_sample')
        right.button('Clear pasted addresses', on_click=clear_batch_text,
                     key='prediction_batch_clear')
        st.download_button('Download sample CSV',
                           pd.DataFrame({'Address': sample_addresses}).to_csv(index=False),
                           file_name='benchmark_batch_sample.csv', mime='text/csv')
        entered = st.text_area('One Ethereum address per line', height=160,
                              placeholder='Paste benchmark addresses here, one per line...',
                              key='prediction_batch_text')
        uploaded = st.file_uploader('Or upload a CSV with an Address column', type='csv',
                                    key='prediction_batch_upload')
        st.caption('Pasted and uploaded addresses are combined. Duplicates are removed. '
                   'Remove an uploaded file using its X button if you want to replace it.')
        values = [v.strip() for v in entered.splitlines() if v.strip()]
        input_error = None
        if uploaded is not None:
            try:
                if uploaded.size > 2 * 1024 * 1024:
                    raise ValueError('Please upload a CSV smaller than 2 MB.')
                uploaded.seek(0)
                csv = pd.read_csv(uploaded, dtype=str, nrows=51)
                column = next((c for c in csv.columns if c.strip().lower() == 'address'), None)
                if column is None:
                    raise ValueError('CSV must contain an Address column. Use the sample CSV above.')
                if len(csv) > 50:
                    raise ValueError('CSV must contain no more than 50 address rows.')
                values.extend(csv[column].dropna().str.strip().tolist())
            except Exception as exc:
                input_error = str(exc)
        values = [v for v in values if v]
        total = len(values)
        values = list(dict.fromkeys(v.lower() for v in values))
        supported = [v for v in values if predictor.resolve(v) is not None]
        invalid = [v for v in values if not re.fullmatch(r'0x[0-9a-f]{40}', v)]
        unknown = [v for v in values if v not in invalid and predictor.resolve(v) is None]
        a, b, c = st.columns(3)
        a.metric('Unique addresses', len(values))
        b.metric('Supported addresses', len(supported))
        c.metric('Unsupported / invalid', len(unknown) + len(invalid))
        if total > len(values):
            st.caption(f'{total - len(values)} duplicate entries removed.')
        if input_error:
            st.error(input_error)
        if invalid or unknown:
            with st.expander('Review addresses that cannot be analysed', expanded=True):
                st.dataframe(pd.DataFrame(
                    [{'Address': v, 'Status': 'Invalid Ethereum address'} for v in invalid] +
                    [{'Address': v, 'Status': 'Not in seed-100 test set'} for v in unknown]
                ), hide_index=True, width='stretch')
                st.caption('These addresses will receive a status only; no prediction will be made.')
        if len(values) > 50:
            st.warning('Reduce the batch to at most 50 unique addresses.')
        signature = (tuple(values), tuple(names))
        ready = bool(names and supported and len(values) <= 50 and not input_error)
        if st.button('Compare batch', type='primary', disabled=not ready,
                     key='prediction_batch_run'):
            output = []
            with st.spinner('Comparing selected trained models...'):
                for value in values:
                    node = predictor.resolve(value)
                    if node is None:
                        output.append({'Address': value, 'Status':
                            'Invalid address' if value in invalid else 'Not in test set'})
                    else:
                        for row in predictor.compare(node, names).to_dict('records'):
                            output.append({'Address': str(predictor.addresses[node]),
                                           'Status': 'Predicted', **row})
            st.session_state['prediction_batch_result'] = (signature, pd.DataFrame(output))
        saved = st.session_state.get('prediction_batch_result')
        if saved is not None:
            saved_signature, result = saved
            if saved_signature != signature or input_error:
                st.info('Inputs changed. Click Compare batch to update the results.')
            else:
                render_batch_summary(result)
                st.subheader('Batch prediction results')
                st.caption(f'{len(supported)} addresses analysed with {len(names)} selected models. '
                           'Scores are model probabilities, not accuracy or calibrated certainty.')
                st.dataframe(result.style.format(
                    {label: '{:.2%}' for label in CLASSES}, na_rep='?'),
                    hide_index=True, width='stretch')
                st.download_button('Download batch report (HTML)',
                    build_analysis_report(predictor, result),
                    file_name='batch_analysis_report.html', mime='text/html', key='batch_report_html')
                st.caption('The HTML report includes scope, class counts, model disagreements and detailed results.')
                st.download_button('Download batch results CSV', result.to_csv(index=False),
                                   file_name='batch_predictions.csv', mime='text/csv')
                st.caption('Exported probabilities use values from 0 to 1; '
                           'for example, 0.95 corresponds to 95%.')
    with st.expander('Benchmark scope and validation'):
        st.write('Models use the released BLTE graph, whose topology is label-correlated. '
                 'Held-out test labels are never supplied to the model forward pass, but inference '
                 'uses full-graph transductive context. These results do not establish performance '
                 'on future, newly collected or cross-chain addresses.')
        report_path = HERE / 'validation_reference.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            st.dataframe(pd.DataFrame(report['models']), hide_index=True, width='stretch')
            st.caption('Reference parity was verified on the packaged inputs; local version differences '
                       'can be checked by running python engine.py.')
        st.caption('Dataset source: salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network. '
                   'Dataset ownership and upstream licensing remain with the original authors.')
