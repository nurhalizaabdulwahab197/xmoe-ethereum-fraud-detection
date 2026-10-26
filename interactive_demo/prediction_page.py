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
    a, b, c = st.columns(3)
    a.metric('Available test addresses', f'{len(predictor.test_indices):,}')
    b.metric('Trained models', '4')
    c.metric('Output classes', '3')
    st.caption('Frozen seed 100 · clean protocol · full released graph context. '
               'Percentage scores are model probabilities, not accuracy or calibrated certainty.')
    names = st.multiselect('Models to compare', list(MODELS), default=list(MODELS), key='prediction_models')
    single, batch = st.tabs(['Single address', 'Batch comparison'])
    with single:
        # ADDRESS_BROWSER_V1
        supported = [str(predictor.addresses[i]) for i in predictor.test_indices]
        category_nodes = {
            label: predictor.test_indices[
                predictor.labels[predictor.test_indices] == label_id
            ].tolist()
            for label_id, label in enumerate(CLASSES)
        }
        st.write(f'**Search {len(supported):,} supported test addresses**')
        st.caption('Type part of an address in the search box, or paste a full address below. '
                   'Only held-out seed-100 benchmark addresses are supported; '
                   'this demo does not fetch live blockchain data.')
        st.caption('Dataset labels: ' + ' | '.join(
            f'{label}: {len(nodes):,}' for label, nodes in category_nodes.items()
        ))

        def choose_supported_address():
            chosen = st.session_state.get('prediction_search')
            if chosen:
                st.session_state['prediction_address'] = chosen
                st.session_state.pop('prediction_result', None)

        def choose_random_address():
            import secrets
            label = st.session_state['prediction_random_category']
            current = st.session_state.get('prediction_address', '').strip().lower()
            candidates = [str(predictor.addresses[i]) for i in category_nodes[label]
                          if str(predictor.addresses[i]).lower() != current]
            if candidates:
                st.session_state['prediction_address'] = secrets.choice(candidates)
                st.session_state['prediction_search'] = None
                st.session_state.pop('prediction_result', None)

        st.selectbox('Search available test addresses', supported, index=None,
                     placeholder='Type or paste an address to search...',
                     key='prediction_search', on_change=choose_supported_address)
        st.radio('Random example category (dataset label)', CLASSES,
                 horizontal=True, key='prediction_random_category')
        st.button('Try a random example', key='prediction_random',
                  on_click=choose_random_address)
        address = st.text_input('Ethereum address', placeholder='0x...', key='prediction_address')
        st.caption('Example categories are dataset labels; the model may predict a different class. '
                   'Click Analyse address to run the selected models.')
        if st.button('Analyse address', type='primary', key='prediction_run'):
            st.session_state.pop('prediction_result', None)
            if not names:
                st.warning('Select at least one model.')
            elif not re.fullmatch(r'0x[0-9a-fA-F]{40}', address.strip()):
                st.warning('Enter an Ethereum address: 0x followed by 40 hexadecimal characters.')
            else:
                node = predictor.resolve(address)
                if node is None:
                    st.warning('This address is not in the seed-100 test set. '
                               'Search the available addresses or try a random example. No prediction was made.')
                    st.session_state.pop('prediction_result', None)
                else:
                    with st.spinner('Running selected trained models...'):
                        table = predictor.compare(node, names)
                    st.session_state['prediction_result'] = (node, table)
        if 'prediction_result' in st.session_state:
            node, table = st.session_state['prediction_result']
            st.subheader('Prediction comparison')
            st.code(str(predictor.addresses[node]), language=None)
            st.caption('Results below belong to the displayed analysed address. '
                       'Click Analyse address again after changing inputs.')
            st.dataframe(table.style.format({c: '{:.2%}' for c in CLASSES}),
                         hide_index=True, width='stretch')
            st.altair_chart(probability_chart(table), width='stretch')
            if st.checkbox('Reveal dataset label and prediction correctness', value=True):
                actual = CLASSES[int(predictor.labels[node])]
                st.write(f'**Dataset label:** {actual}')
                for _, row in table.iterrows():
                    st.write(f"**{row['Model']}:** " +
                             ('matches dataset label' if row['Prediction'] == actual else 'differs from dataset label'))
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
                st.subheader('Batch prediction results')
                st.caption(f'{len(supported)} addresses analysed with {len(names)} selected models. '
                           'Scores are model probabilities, not accuracy or calibrated certainty.')
                st.dataframe(result.style.format(
                    {label: '{:.2%}' for label in CLASSES}, na_rep='?'),
                    hide_index=True, width='stretch')
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
