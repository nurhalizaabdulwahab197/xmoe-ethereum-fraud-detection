from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="X-MoE | Ethereum Fraud Detection",
    page_icon="data:image/png;base64,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",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "demo" / "data" / "verified_model_metrics.csv"

MODEL_ORDER = [
    "GCN",
    "GAT",
    "GraphSAGE",
    "X-MoE",
]

MODEL_COLOURS = {
    "GCN": "#93C5FD",
    "GAT": "#60A5FA",
    "GraphSAGE": "#2563EB",
    "X-MoE": "#0B1F3A",
}


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: #0B1F3A;
    }

    .hero {
        padding: 2.8rem 3rem;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #07172B 0%,
                #0B2F58 55%,
                #164E8B 100%
            );
        box-shadow:
            0 18px 45px rgba(11,31,58,0.15);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 3.3rem;
        margin: 0;
    }

    .hero p {
        color: #DBEAFE;
        font-size: 1.15rem;
        line-height: 1.7;
        max-width: 900px;
    }

    .version {
        color: #93C5FD;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.10rem;
    }

    .tag {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin-right: 0.35rem;
        margin-top: 0.6rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.11);
        color: #EFF6FF;
        font-size: 0.82rem;
    }

    .card {
        padding: 1.35rem;
        min-height: 180px;
        border-radius: 18px;
        border: 1px solid #D6E5F8;
        background: linear-gradient(
            180deg,
            #FFFFFF 0%,
            #F7FBFF 100%
        );
        box-shadow:
            0 5px 20px rgba(18,59,109,0.05);
    }

    .card h4 {
        color: #123B6D;
        margin-top: 0;
    }

    .flow {
        padding: 1rem;
        text-align: center;
        border-radius: 14px;
        border: 1px solid #BFD7F4;
        background: #F7FBFF;
        color: #123B6D;
        font-weight: 650;
    }

    .gate {
        padding: 1.1rem;
        text-align: center;
        border-radius: 14px;
        border: 2px solid #2563EB;
        background: #DBEAFE;
        color: #0B1F3A;
        font-weight: 750;
    }

    .blue-panel {
        padding: 1.3rem 1.5rem;
        border-radius: 17px;
        background: #EFF6FF;
        border-left: 5px solid #2563EB;
    }

    .profile {
        padding: 2rem;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                #07172B,
                #164E8B
            );
        color: white;
    }

    .profile h2,
    .profile h3 {
        color: white;
    }

    .profile p {
        color: #DBEAFE;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: #F8FBFF;
        border: 1px solid #D8E7F8;
        padding: 0.9rem 1rem;
        border-radius: 15px;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07172B 0%,
                #0B2F58 55%,
                #164E8B 100%
            );
        border-right: none;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    section[data-testid="stSidebar"] * {
        color: #EFF6FF;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #BFDBFE !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.18);
    }

    /* Radio navigation container */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.38rem;
    }

    /* Each navigation item */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100%;
        padding: 0.55rem 0.75rem;
        border-radius: 13px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        transition:
            background 0.18s ease,
            border 0.18s ease,
            transform 0.18s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.13);
        border-color: rgba(255,255,255,0.22);
        transform: translateX(2px);
    }

    /* Remove default large radio-circle appearance */
    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label
    > div:first-child {
        transform: scale(0.82);
    }

    /* Selected navigation item */
    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) {
        background: #FFFFFF;
        border-color: #FFFFFF;
        box-shadow:
            0 6px 18px rgba(0,0,0,0.14);
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) * {
        color: #0B2F58 !important;
        font-weight: 700;
    }

    /* Sidebar normal markdown text */
    section[data-testid="stSidebar"] p {
        color: #DBEAFE;
        line-height: 1.65;
    }

    /* Sidebar section labels such as Task / Classes / Evaluation */
    section[data-testid="stSidebar"] strong {
        color: #FFFFFF;
        font-weight: 700;
    }

    /* Collapse control */
    section[data-testid="stSidebar"]
    button[kind="header"] {
        color: #DBEAFE !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_results():
    df = pd.read_csv(DATA)

    df["mean"] = pd.to_numeric(
        df["mean"],
        errors="coerce",
    )

    df["std"] = pd.to_numeric(
        df["std"],
        errors="coerce",
    )

    return df


results = load_results()


def get_metric(model, scope, metric):
    row = results[
        (results["model"] == model)
        & (results["scope"] == scope)
        & (results["metric"] == metric)
    ]

    if row.empty:
        return None, None

    return (
        float(row.iloc[0]["mean"]),
        float(row.iloc[0]["std"]),
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## X-MoE")
    st.caption("Research Demo")

    page = st.radio(
        "Explore",
        [
            "Overview",
            "Architecture Explorer",
            "Results Explorer",
            "Class-wise Performance",
            "About Nur Haliza",
        ],
    )

    st.divider()

    st.markdown(
        """
        **Task**

        Multi-class Ethereum fraud detection

        **Classes**

        • Normal
        • Phishing
        • Scamming
        """
    )

    st.divider()

    st.markdown(
        """
        **Evaluation**

        Clean protocol
        Seeds 100–109
        300 epochs
        Deterministic CPU
        """
    )

    st.divider()

    st.caption(
        "Research prototype — not a production fraud-screening service."
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """<div class="hero">
<div class="version">RESEARCH DEMO</div>
<h1>X-MoE</h1>
<p>
A connectivity-aware Mixture-of-Experts framework for
multi-class Ethereum fraud detection under severe class
imbalance and sparse graph connectivity.
</p>
<span class="tag">Ethereum</span>
<span class="tag">Graph Neural Networks</span>
<span class="tag">Mixture of Experts</span>
<span class="tag">Phishing</span>
<span class="tag">Scamming</span>
<div style="
margin-top:1.4rem;
padding-top:1rem;
border-top:1px solid rgba(255,255,255,0.18);
color:#BFDBFE;
font-size:0.95rem;
">
Developed during Visiting Scholar research at the
<b>Digital Trust Centre (DTC)</b>,
Nanyang Technological University (NTU), Singapore.
</div>
</div>""",
        unsafe_allow_html=True,
    )

    acc, acc_std = get_metric(
        "X-MoE",
        "Overall",
        "Accuracy",
    )

    f1, f1_std = get_metric(
        "X-MoE",
        "Overall",
        "Macro F1",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Addresses",
        "73,033",
    )

    c2.metric(
        "Classes",
        "3",
        help="Normal, Phishing and Scamming",
    )

    c3.metric(
        "X-MoE Accuracy",
        f"{acc:.4f}",
        f"± {acc_std:.4f}",
    )

    c4.metric(
        "X-MoE Macro-F1",
        f"{f1:.4f}",
        f"± {f1_std:.4f}",
    )

    st.markdown("## Why this research matters")

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="card">
            <h4>Extreme class imbalance</h4>
            Normal addresses dominate the dataset. High overall
            accuracy can therefore hide poor fraud-class detection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="card">
            <h4>Sparse connectivity</h4>
            Most addresses in the released graph are isolated.
            Graph evidence is therefore not equally available
            for every node.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            """
            <div class="card">
            <h4>Different evidence requirements</h4>
            Normal, Phishing and Scamming need not rely on the
            same balance of behavioural and relational evidence.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Core idea")

    st.write(
        "X-MoE combines a conflict-aware expert with an external "
        "GraphSAGE expert. Their predictions, disagreement and node "
        "connectivity are supplied to a class-specific routing network. "
        "The router determines how much graph evidence should influence "
        "Normal, Phishing and Scamming predictions."
    )


# ============================================================
# ARCHITECTURE
# ============================================================

elif page == "Architecture Explorer":

    st.title("Architecture Comparison")

    st.markdown(
        """
        All four models solve the same **three-class Ethereum address
        classification problem**:

        **Normal · Phishing · Scamming**

        However, they differ substantially in how node features and
        graph information are processed. GCN, GAT and GraphSAGE provide
        conventional graph-learning baselines, while X-MoE introduces
        complementary experts and adaptive class-specific routing.
        """
    )

    # --------------------------------------------------------
    # Full architecture comparison figure
    # --------------------------------------------------------

    architecture_figure = (
        ROOT
        / "docs"
        / "figures"
        / "architecture_comparison.png"
    )

    if architecture_figure.exists():
        st.image(
            str(architecture_figure),
            caption=(
                "Architecture comparison of GCN, GAT, GraphSAGE "
                "and the proposed X-MoE model."
            ),
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------------
    # Four-model architecture comparison
    # --------------------------------------------------------

    st.header("Explore the Four Model Architectures")

    model_tabs = st.tabs(
        [
            "A · GCN",
            "B · GAT",
            "C · GraphSAGE",
            "D · X-MoE",
        ]
    )

    # ========================================================
    # GCN
    # ========================================================

    with model_tabs[0]:

        st.subheader("Graph Convolutional Network (GCN)")

        st.write(
            "GCN propagates and aggregates information from neighbouring "
            "Ethereum addresses using shared graph convolution."
        )

        c1, c2 = st.columns(
            [0.8, 1.2],
            gap="large",
        )

        with c1:

            st.markdown("#### Implemented pipeline")

            st.code(
                """Node features + graph edges
        │
        ▼
   GCNConv(64)
        │
       ReLU
        │
   Dropout(0.5)
        │
        ▼
   GCNConv(64)
        │
       ReLU
        │
   Dropout(0.5)
        │
 Linear classifier
        │
        ▼
Normal | Phishing | Scamming""",
                language=None,
            )

        with c2:

            st.markdown("#### How it works")

            st.write(
                "Each GCN layer mixes the representation of an address "
                "with information propagated through its graph neighbourhood."
            )

            st.markdown("#### Strength")

            st.write(
                "GCN provides a simple and computationally efficient "
                "mechanism for incorporating transaction-graph structure."
            )

            st.markdown("#### Limitation for this task")

            st.write(
                "Neighbourhood information is aggregated using the same "
                "convolutional mechanism for all nodes. Under severe class "
                "imbalance, minority fraud signals may be difficult to "
                "preserve when fraudulent addresses are sparse relative "
                "to the Normal class."
            )

    # ========================================================
    # GAT
    # ========================================================

    with model_tabs[1]:

        st.subheader("Graph Attention Network (GAT)")

        st.write(
            "GAT extends graph aggregation by learning attention weights "
            "over neighbouring Ethereum addresses."
        )

        c1, c2 = st.columns(
            [0.8, 1.2],
            gap="large",
        )

        with c1:

            st.markdown("#### Implemented pipeline")

            st.code(
                """Node features + graph edges
        │
        ▼
GATConv(32, heads=4)
        │
       ELU
        │
   Dropout(0.5)
        │
        ▼
GATConv(32, heads=1)
        │
       ELU
        │
 Linear classifier
        │
        ▼
Normal | Phishing | Scamming""",
                language=None,
            )

        with c2:

            st.markdown("#### How it works")

            st.write(
                "Unlike GCN, GAT does not treat all neighbours equally. "
                "Attention weights allow the model to assign different "
                "importance to different neighbouring addresses."
            )

            st.markdown("#### Strength")

            st.write(
                "Attention introduces additional flexibility when some "
                "neighbour relationships are more informative than others."
            )

            st.markdown("#### Limitation for this task")

            st.write(
                "Attention alone does not explicitly address severe class "
                "imbalance, expert disagreement, or the fact that many "
                "Ethereum addresses may contain little or no useful "
                "neighbourhood structure."
            )

    # ========================================================
    # GRAPHSAGE
    # ========================================================

    with model_tabs[2]:

        st.subheader("GraphSAGE")

        st.write(
            "GraphSAGE learns node representations by aggregating "
            "information from local graph neighbourhoods."
        )

        c1, c2 = st.columns(
            [0.8, 1.2],
            gap="large",
        )

        with c1:

            st.markdown("#### Implemented pipeline")

            st.code(
                """Node features + graph edges
        │
        ▼
   SAGEConv(64)
        │
       ReLU
        │
   Dropout(0.5)
        │
        ▼
   SAGEConv(64)
        │
       ReLU
        │
   Dropout(0.5)
        │
 Linear classifier
        │
        ▼
Normal | Phishing | Scamming""",
                language=None,
            )

        with c2:

            st.markdown("#### How it works")

            st.write(
                "GraphSAGE aggregates neighbourhood information to build "
                "a relational representation for each Ethereum address."
            )

            st.markdown("#### Strength")

            st.write(
                "Among the conventional GNN baselines evaluated in this "
                "project, GraphSAGE achieved the strongest verified "
                "CPU Macro-F1 performance."
            )

            st.markdown("#### Limitation for this task")

            st.write(
                "GraphSAGE still relies on a single graph-learning "
                "mechanism. Its usefulness can therefore be reduced when "
                "an address is isolated or when discriminative evidence "
                "is primarily contained in node-level features rather "
                "than neighbourhood structure."
            )

    # ========================================================
    # X-MOE
    # ========================================================

    with model_tabs[3]:

        st.subheader("X-MoE · Proposed Architecture")

        st.write(
            "X-MoE differs fundamentally from the three baseline models. "
            "Instead of relying on one graph-learning backbone, it combines "
            "two complementary experts through an adaptive routing mechanism."
        )

        st.markdown("#### High-level pipeline")

        st.code(
            """                         Node features
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
        Conflict-aware              GraphSAGE
        multitask expert              expert
                  │                       │
                  ▼                       ▼
           Multitask logits         GraphSAGE logits
                  │                       │
                  └───────────┬───────────┘
                              │
                     Expert disagreement
                              │
                  Node connectivity / degree
                              │
                              ▼
                       Gating network
                              │
                              ▼
                 Class-specific mixture
                         weights
             w_Normal | w_Phishing | w_Scamming
                              │
                              ▼
                  Adaptive weighted fusion
                              │
                              ▼
               Normal | Phishing | Scamming""",
            language=None,
        )

        st.info(
            "The central architectural idea is that no single source of "
            "evidence is equally reliable for every Ethereum address."
        )

    st.divider()

    # --------------------------------------------------------
    # X-MoE deep dive
    # --------------------------------------------------------

    st.header("X-MoE Deep Dive")

    st.write(
        "Select a component to examine how information moves through "
        "the proposed architecture."
    )

    component = st.selectbox(
        "X-MoE component",
        [
            "Conflict-Aware Multitask Expert",
            "GraphSAGE Expert",
            "Expert Disagreement",
            "Connectivity-Aware Gating",
            "Class-Specific Mixture Weights",
            "Adaptive Weighted Fusion",
        ],
    )

    if component == "Conflict-Aware Multitask Expert":

        st.markdown("### Conflict-Aware Multitask Expert")

        st.write(
            "The conflict-aware multitask expert captures feature-level and task-specific "
            "evidence while also maintaining an internal graph-aware pathway."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Input")
            st.write(
                "Node features, graph edges and connectivity information."
            )

        with c2:
            st.markdown("#### Output")
            st.write(
                "Normal, Phishing and Scamming logits, together with "
                "auxiliary task-specific learning signals."
            )

        with c3:
            st.markdown("#### Purpose")
            st.write(
                "Provides a strong alternative source of evidence when "
                "graph structure is weak, sparse or insufficient."
            )

        st.markdown(
            """
            The expert is trained not only through the multiclass objective
            but also through class-specific auxiliary objectives. This is
            intended to preserve minority-class discrimination under severe
            imbalance.
            """
        )

    elif component == "GraphSAGE Expert":

        st.markdown("### GraphSAGE Expert")

        st.write(
            "The GraphSAGE expert is a dedicated graph-structure expert based "
            "on GraphSAGE."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Input")
            st.write(
                "Node features and Ethereum graph edges."
            )

        with c2:
            st.markdown("#### Output")
            st.write(
                "Independent Normal, Phishing and Scamming logits."
            )

        with c3:
            st.markdown("#### Purpose")
            st.write(
                "Captures relational fraud evidence from neighbouring "
                "Ethereum addresses."
            )

        st.write(
            "This expert becomes particularly useful when graph "
            "connectivity contains meaningful structural information."
        )

    elif component == "Expert Disagreement":

        st.markdown("### Expert Disagreement")

        st.write(
            "X-MoE explicitly measures disagreement between the two experts:"
        )

        st.code(
            "| Multitask logits - GraphSAGE logits |",
            language=None,
        )

        st.write(
            "Agreement provides mutually supporting evidence. When the "
            "experts disagree, the routing network receives this conflict "
            "as an additional signal and can learn which expert should "
            "receive greater influence."
        )

    elif component == "Connectivity-Aware Gating":

        st.markdown("### Connectivity-Aware Gating")

        st.write(
            "The gating network receives four complementary signals:"
        )

        st.markdown(
            """
            1. Multitask logits
            2. GraphSAGE logits
            3. Absolute expert disagreement
            4. Node connectivity represented using log-degree
            """
        )

        st.write(
            "A key design feature is that graph contribution is suppressed "
            "for isolated nodes, where neighbourhood aggregation provides "
            "no meaningful relational evidence."
        )

        st.info(
            "This prevents X-MoE from assuming that every Ethereum address "
            "should rely equally on graph structure."
        )

    elif component == "Class-Specific Mixture Weights":

        st.markdown("### Class-Specific Mixture Weights")

        st.write(
            "X-MoE does not use one global mixture weight for the entire "
            "prediction."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Normal routing weight", "w_Normal")
        c2.metric("Phishing routing weight", "w_Phishing")
        c3.metric("Scamming routing weight", "w_Scamming")

        st.write(
            "Therefore, the contribution of the conflict-aware expert and "
            "GraphSAGE expert can differ for each class."
        )

        st.write(
            "This is important because Normal, Phishing and Scamming "
            "addresses need not exhibit identical behavioural or "
            "structural patterns."
        )

    else:

        st.markdown("### Adaptive Weighted Fusion")

        st.write(
            "The final prediction combines the two experts using the "
            "class-specific routing weights."
        )

        st.code(
            "Conflict-aware logits + w_class × "
            "(GraphSAGE logits - Conflict-aware logits)",
            language=None,
        )

        st.write(
            "The conflict-aware expert acts as the default prediction "
            "pathway, while graph evidence modifies the prediction "
            "according to the learned contribution for each class."
        )

    st.divider()

    # --------------------------------------------------------
    # Why X-MoE helps
    # --------------------------------------------------------

    st.header("Why the X-MoE Architecture Helps Fraud Detection")

    st.markdown(
        """
        The central motivation of X-MoE is that **no single source of
        evidence is equally reliable for every Ethereum address**.

        X-MoE combines four complementary signals:
        """
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.markdown(
            """
            <div class="card">
            <h4>1 · Feature Evidence</h4>
            Behavioural and node-level information that remains available
            even when graph connectivity is limited.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q2:
        st.markdown(
            """
            <div class="card">
            <h4>2 · Graph Evidence</h4>
            Relational information from neighbouring Ethereum addresses.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q3:
        st.markdown(
            """
            <div class="card">
            <h4>3 · Expert Disagreement</h4>
            Indicates when feature/task-aware and graph-based evidence
            support different predictions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q4:
        st.markdown(
            """
            <div class="card">
            <h4>4 · Connectivity</h4>
            Indicates whether meaningful neighbourhood evidence is
            available for the current node.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write(
        "The gating network uses these signals to determine how the two "
        "experts should be combined for each class. This produces an "
        "adaptive decision mechanism instead of applying the same "
        "graph-learning rule to every address."
    )

    st.divider()

    # --------------------------------------------------------
    # Interpretation by class
    # --------------------------------------------------------

    st.header("Architecture-Level Interpretation by Class")

    n, p, s = st.columns(3)

    with n:
        st.markdown(
            """
            <div class="card">
            <h4>Normal</h4>
            The model must avoid incorrectly converting the large majority
            of Normal addresses into fraud predictions. X-MoE can reduce
            unnecessary graph influence when structural evidence is weak
            while preserving strong node-level evidence.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p:
        st.markdown(
            """
            <div class="card">
            <h4>Phishing</h4>
            Discriminative evidence may arise from both behavioural features
            and relational patterns. The two-expert design allows these
            evidence sources to complement each other rather than forcing
            the decision through a single GNN pathway.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s:
        st.markdown(
            """
            <div class="card">
            <h4>Scamming</h4>
            Graph relationships can provide strong structural evidence when
            suspicious interaction patterns exist. Class-specific routing
            allows graph contribution to differ from the Normal and
            Phishing decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --------------------------------------------------------
    # Key takeaway + verified evidence
    # --------------------------------------------------------

    st.header("Key Takeaway")

    st.markdown(
        """
        **GCN, GAT and GraphSAGE each rely primarily on a single
        graph-learning mechanism.**

        **X-MoE instead combines a feature/task-aware expert and a
        graph-structure expert through a connectivity-aware,
        disagreement-aware and class-specific gating mechanism.**

        The purpose of this adaptive fusion is to provide a mechanism
        for more balanced discrimination across Normal, Phishing and
        Scamming addresses.
        """
    )

    evidence_rows = []

    for model in MODEL_ORDER:

        acc, acc_std = get_metric(
            model,
            "Overall",
            "Accuracy",
        )

        mf1, mf1_std = get_metric(
            model,
            "Overall",
            "Macro F1",
        )

        evidence_rows.append(
            {
                "Model": model,
                "Accuracy": acc,
                "Accuracy Std": acc_std,
                "Macro-F1": mf1,
                "Macro-F1 Std": mf1_std,
            }
        )

    evidence = pd.DataFrame(
        evidence_rows
    )

    st.dataframe(
        evidence.style.format(
            {
                "Accuracy": "{:.6f}",
                "Accuracy Std": "{:.6f}",
                "Macro-F1": "{:.6f}",
                "Macro-F1 Std": "{:.6f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    graphsage_f1, _ = get_metric(
        "GraphSAGE",
        "Overall",
        "Macro F1",
    )

    xmoe_f1, _ = get_metric(
        "X-MoE",
        "Overall",
        "Macro F1",
    )

    absolute_gain = (
        xmoe_f1
        - graphsage_f1
    )

    relative_gain = (
        absolute_gain
        / graphsage_f1
        * 100
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "X-MoE Macro-F1",
        f"{xmoe_f1:.6f}",
    )

    m2.metric(
        "Gain vs GraphSAGE",
        f"+{absolute_gain:.6f}",
        "16.59 percentage points",
    )

    m3.metric(
        "Relative improvement",
        f"{relative_gain:.2f}%",
    )

    st.info(
        "Because the dataset is highly class-imbalanced, Macro-F1 is more "
        "informative than Accuracy alone for assessing balanced performance "
        "across Normal, Phishing and Scamming. Detailed class-level results "
        "are available in the Results Explorer and Class-wise Performance pages."
    )



# ============================================================
# RESULTS EXPLORER
# ============================================================

elif page == "Results Explorer":

    st.title("Interactive Results Explorer")

    st.write(
        "Compare all four models using the verified 10-seed fresh-PC "
        "CPU experiments."
    )

    c1, c2 = st.columns(2)

    with c1:
        scope = st.selectbox(
            "Scope",
            [
                "Overall",
                "Normal",
                "Phishing",
                "Scamming",
            ],
        )

    available = (
        results[
            results["scope"] == scope
        ]["metric"]
        .drop_duplicates()
        .tolist()
    )

    defaults = {
        "Overall": "Macro F1",
        "Normal": "F1",
        "Phishing": "F1",
        "Scamming": "F1",
    }

    index = (
        available.index(defaults[scope])
        if defaults[scope] in available
        else 0
    )

    with c2:
        metric = st.selectbox(
            "Metric",
            available,
            index=index,
        )

    models = st.multiselect(
        "Models",
        MODEL_ORDER,
        default=MODEL_ORDER,
    )

    chart_data = results[
        (results["scope"] == scope)
        & (results["metric"] == metric)
        & (results["model"].isin(models))
    ].copy()


    show_variation = st.checkbox(
        "Show variation across seeds (mean +/- SD)",
        value=True,
    )

    chart_data["lower"] = chart_data["mean"] - chart_data["std"]
    chart_data["upper"] = chart_data["mean"] + chart_data["std"]

    axis_top = 1.0
    axis_bottom = 0.0
    if not chart_data.empty:
        axis_top = max(1.0, float(chart_data["upper"].max()) + 0.03)
        axis_bottom = min(0.0, float(chart_data["lower"].min()) - 0.03)

    colour = alt.Scale(
        domain=MODEL_ORDER,
        range=[MODEL_COLOURS[m] for m in MODEL_ORDER],
    )

    base = alt.Chart(chart_data).encode(
        x=alt.X(
            "model:N",
            sort=MODEL_ORDER,
            title=None,
            axis=alt.Axis(
                labelAngle=0,
                labelFontSize=13,
                labelPadding=12,
                ticks=False,
            ),
        ),
        y=alt.Y(
            "mean:Q",
            title=metric,
            scale=alt.Scale(
                domain=[axis_bottom, axis_top],
                nice=False,
            ),
            axis=alt.Axis(
                format=".1f",
                gridColor="#E5EDF5",
                titlePadding=14,
            ),
        ),
    )

    bars = base.mark_bar(
        size=76,
        cornerRadiusTopLeft=6,
        cornerRadiusTopRight=6,
    ).encode(
        color=alt.Color("model:N", scale=colour, legend=None),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("mean:Q", title="Mean", format=".4f"),
            alt.Tooltip("std:Q", title="SD", format=".4f"),
            alt.Tooltip("n_seeds:Q", title="Seeds", format=".0f"),
        ],
    )

    labels = base.mark_text(
        dy=16,
        baseline="top",
        fontSize=13,
        fontWeight="bold",
    ).encode(
        text=alt.Text("mean:Q", format=".3f"),
        color=alt.condition(
            alt.FieldOneOfPredicate(
                field="model",
                oneOf=["GraphSAGE", "X-MoE"],
            ),
            alt.value("white"),
            alt.value("#0B1F3A"),
        ),
    )

    chart = bars

    if show_variation:
        # White outline keeps the interval visible on dark bars.
        halo = base.mark_rule(
            color="white", strokeWidth=5
        ).encode(y="lower:Q", y2="upper:Q")

        interval = base.mark_rule(
            color="#173653", strokeWidth=2
        ).encode(y="lower:Q", y2="upper:Q")

        lower_halo = base.mark_tick(
            orient="horizontal", color="white",
            thickness=5, size=18,
        ).encode(y="lower:Q")

        upper_halo = base.mark_tick(
            orient="horizontal", color="white",
            thickness=5, size=18,
        ).encode(y="upper:Q")

        lower_cap = base.mark_tick(
            orient="horizontal", color="#173653",
            thickness=2, size=16,
        ).encode(y="lower:Q")

        upper_cap = base.mark_tick(
            orient="horizontal", color="#173653",
            thickness=2, size=16,
        ).encode(y="upper:Q")

        chart = (
            chart + halo + lower_halo + upper_halo
            + interval + lower_cap + upper_cap
        )

    chart = (chart + labels).properties(height=420)

    st.altair_chart(
        chart.configure_view(strokeWidth=0),
        use_container_width=True,
    )

    st.caption(
        "Bars show mean scores across seeds. "
        + (
            "Error bars show +/- one standard deviation, "
            "not confidence intervals."
            if show_variation else
            "Variation is hidden; SD remains available in the tooltip."
        )
    )

    table = chart_data[
        [
            "model",
            "mean",
            "std",
            "n_seeds",
        ]
    ].copy()

    table.columns = [
        "Model",
        "Mean",
        "Std",
        "Seeds",
    ]

    st.dataframe(
        table.style.format(
            {
                "Mean": "{:.6f}",
                "Std": "{:.6f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    if scope == "Overall":
        st.info(
            "Macro-F1 is the primary model-comparison metric because "
            "the dataset is severely class-imbalanced."
        )


# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

elif page == "Class-wise Performance":

    st.title("Class-wise Performance")

    fraud_class = st.radio(
        "Select classification class",
        [
            "Normal",
            "Phishing",
            "Scamming",
        ],
        horizontal=True,
    )

    available_metrics = (
        results[
            results["scope"] == fraud_class
        ]["metric"]
        .drop_duplicates()
        .tolist()
    )

    selected_metrics = st.multiselect(
        "Metrics",
        available_metrics,
        default=available_metrics,
    )

    class_data = results[
        (results["scope"] == fraud_class)
        & (
            results["metric"]
            .isin(selected_metrics)
        )
    ].copy()

    colour = alt.Scale(
        domain=MODEL_ORDER,
        range=[
            MODEL_COLOURS[m]
            for m in MODEL_ORDER
        ],
    )


    chart_base = alt.Chart(class_data).encode(
        x=alt.X(
            "metric:N",
            sort=selected_metrics,
            title=None,
            scale=alt.Scale(paddingInner=0.30, paddingOuter=0.15),
            axis=alt.Axis(
                labelAngle=0,
                labelFontSize=14,
                labelPadding=12,
                ticks=False,
            ),
        ),
        xOffset=alt.XOffset(
            "model:N",
            sort=MODEL_ORDER,
        ),
        y=alt.Y(
            "mean:Q",
            title="Mean score",
            scale=alt.Scale(domain=[0, 1.10], nice=False),
            axis=alt.Axis(
                values=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                format=".1f",
                gridColor="#E5EDF5",
                titlePadding=14,
                labelFontSize=12,
            ),
        ),
    )

    bars = chart_base.mark_bar(
        cornerRadiusTopLeft=5,
        cornerRadiusTopRight=5,
    ).encode(
        color=alt.Color(
            "model:N",
            scale=colour,
            sort=MODEL_ORDER,
            legend=alt.Legend(
                title=None,
                orient="bottom",
                direction="horizontal",
                columns=4,
                labelFontSize=12,
                symbolType="square",
                symbolSize=130,
                padding=14,
            ),
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("mean:Q", title="Mean", format=".4f"),
            alt.Tooltip("std:Q", title="SD", format=".4f"),
        ],
    )

    labels = chart_base.mark_text(
        dy=-9,
        fontSize=11,
        fontWeight="bold",
        color="#173653",
    ).encode(
        text=alt.Text("mean:Q", format=".3f"),
    )

    chart = (
        (bars + labels)
        .properties(height=410)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    table = (
        class_data
        .pivot(
            index="model",
            columns="metric",
            values="mean",
        )
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    st.dataframe(
        table.style.format(
            {
                column: "{:.6f}"
                for column in table.columns
                if column != "model"
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    if fraud_class == "Normal":
        st.info(
            "Normal is the dominant class. Strong Normal-class scores "
            "must be interpreted together with fraud-class performance."
        )

    elif fraud_class == "Phishing":
        st.info(
            "Phishing is an extremely rare class. Precision, recall, "
            "F1 and Average Precision are therefore particularly informative."
        )

    else:
        st.info(
            "Scamming performance shows how well each model balances "
            "detection sensitivity and false-positive control for this class."
        )


# ============================================================
# ABOUT
# ============================================================

else:

    st.title("About Nur Haliza")

    left, right = st.columns(
        [1.15, 1],
        gap="large",
    )

    with left:

        st.html(
            """
<div class="profile">
    <h2>Ts. Dr. Nur Haliza Binti Abdul Wahab</h2>

    <p>
        <strong>Visiting Scholar</strong><br>
        Digital Trust Centre (DTC)<br>
        Nanyang Technological University (NTU), Singapore
    </p>

    <hr style="
        border:none;
        border-top:1px solid rgba(255,255,255,0.20);
        margin:1.4rem 0;
    ">

    <p>
        <strong>Senior Lecturer</strong><br>
        Faculty of Computing<br>
        Universiti Teknologi Malaysia (UTM)
    </p>
</div>
            """
        )

        st.markdown("### Research interests")

        st.markdown(
            """
            - Blockchain and blockchain analytics
            - Graph Neural Networks
            - Artificial Intelligence and Machine Learning
            - Fraud detection and digital trust
            - Internet of Things (IoT)
            - Post-quantum computing and post-quantum security
            - Indoor localisation and positioning
            - Augmented Reality
            - Emerging computing technologies
            """
        )

    with right:

        st.markdown("### Current research context")

        st.write(
            "The X-MoE research on multi-class Ethereum fraud detection "
            "is being conducted during my Visiting Scholar appointment "
            "at the Digital Trust Centre (DTC), Nanyang Technological "
            "University (NTU), Singapore."
        )

        st.markdown("### Collaboration")

        st.write(
            "I welcome academic, research and industry collaboration "
            "in blockchain, artificial intelligence, graph learning, "
            "fraud analytics, digital trust, post-quantum security, "
            "Internet of Things and related emerging technologies."
        )

        st.markdown(
            """
            **Contact**

            [CS-NUR.HALIZA@assoc.main.ntu.edu.sg](mailto:CS-NUR.HALIZA@assoc.main.ntu.edu.sg)
            """
        )

        st.link_button(
            "UTM Official Profile",
            "https://www.utm.my/directory/staff/nur.haliza",
            use_container_width=True,
        )

        st.link_button(
            "Professional Website",
            "https://people.utm.my/nur-haliza/",
            use_container_width=True,
        )

        st.link_button(
            "UTM Scholar",
            "https://pure.utm.my/Scholar/ScholarInfoDetails/ag9w",
            use_container_width=True,
        )

        st.link_button(
            "GitHub",
            "https://github.com/nurhalizaabdulwahab197",
            use_container_width=True,
        )

    st.divider()

    st.markdown("### About X-MoE")

    st.write(
        "X-MoE is a research prototype for multi-class Ethereum "
        "address classification into Normal, Phishing and Scamming. "
        "The public repository provides X-MoE together with GCN, "
        "GAT and GraphSAGE baselines and reproducibility material."
    )

    st.warning(
        "This research demo is not a production service for declaring "
        "an arbitrary Ethereum address safe or fraudulent."
    )

    st.link_button(
        "Open X-MoE GitHub Repository",
        "https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection",
    )

