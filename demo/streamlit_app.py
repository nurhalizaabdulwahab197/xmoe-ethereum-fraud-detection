from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="X-MoE | Ethereum Fraud Detection",
    page_icon="â—†",
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
    st.caption("Research Demo Â· V2")

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

        â€¢ Normal
        â€¢ Phishing
        â€¢ Scamming
        """
    )

    st.divider()

    st.markdown(
        """
        **Evaluation**

        Clean protocol
        Seeds 100â€“109
        300 epochs
        Deterministic CPU
        """
    )

    st.divider()

    st.caption(
        "Research prototype â€” not a production fraud-screening service."
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """<div class="hero">
<div class="version">RESEARCH DEMO Â· VERSION 2</div>
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
        f"Â± {acc_std:.4f}",
    )

    c4.metric(
        "X-MoE Macro-F1",
        f"{f1:.4f}",
        f"Â± {f1_std:.4f}",
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

        **Normal Â· Phishing Â· Scamming**

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
            "A Â· GCN",
            "B Â· GAT",
            "C Â· GraphSAGE",
            "D Â· X-MoE",
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
        â”‚
        â–¼
   GCNConv(64)
        â”‚
       ReLU
        â”‚
   Dropout(0.5)
        â”‚
        â–¼
   GCNConv(64)
        â”‚
       ReLU
        â”‚
   Dropout(0.5)
        â”‚
 Linear classifier
        â”‚
        â–¼
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
        â”‚
        â–¼
GATConv(32, heads=4)
        â”‚
       ELU
        â”‚
   Dropout(0.5)
        â”‚
        â–¼
GATConv(32, heads=1)
        â”‚
       ELU
        â”‚
 Linear classifier
        â”‚
        â–¼
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
        â”‚
        â–¼
   SAGEConv(64)
        â”‚
       ReLU
        â”‚
   Dropout(0.5)
        â”‚
        â–¼
   SAGEConv(64)
        â”‚
       ReLU
        â”‚
   Dropout(0.5)
        â”‚
 Linear classifier
        â”‚
        â–¼
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

        st.subheader("X-MoE Â· Proposed Architecture")

        st.write(
            "X-MoE differs fundamentally from the three baseline models. "
            "Instead of relying on one graph-learning backbone, it combines "
            "two complementary experts through an adaptive routing mechanism."
        )

        st.markdown("#### High-level pipeline")

        st.code(
            """                         Node features
                              â”‚
                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                  â”‚                       â”‚
                  â–¼                       â–¼
        Conflict-aware              GraphSAGE
        multitask expert              expert
                  â”‚                       â”‚
                  â–¼                       â–¼
           Expert 1 logits         Expert 2 logits
                  â”‚                       â”‚
                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                     Expert disagreement
                              â”‚
                  Node connectivity / degree
                              â”‚
                              â–¼
                       Gating network
                              â”‚
                              â–¼
                 Class-specific mixture
                         weights
             w_Normal | w_Phishing | w_Scamming
                              â”‚
                              â–¼
                  Adaptive weighted fusion
                              â”‚
                              â–¼
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
            "Expert 1 Â· Conflict-Aware Multitask Expert",
            "Expert 2 Â· GraphSAGE Expert",
            "Expert Disagreement",
            "Connectivity-Aware Gating",
            "Class-Specific Mixture Weights",
            "Adaptive Weighted Fusion",
        ],
    )

    if component == "Expert 1 Â· Conflict-Aware Multitask Expert":

        st.markdown("### Expert 1 Â· Conflict-Aware Multitask Expert")

        st.write(
            "The first expert captures feature-level and task-specific "
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

    elif component == "Expert 2 Â· GraphSAGE Expert":

        st.markdown("### Expert 2 Â· GraphSAGE Expert")

        st.write(
            "The second expert is a dedicated graph-structure expert based "
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
            "| Expert 1 logits - Expert 2 logits |",
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
            1. Expert 1 logits
            2. Expert 2 logits
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
            "Conflict-aware logits + w_class Ã— "
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
            <h4>1 Â· Feature Evidence</h4>
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
            <h4>2 Â· Graph Evidence</h4>
            Relational information from neighbouring Ethereum addresses.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q3:
        st.markdown(
            """
            <div class="card">
            <h4>3 Â· Expert Disagreement</h4>
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
            <h4>4 Â· Connectivity</h4>
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

    chart_data["lower"] = (
        chart_data["mean"]
        - chart_data["std"]
    ).clip(lower=0)

    chart_data["upper"] = (
        chart_data["mean"]
        + chart_data["std"]
    )

    colour = alt.Scale(
        domain=MODEL_ORDER,
        range=[
            MODEL_COLOURS[m]
            for m in MODEL_ORDER
        ],
    )

    bars = (
        alt.Chart(chart_data)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "model:N",
                sort=MODEL_ORDER,
                title=None,
            ),
            y=alt.Y(
                "mean:Q",
                title=metric,
            ),
            color=alt.Color(
                "model:N",
                scale=colour,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "model:N",
                    title="Model",
                ),
                alt.Tooltip(
                    "mean:Q",
                    format=".6f",
                    title="Mean",
                ),
                alt.Tooltip(
                    "std:Q",
                    format=".6f",
                    title="Std",
                ),
            ],
        )
    )

    errors = (
        alt.Chart(chart_data)
        .mark_rule(
            color="#0B1F3A",
            strokeWidth=2,
        )
        .encode(
            x=alt.X(
                "model:N",
                sort=MODEL_ORDER,
            ),
            y="lower:Q",
            y2="upper:Q",
        )
    )

    st.altair_chart(
        bars + errors,
        use_container_width=True,
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

    chart = (
        alt.Chart(class_data)
        .mark_bar(
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        )
        .encode(
            x=alt.X(
                "model:N",
                sort=MODEL_ORDER,
                title=None,
            ),
            y=alt.Y(
                "mean:Q",
                title="Mean score",
                scale=alt.Scale(
                    domain=[0, 1]
                ),
            ),
            color=alt.Color(
                "model:N",
                scale=colour,
                title="Model",
            ),
            column=alt.Column(
                "metric:N",
                title=None,
            ),
            tooltip=[
                "model:N",
                "metric:N",
                alt.Tooltip(
                    "mean:Q",
                    format=".6f",
                ),
                alt.Tooltip(
                    "std:Q",
                    format=".6f",
                ),
            ],
        )
        .properties(
            height=350,
        )
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
