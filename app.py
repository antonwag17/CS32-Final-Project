"""Career-Match — Streamlit web application (v2)."""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from questionnaire import (
    FEATURES, QUESTIONS, TARGETED_QUESTIONS,
    answers_to_vector, apply_targeted_answers,
)
from categorize_careers import (
    ARCHETYPE_NAMES, ARCHETYPE_DESCRIPTIONS, CENTROIDS, CAREERS,
)

# ── Constants ──────────────────────────────────────────────────────────────────

_NEAR_TIE_THRESHOLD = 0.5
_FLAT_PROFILE_STD   = 0.5
_CONFLICT_SCORE     = 4.0
_OPPOSING_PAIRS     = [
    ("structured",      "entrepreneurial"),
    ("independent",     "people_facing"),
    ("technical_depth", "people_facing"),
]
_MAX_DIST = float(np.sqrt(len(FEATURES)) * 4)

# Groups the 20 questions into 10 labeled dimension sections (2 questions each).
_DIMENSION_SECTIONS: list[tuple[str, int, int]] = [
    ("Analytical Thinking",    0,  2),
    ("Creativity",             2,  4),
    ("People & Collaboration", 4,  6),
    ("Independence",           6,  8),
    ("Risk Tolerance",         8,  10),
    ("Prestige & Recognition", 10, 12),
    ("Structure & Process",    12, 14),
    ("Impact & Purpose",       14, 16),
    ("Technical Depth",        16, 18),
    ("Entrepreneurship",       18, 20),
]

# Interleaved question order: one question per dimension per pass (2 passes of 10).
# Prevents adjacent same-dimension questions from anchoring each other's responses.
_QUESTION_ORDER = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
                   1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# Highlight color per archetype — used on cards, borders, and the radar chart.
_ARCHETYPE_COLORS: dict[int, str] = {
    0: "#0ea5e9",   # Consulting       – sky blue
    1: "#10b981",   # Finance          – emerald
    2: "#6366f1",   # Technology       – indigo
    3: "#8b5cf6",   # Academia         – violet
    4: "#f59e0b",   # Engineering      – amber
    5: "#ef4444",   # Public Service   – red
    6: "#ec4899",   # Arts             – pink
    7: "#14b8a6",   # Government       – teal
    8: "#f97316",   # Entrepreneurship – orange
    9: "#64748b",   # Law              – slate
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_match(student: np.ndarray) -> tuple[int, np.ndarray]:
    distances = np.linalg.norm(CENTROIDS - student, axis=1)
    return int(np.argmin(distances)), distances


def _detect_edge_case(
    student: np.ndarray, distances: np.ndarray
) -> tuple[str | None, list[str], list[str]]:
    feat_idx   = {f: i for i, f in enumerate(FEATURES)}
    sorted_ids = np.argsort(distances)

    if distances[sorted_ids[1]] - distances[sorted_ids[0]] < _NEAR_TIE_THRESHOLD:
        c1, c2    = CENTROIDS[sorted_ids[0]], CENTROIDS[sorted_ids[1]]
        top_feats = [FEATURES[i] for i in np.argsort(np.abs(c1 - c2))[::-1][:3]]
        tied      = [ARCHETYPE_NAMES[int(sorted_ids[0])], ARCHETYPE_NAMES[int(sorted_ids[1])]]
        return "near_tie", top_feats, tied

    if float(np.std(student)) < _FLAT_PROFILE_STD:
        top_feats = [FEATURES[i] for i in np.argsort(np.std(CENTROIDS, axis=0))[::-1][:3]]
        return "flat_profile", top_feats, []

    for feat_a, feat_b in _OPPOSING_PAIRS:
        ia, ib = feat_idx[feat_a], feat_idx[feat_b]
        if student[ia] >= _CONFLICT_SCORE and student[ib] >= _CONFLICT_SCORE:
            return "conflicting", [feat_a, feat_b], []

    return None, [], []


def _get_top3(student: np.ndarray) -> list[tuple[str, int, float]]:
    ranked = sorted(
        [(name, cid, np.array(scores)) for name, cid, scores in CAREERS],
        key=lambda x: np.linalg.norm(x[2] - student),
    )
    return [
        (name, cid, float(np.linalg.norm(vec - student)))
        for name, cid, vec in ranked[:3]
    ]


def _match_pct(dist: float) -> float:
    return max(0.0, (1.0 - dist / _MAX_DIST) * 100.0)


def _radar_chart(student: np.ndarray) -> go.Figure:
    labels = [f.replace("_", " ").title() for f in FEATURES]
    vals   = student.tolist()
    # Close the polygon by repeating the first point.
    r     = vals   + [vals[0]]
    theta = labels + [labels[0]]
    fig = go.Figure(go.Scatterpolar(
        r=r, theta=theta,
        fill="toself",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="#6366f1", width=2.5),
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[1, 5],
                tickfont=dict(size=11, color="#64748b"),
                gridcolor="rgba(100,116,139,0.18)",
                linecolor="rgba(100,116,139,0.18)",
            ),
            angularaxis=dict(
                tickfont=dict(
                    size=12, color="#1e293b",
                    family="Space Grotesk, Inter, sans-serif",
                ),
                gridcolor="rgba(100,116,139,0.12)",
                linecolor="rgba(100,116,139,0.18)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=70, r=70, t=50, b=50),
        height=430,
    )
    return fig


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Career Match", layout="wide")

# ── Global CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Font import ────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Variables ──────────────────────────────────────────── */
:root {
  --clr-dominant:     #0f172a;
  --clr-secondary:    #ffffff;
  --clr-accent:       #6366f1;
  --clr-accent-hover: #4f46e5;
  --clr-text:         #1e293b;
  --clr-muted:        #64748b;
  --clr-surface:      #f8fafc;
  --clr-border:       #e2e8f0;

  --sp-1:  0.5rem;
  --sp-2:  1rem;
  --sp-3:  1.5rem;
  --sp-4:  2rem;
  --sp-6:  3rem;
  --sp-8:  4rem;
  --sp-12: 6rem;

  --text-sm:   0.875rem;
  --text-base: 1rem;
  --text-lg:   1.25rem;
  --text-xl:   1.563rem;
  --text-2xl:  1.953rem;
  --text-3xl:  2.441rem;
  --text-4xl:  3.052rem;

  --lh-tight: 1.15;
  --lh-body:  1.6;

  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
}

/* ── Base ───────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

/* ── App background ─────────────────────────────────────── */
.stApp {
    background: linear-gradient(145deg, #0f172a 0%, #1e3a5f 40%, #312e81 75%, #4c1d95 100%) !important;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ──────────────────────────────── */
header[data-testid="stHeader"] { display: none !important; }
footer                          { display: none !important; }
#MainMenu                       { display: none !important; }

/* ── Floating content card ──────────────────────────────── */
section.main > div.block-container,
div.block-container {
    background: rgba(255,255,255,0.97) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.06), 0 30px 80px rgba(0,0,0,.4) !important;
    max-width: 860px !important;
    margin: var(--sp-4) auto !important;
    padding: var(--sp-6) var(--sp-8) var(--sp-8) !important;
}

/* ── Hero ───────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #312e81 55%, #4c1d95 100%);
    border-radius: var(--radius-lg);
    padding: var(--sp-8) var(--sp-6) var(--sp-6);
    text-align: center;
    margin-bottom: var(--sp-6);
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(165,180,252,0.25) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: var(--text-sm);
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: var(--sp-2);
    position: relative;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(var(--text-3xl), 6vw, var(--text-4xl));
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: var(--lh-tight);
    color: #ffffff;
    margin-bottom: var(--sp-2);
    position: relative;
}
.hero-desc {
    font-size: var(--text-base);
    color: #cbd5e1;
    max-width: 55ch;
    margin: 0 auto;
    line-height: var(--lh-body);
    position: relative;
    text-align: center !important;
}
.hero-meta {
    display: flex;
    justify-content: center;
    gap: var(--sp-4);
    margin-top: var(--sp-4);
    position: relative;
}
.hero-stat {
    display: flex;
    align-items: center;
    gap: var(--sp-1);
    font-size: var(--text-sm);
    color: #93c5fd;
    font-weight: 500;
}
.hero-stat-dot {
    width: 6px; height: 6px;
    background: #6366f1;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Survey dimension headers ───────────────────────────── */
.dim-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--sp-2) var(--sp-3);
    background: linear-gradient(90deg, rgba(99,102,241,0.07) 0%, rgba(99,102,241,0.01) 100%);
    border-left: 3px solid var(--clr-accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin-top: var(--sp-4);
    margin-bottom: var(--sp-2);
}
.dim-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--clr-text);
    letter-spacing: -0.01em;
}
.dim-idx {
    font-size: var(--text-sm);
    color: var(--clr-muted);
    font-weight: 500;
}

/* ── Slider cards ───────────────────────────────────────── */
div[data-testid="stSlider"] {
    background: var(--clr-surface) !important;
    border: 1px solid var(--clr-border) !important;
    border-radius: var(--radius-sm) !important;
    padding: var(--sp-2) var(--sp-3) !important;
    margin-bottom: var(--sp-2) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
}
div[data-testid="stSlider"]:hover {
    box-shadow: 0 2px 12px rgba(99,102,241,0.1) !important;
    border-color: #c7d2fe !important;
}
div[data-testid="stSlider"] label p {
    font-size: 1rem !important;
    font-weight: 500 !important;
    line-height: var(--lh-body) !important;
    color: var(--clr-text) !important;
}

/* ── Section headings ───────────────────────────────────── */
h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #1e3a5f !important;
    font-weight: 700 !important;
    line-height: var(--lh-tight) !important;
    letter-spacing: -0.02em !important;
}

/* ── Form submit button ─────────────────────────────────── */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, var(--clr-accent) 0%, #4f46e5 100%) !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: var(--text-base) !important;
    font-weight: 600 !important;
    padding: var(--sp-2) var(--sp-4) !important;
    border-radius: var(--radius-sm) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0) !important;
}

/* ── Regular buttons (Start Over, etc.) ─────────────────── */
div[data-testid="stButton"] button {
    background: transparent !important;
    border: 2px solid var(--clr-accent) !important;
    color: var(--clr-accent) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: var(--text-base) !important;
    font-weight: 600 !important;
    padding: var(--sp-1) var(--sp-4) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] button:hover {
    background: var(--clr-accent) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
div[data-testid="stButton"] button:active {
    transform: translateY(1px) !important;
}

/* ── Archetype result card ──────────────────────────────── */
.archetype-card {
    background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
    border-radius: var(--radius-md);
    border-top: 4px solid var(--arch-color, var(--clr-accent));
    padding: var(--sp-4) var(--sp-4) var(--sp-3);
    margin-bottom: var(--sp-4);
    box-shadow: 0 2px 8px rgba(99,102,241,0.08), 0 1px 2px rgba(0,0,0,.04);
}
.archetype-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: var(--text-sm);
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--arch-color, var(--clr-accent));
    margin-bottom: var(--sp-1);
}
.archetype-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: var(--text-3xl);
    font-weight: 700;
    color: #1e293b;
    letter-spacing: -0.02em;
    line-height: var(--lh-tight);
    margin-bottom: var(--sp-2);
}
.archetype-desc {
    font-size: var(--text-base);
    color: #475569;
    line-height: var(--lh-body);
    max-width: 65ch;
    margin: 0;
    text-align: left;
}

/* ── Career match cards ─────────────────────────────────── */
.career-card {
    background: var(--clr-secondary);
    border: 1px solid var(--clr-border);
    border-left: 4px solid var(--career-color, var(--clr-accent));
    border-radius: var(--radius-sm);
    padding: var(--sp-3);
    margin-bottom: var(--sp-2);
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.career-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,.1);
    transform: translateX(3px);
}
.rank-badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: var(--text-xl);
    font-weight: 700;
    color: var(--career-color, var(--clr-accent));
    min-width: 2.5rem;
    text-align: center;
}
.career-info { flex: 1; }
.career-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--clr-text);
    margin-bottom: 0.2rem;
}
.career-archetype {
    font-size: var(--text-sm);
    color: var(--clr-muted);
    font-weight: 500;
}
.career-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: var(--text-xl);
    font-weight: 700;
    color: var(--career-color, var(--clr-accent));
    min-width: 3.5rem;
    text-align: right;
}

/* ── Progress bars ──────────────────────────────────────── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--clr-accent), #4f46e5) !important;
    border-radius: 4px !important;
}
div[data-testid="stProgress"] > div {
    background: var(--clr-border) !important;
    border-radius: 4px !important;
}

/* ── Alert / info boxes ─────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
}

/* ── Focus styles (accessibility) ───────────────────────── */
:focus-visible {
    outline: 2px solid var(--clr-accent) !important;
    outline-offset: 3px !important;
}

/* ── Dark mode ──────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    section.main > div.block-container,
    div.block-container {
        background: rgba(15,23,42,0.97) !important;
    }
    div[data-testid="stSlider"] {
        background: #1e293b !important;
        border-color: #334155 !important;
    }
    div[data-testid="stSlider"] label p { color: #e2e8f0 !important; }
    div[data-testid="stSlider"]:hover    { border-color: #6366f1 !important; }
    .career-card {
        background: #1e293b !important;
        border-color: #334155 !important;
        border-left-width: 4px !important;
    }
    .career-title   { color: #f1f5f9 !important; }
    .archetype-card { background: linear-gradient(135deg, #1e293b 0%, #1e1b4b 100%) !important; }
    .archetype-name { color: #f1f5f9 !important; }
    .archetype-desc { color: #94a3b8 !important; }
    h2, h3          { color: #93c5fd !important; }
    .dim-name       { color: #e2e8f0 !important; }
    .dim-idx        { color: #64748b !important; }
}

/* ── Reduced motion ─────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration:  0.01ms !important;
        transition-duration: 0.01ms !important;
        scroll-behavior:     auto   !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────

for key, default in [("student", None), ("refined_student", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">CS32 · Career-Match</div>
    <div class="hero-title">Find Your Career Path</div>
    <div class="hero-desc" style="text-align:center;">
        Rate 20 statements to uncover your best-fit career archetype
        and the top roles that match your profile.
    </div>
    <div class="hero-meta">
        <span class="hero-stat">
            <span class="hero-stat-dot"></span>20 questions
        </span>
        <span class="hero-stat">
            <span class="hero-stat-dot"></span>10 dimensions
        </span>
        <span class="hero-stat">
            <span class="hero-stat-dot"></span>~3 minutes
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Survey form ────────────────────────────────────────────────────────────────

with st.form("survey"):
    st.subheader("Career Survey")
    st.markdown(
        '<p style="font-size:1.15rem;font-weight:500;color:#475569;margin-bottom:0.5rem;">'
        "Rate each statement &nbsp;·&nbsp; <strong>1</strong> = strongly disagree"
        " &nbsp;&nbsp;<strong>5</strong> = strongly agree"
        "</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    answers_by_idx: dict[int, int] = {}
    for display_num, q_idx in enumerate(_QUESTION_ORDER, start=1):
        question, _ = QUESTIONS[q_idx]
        val = st.slider(
            f"{display_num}. {question}",
            min_value=1, max_value=5, value=3, key=f"q{q_idx}",
        )
        answers_by_idx[q_idx] = val

    st.write("")
    submitted = st.form_submit_button(
        "Find My Career Match →", use_container_width=True, type="primary",
    )

if submitted:
    answers_in_order = [answers_by_idx[i] for i in range(len(QUESTIONS))]
    st.session_state.student = answers_to_vector(answers_in_order)
    st.session_state.refined_student = None

if st.session_state.student is None:
    st.stop()

# ── Edge-case refinement ───────────────────────────────────────────────────────

student_base: np.ndarray = st.session_state.student
_, distances_base = _run_match(student_base)
edge_case, clarify_features, tied_names = _detect_edge_case(student_base, distances_base)

if edge_case and st.session_state.refined_student is None:
    st.divider()
    if edge_case == "near_tie":
        st.info(
            f"Your profile is a close match between **{tied_names[0]}** and **{tied_names[1]}**. "
            "Answer a few more targeted questions to sharpen your result."
        )
    elif edge_case == "flat_profile":
        st.info(
            "Your responses are broadly balanced with no strong lean. "
            "A few specific questions will help narrow down your best match."
        )
    elif edge_case == "conflicting":
        fa, fb = [f.replace("_", " ") for f in clarify_features]
        st.info(
            f"Your profile scores high on both **{fa}** and **{fb}**, "
            "which pull toward different career types. Let's clarify your priorities."
        )

    with st.form("refinement"):
        st.subheader("Clarifying Questions")
        targeted_raw: dict[str, int] = {}
        for feat in clarify_features:
            targeted_raw[feat] = st.slider(
                TARGETED_QUESTIONS[feat],
                min_value=1, max_value=5, value=3, key=f"t_{feat}",
            )
        refined = st.form_submit_button(
            "Refine My Results →", use_container_width=True,
        )

    if refined:
        st.session_state.refined_student = apply_targeted_answers(
            student_base, {k: float(v) for k, v in targeted_raw.items()}
        )
        st.rerun()

# ── Resolve the active vector ──────────────────────────────────────────────────

display_student = (
    st.session_state.refined_student
    if st.session_state.refined_student is not None
    else student_base
)

cluster_id, _  = _run_match(display_student)
archetype_name = ARCHETYPE_NAMES[cluster_id]
archetype_desc = ARCHETYPE_DESCRIPTIONS[cluster_id]
arch_color     = _ARCHETYPE_COLORS[cluster_id]
top3           = _get_top3(display_student)

# ── Results ────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Your Results")

if st.session_state.refined_student is not None:
    st.success("Results refined based on your clarifying answers.")

# Archetype card
st.markdown(f"""
<div class="archetype-card" style="--arch-color:{arch_color};">
    <div class="archetype-eyebrow">Your Career Archetype</div>
    <div class="archetype-name">{archetype_name}</div>
    <p class="archetype-desc">{archetype_desc}</p>
</div>
""", unsafe_allow_html=True)

# Top 3 career matches
st.subheader("Top 3 Career Matches")
for rank, (career, cid, dist) in enumerate(top3, start=1):
    pct   = _match_pct(dist)
    color = _ARCHETYPE_COLORS[cid]
    aname = ARCHETYPE_NAMES[cid]
    st.markdown(f"""
    <div class="career-card" style="--career-color:{color};">
        <div class="rank-badge">#{rank}</div>
        <div class="career-info">
            <div class="career-title">{career}</div>
            <div class="career-archetype">{aname}</div>
        </div>
        <div class="career-pct">{pct:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct / 100.0)

st.write("")

# Trait profile radar chart
st.subheader("Your Trait Profile")
st.caption("How you score across all 10 career dimensions (scale: 1 – 5)")
st.plotly_chart(_radar_chart(display_student), use_container_width=True)

# Start over
st.write("")
st.divider()
col_l, col_c, col_r = st.columns([1.5, 2, 1.5])
with col_c:
    if st.button("Take the quiz again", use_container_width=True):
        st.session_state.student = None
        st.session_state.refined_student = None
        st.rerun()
