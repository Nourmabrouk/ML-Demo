import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate AI Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f14;
    color: #e8e9ec;
}

/* Main container */
.block-container {
    padding: 2.5rem 3rem;
    max-width: 1200px;
}

/* Hero header */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 600;
    background: linear-gradient(135deg, #e8c87a 0%, #f5e4b0 50%, #c9a84c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}

.hero-subtitle {
    color: #7a7f8e;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.03em;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: #c9a84c;
    text-transform: uppercase;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* Cards for inputs */
.input-card {
    background: #161921;
    border: 1px solid #252830;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Divider */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a84c55, transparent);
    margin: 1.5rem 0;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1a1710 0%, #1e1b0f 100%);
    border: 1px solid #c9a84c44;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-label-text {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #7a7f8e;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #e8c87a;
    font-weight: 600;
}

/* Stat pills */
.stat-row {
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}

.stat-pill {
    background: #1e2029;
    border: 1px solid #2e3140;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    color: #9ca0ae;
}

.stat-pill span {
    color: #e8c87a;
    font-weight: 600;
}

/* Streamlit widget overrides */
div[data-baseweb="select"] > div {
    background-color: #1e2029 !important;
    border-color: #2e3140 !important;
    color: #e8e9ec !important;
    border-radius: 10px !important;
}

.stSlider > div > div > div > div {
    background: #c9a84c !important;
}

.stSlider > div > div > div {
    background: #252830 !important;
}

label {
    color: #9ca0ae !important;
    font-size: 0.85rem !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #c9a84c, #e8c87a);
    color: #0d0f14;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2.5rem;
    width: 100%;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px #c9a84c33;
    color: #0d0f14;
}

/* Metric overrides */
[data-testid="metric-container"] {
    background: #161921;
    border: 1px solid #252830;
    border-radius: 12px;
    padding: 1rem;
}

[data-testid="metric-container"] label {
    color: #7a7f8e !important;
    font-size: 0.75rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8c87a !important;
    font-size: 1.6rem !important;
    font-family: 'Playfair Display', serif !important;
}

/* Info box */
.stInfo {
    background: #141a1e !important;
    border-left-color: #c9a84c !important;
}

/* Success */
.stSuccess {
    background: #0f1a12 !important;
    border-left-color: #4caf82 !important;
}

/* Number input */
.stNumberInput > div > div > input {
    background: #1e2029 !important;
    border-color: #2e3140 !important;
    color: #e8e9ec !important;
    border-radius: 10px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161921;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #7a7f8e;
    border-radius: 8px;
    font-size: 0.85rem;
}

.stTabs [aria-selected="true"] {
    background: #c9a84c22 !important;
    color: #e8c87a !important;
}

</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

model, encoders = load_model()


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Real Estate AI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Predict development potential for vacant land parcels using machine learning</div>', unsafe_allow_html=True)
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)


# ─── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:

    # ── Location & Zone ──────────────────────────────────────
    st.markdown('<div class="section-label">📍 Location & Zoning</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Location Type", ["urban", "suburban", "rural", "new_capital"],
                                format_func=lambda x: {"urban": "🏙️ Urban", "suburban": "🏘️ Suburban",
                                                        "rural": "🌾 Rural", "new_capital": "🌆 New Capital"}[x])
    with col2:
        zone = st.selectbox("Current Zone", ["residential", "commercial", "mixed", "agriculture"],
                            format_func=lambda x: x.capitalize())

    area = st.slider("Area Size (m²)", 1000, 20000, 5000, step=100,
                     help="Total land area in square meters")

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Demographics ─────────────────────────────────────────
    st.markdown('<div class="section-label">👥 Demographics</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        density = st.selectbox("Population Density", ["low", "medium", "high"],
                               format_func=lambda x: x.capitalize())
    with col4:
        income = st.selectbox("Average Income", ["low", "medium", "high"],
                              format_func=lambda x: x.capitalize())

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Market Data ──────────────────────────────────────────
    st.markdown('<div class="section-label">📊 Market & Infrastructure</div>', unsafe_allow_html=True)

    permit = st.selectbox("Permit History",
                          ["commercial", "residential", "industrial", "government", "education", "none"],
                          format_func=lambda x: x.capitalize())

    col5, col6 = st.columns(2)
    with col5:
        dist_city = st.slider("Distance to City Center (km)", 1, 50, 10)
    with col6:
        dist_road = st.slider("Distance to Main Road (km)", 0.1, 5.0, 0.5, step=0.1)

    col7, col8 = st.columns(2)
    with col7:
        projects = st.slider("Nearby Projects", 0, 10, 3)
    with col8:
        price = st.slider("Price per m² (EGP)", 1000, 25000, 12000, step=500)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Predict Button ───────────────────────────────────────
    predict_btn = st.button("🔮  Run Prediction", use_container_width=True)


# ─── Right Column: Results ─────────────────────────────────────────────────────
with right_col:

    # Land summary metrics
    st.markdown('<div class="section-label">📐 Parcel Summary</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Area", f"{area:,} m²")
    m2.metric("Price/m²", f"{price:,} EGP")
    m3.metric("Total Value", f"{area * price / 1_000_000:.1f}M EGP")

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        # ── Build Input ──────────────────────────────────────
        input_data = pd.DataFrame([{
            "area_size": area,
            "location_type": encoders["location_type"].transform([location])[0],
            "distance_to_city_center": dist_city,
            "distance_to_main_road": dist_road,
            "nearby_projects": projects,
            "price_per_meter": price,
            "current_zone": encoders["current_zone"].transform([zone])[0],
            "population_density": encoders["population_density"].transform([density])[0],
            "avg_income": encoders["avg_income"].transform([income])[0],
            "permit_history": encoders["permit_history"].transform([permit])[0],
            "value_impact": 0
        }])

        pred = model.predict(input_data)
        probs = model.predict_proba(input_data)[0]
        classes = encoders["expected_development"].classes_
        label = encoders["expected_development"].inverse_transform(pred)[0]
        top_prob = max(probs) * 100

        # ── Result Card ──────────────────────────────────────
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label-text">Predicted Development</div>
            <div class="result-value">{label.replace("_", " ").title()}</div>
            <div class="stat-row">
                <div class="stat-pill">Confidence <span>{top_prob:.1f}%</span></div>
                <div class="stat-pill">Zone <span>{zone.capitalize()}</span></div>
                <div class="stat-pill">Location <span>{location.replace("_"," ").title()}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Chart ────────────────────────────────
        st.markdown('<div class="section-label">📈 Development Probabilities</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Development": [c.replace("_", " ").title() for c in classes],
            "Probability": [round(p * 100, 1) for p in probs],
            "color": ["#c9a84c" if c == label else "#2e3140" for c in classes]
        }).sort_values("Probability", ascending=True)

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Development"],
            orientation="h",
            marker=dict(
                color=prob_df["color"],
                line=dict(width=0),
            ),
            text=[f"{p}%" for p in prob_df["Probability"]],
            textposition="outside",
            textfont=dict(color="#9ca0ae", size=11),
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=60, t=10, b=10),
            height=280,
            xaxis=dict(
                showgrid=False, showticklabels=False,
                zeroline=False, range=[0, max(probs) * 130],
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color="#9ca0ae", size=11),
            ),
            font=dict(family="DM Sans", color="#e8e9ec"),
            bargap=0.35,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Key Factors ──────────────────────────────────────
        st.markdown('<div class="section-label">🔑 Key Input Factors</div>', unsafe_allow_html=True)

        factors = {
            "Distance to City": f"{dist_city} km",
            "Nearby Projects": f"{projects} projects",
            "Distance to Road": f"{dist_road} km",
            "Permit History": permit.capitalize(),
        }

        fc1, fc2 = st.columns(2)
        for i, (k, v) in enumerate(factors.items()):
            (fc1 if i % 2 == 0 else fc2).metric(k, v)

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            background: #161921;
            border: 1px dashed #2e3140;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            color: #3e4252;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🏙️</div>
            <div style="font-size: 1rem; color: #4e5265;">
                Configure parcel parameters<br>and run prediction to see results
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #3e4252; font-size: 0.75rem; padding: 0.5rem 0;">
    Real Estate AI Predictor · Powered by Machine Learning · For professional use only
</div>
""", unsafe_allow_html=True)