# app.py
"""
Complete Sleep Quality AI Dashboard
- Tech Green-Blue theme
- Camera + upload, photo analysis, numeric model combination
- History, CSV export, charts, radar, save PNG, demo generation
- Safe session_state usage and theme/accent controls
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import random

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Tech Green-Blue Sleep Quality Dashboard", layout="wide", page_icon="💤")

# -------------------------
# HELPERS / UTILITIES
# -------------------------
def load_model(path="sleep_quality_model.joblib"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def analyze_face_brightness(image_path):
    """Simple brightness-based score 0-100 (clipped)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    # Map brightness to a score (darker under-eyes -> lower assumed restfulness)
    score = np.clip((brightness / 255.0) * 100.0, 15.0, 95.0)
    return float(score)

def image_to_bytes(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def pretty_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_history_to_disk(history, filename="predictions_history.csv"):
    try:
        pd.DataFrame(history).to_csv(filename, index=False)
        return True
    except Exception:
        return False

def append_history(state, record):
    state["history"].append(record)
    # also persist to disk
    save_history_to_disk(state["history"])

def create_summary_png(record, theme_dark=True, accent="#00E5A8"):
    """Create a simple PNG summary of the last prediction."""
    W, H = 900, 480
    bg = (12, 18, 24) if theme_dark else (255, 255, 255)
    fg = (235, 255, 245) if theme_dark else (10, 10, 10)
    img = Image.new("RGB", (W, H), color=bg)
    d = ImageDraw.Draw(img)
    try:
        fnt_title = ImageFont.truetype("arialbd.ttf", 36)
        fnt = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        fnt_title = None
        fnt = None
    try:
        acc_rgb = tuple(int(accent.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        acc_rgb = (0, 229, 168)
    d.rectangle([(0,0),(W,80)], fill=acc_rgb)
    d.text((28, 22), "Sleep Quality Report", fill=(255,255,255), font=fnt_title)
    lines = [
        f"Time: {record.get('time','-')}",
        f"Final Score: {record.get('final_score','-'):.2f}/100" if record.get('final_score') is not None else "Final Score: -",
        f"Sleep Duration: {record.get('sleep_duration','-')} hrs",
        f"Screen Time: {record.get('screen_time','-')} hrs",
        f"Caffeine: {record.get('caffeine','-')} cups",
        f"Stress Level: {record.get('stress_level','-')}",
        f"Physical Activity: {record.get('physical_activity','-')} mins",
        f"Age: {record.get('age','-')}"
    ]
    y = 120
    for ln in lines:
        d.text((36, y), ln, fill=fg, font=fnt)
        y += 34
    score = record.get("final_score", None)
    if score is not None:
        cx, cy, r = 700, 240, 100
        d.ellipse([(cx-r, cy-r), (cx+r, cy+r)], outline=fg, width=3)
        txt = f"{score:.0f}"
        d.text((cx-26, cy-20), txt, fill=fg, font=fnt_title)
    b = io.BytesIO()
    img.save(b, format="PNG")
    b.seek(0)
    return b

# -------------------------
# SESSION STATE initialization (before widgets)
# -------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"
if "accent" not in st.session_state:
    st.session_state["accent"] = "#00E5A8"
if "last_image_path" not in st.session_state:
    st.session_state["last_image_path"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

# initialize widget-backed keys with safe defaults
widget_defaults = {
    "sleep_duration": 7.0,
    "screen_time": 2.0,
    "caffeine": 2,
    "stress_level": 5,
    "physical_activity": 45,
    "age": 25,
    "include_photo": True,
    "combine_strategy": "Average (50/50)",
    "save_to_history": True
}
for k, v in widget_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Load numeric model (optional)
# -------------------------
model = load_model("sleep_quality_model.joblib")
if model is None:
    st.sidebar.warning("Model 'sleep_quality_model.joblib' not found — model-based predictions disabled. Photo-only and demo still work.")

# -------------------------
# CSS: Big Tech Green-Blue Theme
# -------------------------
def apply_big_css(theme="Dark", accent="#00E5A8"):
    """
    Replacement: stronger, unmistakable background gradients and colored section panels.
    Paste this function in place of your old apply_big_css.
    """
    # ensure accent is a hex string
    accent_hex = accent if isinstance(accent, str) else "#00E5A8"

    css = f"""
    <style>
    /* ---------- PAGE + STRONG BACKGROUND ---------- */
    html, body, .stApp, .reportview-container .main {{
        height: 100%;
        margin: 0;
        padding: 0;
        background: linear-gradient(120deg, #00131a 0%, #002b2f 30%, #003a3f 60%, #00121a 100%);
        color: #e6f7ef;
        font-family: 'Inter', 'Poppins', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* Animated subtle moving gradient overlay for depth */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: radial-gradient(circle at 10% 10%, rgba(0,229,168,0.03), transparent 12%),
                    radial-gradient(circle at 90% 80%, rgba(0,181,204,0.03), transparent 12%);
        pointer-events: none;
        z-index: 0;
        animation: bgShift 20s linear infinite;
    }}
    @keyframes bgShift {{
        0% {{ transform: translateY(0px) scale(1.00); opacity: 0.96; }}
        50% {{ transform: translateY(-12px) scale(1.02); opacity: 1; }}
        100% {{ transform: translateY(0px) scale(1.00); opacity: 0.96; }}
    }}

    /* Page container padding & stacking */
    .reportview-container .main > div {{
        position: relative;
        z-index: 1;
    }}

    /* ---------- SIDEBAR / HEADER STYLING ---------- */
    .css-1v3fvcr {{ /* header bar container class may vary by streamlit versions */
        background: linear-gradient(90deg, rgba(0,40,40,0.6), rgba(0,20,30,0.6));
        border-bottom: 1px solid rgba(255,255,255,0.03);
        box-shadow: 0 6px 30px rgba(0,0,0,0.6);
    }}

    /* ---------- GLASS CARDS & PANEL COLORS (distinct per-section) ---------- */
    .glass-card {{
        position: relative;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 18px 50px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }}

    /* Left panel - teal gradient */
    .left-panel {{
        background: linear-gradient(135deg, rgba(0,229,168,0.06), rgba(2,60,62,0.06));
        border-left: 4px solid rgba(0,229,168,0.18);
    }}

    /* Middle panel - aqua/blue gradient */
    .middle-panel {{
        background: linear-gradient(135deg, rgba(0,181,204,0.06), rgba(3,45,60,0.06));
        border-left: 4px solid rgba(0,181,204,0.18);
    }}

    /* Right panel - emerald/green gradient */
    .right-panel {{
        background: linear-gradient(135deg, rgba(16,185,129,0.06), rgba(2,45,50,0.06));
        border-left: 4px solid rgba(16,185,129,0.18);
    }}

    /* Preview box darker contrast */
    .preview-box {{
        background: linear-gradient(180deg, rgba(0,0,0,0.35), rgba(255,255,255,0.02));
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.03);
    }}

    /* ---------- NEON ACCENT BUTTONS ---------- */
    .accent-button > button, .stButton>button {{
        background: linear-gradient(90deg, {accent_hex}, #00bfa5) !important;
        color: #001f17 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        font-weight: 700 !important;
        box-shadow: 0 12px 40px rgba(0,229,168,0.12) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .accent-button > button:hover, .stButton>button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,229,168,0.18) !important;
    }}

    /* ---------- METRICS, SLIDERS, INPUT LOOK ---------- */
    .stMetric > div {{
        background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 10px;
        padding: 8px;
        color: #e6fff3;
    }}

    .stSlider > div {{
        height: 8px !important;
        border-radius: 8px;
    }}

    input, textarea {{
        background: rgba(0,20,20,0.55) !important;
        color: #ccffff !important;
        border: 1px solid rgba(0,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 6px !important;
    }}

    /* ---------- FOOTER ---------- */
    footer {{
        color: #9fbfaa;
        text-align:center;
        margin-top: 40px;
        font-size: 0.9rem;
    }}

    /* Small screens - reduce heavy backgrounds */
    @media (max-width: 900px) {{
        .stApp::before {{ display:none; }}
        .left-panel, .middle-panel, .right-panel {{ background: rgba(0,0,0,0.35) !important; border-left: none !important; }}
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


apply_big_css(st.session_state["theme"], st.session_state["accent"])

# -------------------------
# Top header with theme & accent controls (widget keys safe)
# -------------------------
header_c1, header_c2, header_c3 = st.columns([1,1,1])
with header_c1:
    st.markdown("<h2 style='margin:0; color:#E6FFF3;'>✨ Sleep Quality Prediction</h2>", unsafe_allow_html=True)
with header_c2:
    st.markdown("<div style='text-align:center; color:#bfead5;'>Face + Lifestyle hybrid predictor with interactive charts</div>", unsafe_allow_html=True)
with header_c3:
    # theme select (safe, keys initialized earlier)
    theme_choice = st.selectbox("Theme", options=["Dark","Light"], index=0 if st.session_state["theme"]=="Dark" else 1, key="ui_theme")
    if theme_choice != st.session_state["theme"]:
        st.session_state["theme"] = theme_choice
        apply_big_css(st.session_state["theme"], st.session_state["accent"])
    accent_choice = st.color_picker("Accent Color", st.session_state["accent"], key="ui_accent")
    if accent_choice != st.session_state["accent"]:
        st.session_state["accent"] = accent_choice
        apply_big_css(st.session_state["theme"], st.session_state["accent"])

st.markdown("---")

# -------------------------
# Main 3-column layout
# -------------------------
left_col, mid_col, right_col = st.columns([1.05, 1.4, 1])

# -------------------------
# LEFT: Controls & Inputs
# -------------------------
with left_col:
    st.markdown("<div class='glass-card left-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧭 Controls & Inputs</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Use the left panel to set your daily inputs, theme and utilities.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # boxed inputs
    st.markdown("<div class='input-box'><label>Sleep Duration (hours)</label>", unsafe_allow_html=True)
    sleep_duration = st.slider("", 0.0, 12.0, st.session_state["sleep_duration"], step=0.25, key="sleep_duration")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-box'><label>Screen Time before Bed (hours)</label>", unsafe_allow_html=True)
    screen_time = st.slider("", 0.0, 10.0, st.session_state["screen_time"], step=0.25, key="screen_time")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-box'><label>Caffeine Intake (cups/day)</label>", unsafe_allow_html=True)
    caffeine = st.slider("", 0, 10, st.session_state["caffeine"], key="caffeine")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-box'><label>Stress Level (1 = Low, 10 = High)</label>", unsafe_allow_html=True)
    stress_level = st.slider("", 1, 10, st.session_state["stress_level"], key="stress_level")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-box'><label>Physical Activity (minutes/day)</label>", unsafe_allow_html=True)
    physical_activity = st.slider("", 0, 240, st.session_state["physical_activity"], key="physical_activity")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-box'><label>Age</label>", unsafe_allow_html=True)
    age = st.slider("", 10, 90, st.session_state["age"], key="age")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚙️ Extra Controls</div>", unsafe_allow_html=True)
    include_photo = st.checkbox("Use face photo for visual score (middle column)", value=st.session_state["include_photo"], key="include_photo")
    combine_strategy = st.selectbox("Combine Strategy", options=["Average (50/50)","Weighted (model 70% + photo 30%)","Photo only","Model only"], index=0, key="combine_strategy")
    save_to_history = st.checkbox("Save this prediction to history", value=st.session_state["save_to_history"], key="save_to_history")

    # callbacks for autofill & reset (safe)
    def autofill_cb():
        st.session_state["sleep_duration"] = 6.5
        st.session_state["screen_time"] = 3.0
        st.session_state["caffeine"] = 1
        st.session_state["stress_level"] = 6
        st.session_state["physical_activity"] = 30
        st.session_state["age"] = 22

    def reset_history_cb():
        st.session_state["history"] = []
        if os.path.exists("predictions_history.csv"):
            os.remove("predictions_history.csv")

    st.markdown("<div style='display:flex; gap:8px; margin-top:8px;'>", unsafe_allow_html=True)
    st.button("✨ Auto-fill demo inputs", on_click=autofill_cb, key="autofill_btn")
    st.button("🔄 Reset History", on_click=reset_history_cb, key="reset_hist_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# MIDDLE: Camera, Preview, Prediction
# -------------------------
with mid_col:
    st.markdown("<div class='glass-card middle-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📷 Image Capture & Live Preview</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Take a live picture with your webcam or upload a portrait. We compute a simple visual indicator (brightness-based) as proxy for facial tiredness.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    image_file = st.camera_input("Take a Photo", key="camera_input")
    if image_file is None:
        image_file = st.file_uploader("Or upload a photo", type=["jpg","jpeg","png"], key="file_upload")

    photo_score = None
    if image_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tfile.write(image_file.getbuffer())
        tfile.flush()
        tfile.close()
        st.session_state["last_image_path"] = tfile.name

        pil_img = Image.open(tfile.name)
        thumb = pil_img.copy()
        thumb.thumbnail((520,520))
        preview_image_bytes = image_to_bytes(thumb)
        st.markdown("<div class='preview-box'>", unsafe_allow_html=True)
        st.image(preview_image_bytes, caption="Preview", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if include_photo:
            try:
                photo_score = analyze_face_brightness(tfile.name)
                st.markdown(f"**Visual (photo) score:** {photo_score:.2f} / 100")
            except Exception:
                st.error("Failed to analyze photo. Ensure the uploaded file is a valid image.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 Run Prediction</div>", unsafe_allow_html=True)

    # run prediction function used by button
    def run_prediction():
        X = np.array([[st.session_state["sleep_duration"], st.session_state["screen_time"], st.session_state["caffeine"],
                       st.session_state["stress_level"], st.session_state["physical_activity"], st.session_state["age"]]])
        model_score = None
        if model is not None:
            try:
                model_score = float(model.predict(X)[0])
            except Exception:
                st.error("Model prediction failed. Check model compatibility.")
                model_score = None

        # combine according to strategy
        p = photo_score
        strategy = st.session_state["combine_strategy"]
        final_score = None
        if strategy == "Photo only":
            final_score = p if p is not None else model_score
        elif strategy == "Model only":
            final_score = model_score if model_score is not None else p
        elif strategy == "Weighted (model 70% + photo 30%)":
            if (model_score is not None) and (p is not None):
                final_score = 0.7*model_score + 0.3*p
            else:
                final_score = model_score if model_score is not None else p
        else:  # Average
            vals = [v for v in [model_score, p] if v is not None]
            final_score = float(np.mean(vals)) if len(vals) > 0 else None

        if final_score is None:
            st.error("No available prediction (missing both model and photo). Upload a photo or place 'sleep_quality_model.joblib' in project folder.")
            return

        # show result
        st.success(f"💤 Final Sleep Quality Score: **{final_score:.2f} / 100**")
        if final_score >= 90:
            st.balloons()
        st.progress(min(max(int(final_score), 0), 100))

        if final_score >= 80:
            st.markdown("<h3 style='color:#C1FFC1;'>🌟 Excellent Rest</h3>", unsafe_allow_html=True)
        elif final_score >= 60:
            st.markdown("<h3 style='color:#FFE5B4;'>🙂 Good</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#FFB6C1;'>⚠️ Needs Improvement</h3>", unsafe_allow_html=True)

        rec = {
            "time": pretty_time(),
            "sleep_duration": st.session_state["sleep_duration"],
            "screen_time": st.session_state["screen_time"],
            "caffeine": st.session_state["caffeine"],
            "stress_level": st.session_state["stress_level"],
            "physical_activity": st.session_state["physical_activity"],
            "age": st.session_state["age"],
            "model_score": model_score,
            "photo_score": p,
            "final_score": final_score,
            "combine_strategy": st.session_state["combine_strategy"]
        }
        st.session_state["last_prediction"] = rec
        if st.session_state["save_to_history"]:
            append_history(st.session_state, rec)
            st.success("Saved to history (and to predictions_history.csv).")

    st.markdown("<div class='accent-button'>", unsafe_allow_html=True)
    st.button("🌙 Predict Sleep Quality Now", on_click=run_prediction, key="predict_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    # Save summary PNG callback
    def save_png_cb():
        if st.session_state.get("last_prediction") is None:
            st.warning("Run a prediction first.")
            return
        rec = st.session_state["last_prediction"]
        png_b = create_summary_png(rec, theme_dark=(st.session_state["theme"]=="Dark"), accent=st.session_state["accent"])
        st.download_button("⬇️ Download Report PNG", data=png_b, file_name="sleep_report.png", mime="image/png")
    st.button("💾 Save Last Prediction Summary (PNG)", on_click=save_png_cb, key="save_png_btn")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# RIGHT: Charts & History
# -------------------------
with right_col:
    st.markdown("<div class='glass-card right-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Visuals & History</div>", unsafe_allow_html=True)

    if len(st.session_state["history"]) > 0:
        df_hist = pd.DataFrame(st.session_state["history"])
        st.markdown("#### 🔁 Recent Predictions")
        st.dataframe(df_hist.sort_values("time", ascending=False).head(8), use_container_width=True)

        csv_bytes = df_to_csv_bytes(df_hist)
        st.download_button("⬇️ Download History CSV", data=csv_bytes, file_name="predictions_history.csv", mime="text/csv")

        # trend line
        try:
            df_hist['time_dt'] = pd.to_datetime(df_hist['time'])
            df_trend = df_hist.sort_values('time_dt')
            fig_trend = px.line(df_trend, x='time_dt', y='final_score', markers=True, title="Sleep Quality Trend")
            fig_trend.update_layout(margin=dict(l=10,r=10,t=30,b=10), template="plotly_dark" if st.session_state["theme"]=="Dark" else "plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)
        except Exception:
            pass

        # radar of last
        try:
            last = df_hist.iloc[-1]
            categories = ["Sleep Duration", "Screen Time (rev)", "Caffeine (rev)", "Stress (rev)", "Activity"]
            values = [
                float(last['sleep_duration'])/12*100,
                max(0, 10 - float(last['screen_time']))/10*100,
                max(0, 10 - float(last['caffeine']))/10*100,
                max(0, 10 - float(last['stress_level']))/10*100,
                min(1, float(last['physical_activity'])/120)*100
            ]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Profile', line_color=st.session_state["accent"]))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception:
            pass
    else:
        st.info("No saved history yet. Make a prediction and enable 'Save this prediction to history' to populate this section.")

    st.markdown("---")
    st.markdown("### ⚙️ Quick Charts (Current Inputs)")
    quick = {
        "Sleep Duration (hrs)": st.session_state["sleep_duration"],
        "Screen Time (hrs)": st.session_state["screen_time"],
        "Caffeine (cups)": st.session_state["caffeine"],
        "Stress (1-10)": st.session_state["stress_level"],
        "Activity (min)": st.session_state["physical_activity"]
    }
    fig_bar = px.bar(x=list(quick.keys()), y=list(quick.values()), labels={'x':'Metric','y':'Value'}, title="Current Inputs Snapshot")
    fig_bar.update_layout(margin=dict(l=10,r=10,t=30,b=10), template="plotly_dark" if st.session_state["theme"]=="Dark" else "plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# BOTTOM: Extras (generate demo, download disk CSV, save PNG)
# -------------------------
st.markdown("<div style='margin-top:20px; padding: 14px; border-radius:12px;'>", unsafe_allow_html=True)
st.markdown("## 🧾 Export / Extras", unsafe_allow_html=True)
colA, colB, colC = st.columns([1,1,1])

def generate_demo_record():
    sd = round(np.random.uniform(4,9),2)
    demo_record = {
        "time": pretty_time(),
        "sleep_duration": sd,
        "screen_time": round(np.random.uniform(0,5),2),
        "caffeine": int(np.random.randint(0,4)),
        "stress_level": int(np.random.randint(1,9)),
        "physical_activity": int(np.random.randint(10,120)),
        "age": int(np.random.randint(18,60)),
        "model_score": None,
        "photo_score": None,
        "final_score": round(np.random.uniform(50,95),2),
        "combine_strategy": "Demo"
    }
    append_history(st.session_state, demo_record)

with colA:
    if os.path.exists("predictions_history.csv"):
        with open("predictions_history.csv","rb") as f:
            st.download_button("⬇️ Download saved CSV (disk)", f.read(), file_name="predictions_history.csv", mime="text/csv")
    else:
        st.write("No saved CSV on disk yet.")

# Provide safe rerun function that works across Streamlit versions
def rerun_safe():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

with colB:
    st.button("🎯 Generate Random Sample Prediction", on_click=lambda: (generate_demo_record(), rerun_safe()))

with colC:
    def save_last_png_cb():
        if st.session_state.get("last_prediction") is None:
            st.warning("No last prediction — run one first.")
            return
        rec = st.session_state["last_prediction"]
        png_b = create_summary_png(rec, theme_dark=(st.session_state["theme"]=="Dark"), accent=st.session_state["accent"])
        st.download_button("⬇️ Download PNG", data=png_b, file_name="sleep_report.png", mime="image/png")
    st.button("💾 Save Last Prediction as PNG (visual)", on_click=save_last_png_cb)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#9fbfaa; padding-bottom:18px;'>Made with ❤️ by <b>Nidhi Sachdeva</b> — Tech Green-Blue Dashboard</div>", unsafe_allow_html=True)
