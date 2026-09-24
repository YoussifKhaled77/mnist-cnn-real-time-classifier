import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

os.environ.setdefault("KERAS_BACKEND", "torch")


st.set_page_config(
    page_title="Digit Lens",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATHS = [
    Path(__file__).with_name("mnist_cnn_model.keras"),
    Path(__file__).with_name("model.keras"),
]
CANVAS_SIZE = 560


@st.cache_resource(show_spinner="Loading your CNN model...")
def load_model():
    from keras.models import load_model as keras_load_model

    model_path = next((path for path in MODEL_PATHS if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Place model.keras or mnist_cnn_model.keras beside app.py.")
    return keras_load_model(model_path)


def preprocess_canvas(image_data: np.ndarray) -> np.ndarray:
    grayscale = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")
    pixels = np.asarray(grayscale, dtype="uint8")
    ink = pixels > 20
    if not np.any(ink):
        return np.zeros((1, 28, 28, 1), dtype="float32")

    rows, columns = np.where(ink)
    cropped = Image.fromarray(pixels[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1])
    scale = 20 / max(cropped.size)
    digit_size = tuple(max(1, round(value * scale)) for value in cropped.size)
    digit = cropped.resize(digit_size, Image.Resampling.LANCZOS)

    centered = Image.new("L", (28, 28), 0)
    position = ((28 - digit.width) // 2, (28 - digit.height) // 2)
    centered.paste(digit, position)
    normalized = np.asarray(centered, dtype="float32") / 255.0
    return normalized[np.newaxis, ..., np.newaxis]


def predict(image_data: np.ndarray) -> np.ndarray:
    model_input = preprocess_canvas(image_data)
    probabilities = np.asarray(load_model().predict(model_input, verbose=0)[0], dtype="float32")
    return probabilities / probabilities.sum()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #17221d; --muted: #718078; --line: #dbe4df; --mint: #dff5e8; --green: #16794c; --coral: #ef765f; }
    .stApp { background: #f7faf8; color: var(--ink); font-family: 'Manrope', sans-serif; }
    .block-container { max-width: 1180px; padding: 58px 5vw 70px; }
    header[data-testid="stHeader"] { background: transparent; }
    .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 54px; }
    .brand-mark { display: grid; place-items: center; width: 34px; height: 34px; background: var(--ink); color: #e6f8ec; border-radius: 10px; font-size: 18px; }
    .brand-name { font-size: 15px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .eyebrow { color: var(--green); font: 500 12px 'DM Mono', monospace; letter-spacing: .14em; text-transform: uppercase; margin: 0 0 13px; }
    h1 { font-size: clamp(2.6rem, 5vw, 4.8rem); line-height: .98; letter-spacing: -.055em; max-width: 660px; margin: 0 0 22px; }
    .intro { color: var(--muted); font-size: 16px; line-height: 1.65; max-width: 540px; margin-bottom: 42px; }
    .panel { background: white; border: 1px solid var(--line); border-radius: 18px; padding: 24px; height: 100%; }
    .panel-heading { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
    .panel-title { font-weight: 800; font-size: 15px; }
    .panel-meta { color: var(--muted); font: 400 11px 'DM Mono', monospace; }
    .canvas-wrap { background: #0e1512; border-radius: 12px; padding: 12px; display: flex; justify-content: center; }
    .canvas-wrap canvas { border-radius: 7px; cursor: crosshair; max-width: 100%; height: auto !important; }
    .result-label { color: var(--muted); font: 500 11px 'DM Mono', monospace; letter-spacing: .12em; text-transform: uppercase; margin-top: 4px; }
    .digit { font-size: clamp(6rem, 13vw, 10rem); font-weight: 800; line-height: .9; letter-spacing: -.08em; color: var(--ink); margin: 13px 0 17px; }
    .confidence { display: flex; align-items: baseline; gap: 10px; padding-bottom: 25px; border-bottom: 1px solid var(--line); }
    .confidence-value { color: var(--green); font-size: 25px; font-weight: 800; }
    .confidence-text { color: var(--muted); font-size: 13px; }
    .bars-title { font: 500 11px 'DM Mono', monospace; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin: 25px 0 15px; }
    .bar-row { display: grid; grid-template-columns: 18px 1fr 49px; gap: 10px; align-items: center; margin: 10px 0; font: 500 12px 'DM Mono', monospace; }
    .bar-row.active { font-weight: 500; color: var(--green); }
    .bar-track { background: #edf2ef; height: 7px; overflow: hidden; border-radius: 4px; }
    .bar-fill { background: #abc9b8; height: 100%; border-radius: inherit; transition: width .18s ease-out; }
    .bar-row.active .bar-fill { background: var(--coral); }
    .bar-percent { text-align: right; color: var(--muted); }
    .bar-row.active .bar-percent { color: var(--green); }
    div.stButton > button { background: var(--ink); color: white; border: 0; border-radius: 9px; padding: 11px 22px; font-weight: 700; width: 100%; margin-top: 18px; }
    div.stButton > button:hover { background: var(--green); color: white; border: 0; }
    .hint { color: var(--muted); font-size: 12px; text-align: center; margin-top: 14px; }
    @media (max-width: 700px) { .block-container { padding: 32px 20px 45px; } .brand { margin-bottom: 38px; } h1 { font-size: 3rem; } .panel { padding: 16px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="brand"><span class="brand-mark">✦</span><span class="brand-name">Digit Lens</span></div>', unsafe_allow_html=True)
st.markdown('<p class="eyebrow">Live neural vision</p>', unsafe_allow_html=True)
st.markdown("<h1>Draw a digit.<br>Watch it think.</h1>", unsafe_allow_html=True)
st.markdown('<p class="intro">A live window into your MNIST classifier. Every brush stroke is translated into a prediction as you draw.</p>', unsafe_allow_html=True)

try:
    load_model()
except Exception as error:
    st.error(f"Could not load the CNN model: {error}")
    st.stop()

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

left, right = st.columns([1.04, 0.96], gap="large")
with left:
    st.markdown('<div class="panel-heading"><span class="panel-title">Your canvas</span><span class="panel-meta">28 × 28 INPUT</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="canvas-wrap">', unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=19,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"canvas-{st.session_state.canvas_version}",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Clear", type="primary"):
        st.session_state.canvas_version += 1
        st.rerun()
    st.markdown('<p class="hint">Draw anywhere inside the square</p>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if canvas.image_data is None or not np.any(canvas.image_data[:, :, 3] > 0):
        probabilities = np.zeros(10, dtype="float32")
        probabilities[0] = 1.0
        has_drawing = False
    else:
        probabilities = predict(canvas.image_data)
        has_drawing = True

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])
    label = "Predicted digit" if has_drawing else "Ready to recognize"
    st.markdown(f'<div class="result-label">{label}</div><div class="digit">{predicted_digit}</div><div class="confidence"><span class="confidence-value">{confidence * 100:.1f}%</span><span class="confidence-text">confidence</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="bars-title">Probability distribution</div>', unsafe_allow_html=True)
    bars = []
    for digit, probability in enumerate(probabilities):
        active = " active" if digit == predicted_digit and has_drawing else ""
        bars.append(f'<div class="bar-row{active}"><span>{digit}</span><div class="bar-track"><div class="bar-fill" style="width: {probability * 100:.2f}%"></div></div><span class="bar-percent">{probability * 100:.1f}%</span></div>')
    st.markdown("".join(bars), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
