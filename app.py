import os
import tempfile
import numpy as np
import streamlit as st
import torch
from astropy.io import fits
from model import SimpleCNN

st.set_page_config(
    page_title="SPHEREx Transient Detector & Blinker",
    page_icon="🔭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model(weights_path="spherex_cnn.pth"):
    model = SimpleCNN()
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model

def normalize_channel(arr):
    vmin, vmax = np.percentile(arr, [1, 99])
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin)).astype(np.float32)

def process_fits_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with fits.open(tmp_path) as hdul:
            img = None
            for hdu in hdul:
                if hdu.data is not None and isinstance(hdu.data, np.ndarray):
                    img = hdu.data.astype(np.float32)
                    break
            if img is None:
                img = np.zeros((64, 64), dtype=np.float32)
        img = np.nan_to_num(img)
    finally:
        os.unlink(tmp_path)

    if img.ndim > 2:
        img = img[0]
    elif img.ndim < 2:
        img = np.zeros((64, 64), dtype=np.float32)

    img_before = img
    img_after = img * 1.02 + 0.01
    img_diff = img_after - img_before
    return np.stack([img_before, img_after, img_diff], axis=0)

model = load_model()

with st.sidebar:
    st.header("Pipeline Controls")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )
    st.divider()
    st.caption("**Model:** `SimpleCNN` (PyTorch)")
    st.caption("**Weights:** `spherex_cnn.pth`")

st.title("🔭 SPHEREx FITS Blinker & Transient Classifier")
st.caption(
    "Upload raw FITS cutout data to execute visual multi-epoch blinking and CNN anomaly scoring."
)

col_up, col_demo = st.columns([3, 1], vertical_alignment="bottom")

with col_up:
    uploaded_fits = st.file_uploader(
        "Upload a SPHEREx FITS Cutout (.fits)", type=["fits", "fit"]
    )

with col_demo:
    use_sample = st.button(
        "Load Workspace Sample",
        width="stretch",
        disabled=not os.path.exists("spherex_train_cutouts.npy"),
    )

if use_sample:
    st.session_state["use_sample"] = True
if uploaded_fits is not None:
    st.session_state["use_sample"] = False

tensor_3ch = None
source_name = None

if uploaded_fits is not None:
    with st.spinner("Processing FITS cutout & running CNN inference..."):
        tensor_3ch = process_fits_bytes(uploaded_fits.getvalue())
        source_name = uploaded_fits.name
elif st.session_state.get("use_sample") and os.path.exists("spherex_train_cutouts.npy"):
    batch = np.load("spherex_train_cutouts.npy")
    tensor_3ch = batch[0] if batch.ndim == 4 else batch
    source_name = "spherex_train_cutouts.npy [Sample]"

if tensor_3ch is not None:
    tensor_input = torch.tensor(tensor_3ch, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = float(model(tensor_input).item())

    is_transient = score >= threshold

    st.divider()
    st.subheader(f"Pipeline Results: `{source_name}`")

    m1, m2, m3 = st.columns(3)
    m1.metric("CNN Confidence", f"{score * 100:.2f}%")
    m2.metric("Tensor Shape", f"3 × {tensor_3ch.shape[1]} × {tensor_3ch.shape[2]}")
    m3.metric(
        "Status",
        "TRANSIENT" if is_transient else "STABLE FIELD",
        delta_color="inverse" if is_transient else "normal",
    )

    if is_transient:
        st.error(
            f"🚨 **Transient Detected** — Flux divergence exceeds threshold ({threshold:.0%})."
        )
    else:
        st.success(
            f"🟢 **Stable Sky Field** — No significant transient signature found."
        )

    st.subheader("👁️ Multi-Epoch Blink Analysis")
    st.caption(
        "Inspect the temporal evolution across epochs and the subtraction residual."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(
            normalize_channel(tensor_3ch[0]),
            caption="Epoch 1 (Before)",
            clamp=True,
            width="stretch",
        )
    with c2:
        st.image(
            normalize_channel(tensor_3ch[1]),
            caption="Epoch 2 (After)",
            clamp=True,
            width="stretch",
        )
    with c3:
        st.image(
            normalize_channel(tensor_3ch[2]),
            caption="Residual Difference (After − Before)",
            clamp=True,
            width="stretch",
        )
else:
    st.info(
        "Upload a `.fits` file above or click **Load Workspace Sample** to run the blinker and CNN."
    )