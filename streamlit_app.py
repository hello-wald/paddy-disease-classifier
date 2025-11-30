import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import streamlit as st
from PIL import Image, UnidentifiedImageError
import pandas as pd
import altair as alt

from modelling.models.classifier import Classifier
from modelling.utils.transforms import get_transforms
import modelling.config as config


# -----------------------------------------------------------------------
# 🔧 Streamlit Page Config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Paddy Disease Classifier",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------------------------------------------------
# 🔧 Load Model + Transforms
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = Classifier(
        num_classes=config.NUM_CLASSES,
        model_name=config.MODEL_NAME,
        classifier_units=config.CLASSIFIER_UNITS,
        activation_function=config.ACTIVATION_FUNCTION,
        use_pretrained=False,
    )
    weights = torch.load(
        os.path.join("modelling", config.MODEL_SAVE_PATH),
        map_location="cpu"
    )
    model.load_state_dict(weights)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_val_transform():
    _, val_t = get_transforms()
    return val_t


model = load_model()
val_transform = load_val_transform()
class_names = config.CLASS_NAMES


# -----------------------------------------------------------------------
# 🌾 Header Section
# -----------------------------------------------------------------------
st.markdown(
    f"""
    <h1 style="text-align:center; margin-bottom:0;">🌾 Paddy Disease Classifier</h1>

    <p style="text-align:center; font-size:1.05rem; color:gray; margin-top:6px;">
        Upload a rice leaf image to detect the disease class using a fine-tuned EfficientNet model.
    </p>

    <p style="text-align:center; font-size:0.95rem; color:#666; max-width:700px; margin:auto;">
        This model can identify the following paddy leaf conditions:
        <b>{', '.join(class_names)}</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

left, _, right = st.columns([0.8, 0.1, 1.1])


# -----------------------------------------------------------------------
# LEFT COLUMN
# -----------------------------------------------------------------------
with left:
    st.markdown(
        """
        <div style="display:flex; justify-content:center;">
            <div style="width:420px;">
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a rice leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Stop if no image
    if not uploaded_file:
        st.stop()

    # Load image safely
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ Invalid image file. Please upload JPG, PNG, or WEBP.")
        st.stop()

    st.markdown(
        "<div style='display:flex; justify-content:center; margin-top:20px;'>",
        unsafe_allow_html=True,
    )
    st.image(image, caption="Uploaded Image", output_format="PNG", width=420)
    st.markdown("</div>", unsafe_allow_html=True)

    raw_results_placeholder = st.container()


# -----------------------------------------------------------------------
# RUN PREDICTION
# -----------------------------------------------------------------------
img_tensor = val_transform(image).unsqueeze(0)
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

# Most likely class
idx = int(probs.argmax())
pred_class = class_names[idx]
confidence = float(probs[idx])

df = pd.DataFrame({"Disease": class_names, "Probability": probs})
df_sorted = df.sort_values("Probability", ascending=False)


# -----------------------------------------------------------------------
# RIGHT COLUMN
# -----------------------------------------------------------------------
with right:
    st.subheader(f"🔍 Prediction: **{pred_class}**")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Chart
    st.subheader("📊 Probability Distribution")
    chart = (
        alt.Chart(df_sorted)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Disease:N", sort="-x"),
            color="Disease:N",
            tooltip=[
                alt.Tooltip("Disease:N"),
                alt.Tooltip("Probability:Q", format=".3f")
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, width="stretch")

    # Table
    st.subheader("📋 Probability Table")
    st.dataframe(
        df_sorted.style.format({"Probability": "{:.3f}"}),
        width="stretch",
    )


# -----------------------------------------------------------------------
# RAW RESULTS (LEFT COLUMN)
# -----------------------------------------------------------------------
with raw_results_placeholder:
    st.subheader("📘 Raw Probabilities")
    st.write({cls: f"{p:.3f}" for cls, p in zip(class_names, probs)})
