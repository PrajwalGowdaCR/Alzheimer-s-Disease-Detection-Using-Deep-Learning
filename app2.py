import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_option_menu import option_menu
import re
import base64
from fpdf import FPDF

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer Detection",
    page_icon="🧠",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    button.step-up {display: none;}
    button.step-down {display: none;}
    div[data-baseweb] {border-radius: 4px;}
</style>""", unsafe_allow_html=True)


def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    try:
        bin_str = get_base64(png_file)
        st.markdown(
            f"""<style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-position: center;
                background-size: cover;
            }}
            </style>""",
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass   # background is optional; skip silently if missing


set_background("./images/bg3.png")

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Class labels — must match the order the model was trained with
CLASS_LABELS = ["Non Demented", "Very Mild Demented",
                "Mild Demented", "Moderate Demented"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((160, 160))   # matches MobileNetV2 training size
    arr   = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def validate_phone_number(phone_number: str) -> bool:
    pattern = r"^\d{10}$"
    if not re.match(pattern, str(phone_number)):
        st.error("Please enter a 10-digit number!")
        return False
    return True


def validate_name(name: str) -> bool:
    if not all(ch.isalpha() or ch.isspace() for ch in name):
        st.error("Name should not contain numbers or special characters.")
        return False
    return True


def validate_input(name, age, contact, file) -> bool:
    if not name:
        st.error("Please enter the patient's name!")
        return False
    if not age:
        st.error("Please enter the age!")
        return False
    if not contact:
        st.error("Please enter a contact number!")
        return False
    if file is None:
        st.error("Please upload an MRI scan!")
        return False
    return True


def create_download_link(val: bytes, filename: str) -> str:
    b64 = base64.b64encode(val).decode()
    return (f'<a href="data:application/octet-stream;base64,{b64}" '
            f'download="{filename}.pdf">⬇ Download Report (PDF)</a>')


# ── Navigation ────────────────────────────────────────────────────────────────
selected = option_menu(
    menu_title=None,
    options=["Home", "Alzheimer Detection", "About Us"],
    icons=["house", "activity", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0f1932"},
        "icon":      {"color": "#00c8ff", "font-size": "16px"},
        "nav-link":  {"font-size": "15px", "color": "white",
                      "--hover-color": "#1a2a50"},
        "nav-link-selected": {"background-color": "#1a4080"},
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Home":
    st.title("🧠 Alzheimer's Disease Detection")
    st.markdown("""
Alzheimer's disease is the most common type of dementia. It is a progressive
disease beginning with mild memory loss that may eventually lead to loss of
the ability to carry on a conversation and respond to the environment.

**Using this app**, you can upload an MRI brain scan and instantly find out
which stage of Alzheimer's disease — if any — is present.

### Four stages detected:
| # | Stage |
|---|-------|
| 1 | Non Demented |
| 2 | Very Mild Demented |
| 3 | Mild Demented |
| 4 | Moderate Demented |
""")

# ─────────────────────────────────────────────────────────────────────────────
# ABOUT US
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "About Us":
    st.title("Welcome!")
    st.write(
        "This web app uses a CNN model (MobileNetV2 transfer learning) to "
        "recognise the presence of Alzheimer's disease across four stages. "
        "Upload an MRI brain scan and receive an instant prediction along "
        "with a downloadable PDF report."
    )
    st.info("This is a mini-project for educational purposes only. "
            "Always consult a qualified medical professional for diagnosis.")

# ─────────────────────────────────────────────────────────────────────────────
# ALZHEIMER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Alzheimer Detection":
    st.title("🔬 Alzheimer Detection")
    st.write("Enter patient details and upload an MRI brain scan.")

    with st.form(key="myform", clear_on_submit=False):
        name    = st.text_input("Patient Name")
        age     = st.number_input("Age", min_value=1, max_value=150, value=40)
        gender  = st.radio("Gender", ("Male", "Female", "Other"))
        contact = st.text_input("Contact Number (10 digits)")
        file    = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
        submit  = st.form_submit_button("🔍 Analyse")

    if submit:
        if (
            validate_input(name, age, contact, file)
            and validate_phone_number(contact)
            and validate_name(name)
        ):
            st.success("Personal information recorded ✅")

            # ── Show uploaded image ──────────────────────────────────────────
            pil_image = Image.open(file)
            png_image = pil_image.convert("RGBA")
            st.image(pil_image, caption="Uploaded MRI Scan", width=220)

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {name}")
                st.write(f"**Age:** {age}")
            with col2:
                st.write(f"**Gender:** {gender}")
                st.write(f"**Contact:** {contact}")

            # ── Predict ─────────────────────────────────────────────────────
            with st.spinner("Analysing scan …"):
                arr        = preprocess_image(pil_image)
                probs      = model.predict(arr, verbose=0)[0]
                pred_idx   = int(np.argmax(probs))
                pred_label = CLASS_LABELS[pred_idx]
                confidence = float(probs[pred_idx])

            # ── Result ──────────────────────────────────────────────────────
            st.markdown("---")
            if pred_idx == 0:   # Non Demented
                st.success(f"✅ Prediction: **{pred_label}**  ({confidence:.1%} confidence)")
            else:
                st.error(f"⚠️ Prediction: **{pred_label}**  ({confidence:.1%} confidence)")

            # Confidence bar chart
            st.markdown("#### Confidence per class")
            import pandas as pd
            prob_df = pd.DataFrame(
                {"Class": CLASS_LABELS, "Confidence": probs}
            ).set_index("Class")
            st.bar_chart(prob_df)

            # ── PDF Report ──────────────────────────────────────────────────
            st.markdown("---")
            if st.button("📄 Export PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_draw_color(0, 0, 0)
                pdf.set_line_width(1)
                pdf.rect(5.0, 5.0, 200.0, 287.0, "D")

                pdf.set_font("Times", "B", 24)
                pdf.cell(200, 20, "Alzheimer Detection Report", 0, 1, "C")

                pdf.set_font("Arial", "B", 16)
                pdf.cell(200, 10, "Patient Details", 0, 1)

                pdf.set_font("Arial", "", 12)
                pdf.cell(200, 10, f"Name: {name}", 0, 1)
                pdf.cell(200, 10, f"Age: {int(age)}", 0, 1)
                pdf.cell(200, 10, f"Gender: {gender}", 0, 1)
                pdf.cell(200, 10, f"Contact: {contact}", 0, 1)
                pdf.ln(4)

                # Embed MRI image
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                png_image.save(tmp_path, "PNG")
                pdf.cell(200, 10, "MRI Scan:", 0, 1)
                pdf.image(tmp_path, x=40, y=80, w=50, h=50)
                os.unlink(tmp_path)

                pdf.ln(55)
                pdf.set_font("Arial", "B", 16)
                pdf.cell(200, 10,
                         f"Prediction: {pred_label}  ({confidence:.1%})",
                         0, 1)
                pdf.ln(4)

                pdf.set_font("Arial", "B", 12)
                if pred_idx != 0:
                    pdf.set_text_color(255, 0, 0)
                    pdf.cell(200, 10,
                             "Dementia detected. Consult a neurologist immediately.",
                             0, 1)
                    pdf.set_text_color(0, 0, 200)
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(200, 10, "Precautions you can take:", 0, 1)
                    pdf.ln(2)
                    precautions = [
                        "1. Stay mentally active with reading, puzzles, and games.",
                        "2. Exercise regularly to improve brain blood flow.",
                        "3. Eat a balanced diet rich in fruits and vegetables.",
                        "4. Maintain social connections to prevent isolation.",
                        "5. Aim for 7-8 hours of sleep per night.",
                    ]
                    pdf.set_font("Arial", "", 11)
                    for p in precautions:
                        pdf.multi_cell(190, 9, p, 0, "L")
                        pdf.ln(1)
                else:
                    pdf.set_text_color(0, 160, 0)
                    pdf.cell(200, 10,
                             "No signs of dementia detected in this MRI.",
                             0, 1)

                html = create_download_link(
                    pdf.output(dest="S").encode("latin-1"), "alzheimer_report"
                )
                st.markdown(html, unsafe_allow_html=True)
