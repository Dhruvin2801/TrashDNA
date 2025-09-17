import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

# -----------------------
# 1. Load CLIP model
# -----------------------
@st.cache(allow_output_mutation=True)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = load_model()

# -----------------------
# 2. Material classes
# -----------------------
material_classes = [
    "Plastic bottle",
    "Cardboard box",
    "Foil pack",
    "Paper bag",
    "Metal can",
    "Glass bottle"
]

material_bins = {
    "Plastic bottle": ("Plastic", "Recycle Bin A", "icons/plastic.jpg"),
    "Cardboard box": ("Paper/Cardboard", "Recycle Bin B", "icons/cardboard.jpg"),
    "Foil pack": ("Metal/Foil", "Recycle Bin C", "icons/foil.jpg"),
    "Paper bag": ("Paper/Cardboard", "Recycle Bin B", "icons/cardboard.jpg"),
    "Metal can": ("Metal/Foil", "Recycle Bin C", "icons/metal.jpg"),
    "Glass bottle": ("Glass", "Recycle Bin D", "icons/glass.jpg")
}

# -----------------------
# 3. Streamlit UI Setup
# -----------------------
st.set_page_config(page_title="TrashDNA Smart Recycling", layout="wide")
st.markdown("<h1 style='text-align:center; color:green;'>♻️ TrashDNA: Smart Recycling Demo</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Upload any packaging image and see AI classify material and assign recycling bin!</h4>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session scoreboard
if "total_points" not in st.session_state:
    st.session_state.total_points = 0

# Upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # -----------------------
    # 4. Layout: Image left, Prediction right
    # -----------------------
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing image..."):
            inputs = processor(text=[f"a photo of a {c}" for c in material_classes],
                               images=img, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            pred_idx = probs.argmax().item()
            predicted_material = material_classes[pred_idx]

        # Display results
        if predicted_material in material_bins:
            material, bin_name, icon_path = material_bins[predicted_material]
            points = 10
            st.session_state.total_points += points

            # Display bin icon and prediction
            st.markdown(f"### ✅ Predicted Material: **{material}**")
            st.image(icon_path, width=100)
            st.markdown(f"### 📦 Place in: **{bin_name}**")
            st.balloons()
            st.markdown(f"<h3 style='color:green;'>Points Awarded: {points} 🎉</h3>", unsafe_allow_html=True)
        else:
            st.warning("Material not recognized. Try another image.")

# -----------------------
# 5. Display total points scoreboard
# -----------------------
st.markdown("---")
st.markdown(f"<h2 style='text-align:center; color:blue;'>🏆 Total Points this session: {st.session_state.total_points}</h2>", unsafe_allow_html=True)

