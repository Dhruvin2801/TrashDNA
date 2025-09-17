import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------------
# 1. Load CLIP (Hugging Face)
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
    "Plastic bottle": ("Plastic", "Recycle Bin A"),
    "Cardboard box": ("Paper/Cardboard", "Recycle Bin B"),
    "Foil pack": ("Metal/Foil", "Recycle Bin C"),
    "Paper bag": ("Paper/Cardboard", "Recycle Bin B"),
    "Metal can": ("Metal/Foil", "Recycle Bin C"),
    "Glass bottle": ("Glass", "Recycle Bin D")
}

# -----------------------
# 3. Streamlit UI
# -----------------------
st.set_page_config(page_title="TrashDNA Demo", layout="wide")
st.title("♻️ TrashDNA: Smart Recycling Demo (CLIP)")
st.write("Upload any packaging image. AI predicts material, correct bin, and awards points!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Identifying material..."):
        inputs = processor(text=[f"a photo of a {c}" for c in material_classes], images=img, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred_idx = probs.argmax().item()
        predicted_material = material_classes[pred_idx]

    if predicted_material in material_bins:
        material, bin_name = material_bins[predicted_material]
        points = 10
        st.success(f"Predicted Material: **{material}** ✅")
        st.info(f"Place in: **{bin_name}**")
        st.balloons()
        st.markdown(f"<h3 style='color:green;'>Points Awarded: {points} 🎉</h3>", unsafe_allow_html=True)
    else:
        st.warning("Material not recognized. Try another image.")
