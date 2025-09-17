import streamlit as st
from PIL import Image
import torch
import clip

# -----------------------
# 1. Load CLIP model
# -----------------------
@st.cache(allow_output_mutation=True)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# -----------------------
# 2. Material classes
# -----------------------
material_classes = [
    "Plastic bottle",
    "Cardboard box",
    "Foil pack",
    "Paper bag",
    "Metal can",
    "Glass bottle",
]

material_bins = {
    "Plastic bottle": ("Plastic", "Recycle Bin A"),
    "Cardboard box": ("Paper/Cardboard", "Recycle Bin B"),
    "Foil pack": ("Metal/Foil", "Recycle Bin C"),
    "Paper bag": ("Paper/Cardboard", "Recycle Bin B"),
    "Metal can": ("Metal/Foil", "Recycle Bin C"),
    "Glass bottle": ("Glass", "Recycle Bin D"),
}

# -----------------------
# 3. Streamlit UI
# -----------------------
st.set_page_config(page_title="TrashDNA: AI Material Classification", layout="wide")
st.title("♻️ TrashDNA: Smart Recycling Demo")
st.markdown(
    "Upload any packaging image, and our AI will identify the material and suggest the correct recycling bin. Points awarded for each correct identification!"
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Identifying material..."):
        # Preprocess and predict using CLIP
        image_input = preprocess(img).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in material_classes]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarities[0].topk(1)
            predicted_material = material_classes[indices[0]]

    # Display results
    if predicted_material in material_bins:
        material, bin_name = material_bins[predicted_material]
        points = 10
        st.success(f"Predicted Material: **{material}** ✅")
        st.info(f"Place in: **{bin_name}**")
        st.balloons()
        st.markdown(f"<h3 style='color:green;'>Points Awarded: {points} 🎉</h3>", unsafe_allow_html=True)
    else:
        st.warning("Material not recognized. Try another image.")

# -----------------------
# Optional: Footer
# -----------------------
st.markdown("---")
st.markdown("💡 Demo by TrashDNA | TRBS 2025")
