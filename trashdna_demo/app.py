import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# -----------------------
# 1. Load pre-trained model
# -----------------------
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

# -----------------------
# 2. Define preprocessing
# -----------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# 3. Define material-bin mapping (simplified)
# -----------------------
material_bins = {
    "plastic_bottle": ("Plastic", "Recycle Bin A"),
    "cardboard_box": ("Paper/Cardboard", "Recycle Bin B"),
    "foil_pack": ("Metal/Foil", "Recycle Bin C")
}

# -----------------------
# 4. Streamlit UI
# -----------------------
st.title("TrashDNA: AI Material Classification Demo")
st.write("Upload an image of packaging to identify its material and the correct recycling bin.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    input_tensor = preprocess(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = outputs.max(1)
    
    # Map predicted index to simplified label (for demo purposes)
    # In practice, you can map more classes or train a custom dataset
    # Here we just simulate mapping for demo images
    filename = uploaded_file.name.lower()
    if "plastic" in filename:
        label = "plastic_bottle"
    elif "cardboard" in filename:
        label = "cardboard_box"
    elif "foil" in filename:
        label = "foil_pack"
    else:
        label = "unknown"

    if label in material_bins:
        material, bin_name = material_bins[label]
        points = 10  # simple reward system
        st.success(f"Predicted Material: **{material}**")
        st.info(f"Place in: **{bin_name}**")
        st.balloons()
        st.write(f"Points Awarded: **{points}** 🎉")
    else:
        st.warning("Material not recognized. Try another image.")
