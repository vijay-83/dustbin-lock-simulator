import streamlit as st
from PIL import Image
import io
import torch
from torchvision import models, transforms
import numpy as np

def main():
    # dustbin_simulator_streamlit_torch.py
    # Dustbin Lock Simulator using PyTorch + Streamlit (no TensorFlow required)
    # Categories: veg, paper, plastic_metal, glass, other
    # Run: pip install streamlit torch torchvision pillow numpy
    # Then: streamlit run dustbin_simulator_streamlit_torch.py
    # -----------------------------
    # Model setup
    # -----------------------------
    @st.cache_resource
    def load_torch_model():
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return model, preprocess

    model, preprocess = load_torch_model()

    # ImageNet index-to-label mapping
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    imagenet_labels = MobileNet_V2_Weights.IMAGENET1K_V1.meta['categories']

    IMAGENET_TO_CATEGORY = {
        'banana': 'veg', 'apple': 'veg', 'orange': 'veg', 'lemon': 'veg', 'pineapple': 'veg',
        'cucumber': 'veg', 'fig': 'veg', 'bell_pepper': 'veg', 'broccoli': 'veg', 'cauliflower': 'veg',
        'carrot': 'veg', 'mushroom': 'veg', 'potato': 'veg', 'strawberry': 'veg', 'granny_smith': 'veg',
        'spoon': 'plastic_metal', 'fork': 'plastic_metal', 'bottle': 'plastic_metal', 'water_bottle': 'plastic_metal',
        'beer_bottle': 'plastic_metal', 'wine_bottle': 'glass', 'glass': 'glass', 'wine_glass': 'glass',
        'beer_glass': 'glass', 'cup': 'glass',
        'envelope': 'paper', 'notebook': 'paper', 'paper_towel': 'paper', 'newspaper': 'paper', 'book': 'paper',
        'tray': 'plastic_metal', 'can': 'plastic_metal', 'tin': 'plastic_metal', 'steel_drum': 'plastic_metal',
    }

    CATEGORY_LABELS = {
        'veg': 'Vegetable / Food Waste',
        'paper': 'Paper',
        'plastic_metal': 'Bottle / Stainless Steel / Plastic / Metal',
        'glass': 'Glass',
        'other': 'Other / Unknown'
    }

    # -----------------------------
    # Streamlit UI setup
    # -----------------------------

    st.set_page_config(page_title='Dustbin Lock Simulator', layout='centered')
    st.title('🗑️ Dustbin Lock Simulator (PyTorch Version)')
    st.write('Upload or capture an image — the simulator will detect waste type and open the correct bin lid.')

    col1, col2 = st.columns([2,1])
    with col1:
        img_file = st.file_uploader('Upload image', type=['png','jpg','jpeg'])
        camera_img = st.camera_input('Or take a photo')

    with col2:
        st.markdown('**Instructions:**')
        st.markdown('- Use clear object photos.\n- Works best for single items.\n- Example: banana, glass, paper sheet, metal bottle.')

    image = None
    if img_file:
        image = Image.open(img_file).convert('RGB')
    elif camera_img:
        image = Image.open(io.BytesIO(camera_img.getvalue())).convert('RGB')

    if image is None:
        st.info('Please upload or capture an image to start.')
        st.stop()

    st.image(image, caption='Input image', use_column_width=True)

    # -----------------------------
    # Prediction logic
    # -----------------------------

    def predict_image(img):
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            results = [(imagenet_labels[i], float(top5_prob[j])) for j, i in enumerate(top5_catid)]
        return results

    results = predict_image(image)

    # Map label to category
    def map_label_to_category(results):
        for label, prob in results:
            label_low = label.lower()
            for keyword, cat in IMAGENET_TO_CATEGORY.items():
                if keyword in label_low:
                    return cat, label, prob
            if 'bottle' in label_low:
                return 'plastic_metal', label, prob
            if 'glass' in label_low:
                return 'glass', label, prob
            if 'paper' in label_low or 'book' in label_low or 'notebook' in label_low:
                return 'paper', label, prob
            if any(x in label_low for x in ['vegetable','fruit','banana','apple','orange','tomato','carrot','pepper','potato']):
                return 'veg', label, prob
        return 'other', results[0][0], results[0][1]

    category, matched_label, probability = map_label_to_category(results)

    st.markdown('---')
    st.subheader('Top Predictions:')
    for i, (label, prob) in enumerate(results, start=1):
        st.write(f"{i}. **{label}** — {prob*100:.1f}%")

    st.markdown('---')
    st.subheader('Detected Category:')
    st.write(f"**{CATEGORY_LABELS.get(category,'Other')}** (matched label: *{matched_label}*, confidence {probability*100:.1f}%)")

    # -----------------------------
    # Dustbin Lid Simulation
    # -----------------------------

    st.markdown('### Dustbin Lid Simulation')
    if category == 'veg':
        st.balloons()
        st.success('✅ Organic lid opened!')
    elif category == 'paper':
        st.success('✅ Paper lid opened!')
    elif category == 'plastic_metal':
        st.success('✅ Plastic/Metal lid opened!')
    elif category == 'glass':
        st.success('✅ Glass lid opened!')
    else:
        st.warning('⚠️ Unknown type — manual check recommended.')
    st.markdown('---')
    st.caption('This PyTorch version avoids TensorFlow installation issues and uses MobileNetV2 pretrained on ImageNet for basic category inference. For real deployments, fine-tune on a custom waste dataset.')


if __name__ == "__main__":
    main()
