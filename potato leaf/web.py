import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    .stSidebar {
        background-color: #2c3e50;
        color: white;
    }
    .stFileUploader>div>div>div>div {
        color: #2c3e50;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load the model and make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model("Test_model_o11.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar for navigation
st.sidebar.title("🌱 Plant Disease Recognition")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.markdown("<h1 class='stHeader'>Plant Disease Recognition for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### Welcome to the Plant Disease Recognition App!
    This app helps farmers and agricultural enthusiasts identify diseases in potato plants using AI.
    - **How to Use:**
        1. Go to the **Disease Recognition** page.
        2. Upload an image of a potato leaf.
        3. Click **Predict** to see the result.
    """)
    st.image("ecosystem.jpg", use_column_width=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 class='stHeader'>🌿 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # File uploader for image
    test_image = st.file_uploader("Upload an image of a potato leaf", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("🔍 Analyzing the leaf..."):
                try:
                    result_index = model_prediction(test_image)
                    class_name = ['Potato__Early_blight', 'Potato__healthy', 'Potato__Late_blight']
                    prediction = class_name[result_index]

                    # Display result with dynamic emojis
                    if prediction == "Potato__healthy":
                        st.snow()  # Celebrate healthy prediction
                        st.markdown(
                            f"<div class='stSuccess'>🎉 Yay! The leaf is <strong>healthy</strong>! 🌱</div>",
                            unsafe_allow_html=True
                        )
                    elif prediction == "Potato__Early_blight":
                        st.markdown(
                            f"<div class='stWarning'>⚠️ The leaf has <strong>Early Blight</strong>. Take action! 🍂</div>",
                            unsafe_allow_html=True
                        )
                    elif prediction == "Potato__Late_blight":
                        st.markdown(
                            f"<div class='stError'>🚨 The leaf has <strong>Late Blight</strong>. Immediate action required! 🍁</div>",
                            unsafe_allow_html=True
                        )

                    # Show additional tips based on prediction
                    if prediction == "Potato__Early_blight":
                        st.markdown("""
                        ### Tips for Early Blight:
                        - Remove infected leaves.
                        - Apply fungicides like chlorothalonil or copper-based sprays.
                        - Ensure proper spacing between plants for air circulation.
                        """)
                    elif prediction == "Potato__Late_blight":
                        st.markdown("""
                        ### Tips for Late Blight:
                        - Remove and destroy infected plants.
                        - Apply fungicides like mancozeb or metalaxyl.
                        - Avoid overhead watering to reduce humidity.
                        """)
                    elif prediction == "Potato__healthy":
                        st.markdown("""
                        ### Tips for Healthy Plants:
                        - Continue regular watering and fertilization.
                        - Monitor for pests and diseases.
                        - Maintain proper spacing and sunlight exposure.
                        """)

                except Exception as e:
                    st.markdown(
                        f"<div class='stError'>❌ Error: {str(e)}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                "<div class='stWarning'>⚠️ Please upload an image first.</div>",
                unsafe_allow_html=True
            )