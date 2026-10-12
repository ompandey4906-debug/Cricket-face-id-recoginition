
import streamlit as st
import os
import sys
import tempfile

# Add the src folder to Python's import path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_PATH, "src")

sys.path.insert(0, SRC_PATH)

from predict import predict_player


st.set_page_config(
    page_title="Indian Cricket Face ID",
    page_icon="🏏",
    layout="wide"
)


st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)



st.markdown(
    '<div class="main-title">🏏 Indian Cricket Face ID</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Indian cricket player face identification</div>',
    unsafe_allow_html=True
)



st.sidebar.title("About the Project")

st.sidebar.write(
    """
This application identifies selected Indian cricket
players using facial features extracted from an input image.
"""
)

st.sidebar.subheader("Pipeline")

st.sidebar.write(
    """
1. Face Detection — MTCNN
2. Face Embedding — FaceNet
3. Classification — SVM
4. Unknown Detection — Similarity Threshold
"""
)

st.sidebar.subheader("Players")

players = [
    "Virat Kohli",
    "Rohit Sharma",
    "Jasprit Bumrah",
    "Hardik Pandya",
    "Ravindra Jadeja",
    "Rishabh Pant",
    "KL Rahul",
    "Shubman Gill",
    "Mohammed Siraj",
    "Suryakumar Yadav"
]

for player in players:
    st.sidebar.write("• " + player)



st.header("Upload an image")

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "gif"]
)




if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=400
    )

    if st.button("🔍 Identify Player"):

        with st.spinner("Analyzing the face..."):

            temp_path = None

            try:

                # Save uploaded image temporarily
                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                # Predict
                result = predict_player(
                    temp_path
                )

                prediction = result[
                    "prediction"
                ]

                similarity = result[
                    "similarity"
                ]


                # -------------------------
                # Display result
                # -------------------------

                if prediction == "No face detected":

                    st.error(
                        "❌ No face detected in the image."
                    )


                elif prediction == "UNKNOWN":

                    st.warning(
                        "⚠️ Unknown person"
                    )

                    st.write(
                        f"Similarity: {similarity * 100:.2f}%"
                    )


                else:

                    st.success(
                        f"🏏 Identified Player: {prediction.replace('_', ' ')}"
                    )

                    st.write(
                        f"Face similarity: {similarity * 100:.2f}%"
                    )

                    st.progress(
                        min(float(similarity), 1.0)
                    )


            except Exception as e:

                st.error(
                    "An error occurred while processing the image."
                )

                st.exception(e)


            finally:

                if temp_path and os.path.exists(
                    temp_path
                ):

                    os.remove(temp_path)



st.markdown("---")

st.caption(
    "Built with Python • Streamlit • MTCNN • FaceNet • SVM"
)