import tempfile

import streamlit as st

from semantic_core.run_image import run_image_pipeline
from semantic_core.run_text import run_text_pipeline
from semantic_core.run_video import run_video_pipeline


st.set_page_config(
    page_title="Semantic AI 6G System",
    layout="wide"
)

st.title("Multimodal Semantic Communication System")
st.write("Image + Text + Video Semantic Communication")

module = st.selectbox(
    "Choose Module",
    ["Image", "Text", "Video"]
)

noise_type = st.selectbox(
    "Channel Impairment",
    ["awgn", "packet_loss"]
)

noise_level = st.slider(
    "Noise Level / Packet Loss Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05
)


if module == "Image":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        suffix = ".jpg"

        if uploaded_file.name.lower().endswith(".png"):
            suffix = ".png"
        elif uploaded_file.name.lower().endswith(".jpeg"):
            suffix = ".jpeg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            image_path = temporary_file.name

        st.image(
            image_path,
            caption="Uploaded Image",
            width=400
        )

        if st.button("Run Image Semantic Communication"):
            try:
                with st.spinner(
                    "Extracting image semantic features and simulating transmission..."
                ):
                    result = run_image_pipeline(
                        image_path=image_path,
                        noise_type=noise_type,
                        noise_level=noise_level
                    )

                st.success("Image pipeline completed successfully.")

                st.subheader("Semantic Classification")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Original Prediction",
                    result["original_prediction"]
                )

                col2.metric(
                    "Recovered Prediction",
                    result["recovered_prediction"]
                )

                col3.metric(
                    "Meaning Preserved",
                    "Yes" if result["semantic_preserved"] else "No"
                )

                confidence_col1, confidence_col2 = st.columns(2)

                confidence_col1.metric(
                    "Original Confidence",
                    f"{result['original_confidence'] * 100:.2f}%"
                )

                confidence_col2.metric(
                    "Recovered Confidence",
                    f"{result['recovered_confidence'] * 100:.2f}%"
                )

                st.subheader("Semantic Compression")

                original_col, compressed_col, recovered_col, saving_col = (
                    st.columns(4)
                )

                original_col.metric(
                    "Original Embedding",
                    result["original_shape"][0]
                )

                compressed_col.metric(
                    "Compressed Embedding",
                    result["compressed_shape"][0]
                )

                recovered_col.metric(
                    "Recovered Embedding",
                    result["recovered_shape"][0]
                )

                saving_col.metric(
                    "Bandwidth Saving",
                    result["bandwidth_saving"]
                )

                st.subheader("Channel Information")

                channel_col, level_col, model_col = st.columns(3)

                channel_col.metric(
                    "Channel Type",
                    noise_type.upper()
                )

                level_col.metric(
                    "Impairment Level",
                    f"{noise_level:.2f}"
                )

                model_col.metric(
                    "Autoencoder Loaded",
                    "Yes" if result["autoencoder_model_loaded"] else "No"
                )

                st.subheader("Quality Evaluation")

                mse_col, psnr_col, cosine_col = st.columns(3)

                mse_col.metric(
                    "MSE",
                    f"{result['mse']:.6f}"
                )

                psnr_col.metric(
                    "PSNR",
                    f"{result['psnr']:.2f} dB"
                )

                cosine_col.metric(
                    "Cosine Similarity",
                    f"{result['cosine_similarity']:.4f}"
                )

                st.subheader("Agentic AI and Generative AI Report")

                st.markdown(
                    result["agentic_ai_report"]
                )

            except Exception as error:
                st.error(f"Image pipeline failed: {error}")


elif module == "Text":
    text = st.text_area(
        "Enter Text",
        placeholder="Enter a sentence or news article..."
    )

    if st.button("Run Text Semantic Communication"):
        try:
            if not text.strip():
                st.warning("Please enter some text.")

            else:
                with st.spinner(
                    "Encoding text and simulating semantic transmission..."
                ):
                    result = run_text_pipeline(
                        text=text,
                        noise_type=noise_type,
                        noise_level=noise_level
                    )

                st.success("Text pipeline completed successfully.")

                st.subheader("Text Classification")

                prediction_col, confidence_col = st.columns(2)

                prediction_col.metric(
                    "Prediction",
                    result["original_prediction"]
                )

                confidence_col.metric(
                    "Confidence",
                    f"{result['original_confidence'] * 100:.2f}%"
                )

                st.subheader("Semantic Compression")

                original_col, compressed_col, recovered_col, saving_col = (
                    st.columns(4)
                )

                original_col.metric(
                    "Original Embedding",
                    result["original_shape"][0]
                )

                compressed_col.metric(
                    "Compressed Embedding",
                    result["compressed_shape"][0]
                )

                recovered_col.metric(
                    "Recovered Embedding",
                    result["recovered_shape"][0]
                )

                saving_col.metric(
                    "Bandwidth Saving",
                    result["bandwidth_saving"]
                )

                st.subheader("Channel Information")

                channel_col, level_col, model_col = st.columns(3)

                channel_col.metric(
                    "Channel Type",
                    noise_type.upper()
                )

                level_col.metric(
                    "Impairment Level",
                    f"{noise_level:.2f}"
                )

                model_col.metric(
                    "Autoencoder Loaded",
                    "Yes" if result["autoencoder_model_loaded"] else "No"
                )

                st.subheader("Quality Evaluation")

                mse_col, psnr_col, cosine_col = st.columns(3)

                mse_col.metric(
                    "MSE",
                    f"{result['mse']:.6f}"
                )

                psnr_col.metric(
                    "PSNR",
                    f"{result['psnr']:.2f} dB"
                )

                cosine_col.metric(
                    "Cosine Similarity",
                    f"{result['cosine_similarity']:.4f}"
                )

                st.subheader("Agentic AI and Generative AI Report")

                st.markdown(
                    result["agentic_ai_report"]
                )

        except Exception as error:
            st.error(f"Text pipeline failed: {error}")


elif module == "Video":
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        suffix = ".mp4"

        if uploaded_video.name.lower().endswith(".avi"):
            suffix = ".avi"
        elif uploaded_video.name.lower().endswith(".mov"):
            suffix = ".mov"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temporary_file:
            temporary_file.write(uploaded_video.getbuffer())
            video_path = temporary_file.name

        st.video(video_path)

        if st.button("Run Video Semantic Communication"):
            try:
                with st.spinner(
                    "Extracting video frames and simulating semantic transmission..."
                ):
                    result = run_video_pipeline(
                        video_path=video_path,
                        noise_type=noise_type,
                        noise_level=noise_level
                    )

                st.success("Video pipeline completed successfully.")

                st.subheader("Video Classification")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Original Prediction",
                    result["original_prediction"]
                )

                col2.metric(
                    "Recovered Prediction",
                    result["recovered_prediction"]
                )

                col3.metric(
                    "Meaning Preserved",
                    "Yes" if result["semantic_preserved"] else "No"
                )

                confidence_col1, confidence_col2 = st.columns(2)

                confidence_col1.metric(
                    "Original Confidence",
                    f"{result['original_confidence'] * 100:.2f}%"
                )

                confidence_col2.metric(
                    "Recovered Confidence",
                    f"{result['recovered_confidence'] * 100:.2f}%"
                )

                st.subheader("Video Statistics")

                frame_col1, frame_col2 = st.columns(2)

                frame_col1.metric(
                    "Frames Processed",
                    result["processed_frames"]
                )

                frame_col2.metric(
                    "Total Frames",
                    result["total_frames"]
                )

                st.subheader("Semantic Compression")

                original_col, compressed_col, recovered_col, saving_col = (
                    st.columns(4)
                )

                original_col.metric(
                    "Original Embedding",
                    result["original_shape"][0]
                )

                compressed_col.metric(
                    "Compressed Embedding",
                    result["compressed_shape"][0]
                )

                recovered_col.metric(
                    "Recovered Embedding",
                    result["recovered_shape"][0]
                )

                saving_col.metric(
                    "Bandwidth Saving",
                    result["bandwidth_saving"]
                )

                st.subheader("Channel Information")

                channel_col, level_col, model_col = st.columns(3)

                channel_col.metric(
                    "Channel Type",
                    noise_type.upper()
                )

                level_col.metric(
                    "Impairment Level",
                    f"{noise_level:.2f}"
                )

                model_col.metric(
                    "Autoencoder Loaded",
                    "Yes" if result["autoencoder_model_loaded"] else "No"
                )

                st.subheader("Quality Evaluation")

                mse_col, psnr_col, cosine_col = st.columns(3)

                mse_col.metric(
                    "MSE",
                    f"{result['mse']:.6f}"
                )

                psnr_col.metric(
                    "PSNR",
                    f"{result['psnr']:.2f} dB"
                )

                cosine_col.metric(
                    "Cosine Similarity",
                    f"{result['cosine_similarity']:.4f}"
                )

                st.subheader("Agentic AI and Generative AI Report")

                st.markdown(
                    result["agentic_ai_report"]
                )

            except Exception as error:
                st.error(f"Video pipeline failed: {error}")