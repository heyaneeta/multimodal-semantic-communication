import tempfile
from pathlib import Path

import streamlit as st

from semantic_core.run_image import run_image_pipeline
from semantic_core.run_text import run_text_pipeline
from semantic_core.run_video import run_video_pipeline


st.set_page_config(
    page_title="Semantic AI 6G System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(40, 90, 180, 0.16),
                transparent 34%
            ),
            #0b1120;
    }

    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero-card {
        padding: 2rem;
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.95),
                rgba(30, 41, 59, 0.82)
            );
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.30);
        margin-bottom: 1.4rem;
    }

    .hero-title {
        font-size: 2.35rem;
        line-height: 1.1;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        max-width: 900px;
        margin-bottom: 0;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.14);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #86efac;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    .section-card {
        padding: 1.3rem 1.4rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(15, 23, 42, 0.74);
        margin-bottom: 1rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .section-caption {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 1rem;
        border-radius: 16px;
        min-height: 112px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 800;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.50);
        border-radius: 16px;
        padding: 0.3rem;
    }

    div[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    div[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    .success-box {
        padding: 0.85rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(34, 197, 94, 0.28);
        background: rgba(34, 197, 94, 0.10);
        color: #bbf7d0;
        font-weight: 650;
        margin: 0.6rem 0 1rem 0;
    }

    .warning-box {
        padding: 0.85rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(245, 158, 11, 0.28);
        background: rgba(245, 158, 11, 0.10);
        color: #fde68a;
        margin: 0.6rem 0 1rem 0;
    }

    .footer-note {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        color: #64748b;
        font-size: 0.8rem;
        text-align: center;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        border: 1px solid rgba(96, 165, 250, 0.35);
        background:
            linear-gradient(
                135deg,
                #2563eb,
                #4f46e5
            );
        color: white;
    }

    .stButton > button:hover {
        border-color: #93c5fd;
        box-shadow: 0 8px 28px rgba(37, 99, 235, 0.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">
                Multimodal Semantic Communication System
            </div>
            <div class="hero-subtitle">
                Intelligent 6G semantic transmission for image, text, and video
                using MobileNetV3, BERT, autoencoder compression, channel
                simulation, Agentic AI, and local Generative AI.
            </div>
            <div class="status-pill">
                ● System Online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_overview():
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">System Pipeline</div>
            <div class="section-caption">
                Input → Semantic Encoding → 576-D Embedding → 128-D Compression
                → AWGN / Packet Loss → Recovery → Metrics → AI Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_semantic_metrics(result):
    st.markdown("### Semantic Compression")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Original Embedding",
        result["original_shape"][0],
        help="Original semantic representation size",
    )

    col2.metric(
        "Compressed Embedding",
        result["compressed_shape"][0],
        help="Latent representation transmitted through the channel",
    )

    col3.metric(
        "Recovered Embedding",
        result["recovered_shape"][0],
        help="Semantic representation reconstructed by the decoder",
    )

    col4.metric(
        "Bandwidth Saving",
        result["bandwidth_saving"],
        help="Reduction from 576 dimensions to 128 dimensions",
    )

    st.markdown("### Channel and Model Status")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Channel",
        st.session_state.noise_type.upper(),
    )

    col2.metric(
        "Impairment Level",
        f"{st.session_state.noise_level:.2f}",
    )

    col3.metric(
        "Autoencoder",
        "Loaded" if result["autoencoder_model_loaded"] else "Unavailable",
    )

    st.markdown("### Reconstruction Quality")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MSE",
        f"{result['mse']:.6f}",
        help="Lower is better",
    )

    col2.metric(
        "PSNR",
        f"{result['psnr']:.2f} dB",
        help="Higher is better",
    )

    col3.metric(
        "Cosine Similarity",
        f"{result['cosine_similarity']:.4f}",
        help="Closer to 1 means better semantic preservation",
    )


def render_report(result, filename):
    report = result.get(
        "agentic_ai_report",
        "No AI report was generated.",
    )

    st.markdown("### Agentic AI and Generative AI Report")

    st.markdown(
        """
        <div class="section-card">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(report)

    st.download_button(
        label="Download Technical Report",
        data=report,
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


render_hero()
render_pipeline_overview()


with st.sidebar:
    st.markdown("## Control Panel")

    module = st.selectbox(
        "Input Modality",
        ["Image", "Text", "Video"],
    )

    noise_type = st.selectbox(
        "Channel Impairment",
        ["awgn", "packet_loss"],
    )

    noise_level = st.slider(
        "Noise / Packet Loss Level",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
    )

    st.session_state.noise_type = noise_type
    st.session_state.noise_level = noise_level

    st.markdown("---")

    st.markdown(
        """
        <div class="small-note">
        <strong>Embedding flow</strong><br>
        576 dimensions → 128 latent dimensions → 576 recovered dimensions
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="small-note" style="margin-top: 1rem;">
        <strong>Supported channels</strong><br>
        AWGN and packet-loss simulation
        </div>
        """,
        unsafe_allow_html=True,
    )


input_tab, results_tab, report_tab = st.tabs(
    ["Input and Processing", "Results Dashboard", "AI Technical Report"]
)


if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_modality" not in st.session_state:
    st.session_state.last_modality = None


with input_tab:
    if module == "Image":
        st.markdown("## Image Semantic Communication")
        st.caption(
            "Upload a cat or dog image for semantic extraction, compression, "
            "channel transmission, reconstruction, and AI analysis."
        )

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            key="image_upload",
        )

        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:
                temporary_file.write(uploaded_file.getbuffer())
                image_path = temporary_file.name

            preview_col, action_col = st.columns([1, 2])

            with preview_col:
                st.image(
                    image_path,
                    caption="Uploaded Image",
                    use_container_width=True,
                )

            with action_col:
                st.markdown("### Ready for Transmission")
                st.write(
                    "The image will be encoded into a 576-dimensional "
                    "semantic embedding and transmitted through the selected "
                    "channel."
                )

                run_image = st.button(
                    "Run Image Pipeline",
                    use_container_width=True,
                )

            if run_image:
                try:
                    with st.spinner(
                        "Running image semantic communication pipeline..."
                    ):
                        result = run_image_pipeline(
                            image_path=image_path,
                            noise_type=noise_type,
                            noise_level=noise_level,
                        )

                    st.session_state.last_result = result
                    st.session_state.last_modality = "Image"

                    st.markdown(
                        """
                        <div class="success-box">
                            Image pipeline completed successfully.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.info(
                        "Open the Results Dashboard and AI Technical Report tabs."
                    )

                except Exception as error:
                    st.error(f"Image pipeline failed: {error}")

    elif module == "Text":
        st.markdown("## Text Semantic Communication")
        st.caption(
            "Enter a news sentence or article for BERT classification and "
            "semantic transmission."
        )

        text = st.text_area(
            "Text Input",
            placeholder="Example: Apple announced a new processor...",
            height=180,
        )

        if st.button(
            "Run Text Pipeline",
            use_container_width=True,
        ):
            try:
                if not text.strip():
                    st.warning("Enter some text before running the pipeline.")
                else:
                    with st.spinner(
                        "Running text semantic communication pipeline..."
                    ):
                        result = run_text_pipeline(
                            text=text,
                            noise_type=noise_type,
                            noise_level=noise_level,
                        )

                    st.session_state.last_result = result
                    st.session_state.last_modality = "Text"

                    st.markdown(
                        """
                        <div class="success-box">
                            Text pipeline completed successfully.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.info(
                        "Open the Results Dashboard and AI Technical Report tabs."
                    )

            except Exception as error:
                st.error(f"Text pipeline failed: {error}")

    elif module == "Video":
        st.markdown("## Video Semantic Communication")
        st.caption(
            "Upload a video for frame sampling, MobileNetV3 semantic extraction, "
            "compression, transmission, and recovery."
        )

        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"],
            key="video_upload",
        )

        if uploaded_video is not None:
            suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:
                temporary_file.write(uploaded_video.getbuffer())
                video_path = temporary_file.name

            if suffix == ".mp4":
                st.video(video_path)
            else:
                st.markdown(
                    """
                    <div class="warning-box">
                        Browser preview may not support AVI or MOV.
                        OpenCV can still process the uploaded video.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write(f"Uploaded file: **{uploaded_video.name}**")

            if st.button(
                "Run Video Pipeline",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Sampling frames and running semantic transmission..."
                    ):
                        result = run_video_pipeline(
                            video_path=video_path,
                            noise_type=noise_type,
                            noise_level=noise_level,
                        )

                    st.session_state.last_result = result
                    st.session_state.last_modality = "Video"

                    st.markdown(
                        """
                        <div class="success-box">
                            Video pipeline completed successfully.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.info(
                        "Open the Results Dashboard and AI Technical Report tabs."
                    )

                except Exception as error:
                    st.error(f"Video pipeline failed: {error}")


with results_tab:
    result = st.session_state.last_result
    modality = st.session_state.last_modality

    if result is None:
        st.info(
            "Run an Image, Text, or Video pipeline to display results here."
        )

    else:
        st.markdown(f"## {modality} Transmission Results")

        if modality == "Image":
            st.markdown("### Semantic Classification")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Original Prediction",
                result["original_prediction"],
            )

            col2.metric(
                "Recovered Prediction",
                result["recovered_prediction"],
            )

            col3.metric(
                "Semantic Meaning",
                "Preserved" if result["semantic_preserved"] else "Changed",
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Original Confidence",
                f"{result['original_confidence'] * 100:.2f}%",
            )

            col2.metric(
                "Recovered Confidence",
                f"{result['recovered_confidence'] * 100:.2f}%",
            )

        elif modality == "Text":
            st.markdown("### Text Classification")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Category",
                result["original_prediction"],
            )

            col2.metric(
                "Confidence",
                f"{result['original_confidence'] * 100:.2f}%",
            )

        elif modality == "Video":
            st.markdown("### Video Classification")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Original Prediction",
                result["original_prediction"],
            )

            col2.metric(
                "Recovered Prediction",
                result["recovered_prediction"],
            )

            col3.metric(
                "Semantic Meaning",
                "Preserved" if result["semantic_preserved"] else "Changed",
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Original Confidence",
                f"{result['original_confidence'] * 100:.2f}%",
            )

            col2.metric(
                "Recovered Confidence",
                f"{result['recovered_confidence'] * 100:.2f}%",
            )

            st.markdown("### Video Processing")

            col1, col2 = st.columns(2)

            col1.metric(
                "Processed Frames",
                result["processed_frames"],
            )

            col2.metric(
                "Total Frames",
                result["total_frames"],
            )

        st.divider()

        render_semantic_metrics(result)

        status = (
            "Semantic transmission successful"
            if result.get("cosine_similarity", 0) >= 0.8
            else "Semantic quality degraded"
        )

        if result.get("cosine_similarity", 0) >= 0.8:
            st.success(status)
        else:
            st.warning(status)


with report_tab:
    result = st.session_state.last_result
    modality = st.session_state.last_modality

    if result is None:
        st.info(
            "Run a pipeline first to generate an Agentic AI and Generative AI report."
        )

    else:
        filename = (
            f"{modality.lower()}_semantic_communication_report.md"
        )

        render_report(
            result=result,
            filename=filename,
        )


st.markdown(
    """
    <div class="footer-note">
        Semantic AI 6G System · MobileNetV3 · BERT · Autoencoder ·
        AWGN · Packet Loss · Agentic AI · Ollama
    </div>
    """,
    unsafe_allow_html=True,
)