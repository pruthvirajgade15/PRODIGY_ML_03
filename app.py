# pyrefly: ignore [missing-import]
import streamlit as st
import os
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.Pipeline.predict_pipeline import PredictPipeline, CustomData
from src.Pipeline.train_pipeline import TrainPipeline
from src.utils import load_object


# ─────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐾 CatDog AI — SVM Image Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Styles ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Section ── */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
        animation: pulse-bg 4s ease-in-out infinite;
    }

    @keyframes pulse-bg {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #a5b4fc;
        font-weight: 300;
        letter-spacing: 0.05em;
        position: relative;
        z-index: 1;
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
    }

    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    /* ── Prediction Result ── */
    .prediction-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        animation: slideUp 0.6s ease-out;
    }

    .prediction-cat {
        background: linear-gradient(135deg, #1e1b4b, #4c1d95);
        border: 2px solid #8b5cf6;
    }

    .prediction-dog {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border: 2px solid #3b82f6;
    }

    .prediction-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .prediction-cat .prediction-label {
        color: #c4b5fd;
    }

    .prediction-dog .prediction-label {
        color: #93c5fd;
    }

    .prediction-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .confidence-text {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 500;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Upload Area ── */
    .upload-area {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 2px dashed rgba(139, 92, 246, 0.4);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: rgba(139, 92, 246, 0.8);
        background: linear-gradient(135deg, #1e1b4b, #3b2f80);
    }

    /* ── Info Cards ── */
    .info-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }

    .info-card h3 {
        color: #a78bfa;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .info-card p {
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* ── Tech Badge ── */
    .tech-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #a78bfa;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
        transition: all 0.2s ease;
    }

    .tech-badge:hover {
        background: rgba(139, 92, 246, 0.3);
        transform: scale(1.05);
    }

    /* ── Progress Section ── */
    .pipeline-step {
        background: rgba(30, 27, 75, 0.6);
        border-left: 3px solid #6366f1;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        color: #cbd5e1;
    }

    .pipeline-step.active {
        border-left-color: #a78bfa;
        background: rgba(139, 92, 246, 0.1);
    }

    .pipeline-step.done {
        border-left-color: #34d399;
        background: rgba(52, 211, 153, 0.1);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1e1b4b 100%);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    /* ── Metrics Row ── */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    /* ── Architecture Diagram ── */
    .arch-container {
        background: linear-gradient(135deg, #0f0c29, #1e1b4b);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }

    .arch-step {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(139, 92, 246, 0.15);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        color: #c4b5fd;
        font-weight: 500;
        margin: 0.3rem;
    }

    .arch-arrow {
        color: #6366f1;
        font-size: 1.3rem;
        margin: 0 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3rem;">🐾</span>
        <h2 style="background: linear-gradient(135deg, #a78bfa, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-weight: 800; margin-top: 0.5rem;">
            CatDog AI
        </h2>
        <p style="color: #94a3b8; font-size: 0.85rem;">SVM Image Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict", "🏋️ Train Model", "📊 Performance", "📖 About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Model Status
    predict_pipeline = PredictPipeline()
    model_available = predict_pipeline.is_model_available()

    if model_available:
        st.markdown("""
        <div style="background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3);
            border-radius: 12px; padding: 1rem; text-align: center;">
            <span style="font-size: 1.5rem;">✅</span>
            <p style="color: #34d399; font-weight: 600; margin: 0.3rem 0 0 0;">Model Ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3);
            border-radius: 12px; padding: 1rem; text-align: center;">
            <span style="font-size: 1.5rem;">⚠️</span>
            <p style="color: #fbbf24; font-weight: 600; margin: 0.3rem 0 0 0;">Model Not Trained</p>
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Go to Train Model</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Page: HOME
# ─────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🐱 CatDog AI 🐶</div>
        <div class="hero-subtitle">
            Production-ready SVM Image Classifier — Cats vs Dogs
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">SVM</div>
            <div class="stat-label">Algorithm</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">HOG</div>
            <div class="stat-label">Feature Extraction</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">64×64</div>
            <div class="stat-label">Image Resolution</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">GridCV</div>
            <div class="stat-label">Hyperparameter Tuning</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 How It Works</h3>
            <p>
                This system uses <strong>Support Vector Machine (SVM)</strong> with
                <strong>HOG (Histogram of Oriented Gradients)</strong> features to
                classify images of cats and dogs. Images are resized to 64×64 pixels,
                converted to grayscale, and HOG features are extracted to capture
                edge and gradient patterns that distinguish cats from dogs.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>📐 Architecture</h3>
            <p>
                <span class="arch-step">📷 Input Image</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">🔄 Resize & Grayscale</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">📊 HOG Features</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">⚖️ StandardScaler</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">🤖 SVM Classifier</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">🏷️ Cat / Dog</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🚀 Getting Started</h3>
            <p>
                <strong>Step 1:</strong> Download the Kaggle Dogs vs Cats dataset<br>
                <strong>Step 2:</strong> Go to <em>Train Model</em> → provide dataset path<br>
                <strong>Step 3:</strong> Wait for training to complete<br>
                <strong>Step 4:</strong> Go to <em>Predict</em> → upload any cat/dog image<br>
                <strong>Step 5:</strong> Get instant classification with confidence score!
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🛠️ Tech Stack</h3>
            <p>
                <span class="tech-badge">Python</span>
                <span class="tech-badge">Scikit-Learn</span>
                <span class="tech-badge">OpenCV</span>
                <span class="tech-badge">scikit-image</span>
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">Plotly</span>
                <span class="tech-badge">NumPy</span>
                <span class="tech-badge">Pandas</span>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Page: PREDICT
# ─────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(135deg, #a78bfa, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-weight: 800;">
            🔮 Image Prediction
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            Upload a cat or dog image and let our SVM classifier do its magic
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not model_available:
        st.warning("⚠️ No trained model found. Please go to **Train Model** first to train the SVM classifier.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="upload-area">
                <span style="font-size: 3rem;">📤</span>
                <p style="color: #a5b4fc; font-weight: 600; font-size: 1.1rem; margin: 0.5rem 0;">
                    Drag & Drop or Browse
                </p>
                <p style="color: #64748b; font-size: 0.85rem;">
                    Supports JPG, JPEG, PNG
                </p>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

        with col2:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button("🔍 Classify Image", use_container_width=True, type="primary"):
                    with st.spinner("Analyzing image..."):
                        time.sleep(0.5)  # Brief pause for UX

                        custom_data = CustomData(image=image)
                        result = predict_pipeline.predict(custom_data)

                    prediction = result['prediction']
                    confidence = result['confidence']
                    probs = result['probabilities']

                    # Prediction Card
                    if prediction == 'Cat':
                        emoji = "🐱"
                        card_class = "prediction-cat"
                    else:
                        emoji = "🐶"
                        card_class = "prediction-dog"

                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <div class="prediction-emoji">{emoji}</div>
                        <div class="prediction-label">It's a {prediction}!</div>
                        <div class="confidence-text">
                            Confidence: {confidence * 100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability Chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Cat 🐱', 'Dog 🐶'],
                        y=[probs['Cat'] * 100, probs['Dog'] * 100],
                        marker=dict(
                            color=['#8b5cf6', '#3b82f6'],
                            line=dict(color=['#a78bfa', '#60a5fa'], width=2)
                        ),
                        text=[f"{probs['Cat']*100:.1f}%", f"{probs['Dog']*100:.1f}%"],
                        textposition='outside',
                        textfont=dict(color='white', size=14, family='Inter')
                    ))
                    fig.update_layout(
                        title=dict(
                            text="Class Probabilities",
                            font=dict(color='#a78bfa', size=16, family='Inter')
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#cbd5e1', family='Inter'),
                        yaxis=dict(
                            title='Probability (%)',
                            range=[0, 110],
                            gridcolor='rgba(139,92,246,0.1)'
                        ),
                        xaxis=dict(gridcolor='rgba(139,92,246,0.1)'),
                        height=350,
                        margin=dict(t=50, b=30)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.markdown("""
                <div style="background: rgba(30, 27, 75, 0.4); border-radius: 16px;
                    padding: 3rem; text-align: center; border: 1px solid rgba(139, 92, 246, 0.1);">
                    <span style="font-size: 4rem; opacity: 0.5;">🖼️</span>
                    <p style="color: #64748b; margin-top: 1rem;">
                        Upload an image to see the prediction
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Page: TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────
elif page == "🏋️ Train Model":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(135deg, #a78bfa, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-weight: 800;">
            🏋️ Train SVM Model
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            Configure and train the SVM classifier on the Cats vs Dogs dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Training Configuration
    st.markdown("""
    <div class="info-card">
        <h3>⚙️ Training Configuration</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.2);
            border-radius: 12px; padding: 1rem;">
            <p style="color: #34d399; margin: 0; font-weight: 600;">📁 Data Source: Local Dataset</p>
            <p style="color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Using pre-extracted images from artifacts/raw_data
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        max_samples = st.slider(
            "📊 Max Samples Per Class",
            min_value=100,
            max_value=12500,
            value=2000,
            step=100,
            help="More samples = better accuracy but longer training time"
        )

        st.markdown(f"""
        <div style="background: rgba(139, 92, 246, 0.1); border-radius: 10px;
            padding: 0.8rem; margin-top: 0.5rem;">
            <p style="color: #a78bfa; margin: 0; font-size: 0.9rem;">
                📝 Total training images: <strong>{max_samples * 2:,}</strong><br>
                ⏱️ Estimated time: <strong>{max(1, max_samples // 500)}-{max(2, max_samples // 300)} min</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training Pipeline Steps
    st.markdown("""
    <div class="info-card">
        <h3>🔄 Pipeline Steps</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="pipeline-step">
            <strong>Step 1:</strong> Data Ingestion<br>
            <span style="font-size: 0.85rem; color: #94a3b8;">
                Download → Extract → Split (80/20)
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="pipeline-step">
            <strong>Step 2:</strong> Data Transformation<br>
            <span style="font-size: 0.85rem; color: #94a3b8;">
                Resize → Grayscale → HOG → Scale
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="pipeline-step">
            <strong>Step 3:</strong> Model Training<br>
            <span style="font-size: 0.85rem; color: #94a3b8;">
                SVM + GridSearchCV → Evaluate → Save
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Start Training Button
    if st.button("🚀 Start Training Pipeline", use_container_width=True, type="primary"):
        try:
            pipeline = TrainPipeline()

            # Progress tracking
            progress_bar = st.progress(0, text="Initializing pipeline...")
            status_container = st.empty()

            # Step 1: Data Ingestion
            progress_bar.progress(10, text="📥 Step 1/3: Data Ingestion...")
            status_container.info("📥 Scanning and organizing dataset...")

            train_path, test_path = pipeline.data_ingestion.initiate_data_ingestion()
            progress_bar.progress(30, text="✅ Step 1/3: Data Ingestion Complete")

            # Step 2: Data Transformation
            progress_bar.progress(35, text="🔄 Step 2/3: Data Transformation...")
            status_container.info("🔄 Extracting HOG features from images...")

            pipeline.data_transformation.transformation_config.max_samples_per_class = max_samples
            X_train, y_train, X_test, y_test, preprocessor_path = (
                pipeline.data_transformation.initiate_data_transformation(
                    train_path, test_path
                )
            )
            progress_bar.progress(65, text="✅ Step 2/3: Data Transformation Complete")

            # Step 3: Model Training
            progress_bar.progress(70, text="🤖 Step 3/3: Training SVM with GridSearchCV...")
            status_container.info("🤖 Training SVM classifier with hyperparameter tuning...")

            results = pipeline.model_trainer.initiate_model_trainer(
                X_train, y_train, X_test, y_test
            )
            progress_bar.progress(100, text="✅ Training Pipeline Complete!")

            # Success Message
            status_container.empty()
            st.balloons()

            st.success(f"""
            🎉 **Training Complete!**

            - **Accuracy:** {results['accuracy'] * 100:.2f}%
            - **Best Parameters:** {results['best_params']}
            - **CV Score:** {results['best_cv_score'] * 100:.2f}%
            """)

            # Quick Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Accuracy", f"{results['accuracy'] * 100:.2f}%")
            with col2:
                st.metric("CV Score", f"{results['best_cv_score'] * 100:.2f}%")
            with col3:
                st.metric("Best Kernel", results['best_params'].get('kernel', 'N/A'))

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.exception(e)


# ─────────────────────────────────────────────────────────────────────
# Page: PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
elif page == "📊 Performance":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(135deg, #a78bfa, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-weight: 800;">
            📊 Model Performance
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            Detailed metrics and visualizations of the trained model
        </p>
    </div>
    """, unsafe_allow_html=True)

    metrics_path = os.path.join('artifacts', 'metrics.pkl')

    if not os.path.exists(metrics_path):
        st.warning("⚠️ No training metrics found. Please train the model first.")
    else:
        results = load_object(metrics_path)

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        report = results['classification_report']

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{results['accuracy'] * 100:.1f}%</div>
                <div class="stat-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            cat_precision = report.get('Cat', {}).get('precision', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{cat_precision * 100:.1f}%</div>
                <div class="stat-label">Cat Precision</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            dog_precision = report.get('Dog', {}).get('precision', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{dog_precision * 100:.1f}%</div>
                <div class="stat-label">Dog Precision</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            cv_score = results.get('best_cv_score', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{cv_score * 100:.1f}%</div>
                <div class="stat-label">CV Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Confusion Matrix
            cm = results['confusion_matrix']
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Cat', 'Predicted Dog'],
                y=['Actual Cat', 'Actual Dog'],
                colorscale=[
                    [0, '#1e1b4b'],
                    [0.5, '#6366f1'],
                    [1, '#a78bfa']
                ],
                text=cm,
                texttemplate='<b>%{text}</b>',
                textfont=dict(size=18, color='white', family='Inter'),
                hovertemplate='%{y} → %{x}<br>Count: %{z}<extra></extra>',
                showscale=True,
                colorbar=dict(
                    title=dict(text='Count', font=dict(color='#cbd5e1')),
                    tickfont=dict(color='#cbd5e1')
                )
            ))
            fig_cm.update_layout(
                title=dict(
                    text='Confusion Matrix',
                    font=dict(color='#a78bfa', size=18, family='Inter')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Inter'),
                height=400,
                xaxis=dict(side='bottom'),
                margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            # Per-Class Metrics
            cats_report = report.get('Cat', {})
            dogs_report = report.get('Dog', {})

            metrics_names = ['Precision', 'Recall', 'F1-Score']
            cat_values = [
                cats_report.get('precision', 0) * 100,
                cats_report.get('recall', 0) * 100,
                cats_report.get('f1-score', 0) * 100
            ]
            dog_values = [
                dogs_report.get('precision', 0) * 100,
                dogs_report.get('recall', 0) * 100,
                dogs_report.get('f1-score', 0) * 100
            ]

            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Bar(
                name='Cat 🐱',
                x=metrics_names,
                y=cat_values,
                marker_color='#8b5cf6',
                text=[f'{v:.1f}%' for v in cat_values],
                textposition='outside',
                textfont=dict(color='#c4b5fd', size=12, family='Inter')
            ))
            fig_metrics.add_trace(go.Bar(
                name='Dog 🐶',
                x=metrics_names,
                y=dog_values,
                marker_color='#3b82f6',
                text=[f'{v:.1f}%' for v in dog_values],
                textposition='outside',
                textfont=dict(color='#93c5fd', size=12, family='Inter')
            ))
            fig_metrics.update_layout(
                title=dict(
                    text='Per-Class Metrics',
                    font=dict(color='#a78bfa', size=18, family='Inter')
                ),
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Inter'),
                yaxis=dict(
                    title='Score (%)',
                    range=[0, 115],
                    gridcolor='rgba(139,92,246,0.1)'
                ),
                xaxis=dict(gridcolor='rgba(139,92,246,0.1)'),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1')
                ),
                height=400,
                margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        # Classification Report
        st.markdown("""
        <div class="info-card">
            <h3>📋 Detailed Classification Report</h3>
        </div>
        """, unsafe_allow_html=True)

        st.code(results.get('classification_report_text', 'No report available'), language='text')

        # Best Parameters
        best_params = results.get('best_params', {})
        if best_params:
            st.markdown("""
            <div class="info-card">
                <h3>🎯 Best Hyperparameters (GridSearchCV)</h3>
            </div>
            """, unsafe_allow_html=True)

            param_cols = st.columns(len(best_params))
            for i, (param, value) in enumerate(best_params.items()):
                with param_cols[i]:
                    st.metric(param.upper(), str(value))


# ─────────────────────────────────────────────────────────────────────
# Page: ABOUT
# ─────────────────────────────────────────────────────────────────────
elif page == "📖 About":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(135deg, #a78bfa, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-weight: 800;">
            📖 About This Project
        </h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Project Overview</h3>
            <p>
                <strong>PRODIGY_ML_03</strong> is a production-ready machine learning project
                that implements a <strong>Support Vector Machine (SVM)</strong> to classify
                images of cats and dogs from the famous Kaggle Dogs vs Cats dataset.
            </p>
            <p>
                The project follows a modular architecture with separate components for
                data ingestion, transformation, and model training — making it easy to
                maintain, test, and extend.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>📐 ML Pipeline Architecture</h3>
            <p>
                <strong>1. Data Ingestion</strong><br>
                Downloads the Kaggle dataset, extracts images, and splits into
                train/test sets (80/20 ratio).
            </p>
            <p>
                <strong>2. Data Transformation</strong><br>
                Resizes images to 64×64, converts to grayscale, extracts HOG
                (Histogram of Oriented Gradients) features, and normalizes
                with StandardScaler.
            </p>
            <p>
                <strong>3. Model Training</strong><br>
                Trains an SVM classifier with RBF/Linear kernel using
                GridSearchCV for hyperparameter optimization (C, gamma, kernel).
            </p>
            <p>
                <strong>4. Prediction</strong><br>
                Loads the trained model and preprocessor to classify new images
                with confidence scores.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🛠️ Technology Stack</h3>
            <p>
                <span class="tech-badge">Python 3.8+</span>
                <span class="tech-badge">Scikit-Learn</span>
                <span class="tech-badge">OpenCV</span>
                <span class="tech-badge">scikit-image</span>
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">Plotly</span>
                <span class="tech-badge">NumPy</span>
                <span class="tech-badge">Pandas</span>
                <span class="tech-badge">Matplotlib</span>
                <span class="tech-badge">Seaborn</span>
                <span class="tech-badge">Kaggle API</span>
                <span class="tech-badge">Pillow</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>📁 Project Structure</h3>
            <p style="font-family: monospace; font-size: 0.85rem; line-height: 1.8;">
                PRODIGY_ML_03/<br>
                ├── app.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #6366f1;"># Streamlit UI</span><br>
                ├── requirements.txt<br>
                ├── setup.py<br>
                ├── artifacts/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #6366f1;"># Model & data</span><br>
                ├── logs/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #6366f1;"># Log files</span><br>
                └── src/<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├── Components/ <span style="color: #6366f1;"># ML modules</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;├── Pipeline/ &nbsp;&nbsp; <span style="color: #6366f1;"># Orchestration</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;└── Notebooks/ &nbsp; <span style="color: #6366f1;"># EDA & experiments</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>👤 Author</h3>
            <p>
                <strong>Pruthviraj Gade</strong><br>
                Prodigy InfoTech — Machine Learning Internship<br>
                Task 03: SVM Image Classification<br><br>
                <span class="tech-badge">📊 Dataset: Kaggle Dogs vs Cats</span>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1.5rem; border-top: 1px solid rgba(139, 92, 246, 0.2);">
    <p style="color: #64748b; font-size: 0.85rem; margin: 0;">
        Built with ❤️ by <strong style="color: #a78bfa;">Pruthviraj Gade</strong> |
        PRODIGY_ML_03 — SVM Cats vs Dogs Classifier |
        <span style="color: #6366f1;">Prodigy InfoTech</span>
    </p>
</div>
""", unsafe_allow_html=True)
