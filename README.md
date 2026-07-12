# 🐾 PRODIGY_ML_03 — SVM Cats vs Dogs Image Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/app.py)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/src/Components/model_trainer.py)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-5c3ee8?style=for-the-badge&logo=opencv&logoColor=white)](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/src/Components/data_transformation.py)
[![Streamlit](https://img.shields.io/badge/Streamlit-App_UI-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/app.py)

Production-ready **Support Vector Machine (SVM)** image classifier that distinguishes between cats and dogs using HOG features, built with a modular ML pipeline and a premium **Streamlit** web interface.

---

## 📸 Features

- 🤖 **SVM Classifier** with GridSearchCV hyperparameter tuning
- 📊 **HOG Feature Extraction** (Histogram of Oriented Gradients)
- 🔄 **Modular ML Pipeline** — Data Ingestion → Transformation → Training
- 🌐 **Streamlit Web App** with dark theme and interactive visualizations
- 📈 **Plotly Charts** — Confusion matrix, per-class metrics, probability bars
- 📥 **Kaggle Integration** — Direct dataset download via API with automatic fallback

---

## 🏗️ Architecture

```
Input Image → Resize (64×64) → Grayscale → HOG Features → StandardScaler → SVM → Cat/Dog
```

### Project Structure

```
PRODIGY_ML_03/
├── app.py                          # Streamlit web application
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
│
├── artifacts/
│   ├── raw_data/                   # Downloaded and unzipped dataset images
│   ├── preprocessor.pkl            # Trained StandardScaler
│   ├── model.pkl                   # Trained SVM model
│   └── metrics.pkl                 # Training evaluation metrics
│
├── logs/                           # Timestamped execution log files
│
└── src/
    ├── exception.py                # Custom exception handling
    ├── logger.py                   # Logging configuration
    ├── utils.py                    # Utility helper functions
    │
    ├── Components/
    │   ├── data_ingestion.py       # Dataset download, extraction, & organization
    │   ├── data_transformation.py  # Image preprocessing & HOG extraction
    │   └── model_trainer.py        # SVM classifier training & tuning
    │
    ├── Pipeline/
    │   ├── train_pipeline.py       # End-to-end training orchestrator
    │   └── predict_pipeline.py     # Single image prediction orchestrator
    │
    └── Notebooks/
        └── Data/                   # Sample data for exploratory notebooks
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/pruthvirajgade15/PRODIGY_ML_03.git
cd PRODIGY_ML_03
pip install -r requirements.txt
```

### 2. Setup Kaggle API (Optional)

Download `kaggle.json` from [Kaggle settings](https://www.kaggle.com/settings) and place it in:
* **Windows:** `C:\Users\<Your-Username>\.kaggle\kaggle.json`
* **Linux/Mac:** `~/.kaggle/kaggle.json`

*(Note: The ingestion component automatically falls back to downloading the public 'tongpython/cat-and-dog' dataset if Kaggle API keys are not set up).*

### 3. Run the App

Start the Streamlit dashboard:
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Or simply double-click [`run_app.bat`](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/run_app.bat).

### 4. Train the Model

To run the training pipeline via terminal:
```powershell
.\venv\python.exe -X utf8 -m src.Pipeline.train_pipeline
```
Or double-click [`train.bat`](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/train.bat). To run a fast validation test with fewer images (e.g. 500 per class), run:
```powershell
.\venv\python.exe -X utf8 -m src.Pipeline.train_pipeline --max-samples 500
```

---

## 🧠 How It Works

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Data Ingestion** ([`data_ingestion.py`](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/src/Components/data_ingestion.py)) | Downloads dataset (with automatic fallback), extracts it, and organizes classes. |
| 2 | **Data Transformation** ([`data_transformation.py`](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/src/Components/data_transformation.py)) | Resize → Grayscale → HOG features → StandardScaler. |
| 3 | **Model Training** ([`model_trainer.py`](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/src/Components/model_trainer.py)) | SVM + GridSearchCV hyperparameter tuning (kernel, C, gamma). |
| 4 | **Prediction** ([`predict_pipeline.py`](file:///C:/Users/Pruthviraj/PRODIGY_ML_03/src/Pipeline/predict_pipeline.py)) | Preprocess input image → Extract HOG features → StandardScaler → Classify. |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Scikit-Learn** | SVM classifier, GridSearchCV, StandardScaler |
| **OpenCV** | Image reading and preprocessing |
| **scikit-image** | HOG feature extraction |
| **Streamlit** | Web application interface |
| **Plotly** | Interactive visualizations |
| **Kaggle API** | Dataset download |

---

## 📊 Dataset

- **Source:** [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Size:** 25,000 labeled images (12,500 cats + 12,500 dogs)
- **Usage:** Configurable subset (default: 2,000 per class to balance training time)

---

## 👤 Author

**Pruthviraj Gade**  
Prodigy InfoTech — Machine Learning Internship  
Task 03: Implement SVM for Image Classification

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
