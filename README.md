# 🩺 Disease Prediction Using Machine Learning

An AI-powered clinical decision-support prototype that uses machine learning classification models to predict disease classes from structured patient data. The project includes an interactive Streamlit dashboard, model evaluation metrics, prediction probabilities, and visual comparisons of model performance.

> **Disclaimer:** This project is for educational and research purposes only. Its predictions are not medically validated and must not be used for diagnosis, treatment, or other clinical decisions.

## 🚀 Project Overview

The objective of this project is to explore how machine learning can classify medical datasets and present the results through an interactive dashboard.

### Key Features

- 🧬 Three datasets: Breast Cancer, Heart Disease, and Diabetes
- 🤖 Four model families: Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost
- 📊 Interactive dashboard built with Streamlit
- 🔎 Prediction results with model-estimated class probabilities
- 📈 Model performance comparison using accuracy, precision, recall, and F1-score
- 📉 Confusion matrices for evaluating classification performance
- 🎨 Dark-themed dashboard with interactive charts
- 🧪 Reproducible data preparation and model training scripts

## 🗂️ Project Structure

```text
Disease-Prediction-ML/
├── app.py
├── .gitignore
├── README.md
├── data/
│   ├── breast_cancer.csv
│   ├── heart_disease.csv
│   └── diabetes.csv
├── models/
│   ├── breast_cancer_logistic_regression.joblib
│   ├── breast_cancer_random_forest.joblib
│   ├── breast_cancer_svm.joblib
│   ├── breast_cancer_xgboost.joblib
│   ├── diabetes_logistic_regression.joblib
│   ├── diabetes_svm.joblib
│   ├── diabetes_xgboost.joblib
│   └── heart_disease_*.joblib
├── outputs/
│   ├── model_metrics.csv
│   └── *_confusion_matrix.png
└── src/
    ├── download_data.py
    └── train_models.py
```

*Note: This tree is representative; some local files, including the large Diabetes Random Forest model, are not included in the GitHub repository.*

## 🧰 Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Matplotlib and Seaborn
- SHAP (installed for planned explainability improvements)
- Joblib

## 📚 Datasets

| Dataset | Purpose | Source |
|---|---|---|
| Breast Cancer Wisconsin | Classify breast tumours as benign or malignant | [Scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |
| Heart Disease | Classify the presence or absence of heart disease | [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Diabetes Health Indicators | Classify diabetes status from health indicators | [UCI CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) |

The Breast Cancer CSV was prepared from Scikit-learn's bundled Wisconsin Breast Cancer dataset. The Heart Disease and Diabetes datasets were sourced from the UCI Machine Learning Repository.

## 🧠 Machine Learning Models

The project explores four classification algorithms:

1. **Logistic Regression** — a linear classification baseline.
2. **Support Vector Machine (SVM)** — a margin-based classification algorithm.
3. **Random Forest** — an ensemble of decision trees.
4. **XGBoost** — a gradient-boosted tree model.

The saved model files and evaluation outputs are stored in the `models/` and `outputs/` directories respectively.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.13 or a compatible Python version supported by the dependencies
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Ramk09/Disease-Prediction-ML.git
cd Disease-Prediction-ML
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn xgboost streamlit shap plotly joblib ucimlrepo
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 🔄 Data Preparation and Model Training

The project includes scripts for preparing datasets and training classification models.

To prepare or download the datasets:

```bash
python src/download_data.py
```

To train the models:

```bash
python src/train_models.py
```

**Important:** Running these scripts may regenerate datasets, model files, or evaluation outputs. Training time depends on the dataset, algorithm, and hardware.

## 📊 Model Evaluation

Model performance is evaluated using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrices

The generated metrics are stored in `outputs/model_metrics.csv`, and confusion-matrix visualizations are saved in `outputs/`.

Performance can vary with preprocessing, data splits, class imbalance, and training configuration. Metrics from this project should not be interpreted as evidence of clinical effectiveness.

## 📦 Large Model File

The trained Diabetes Random Forest model, `diabetes_random_forest.joblib`, is approximately 1.2 GB and is intentionally excluded from this Git repository because of GitHub's regular file-size limit.

The model remains available in the original local project directory, but it will not be available to someone who clones this repository. To use that model in a fresh clone, retrain it using the project's training script or arrange a separate, documented model download.

## 🔮 Future Improvements

- Add SHAP-based explanations for individual predictions.
- Improve feature-contribution visualizations across supported models.
- Improve handling of class imbalance and probability calibration.
- Add automated tests for data preparation, model loading, and predictions.
- Improve dashboard responsiveness and accessibility.
- Document model limitations and reproducibility details.

## 👨‍💻 Author

**Ram Kurra**

GitHub: [@Ramk09](https://github.com/Ramk09)

## 📄 License

No license has been added yet. If you intend to distribute or reuse this project publicly, choose and add an appropriate open-source license.
