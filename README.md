# Comprehensive Machine Learning Pipeline on UCI / Kaggle Heart Disease

This repository implements a full machine learning pipeline around heart disease prediction. It handles data preprocessing, dimensionality reduction, feature selection, supervised and unsupervised modeling, hyperparameter tuning, and final deployment via a Streamlit user interface.

---

## Quickstart

```bash
# 1) Create a virtual environment and install dependencies
python -m venv .venv && . .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Obtain the dataset
#    The project will try to fetch the data in notebook 01; alternatively, you can manually download it from:
#    https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
#    and place it at data/heart_disease.csv

# 3) Run the notebooks in order (notebooks/)
# 4) After training, the pipeline will be saved to models/final_model.pkl

# 5) Launch the Streamlit app locally
streamlit run ui/app.py
````

---

## Live Demo (Ngrok)

You can also access the deployed app via Ngrok at the following link:

👉 [Heart Disease Predictor (Ngrok Deployment)](https://fugacious-lonnie-nonsalaried.ngrok-free.dev)

> Note: Ngrok links are temporary and may change when restarted. If the above link is inactive, please run the app locally as described in the Quickstart section.

---

## Repository Structure

```
Heart_Disease_Project/
├─ data/
│   └─ heart_disease.csv
├─ notebooks/
│   ├─ 01_data_preprocessing.ipynb
│   ├─ 02_pca_analysis.ipynb
│   ├─ 03_feature_selection.ipynb
│   ├─ 04_supervised_learning.ipynb
│   ├─ 05_unsupervised_learning.ipynb
│   └─ 06_hyperparameter_tuning.ipynb
├─ models/
│   └─ final_model.pkl
├─ ui/
│   └─ app.py
├─ deployment/
│   └─ ngrok_setup.txt
├─ results/
│   └─ evaluation_metrics.txt
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## Project Workflow

1. **Data Preprocessing**
   Handle missing values, encode categorical variables, scale numerical features, perform exploratory data analysis.

2. **Dimensionality Reduction**
   Apply PCA, visualize explained variance, and determine how many components to keep.

3. **Feature Selection**
   Use methods like feature importance (e.g. Random Forest), recursive feature elimination (RFE), and statistical tests.

4. **Supervised Learning**
   Train Logistic Regression, Decision Tree, Random Forest, SVM. Evaluate with accuracy, precision, recall, F1-score, ROC-AUC.

5. **Unsupervised Learning**
   Apply clustering techniques like K-Means and hierarchical clustering.

6. **Hyperparameter Tuning**
   Use GridSearchCV or RandomizedSearchCV to optimize models.

7. **Deployment**
   Export the final pipeline (`.pkl`) and serve it with Streamlit. Optionally deploy using Ngrok for public access.

---

## Final Deliverables

* Preprocessed dataset
* PCA analysis and visualizations
* Feature importance rankings
* Trained supervised and unsupervised models
* Model evaluation metrics
* Best tuned model
* Exported pipeline (`models/final_model.pkl`)
* Streamlit app for interactive predictions
* Public Ngrok link for remote access

---

## Dataset

This project uses the **Heart Disease Dataset** from Kaggle (Cleveland, Hungary, Switzerland, Long Beach V).
Dataset: [Heart Disease on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---
