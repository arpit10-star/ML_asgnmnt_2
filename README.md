# Heart Disease Prediction

## Project Overview
This project focuses on predicting heart disease based on various health indicators and lifestyle factors. The goal is to build and evaluate several machine learning models to identify individuals at risk of heart disease, with a particular emphasis on handling imbalanced datasets.

## Dataset
The dataset used is `heart_2020_cleaned.csv`, which contains information about individuals' health, lifestyle, and whether they have heart disease. Key characteristics of the dataset include:
- **Size**: Approximately 300,000 entries with 18 features after cleaning.
- **Features**: Includes numerical (BMI, PhysicalHealth, MentalHealth, SleepTime) and categorical variables (Smoking, AlcoholDrinking, Stroke, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, Asthma, KidneyDisease, SkinCancer).
- **Target Variable**: `HeartDisease` (Binary: Yes/No).
- **Class Imbalance**: The target variable is highly imbalanced, with only about 9% of the entries indicating 'Yes' for heart disease.

## Methodology

1.  **Data Loading and Initial Exploration**: Loaded the dataset and performed initial checks for shape, data types, and basic statistics.
2.  **Data Cleaning**: Identified and removed duplicate rows. Checked for null values (none found).
3.  **Outlier Treatment**: Numerical features were analyzed for outliers using histograms and box plots. Outliers were capped using the Interquartile Range (IQR) method to prevent extreme values from disproportionately influencing the models.
4.  **Feature Engineering**: The target variable `HeartDisease` was mapped to numerical values (Yes: 1, No: 0). Categorical features were identified for one-hot encoding.
5.  **Data Splitting**: The dataset was split into training and testing sets using `StratifiedKFold` to maintain the class distribution of the imbalanced target variable in both sets.
6.  **Preprocessing Pipeline**: A `ColumnTransformer` and `Pipeline` were used to encapsulate preprocessing steps for both numerical and categorical features:
    *   **Numerical Features**: Winsorization (capping outliers) followed by `QuantileTransformer` (to transform features to a uniform distribution).
    *   **Categorical Features**: One-Hot Encoding with `drop='first'` to avoid multicollinearity.
7.  **Model Training and Evaluation**: The following classification models were trained and evaluated using `StratifiedKFold` cross-validation on the training data and then assessed on the test set:
    *   **Logistic Regression**
    *   **Decision Tree Classifier**
    *   **K-Nearest Neighbors (KNN) Classifier**
    *   **Gaussian Naive Bayes Classifier**
    *   **Random Forest Classifier**
    *   **XGBoost Classifier**

    Models were configured to handle class imbalance (e.g., `class_weight='balanced'` for some models).

8.  **Evaluation Metrics**: Performance was evaluated using:
    *   **ROC AUC Score**: A primary metric due to class imbalance.
    *   **Precision (Class 1)**, **Recall (Class 1)**, **F1-Score (Class 1)**: Crucial metrics for the minority class (Heart Disease: Yes).

## Results

The models were compared based on their ROC AUC scores, as well as precision, recall, and F1-score for the positive class (Heart Disease: Yes).

| Model                  | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC AUC Score |
|:-----------------------|:--------------------|:-----------------|:-------------------|:--------------|
| Logistic Regression    | 0.23                | 0.79             | 0.36               | 0.759         |
| Gaussian Naive Bayes   | 0.20                | 0.76             | 0.32               | 0.733         |
| Decision Tree          | 0.22                | 0.23             | 0.22               | 0.577         |
| KNN                    | 0.28                | 0.10             | 0.14               | 0.556         |
| XGBoost                | 0.52                | 0.10             | 0.17               | 0.546         |
| Random Forest          | 0.31                | 0.09             | 0.14               | 0.542         |

**Logistic Regression** performed the best in terms of ROC AUC score and achieved a good recall for the positive class, indicating its ability to identify a high proportion of actual heart disease cases, despite a lower precision.

## Dependencies

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `feature_engine`
*   `xgboost`
*   `joblib`

## How to Run

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install dependencies**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn feature_engine xgboost joblib
    ```
3.  **Place the dataset**: Ensure `heart_2020_cleaned.csv` is in the appropriate directory (e.g., `/content/drive/MyDrive/` if running in Colab as per the notebook, or adjust path).
4.  **Open and run the Jupyter Notebook/Colab Notebook** (`your_notebook_name.ipynb`).
5.  The trained models are saved as `.pkl` files (e.g., `lr_pipeline.pkl`, `XGB_pipeline.pkl`) in the project directory.
