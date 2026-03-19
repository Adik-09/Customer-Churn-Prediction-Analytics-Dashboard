# Customer Churn Prediction System

An end-to-end Machine Learning pipeline to predict customer churn, identify at-risk customers, and generate actionable risk assessments. Developed using Python, Scikit-Learn, and XGBoost.

## 🚀 Features

- **Data Preprocessing**: Handles missing values, correctly formats data types, and processes categorical columns using one-hot encoding.
- **Model Training**: Evaluates both **Random Forest** and **XGBoost** classifiers to predict customer churn.
- **Model Evaluation**: Calculates comprehensive metrics including Accuracy, Precision, Recall, and Confusion Matrix.
- **Risk Assessment**: Not only predicts churn but provides a probabilistic `Risk_Score` and categorizes customers into `Low`, `Medium`, or `High` risk segments for targeted retention strategies.
- **Automated Pipeline**: End-to-end execution script (`main.py`) handles data loading, preprocessing, training, evaluating, and outputting final predictions to a structured CSV file.

## 🛠️ Technologies & Libraries

- **Language**: Python 3.x
- **Data Manipulation**: `pandas`
- **Machine Learning**: `scikit-learn`, `xgboost`

## 📁 Project Structure

```text
├── data/
│   └── churn.csv                # Raw customer dataset (Input)
├── src/
│   ├── data_preprocessing.py    # Data cleaning and feature engineering
│   ├── evaluate.py              # Model evaluation metrics calculation
│   ├── model.py                 # Random Forest and XGBoost model definitions
│   └── utils.py                 # Helper functions (e.g., risk categorization)
├── main.py                      # Main execution script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation 
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd churn-ai-project
   ```

2. **Install dependencies:**
   Ensure you have the necessary libraries installed. 
   ```bash
   pip install pandas scikit-learn xgboost
   ```

3. **Run the Application:**
   Execute the main script to run the pipeline.
   ```bash
   python main.py
   ```

## 📊 Output
After running the script, the system will output the evaluation metrics for both models in the terminal and generate a `churn_predictions.csv` file. This file contains the original data appended with:
- `Churn_Prediction`: Model's discrete classification (0 or 1).
- `Risk_Score`: Probability of churn (0.0 to 1.0).
- `Risk_Category`: Stratification into Low, Medium, or High risk for easy analysis.
