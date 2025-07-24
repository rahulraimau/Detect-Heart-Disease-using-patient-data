Heart Disease Prediction Project
Project Summary
This project aims to build machine learning models to predict the presence of heart disease based on various patient health metrics provided in the dataset. The notebook explores the data, preprocesses it, trains and evaluates different classification models, and provides an initial interpretation of the best performing model using SHAP values.

Notebook Structure
This notebook follows a standard machine learning workflow:

Data Loading: The dataset is loaded into a pandas DataFrame.
Data Exploration (EDA):
Checking data types and non-null values (.info()).
Generating descriptive statistics (.describe()).
Examining the distribution of the target variable (.value_counts()).
Visualizing the distribution of numerical features by target class.
Visualizing the distribution of categorical features by target class.
Data Preprocessing:
Standardizing column names (lowercasing, replacing spaces with underscores).
Converting categorical columns to string type for one-hot encoding.
Applying one-hot encoding to categorical features.
Separating features (X) and target (y).
Scaling numerical features using StandardScaler.
Model Training:
Splitting the data into training and testing sets.
Training a Logistic Regression model.
Training a Random Forest Classifier model.
Training an XGBoost Classifier model.
Model Evaluation:
Evaluating models using classification_report (precision, recall, f1-score, support, accuracy).
Plotting ROC curves and calculating AUC for each model.
Displaying the confusion matrix for the best-performing model (XGBoost).
Model Interpretation:
Using SHAP (SHapley Additive exPlanations) to interpret the predictions of the trained XGBoost model.
Visualizing individual predictions with waterfall plots.
Visualizing overall feature importance with bar plots.
Saved Artifacts:
Saving the trained XGBoost model using joblib.
Saving the fitted StandardScaler object using joblib.
Key Findings
Data Exploration Insights
The dataset contains 1190 entries with no missing values, making it ready for preprocessing and modeling.
The target variable, indicating the presence of heart disease, is relatively balanced, with approximately 53% positive cases (target=1) and 47% negative cases (target=0). This suggests that standard classification metrics are appropriate without significant concerns about class imbalance.
Visualizations of numerical features showed some differentiation between the target classes, particularly in 'age', 'max_heart_rate', and 'oldpeak'. For instance, patients with heart disease (target=1) tend to have a lower max heart rate achieved during exercise and higher oldpeak values compared to those without heart disease (target=0).
Categorical features also showed varying distributions across the target classes. 'Chest pain type' and 'exercise angina' appear to be particularly strong indicators, with certain categories within these features being more prevalent in patients with heart disease.
Model Performance Summary
Three different classification models were trained and evaluated: Logistic Regression, Random Forest, and XGBoost.

Logistic Regression: Achieved an accuracy of 86%, with balanced precision and recall for both classes (around 0.85-0.89).
Random Forest: Showed improved performance with an accuracy of 93%. It demonstrated high precision and recall for both classes (around 0.91-0.95), indicating strong predictive capability.
XGBoost: Also performed well with an accuracy of 92%, slightly lower than Random Forest but still very high. Its precision and recall were also strong (around 0.91-0.94).
Comparing the models based on the classification reports and ROC curves, the Random Forest Classifier appears to be the best-performing model on this dataset, exhibiting the highest overall accuracy and strong F1-scores for both classes. The ROC curves further illustrate that all three models perform significantly better than random chance, with Random Forest and XGBoost showing the largest Area Under the Curve (AUC).

Model Details and Performance
Three different classification models were trained and evaluated to predict heart disease:

Logistic Regression:

A linear model providing a baseline for comparison.
Performance:
Accuracy: 0.86
Precision (Class 0): 0.87
Recall (Class 0): 0.83
F1-score (Class 0): 0.85
Precision (Class 1): 0.85
Recall (Class 1): 0.89
F1-score (Class 1): 0.87
Random Forest Classifier:

An ensemble method using multiple decision trees.
Performance:
Accuracy: 0.93
Precision (Class 0): 0.91
Recall (Class 0): 0.95
F1-score (Class 0): 0.93
Precision (Class 1): 0.95
Recall (Class 1): 0.92
F1-score (Class 1): 0.94
XGBoost Classifier:

A gradient boosting framework known for its performance.
Performance:
Accuracy: 0.92
Precision (Class 0): 0.91
Recall (Class 0): 0.93
F1-score (Class 0): 0.92
Precision (Class 1): 0.94
Recall (Class 1): 0.92
F1-score (Class 1): 0.93
Based on these metrics, the Random Forest and XGBoost models significantly outperformed the Logistic Regression model, with the Random Forest Classifier showing slightly better overall accuracy and F1-scores on the test set used in the notebook.

SHAP Value Explanation
SHAP (SHapley Additive exPlanations) values are a game-theoretic approach used to explain the output of any machine learning model. They connect optimal credit allocation with local explanations using the classic Shapley values from game theory.

In this notebook, SHAP values are used to interpret the predictions of the trained XGBoost model. Their significance lies in providing model interpretability by:

Local Interpretability: Explaining how each feature contributes to the prediction for a single instance (e.g., why a specific patient was predicted to have heart disease). The waterfall plot shown visualizes the impact of each feature value on the individual prediction, pushing it from the base value towards the final output.
Global Interpretability: Providing an overview of the overall feature importance across the entire dataset. The bar plot of mean absolute SHAP values shows which features have the biggest impact on the model's output on average.
By using SHAP, we gain insight into the model's decision-making process, understanding which features are most influential in predicting heart disease and how they affect the predictions for individual patients.

Saved Artifacts
For future use and deployment, the following trained model and preprocessor have been saved to disk:

Trained XGBoost Model: Saved as heart_disease_model_xgb.pkl. This file contains the fitted XGBoost classifier that can be loaded to make new predictions without retraining.
Fitted StandardScaler: Saved as scaler.pkl. This file contains the fitted scaler object, which is necessary to preprocess new data in the same way the training data was scaled before making predictions with the saved model.
