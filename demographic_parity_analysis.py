import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import json

# Load the original data and evaluation results
data = pd.read_csv('data_more_factors.csv')
evaluation_results = pd.read_csv('evaluation_scores_pca_post.csv')

# Get the best classifier from evaluation results
best_classifier_row = evaluation_results.loc[evaluation_results['Accuracy'].idxmax()]
best_classifier_name = best_classifier_row.name

# Initialize the corresponding classifier based on the best results
classifier_mapping = {
    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(600, 500, 300, 200, 100), 
                                  activation='relu', solver='adam', alpha=0.0001,
                                  learning_rate='adaptive', max_iter=1000),
    'LogisticRegression': LogisticRegression(C=10, solver='saga', max_iter=500),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=200, max_depth=None, 
                                                    min_samples_split=5),
    'SVC': SVC(C=10, kernel='rbf', gamma='scale', probability=True),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
}

best_model = classifier_mapping['MLPClassifier']

# Prepare the data
X = data.drop('patient_category', axis=1)
y = data['patient_category'].map({'Barotrauma': 0, 'VAP': 1, 'VILI': 2})

# Save demographic columns before transformation
demographic_cols = ['race', 'gender']
demographics = X[demographic_cols].copy()

# Remove demographic columns for PCA
X_for_pca = X.drop(columns=demographic_cols)

# Split the data normally (just X and y)
X_train, X_test, y_train, y_test = train_test_split(
    X_for_pca, y, 
    test_size=0.2, random_state=42, stratify=y
)

# Get the corresponding demographic data for test set using the same indices
demographics_test = demographics.iloc[X_test.index]

# Apply PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the best model
best_model.fit(X_train_pca, y_train)

# Make predictions
y_pred = best_model.predict(X_test_pca)
y_pred_proba = best_model.predict_proba(X_test_pca)  # For Demographic Parity calculation

# Function to calculate all metrics for each class and gender
def calculate_all_metrics(y_true, y_pred, y_pred_proba, group_data):
    """Calculate all metrics for each class across gender groups."""
    results = []
    
    for class_label in range(3):  # 3 classes: Barotrauma (0), VAP (1), VILI (2)
        for gender in group_data.unique():
            mask = group_data == gender
            
            # Convert to binary classification problem for this class
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            
            # Calculate metrics
            metrics = {
                'Class': ['Barotrauma', 'VAP', 'VILI'][class_label],
                'Gender': 'Female' if gender == -0.806226 else 'Male',
                'Accuracy': accuracy_score(y_true_binary[mask], y_pred_binary[mask]),
                'Precision': precision_score(y_true_binary[mask], y_pred_binary[mask], zero_division=0),
                'Recall': recall_score(y_true_binary[mask], y_pred_binary[mask], zero_division=0),
                'F1': f1_score(y_true_binary[mask], y_pred_binary[mask], zero_division=0),
                'Demographic_Parity': y_pred_proba[mask, class_label].mean()
            }
            results.append(metrics)
        
    return pd.DataFrame(results)

def calculate_all_metrics_race(y_true, y_pred, y_pred_proba, group_data):
    """Calculate all metrics for each class across race groups."""
    results = []
    
    def clean_race_label(race_str):
        major_races = ['ASIAN', 'BLACK', 'WHITE', 'HISPANIC/LATINO']
        for major_race in major_races:
            if major_race in str(race_str).upper():
                return major_race
        return None  # Return None for non-major races
    
    # First clean all race labels and create a new series
    cleaned_races = group_data.apply(clean_race_label)
    
    for class_label in range(3):  # 3 classes: Barotrauma (0), VAP (1), VILI (2)
        for race in cleaned_races.unique():
            # Skip None values (non-major races)
            if race is None:
                continue
                
            # Use the cleaned race labels for masking
            mask = cleaned_races == race
            
            # Convert to binary classification problem for this class
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            
            # Calculate metrics
            metrics = {
                'Class': ['Barotrauma', 'VAP', 'VILI'][class_label],
                'Race': race,
                'Accuracy': accuracy_score(y_true_binary[mask], y_pred_binary[mask]),
                'Precision': precision_score(y_true_binary[mask], y_pred_binary[mask], zero_division=0),
                'Recall': recall_score(y_true_binary[mask], y_pred_binary[mask], zero_division=0),
                'F1': f1_score(y_true_binary[mask], y_pred_binary[mask], zero_division=0),
                'Demographic_Parity': y_pred_proba[mask, class_label].mean()
            }
            results.append(metrics)
    
    return pd.DataFrame(results)

# Calculate all metrics
demographics_test['gender'] = demographics_test['gender'].apply(
    lambda x: -0.806226 if abs(x - -0.806226) < abs(x - 1.240347) else 1.240347
)

# load race_mapping from json file
with open('race_mapping.json', 'r') as f:
    race_mapping = json.load(f)

# for race, search the most close standardized_value in race_mapping dictionary, then use the corresponding encoded_value
demographics_test['race'] = demographics_test['race'].apply(
    lambda x: min(race_mapping, key=lambda k: abs(race_mapping[k]['standardized_value'] - x))
)

# For gender metrics
print("\nGender Distribution in Test Set:")
gender_counts = demographics_test['gender'].apply(
    lambda x: 'Female' if x == -0.806226 else 'Male'
).value_counts()
print(gender_counts)

results_df = calculate_all_metrics(y_test, y_pred, y_pred_proba, demographics_test['gender'])

# For race metrics
def clean_race_label(race_str):
    major_races = ['ASIAN', 'BLACK', 'WHITE', 'HISPANIC/LATINO']
    for major_race in major_races:
        if major_race in str(race_str).upper():
            return major_race
    return None  # Return None for non-major races

# Clean race labels and get counts
cleaned_races = demographics_test['race'].apply(clean_race_label)
print("\nMajor Race Distribution in Test Set:")
race_counts = cleaned_races[cleaned_races.notna()].value_counts()
print(race_counts)

results_df_race = calculate_all_metrics_race(y_test, y_pred, y_pred_proba, demographics_test['race'])

# Round all numeric columns to 3 decimal places
numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Demographic_Parity']
results_df[numeric_columns] = results_df[numeric_columns].round(3)
results_df_race[numeric_columns] = results_df_race[numeric_columns].round(3)

# Display results
print("\nComplete Metrics by Gender and Class:")
print(results_df.to_string(index=False))

print("\nComplete Metrics by Race and Class:")
print(results_df_race.to_string(index=False))

# Save results to CSV
results_df.to_csv('gender_metrics_by_class.csv', index=False)
results_df_race.to_csv('race_metrics_by_class.csv', index=False)
