import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
# Ignore warnings
import warnings
import logging

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
logging.getLogger().setLevel(logging.ERROR)


import json

data = pd.read_csv('data_more_factors.csv')
X = data.drop('patient_category', axis=1)
y = data['patient_category'].map({'Barotrauma': 0, 'VAP': 1, 'VILI': 2})
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print("Training set label distribution:\n", y_train.value_counts())
print("Test set label distribution:\n", y_test.value_counts())

importance_threshold = 0.014

# Fit RandomForestClassifier to calculate feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Sort features by importance and filter by threshold
feature_importances_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

important_features = feature_importances_df[
    feature_importances_df["Importance"] >= importance_threshold
]
selected_features = important_features["Feature"].tolist()
print(f"Selected {len(selected_features)} features with importance >= {importance_threshold}")
print("Selected Features and Importances:\n", important_features)

X_train_rf = X_train[selected_features]
X_test_rf = X_test[selected_features]

plt.figure(figsize=(10, 6))
plt.bar(range(len(important_features)), important_features["Importance"], align='center')
plt.xticks(range(len(important_features)), important_features["Feature"], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Selected Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig('selected_feature_importances_rf.png')
plt.show()
plt.clf()

# PCA with variance-based component selection
pca = PCA(n_components=0.95)  # Retain components that explain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Output PCA results
print("Number of PCA Components:", pca.n_components_)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance Ratio:", np.sum(pca.explained_variance_ratio_))

classifiers = [
    ("MLPClassifier", MLPClassifier(hidden_layer_sizes=(300, 200, 100), max_iter=3000)),
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVC", SVC(probability=True, random_state=42)),
    ("XGBoost", XGBClassifier(random_state=42)),
]

y_test_binarized = label_binarize(y_test, classes=[0, 1, 2]) if len(set(y_test)) > 2 else y_test

# Function to evaluate classifiers and store metrics
def evaluate_classifiers(X_train, X_test, y_train, y_test, y_test_binarized, method_name):
    scores = {}
    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate AUC-ROC for binary or multi-class cases
        if y_proba is not None:
            auc_roc = roc_auc_score(y_test_binarized, y_proba, multi_class="ovr") if y_proba.shape[1] > 2 else roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc_roc = None

        # Calculate F1-score, Recall, and Precision
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        # Store metrics
        scores[clf_name] = {
            "Accuracy": accuracy,
            "AUC-ROC": auc_roc,
            "F1-score": f1,
            "Recall": recall,
            "Precision": precision
        }
        print(f"{method_name} - {clf_name} - Accuracy: {accuracy:.4f}, AUC-ROC: {auc_roc:.4f}, F1-score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}" if auc_roc else f"{method_name} - {clf_name} - Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    return scores

# Evaluate classifiers using Random Forest-selected features
print("Evaluation using Random Forest-selected features:")
evaluation_scores_rf_pre = evaluate_classifiers(
    X_train_rf, X_test_rf, y_train, y_test, y_test_binarized, "Random Forest Features"
)

# Evaluate classifiers using PCA-transformed features
print("\nEvaluation using PCA-transformed features:")
evaluation_scores_pca_pre = evaluate_classifiers(
    X_train_pca, X_test_pca, y_train, y_test, y_test_binarized, "PCA Features"
)

# Find best classifier for each feature selection method
best_rf_classifier_name = max(evaluation_scores_rf_pre, key=lambda x: evaluation_scores_rf_pre[x]["Accuracy"])
best_pca_classifier_name = max(evaluation_scores_pca_pre, key=lambda x: evaluation_scores_pca_pre[x]["Accuracy"])

print(f"\nBest Classifier (Random Forest Features): {best_rf_classifier_name} with Accuracy: {evaluation_scores_rf_pre[best_rf_classifier_name]['Accuracy']:.4f}, AUC-ROC: {evaluation_scores_rf_pre[best_rf_classifier_name]['AUC-ROC']:.4f}, F1-score: {evaluation_scores_rf_pre[best_rf_classifier_name]['F1-score']:.4f}, Recall: {evaluation_scores_rf_pre[best_rf_classifier_name]['Recall']:.4f}, Precision: {evaluation_scores_rf_pre[best_rf_classifier_name]['Precision']:.4f}")
print(f"Best Classifier (PCA Features): {best_pca_classifier_name} with Accuracy: {evaluation_scores_pca_pre[best_pca_classifier_name]['Accuracy']:.4f}, AUC-ROC: {evaluation_scores_pca_pre[best_pca_classifier_name]['AUC-ROC']:.4f}, F1-score: {evaluation_scores_pca_pre[best_pca_classifier_name]['F1-score']:.4f}, Recall: {evaluation_scores_pca_pre[best_pca_classifier_name]['Recall']:.4f}, Precision: {evaluation_scores_pca_pre[best_pca_classifier_name]['Precision']:.4f}")

from sklearn.model_selection import GridSearchCV

# param_grids = {
#     "MLPClassifier": {
#         "hidden_layer_sizes": [
#             (100,), (100, 50), (300, 200, 100), (500, 300, 200, 100), (600, 500, 300, 200, 100)
#         ],
#         "activation": ["relu", "tanh"],
#         "solver": ["adam"],
#         "alpha": [0.0001, 0.001, 0.01],
#         "learning_rate": ["constant", "adaptive"],
#         "max_iter": [500, 1000],
#         "early_stopping": [False, True]
#     },
#     "LogisticRegression": {
#         "C": [0.1, 1, 10],
#         "solver": ["liblinear", "saga"],
#         "max_iter": [100, 500, 1000]
#     },
#     "RandomForestClassifier": {
#         "n_estimators": [50, 100, 200],
#         "max_depth": [None, 10, 20],
#         "min_samples_split": [2, 5, 10]
#     },
#     "SVC": {
#         "C": [10],
#         "kernel": ["rbf"],
#         "gamma": ["scale"]
#     },
#     "XGBoost": {
#         "n_estimators": [50, 100, 200],
#         "max_depth": [3, 6, 10],
#         "learning_rate": [0.01, 0.1, 0.2],
#         "early_stopping_rounds": [None]
#     }
# }

param_grids = {
    "MLPClassifier": {
        "hidden_layer_sizes": [
            (600, 500, 300, 200, 100)
        ],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001],
        "learning_rate": ["adaptive"],
        "max_iter": [1000],
        "early_stopping": [False]
    },
    "LogisticRegression": {
        "C": [10],
        "solver": ["saga"],
        "max_iter": [500]
    },
    "RandomForestClassifier": {
        "n_estimators": [200],
        "max_depth": [None],
        "min_samples_split": [5,]
    },
    "SVC": {
        "C": [10],
        "kernel": ["rbf"],
        "gamma": ["scale"]
    },
    "XGBoost": {
        "n_estimators": [200],
        "max_depth": [6,],
        "learning_rate": [0.1],
        "early_stopping_rounds": [None]
    }
}


# Modified evaluation function to include GridSearchCV
def evaluate_classifiers_with_grid_search(X_train, X_test, y_train, y_test, y_test_binarized, method_name):
    scores = {}
    for clf_name, clf in classifiers:
        # Apply GridSearchCV with the parameter grid for each classifier
        grid_search = GridSearchCV(
            clf, param_grids[clf_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        # Get the best estimator and evaluate on the test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate AUC-ROC for binary or multi-class cases
        if y_proba is not None:
            auc_roc = roc_auc_score(y_test_binarized, y_proba, multi_class="ovr") if y_proba.shape[1] > 2 else roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc_roc = None

        # Calculate F1-score, Recall, and Precision
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        # Store metrics and best parameters
        scores[clf_name] = {
            "Best Params": grid_search.best_params_,
            "Accuracy": accuracy,
            "AUC-ROC": auc_roc,
            "F1-score": f1,
            "Recall": recall,
            "Precision": precision
        }
        print(f"{method_name} - {clf_name} - Best Params: {grid_search.best_params_}, Accuracy: {accuracy:.4f}, AUC-ROC: {auc_roc:.4f}, F1-score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}" if auc_roc else f"{method_name} - {clf_name} - Best Params: {grid_search.best_params_}, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    return scores

# Evaluate classifiers with GridSearchCV using Random Forest-selected features
# print("Grid Search Evaluation using Random Forest-selected features:")
# evaluation_scores_rf_post = evaluate_classifiers_with_grid_search(
#     X_train_rf, X_test_rf, y_train, y_test, y_test_binarized, "Random Forest Features"
# )

# Evaluate classifiers with GridSearchCV using PCA-transformed features
print("\nGrid Search Evaluation using PCA-transformed features:")
evaluation_scores_pca_post = evaluate_classifiers_with_grid_search(
    X_train_pca, X_test_pca, y_train, y_test, y_test_binarized, "PCA Features"
)

# Find best classifier for each feature selection method
# best_rf_classifier_name = max(evaluation_scores_rf_post, key=lambda x: evaluation_scores_rf_post[x]["Accuracy"])
best_pca_classifier_name = max(evaluation_scores_pca_post, key=lambda x: evaluation_scores_pca_post[x]["Accuracy"])

# print(f"\nBest Classifier (Random Forest Features): {best_rf_classifier_name} with Best Params: {evaluation_scores_rf_post[best_rf_classifier_name]['Best Params']}, Accuracy: {evaluation_scores_rf_post[best_rf_classifier_name]['Accuracy']:.4f}, AUC-ROC: {evaluation_scores_rf_post[best_rf_classifier_name]['AUC-ROC']:.4f}, F1-score: {evaluation_scores_rf_post[best_rf_classifier_name]['F1-score']:.4f}, Recall: {evaluation_scores_rf_post[best_rf_classifier_name]['Recall']:.4f}, Precision: {evaluation_scores_rf_post[best_rf_classifier_name]['Precision']:.4f}")
print(f"Best Classifier (PCA Features): {best_pca_classifier_name} with Best Params: {evaluation_scores_pca_post[best_pca_classifier_name]['Best Params']}, Accuracy: {evaluation_scores_pca_post[best_pca_classifier_name]['Accuracy']:.4f}, AUC-ROC: {evaluation_scores_pca_post[best_pca_classifier_name]['AUC-ROC']:.4f}, F1-score: {evaluation_scores_pca_post[best_pca_classifier_name]['F1-score']:.4f}, Recall: {evaluation_scores_pca_post[best_pca_classifier_name]['Recall']:.4f}, Precision: {evaluation_scores_pca_post[best_pca_classifier_name]['Precision']:.4f}")

# Save evaluation results to CSV files
evaluation_df_rf_pre = pd.DataFrame(evaluation_scores_rf_pre).T
evaluation_df_rf_pre.to_csv('evaluation_scores_rf_pre.csv')

evaluation_df_pca_pre = pd.DataFrame(evaluation_scores_pca_pre).T
evaluation_df_pca_pre.to_csv('evaluation_scores_pca_pre.csv')

# evaluation_df_rf_post = pd.DataFrame(evaluation_scores_rf_post).T
# evaluation_df_rf_post.to_csv('evaluation_scores_rf_post.csv')

evaluation_df_pca_post = pd.DataFrame(evaluation_scores_pca_post).T
evaluation_df_pca_post.to_csv('evaluation_scores_pca_post.csv')

# Ensure correct feature dimensions for SHAP analysis based on selected model and features
# if best_rf_classifier_name in dict(classifiers):
#     # Train and explain with Random Forest-selected features
#     best_rf_model = dict(classifiers)[best_rf_classifier_name]
#     best_rf_model.fit(X_train_rf, y_train)

#     print("\nSHAP Analysis for Best Model (Random Forest-selected features):")
#     explainer_rf = shap.KernelExplainer(best_rf_model.predict, X_train_rf)
#     shap_values_rf = explainer_rf.shap_values(X_test_rf)

#     # Global feature importance using SHAP for Random Forest-selected features
#     shap.summary_plot(shap_values_rf, X_test_rf, plot_type="bar", show=False)
#     plt.tight_layout()
#     plt.savefig('shap_summary_bar_rf.png')
#     plt.clf()

#     shap.summary_plot(shap_values_rf, X_test_rf, plot_type="dot", show=False)
#     plt.tight_layout()
#     plt.savefig('shap_summary_dot_rf.png')
#     plt.clf()

#     shap.summary_plot(shap_values_rf, X_test_rf, show=False)
#     plt.tight_layout()
#     plt.savefig('shap_summary_rf.png')
#     plt.clf()

if best_pca_classifier_name in dict(classifiers):
    # Train and explain with PCA-transformed features
    best_pca_model = dict(classifiers)[best_pca_classifier_name]
    best_pca_model.fit(X_train_pca, y_train)

    print("\nSHAP Analysis for Best Model (PCA-transformed features):")
    explainer_pca = shap.KernelExplainer(best_pca_model.predict, X_train_pca)
    shap_values_pca = explainer_pca.shap_values(X_test_pca)

    # Global feature importance using SHAP for PCA-transformed features
    shap.summary_plot(shap_values_pca, X_test_pca, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_bar_pca.png')
    plt.clf()

    shap.summary_plot(shap_values_pca, X_test_pca, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_dot_pca.png')
    plt.clf()

    shap.summary_plot(shap_values_pca, X_test_pca, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_pca.png')
    plt.clf()

