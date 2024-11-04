import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score

def data_preprocess(data, respone_var):
    # Drop columns nan above 0.6
    data = data.dropna(axis=1, thresh=int(0.95*data.shape[0]))
    # data = data[data['patient_category'] != 'VAT']
    # Drop rows with nan
    # data = data.dropna(axis=0)

    # fill antibiotic_given col nan as 0
    data['antibiotic_given'] = data['antibiotic_given'].fillna(0)

    #fill sputum_culture_positive col nan as 1
    data['sputum_culture_positive'] = data['sputum_culture_positive'].fillna(1)

    # fill blood_culture_positive col nan as 1
    data['blood_culture_positive'] = data['blood_culture_positive'].fillna(1)


    #encode response variable
    data[respone_var] = data[respone_var].astype('category')
    #data[respone_var] = data[respone_var].cat.codes


    # standardize data except response variable
    for col in data.columns:
        if col != respone_var:
            data[col] = (data[col] - np.mean(data[col]))/np.std(data[col])
    
    # drop col with nan
    data = data.dropna(axis=0)

    return data

def data_augmentation_with_smote(data):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(data.drop('patient_category', axis=1), data['patient_category'])
    data = pd.concat([X_res, y_res], axis=1)
    return data

# duplicate the less data in the dataset
def data_arguementation(data):
    # data_1 = data[data['patient_category'] == 'VAP']
    data_2 = data[data['patient_category'] == 'VAT']
    # data_3 = data[data['patient_category'] == 'Barotrauma']

    # data_1 = data_1.sample(n=100, replace=True)
    data_2 = data_2.sample(n=100, replace=True)
    data_3 = data_2.sample(n=100, replace=True)
    data_4 = data_2.sample(n=100, replace=True)
    data_5 = data_2.sample(n=100, replace=True)
    data_6 = data_2.sample(n=100, replace=True)
    data_7 = data_2.sample(n=100, replace=True)
    # data_3 = data_3.sample(n=100, replace=True)

    data = pd.concat([data, data_2, data_3, data_4, data_5, data_6, data_7], axis=0)

    return data

# load data
data = pd.read_csv('data_all.csv')

# data['patient_category'] = data['patient_category'].replace(['Unknown'], 'VILI', regex=True)

# drop first three columns
data = data.drop(data.columns[0:3], axis=1)
# drop comorbidity_icd_codes column
data = data.drop('comorbidity_icd_codes', axis=1)

data = data_preprocess(data, 'patient_category')
data = data_augmentation_with_smote(data)
#save data
data.to_csv('data_all_preprocessed_working.csv', index=False)
# show data catgories summary
#drop data patient_category VAT

# print(data['patient_category'].value_counts())

# # train a classifier
# # split data into training and testing
# # 80% training, 20% testing by sklearn

X_train, X_test, y_train, y_test = train_test_split(data.drop('patient_category', axis=1), data['patient_category'], test_size=0.2, random_state=42)

# # train a random forest classifier
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)

# # predict on test set
# y_pred = clf.predict(X_test)

# # evaluate the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print('Random forest Accuracy: ', accuracy)

# # fit LR classifier
# clf = LogisticRegression()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('LR Accuracy: ', accuracy)

# #fit pca
# pca = PCA(n_components=3)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
# # print the explained variance ratio
# # print('Explained variance ratio: ', pca.explained_variance_ratio_)

# # fit a random forest classifier on pca data
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train_pca, y_train)
# y_pred = clf.predict(X_test_pca)
# accuracy = accuracy_score(y_test, y_pred)
# print('Random Forest with PCA Accuracy: ', accuracy)

# # fit a neural network classifier with 3 hidden layers

clf = MLPClassifier(hidden_layer_sizes=(300, 200, 100), max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('MLP Accuracy: ', accuracy)
# print AUROC score 
y_pred_proba = clf.predict_proba(X_test)
auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print('MLP AUROC: ', auroc)
# print recall score
recall = recall_score(y_test, y_pred, average='weighted')
print('MLP Recall: ', recall)
# print F1 score
f1 = 2 * (recall * auroc) / (recall + auroc)
print('MLP F1: ', f1)



# # fit a support vector machine
# clf = SVC()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('SVC Accuracy: ', accuracy)

