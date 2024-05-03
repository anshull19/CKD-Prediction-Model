import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score

# Load and preprocess the data
df = pd.read_csv('./data/kidney_disease.csv')

columns_to_retain = ['sg', 'al', 'sc', 'hemo', 'pcv', 'htn', 'classification']
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)
df = df.dropna(axis=0)

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        continue
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(['classification'], axis=1)
y = df['classification']

x_scaler = MinMaxScaler()
x_scaler.fit(X)
column_names = X.columns
X[column_names] = x_scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Train the SVM model
svc_model = SVC(C=0.1, kernel='linear', gamma=1)
svc_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svc_model.predict(X_test)

# Calculate recall, precision, and F1-score
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

# Print the results
print(svc_model.predict(X_test))
print("Accuracy: ",svc_model.score(X_test, y_test)*100)
print("Recall: {:.2f}".format(recall))
print("Precision: {:.2f}".format(precision))
print("F1-score: {:.2f}".format(f1_score))