#importing required library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from flask import Flask, request, render_template
from sklearn.metrics import recall_score, precision_score, f1_score

app = Flask(__name__)

# Loading and preprocessing the dataset
df = pd.read_csv('./data/kidney_disease.csv')
columns_to_retain=['sg','al','sc','hemo','pcv','htn','classification']
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)
df = df.dropna(axis=0)

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        continue
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(['classification'],axis=1)
y = df['classification']
x_scaler = MinMaxScaler()
x_scaler.fit(X)
column_names = X.columns
X[column_names] = x_scaler.transform(X)

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)

# Training the SVM model
svc_model = SVC(C= .1, kernel='linear', gamma=1, probability=True)
svc_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svc_model.predict(X_test)

# Calculate recall, precision, and F1-score
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

print(svc_model.predict(X_test))
print("The accuracy of this model is",svc_model.score(X_test, y_test)*100)
print("Recall: {:.2f}".format(recall))
print("Precision: {:.2f}".format(precision))
print("F1-score: {:.2f}".format(f1_score))

# function to predict the class and probability of CKD
def predict_ckd(features):
    x = np.array(features).reshape(1,-1)
    x = x_scaler.transform(x)
    y_pred = svc_model.predict(x)
    proba = svc_model.predict_proba(x)
    return y_pred[0], proba.max()

import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.template_folder = os.path.abspath('templates')

# Defining the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Defining the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    for feature in features:
        features[feature] = float(features[feature])
    pred, proba = predict_ckd(list(features.values()))
    if pred == 0:
        return render_template('index.html', prediction_text='The patient has CKD with probability {:.2f}%'.format(proba*100))
    else:
        return render_template('index.html', prediction_text='The patient does not have CKD with probability {:.2f}%'.format(proba*100))

# Running the web application
if __name__ == '__main__':
    app.run(debug=True)
