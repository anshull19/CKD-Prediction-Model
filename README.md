# CKD Predictor

## Introduction

This project is a web application for predicting Chronic Kidney Disease (CKD) based on specific medical parameters provided by the user. The application uses a Support Vector Machine (SVM) model to make predictions based on data from a chronic kidney disease dataset.

## Features

- User-friendly web interface for inputting medical data.
- Predicts whether a patient has CKD or not based on provided data.
- Displays the probability of the prediction.
- Provides metrics such as recall, precision, and F1-score for model performance evaluation.

## How It Works

The application uses a machine learning model to predict CKD based on the following medical parameters provided by the user:

- **Specific Gravity** (`sg`): A measure of urine concentration.
- **Albumin** (`al`): Protein found in urine.
- **Serum Creatinine** (`sc`): A measure of kidney function.
- **Hemoglobin** (`hemo`): Red blood cell count.
- **Packed Cell Volume** (`pcv`): Percentage of red blood cells in blood.
- **Hypertension** (`htn`): Indicates whether the patient has high blood pressure.

The user inputs these parameters through the web interface, and the application uses the trained SVM model to make a prediction about the likelihood of CKD. The application then displays the prediction along with the associated probability.

## Dataset

The application uses the `kidney_disease.csv` dataset to train the machine learning model. This dataset is expected to be in the `Data/` directory.

## Technologies Used

- Python
- Flask
- scikit-learn (for machine learning)
- NumPy and pandas (for data handling and preprocessing)

## License

This project is open source and available under the [LICENSE](LICENSE) file.

---

Thank you for checking out the CKD Predictor project! If you have any questions or feedback, feel free to reach out.

