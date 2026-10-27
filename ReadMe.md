🎗️ Breast Cancer Prediction with PySpark and Streamlit

<!-- confirm this image  displays -->

![alt text](<Screenshots/cover_image.jpg>)

📌 Project Overview
This project demonstrates the use of machine learning to classify breast cancer cases as Benign or Malignant using the Breast Cancer Wisconsin (Diagnostic) dataset.
The model is trained using PySpark's Logistic Regression algorithm, with feature scaling applied during model development. The trained model parameters are then used to build an interactive web application with Streamlit, allowing users to submit input features and obtain predictions.
The project combines distributed data processing, machine learning and web application development in a single workflow.
Disclaimer: This project is intended for educational and demonstration purposes only. It is not a clinically validated diagnostic tool and must not be used to make medical decisions.

🎯 Project Objectives
Apply PySpark to a machine learning classification problem.
Develop a Logistic Regression model for binary classification.
Apply feature scaling consistently between training and prediction.
Export the trained model parameters for use outside the training environment.
Build an interactive prediction interface using Streamlit.
Support individual predictions and batch predictions through CSV uploads.

🛠️ Technologies Used
Python — programming language.
PySpark — data processing and machine learning.
Spark MLlib — Logistic Regression and model training.
NumPy — numerical computations.
Pandas — tabular data manipulation and CSV processing.
Streamlit — interactive web application.
Joblib — loading the exported model artifact.
Google Colab — Prototyping environment
Visual Studio Code — local development environment.

📊 Dataset
The project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which contains numerical features computed from digitised images of breast mass samples.
The dataset includes 30 numerical features describing characteristics such as:
Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Concave points
Symmetry
Fractal dimension
These characteristics are represented through mean, standard error and worst-value measurements.
The classification task is to distinguish between benign and malignant cases.
Dataset source: UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)
The features have a mix of positive and negative correlation with the diagnosis, as shown in the image below:

![alt text](<Screenshots/correlation_plot.png>)

⚙️ Project Workflow
Data preparation: Prepare the dataset for machine learning.
Feature processing: Select the numerical input features and apply the required preprocessing.
Model training: Train a Logistic Regression classifier using PySpark.
Model export: Save the parameters required to reproduce predictions.
Application development: Build a Streamlit interface for manual input and CSV uploads.
Prediction: Apply the same feature scaling and classification logic used during training.
Results: Display the predicted class and estimated malignant probability.
🚀 Features
Interactive Streamlit interface.
Manual entry of the 30 numerical features.
CSV template download for batch predictions.
CSV upload and input validation.
Individual and batch classification.
Estimated malignant probability.
Downloadable batch prediction results.
📁 Project Structure
breast-cancer-prediction-with-pyspark-and-streamlit/
│
├── bcw_app.py
├── bcw_model_training.py
├── wdbc.data
├── column_names.txt
├── requirements.txt
├── README.md
│
└── Screenshots/
    └──correlation_plot.png
    └──cover_image.jpg

💻 Installation and Setup
Prerequisites
Python installed on your system.
Visual Studio Code.
Git, if you want to clone the repository.
The exported model artifact: breast_cancer_model.pkl.
1. Clone the repository
git clone breast-cancer-prediction-with-pyspark-and-streamlit.git
cd breast-cancer-prediction-with-pyspark
Alternatively, download the repository and open the project folder in VS Code.
2. Create a virtual environment
Windows:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Run the application
streamlit run app.py
Open the local URL displayed in the terminal, usually:
http://localhost:8501

📈 Model Evaluation

Evaluation Metrics
Accuracy:
97%
Precision:
94%
Recall:
100%
F1-score:
97%

⚠️ Limitations
Predictions depend on the quality of the input features and the trained model.
The model has not been established as a clinically validated diagnostic system.
The estimated probability should not be interpreted as an individual's actual cancer risk.
The application requires the compatible exported model artifact to be available locally.

👨‍💻 Author: [Ibikunle Gabriel]
GitHub: [GitHub Profile](https://github.com/Ibikunleg/)
LinkedIn: [Linkedin](https://linkedin.com/in/ibikunle-gabriel)
Email: ibikunlesgabriel@gmail.com

If you find this project useful, consider giving the repository a ⭐ on GitHub.
