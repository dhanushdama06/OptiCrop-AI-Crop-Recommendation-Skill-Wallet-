# 🌱 OptiCrop – AI Crop Recommendation System

## 📌 Overview

OptiCrop is an AI-powered Crop Recommendation System that helps farmers, agricultural researchers, and agribusiness professionals select the most suitable crop based on soil nutrients and environmental conditions.

The application uses Machine Learning algorithms trained on agricultural data to predict the best crop using:

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The project is developed using Flask for the web application and Scikit-learn for machine learning.

---

# 🚀 Features

- AI-powered crop recommendation
- User-friendly web interface
- Soil nutrient analysis
- Climate-based crop prediction
- Machine Learning model comparison
- Data preprocessing pipeline
- Model serialization using Pickle
- Responsive UI
- Error handling and input validation

---

# 🛠️ Technology Stack

## Frontend

- HTML5
- CSS3
- Bootstrap
- JavaScript

## Backend

- Python
- Flask

## Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib
- Pickle

## Data Visualization

- Matplotlib
- Seaborn

---

# 📂 Project Structure

```
smartBridge/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── Procfile
├── render.yaml
│
├── dataset/
│
├── models/
│   ├── prediction.py
│   ├── preprocessing.py
│   └── training.py
│
├── templates/
│
├── static/
│
├── notebooks/
│
└── utils/
```

---

# 📊 Dataset

The project uses the Crop Recommendation Dataset.

### Input Features

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

### Output

Recommended Crop

---

# 🤖 Machine Learning Models

The project evaluates multiple algorithms including:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- K-Means Clustering (comparison)

The best-performing model is saved and used for prediction.

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
```

Go to the project folder

```bash
cd your-repository
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🧠 How It Works

1. User enters soil nutrient values.
2. Input data is validated.
3. Data is preprocessed.
4. Machine Learning model predicts the best crop.
5. Result is displayed with crop information.

---

# 📈 Future Improvements

- Fertilizer Recommendation
- Crop Yield Prediction
- Weather API Integration
- Disease Detection
- Satellite Image Analysis
- Mobile Application
- Multi-language Support

---

# 📷 Screenshots

Add screenshots here.

Example:

- Home Page
- Prediction Page
- Result Page

---

# 🌍 Deployment

The project can be deployed using:

- Render
- Railway
- PythonAnywhere
- Heroku

---

# 📄 License

This project is created for educational and learning purposes.
