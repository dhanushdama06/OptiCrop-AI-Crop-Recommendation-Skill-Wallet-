# 🌱 OptiCrop – AI Crop Recommendation System

## 📌 Overview

OptiCrop is an AI-powered Crop Recommendation System that helps farmers, agricultural researchers, and agribusiness professionals select the most suitable crop based on soil nutrients and environmental conditions.

The application uses Machine Learning algorithms trained on agricultural data to predict the most suitable crop using the following input parameters:

* Nitrogen (N)
* Phosphorous (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

The project is developed using **Flask** for the web application and **Scikit-learn** for building and deploying the Machine Learning model.

---

# 🚀 Features

* 🌱 AI-powered crop recommendation
* 💻 User-friendly web interface
* 🌾 Soil nutrient analysis
* 🌦️ Climate-based crop prediction
* 🤖 Machine Learning model comparison
* 📊 Data preprocessing pipeline
* 💾 Model serialization using Pickle
* 📱 Responsive user interface
* ✅ Error handling and input validation

---

# 🛠️ Technology Stack

## Frontend

* HTML5
* CSS3
* Bootstrap
* JavaScript

## Backend

* Python
* Flask

## Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib
* Pickle

## Data Visualization

* Matplotlib
* Seaborn

---

# 📂 Project Structure

```text
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

The project uses the **Crop Recommendation Dataset**.

### Input Features

* Nitrogen (N)
* Phosphorous (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

### Output

* Recommended Crop

---

# 🤖 Machine Learning Models

The project evaluates multiple Machine Learning algorithms, including:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* K-Means Clustering (for comparison)

After evaluation, the best-performing model is serialized using **Pickle** and used for real-time crop prediction.

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
```

## Navigate to the Project Folder

```bash
cd your-repository
```

## Create a Virtual Environment

```bash
python -m venv venv
```

## Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

## Open in Browser

```text
http://127.0.0.1:5000
```

---

# 🧠 How It Works

1. The user enters soil nutrient and environmental values.
2. The application validates the input.
3. Data is preprocessed before prediction.
4. The trained Machine Learning model predicts the most suitable crop.
5. The recommended crop is displayed to the user through the web interface.

---

# 🎥 Project Demo

Watch the complete demonstration of the project here:

**Demo Video:**
https://drive.google.com/file/d/1OQef_NYbdbY9Ulw7Mxl5vK7ooMGQgZYz/view?usp=sharing

---

# 🌐 Live Website

Try the deployed application here:

**OptiCrop Live Website:**
https://smartbridge-opticrop.onrender.com

---

# 📈 Future Improvements

* 🌱 Fertilizer Recommendation System
* 📈 Crop Yield Prediction
* ☁️ Weather API Integration
* 🍃 Plant Disease Detection
* 🛰️ Satellite Image Analysis
* 📱 Mobile Application
* 🌍 Multi-language Support

---

# 📷 Screenshots

Add screenshots of the application here.

Suggested screenshots:

* Home Page
* About Page
* Prediction Page
* Prediction Result Page

---

# 🌍 Deployment

This project can be deployed using platforms such as:

* Render
* Railway
* PythonAnywhere
* Heroku

---

# 📄 License

This project is developed for educational and learning purposes.
