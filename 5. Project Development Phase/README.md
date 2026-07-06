# OptiCrop - Smart Agricultural Production Optimization Engine

OptiCrop is a production-ready, AI-powered Agricultural Production Optimization Engine. It helps farmers, researchers, agribusiness companies, and policymakers make intelligent crop selection decisions based on soil nutrients and environmental conditions.

---

## 🌟 Features
- **Crop Recommendation Engine**: Predicts the most suitable crop based on N, P, K, temperature, humidity, pH, and rainfall.
- **Environmental Feasibility Analysis**: Evaluates whether soil and climate variables are optimal for particular types of crops.
- **Multi-Algorithm Comparison**: Trains and compares 5 models (Logistic Regression, KNN, Decision Tree, Random Forest, and K-Means Clustering) to automatically select the highest accuracy model.
- **Glassmorphic Agricultural UI**: A premium, responsive web interface built with HTML5, CSS3, and Bootstrap 5.
- **Modular Data Pipeline**: Includes clean data loading, missing value imputation, duplicate removal, outlier logging, and feature scaling.
- **Persisted Artifacts**: Uses joblib/pickle serialization for saving the best model, standard scaler, and label encoders.

---

## 🎯 Objectives
- **Maximize Yield**: Guide crop selection to leverage maximum agricultural productivity.
- **Soil Conservation**: Analyze macronutrient profiles to prevent crop-soil mismatching.
- **Sustainable Farming**: Match crop water demands (rainfall) and pH thresholds for resource efficiency.
- **Support Researchers**: Provide complete EDA visualizations and model comparison tables for research.

---

## 📊 Dataset & Features
The application uses the standard **Crop Recommendation Dataset** containing 2200 rows of agricultural samples across 22 crops.

### Features analyzed:
1. **Nitrogen (N)**: Ratio of Nitrogen content in soil (mg/kg).
2. **Phosphorous (P)**: Ratio of Phosphorous content in soil (mg/kg).
3. **Potassium (K)**: Ratio of Potassium content in soil (mg/kg).
4. **Temperature**: Temperature in Celsius (°C).
5. **Humidity**: Relative humidity in percentage (%).
6. **pH**: pH value of the soil (0.0 to 14.0).
7. **Rainfall**: Rainfall in mm.
8. **Label (Target)**: Recommended crop type (Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee).

---

## 🛠️ Tech Stack
- **Backend**: Python 3.10+, Flask
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3 (Vanilla + Custom Animations), Bootstrap 5.3, FontAwesome 6.4

---

## 📁 Project Structure
```
smartBridge/ (OptiCrop Root)
├── app.py                      # Flask Web Application Server
├── train_model.py              # Orchestration script to preprocess, train, and save best model
├── requirements.txt            # Python dependencies
├── README.md                   # Project Documentation
│
├── dataset/
│   └── crop_recommendation.csv # Harvest dataset (downloaded automatically or placed manually)
│
├── models/
│   ├── preprocessing.py        # Data cleaning, outlier handling, scaling & encoding modules
│   ├── training.py             # Supervised/Unsupervised ML training, evaluation metrics
│   └── prediction.py           # Verification wrapper, bounds-checking and prediction execution
│
├── templates/
│   ├── layout.html             # Bootstrap master navbar/footer layout template
│   ├── index.html              # Soil nutrient input dashboard form
│   └── result.html             # Crop recommendation results presentation card
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom theme colors, glassmorphic card stylings, animations
│   ├── js/
│   │   └── script.js           # Client-side input validation and error prompts
│   └── images/                 # Generated EDA visualization figures
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       ├── feature_distributions.png
│       ├── feature_boxplots.png
│       └── feature_pairplot.png
│
├── notebooks/
│   └── Crop_Recommendation.ipynb # Jupyter notebook for interactive model development
│
└── utils/                      # Helper modules
```

---

## 🚀 Execution & Setup Guide

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Open terminal in the project directory and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Train Models & Run Preprocessing
Run the training script. This script will:
- Download the dataset to `dataset/` if not present.
- Perform Exploratory Data Analysis (EDA) and save visual plots under `static/images/`.
- Train and compare Logistic Regression, KNN, Decision Tree, and Random Forest.
- Output comparison metrics and save the best model to `model.pkl` and standard scaler to `scaler.pkl`.
```bash
python train_model.py
```

### 3. Launch Flask Server
Start the Flask web server:
```bash
python app.py
```
Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📈 Model Comparison Example
Below is an illustrative comparison of the accuracy metrics printed by the training execution:
| Model | Accuracy | Precision | Recall | F1-Score | Cross-Validation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 96.14% | 96.39% | 96.14% | 96.11% | 95.85% |
| K-Nearest Neighbors | 97.27% | 97.48% | 97.27% | 97.24% | 97.05% |
| Decision Tree | 98.64% | 98.71% | 98.64% | 98.63% | 98.35% |
| **Random Forest** | **99.55%** | **99.60%** | **99.55%** | **99.55%** | **99.20%** |

*Note: Random Forest is automatically selected and saved due to its near-perfect accuracy and high stability under cross-validation.*

---

## 🔮 Future Scope
- **Real-time API Integration**: Hook up coordinates to Weather APIs (OpenWeatherMap) to fetch temperature, humidity, and rainfall automatically based on GPS location.
- **Yield Forecasting**: Train regression models to predict actual crop yields (tons per hectare) rather than just recommending crop types.
- **Pesticide / Fertilizer Optimization**: Add recommendations for fertilizer doses (exact N-P-K additions needed to reach the recommended crop threshold).

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE details.
