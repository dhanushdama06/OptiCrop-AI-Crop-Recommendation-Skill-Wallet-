import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from models.prediction import CropPredictor, PredictionError

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "opti_crop_secret_key_for_flash_messages"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Crop Predictor
predictor = CropPredictor()

@app.route("/")
def home():
    """Renders the Home Page where users input variables."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles crop prediction form submission."""
    # Retrieve inputs from POST request
    form_data = {
        'N': request.form.get("nitrogen"),
        'P': request.form.get("phosphorous"),
        'K': request.form.get("potassium"),
        'temperature': request.form.get("temperature"),
        'humidity': request.form.get("humidity"),
        'ph': request.form.get("ph"),
        'rainfall': request.form.get("rainfall")
    }
    
    logger.info(f"Received prediction request with raw form data: {form_data}")
    
    try:
        # Load models and predict
        crop_prediction = predictor.predict(form_data)
        
        # Format input values for displaying in result
        formatted_inputs = {
            "Nitrogen (N)": f"{float(form_data['N']):.1f} mg/kg",
            "Phosphorous (P)": f"{float(form_data['P']):.1f} mg/kg",
            "Potassium (K)": f"{float(form_data['K']):.1f} mg/kg",
            "Temperature": f"{float(form_data['temperature']):.1f} °C",
            "Humidity": f"{float(form_data['humidity']):.1f} %",
            "pH level": f"{float(form_data['ph']):.1f}",
            "Rainfall": f"{float(form_data['rainfall']):.1f} mm"
        }
        
        # Define informational details about the crop to display
        # We can write a dictionary mapping crops to their nutritional requirements and brief descriptions
        crop_info = get_crop_details(crop_prediction)
        
        return render_template("result.html", 
                               prediction=crop_prediction, 
                               inputs=formatted_inputs,
                               details=crop_info,
                               raw_inputs=form_data)
                               
    except PredictionError as e:
        logger.warning(f"Validation/Prediction error: {str(e)}")
        flash(str(e), "danger")
        return render_template("index.html", form_values=form_data)
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {str(e)}")
        flash("The machine learning models are not trained yet. Please run 'train_model.py' to initialize.", "warning")
        return render_template("index.html", form_values=form_data)
    except Exception as e:
        logger.exception("Unexpected error in prediction route:")
        flash("An unexpected error occurred. Please verify your inputs and try again.", "danger")
        return render_template("index.html", form_values=form_data)

def get_crop_details(crop_name: str) -> dict:
    """Returns description and characteristics for the recommended crop."""
    crop_info_dict = {
        "rice": {
            "category": "Cereal",
            "description": "Rice is a staple food starch source. It thrives in high rainfall, heavy soils, and high temperatures.",
            "care": "Requires continuous standing water during growth. Prefers nitrogen-rich clay soils.",
            "duration": "100 - 150 days"
        },
        "maize": {
            "category": "Cereal",
            "description": "Maize (Corn) is widely grown globally and is a versatile crop for food, feed, and biofuel.",
            "care": "Requires well-drained loamy soil, moderate rainfall, and high nitrogen/potassium levels.",
            "duration": "90 - 120 days"
        },
        "chickpea": {
            "category": "Legume / Pulse",
            "description": "Chickpeas (Garbanzo beans) are highly nutritious pulses that help fix nitrogen in the soil.",
            "care": "Requires dry weather, low humidity, and sandy-loam soils. Highly drought-tolerant.",
            "duration": "90 - 110 days"
        },
        "kidneybeans": {
            "category": "Legume / Pulse",
            "description": "Kidney beans are rich in plant proteins and complex carbohydrates.",
            "care": "Requires moderate temperature, average rainfall, and slightly acidic pH soils.",
            "duration": "80 - 100 days"
        },
        "pigeonpeas": {
            "category": "Legume / Pulse",
            "description": "Pigeonpeas are drought-resistant pulses essential for soil nitrogen enrichment.",
            "care": "Grows well in warm climates with low rainfall and deep loam soils.",
            "duration": "140 - 180 days"
        },
        "mothbeans": {
            "category": "Legume / Pulse",
            "description": "Mothbeans are extremely drought-resistant pulses primarily grown in arid regions.",
            "care": "Requires dry, hot climates and sandy soils. Extremely low water demand.",
            "duration": "75 - 90 days"
        },
        "mungbean": {
            "category": "Legume / Pulse",
            "description": "Mungbeans are fast-maturing crops with high nutrient value and low water requirements.",
            "care": "Thrives in warm temperatures, sandy loam soils with good drainage.",
            "duration": "60 - 75 days"
        },
        "blackgram": {
            "category": "Legume / Pulse",
            "description": "Blackgram is a popular Indian pulse containing high protein. It helps replenish soil nitrogen.",
            "care": "Requires loamy soil and moderate rainfall with warm temperatures.",
            "duration": "80 - 90 days"
        },
        "lentil": {
            "category": "Legume / Pulse",
            "description": "Lentils are highly nutritious cool-season pulses grown extensively in dry climates.",
            "care": "Prefers cool conditions, moderate rainfall, and deep, fertile sandy loam soils.",
            "duration": "100 - 120 days"
        },
        "pomegranate": {
            "category": "Fruit",
            "description": "Pomegranates are high-value fruits loaded with antioxidants and adapt to arid conditions.",
            "care": "Requires warm, dry summers, well-drained soils, and moderate nitrogen inputs.",
            "duration": "2 - 3 years to bear fruit"
        },
        "banana": {
            "category": "Fruit",
            "description": "Bananas are high-energy fruits grown in humid tropical regions.",
            "care": "Needs very rich, well-draining organic soil, high nitrogen, high potassium, and abundant water.",
            "duration": "9 - 12 months to harvest"
        },
        "mango": {
            "category": "Fruit",
            "description": "Mangoes are tropical stone fruits known as the king of fruits, thriving in warm climates.",
            "care": "Prefers dry weather during flowering, deep alluvial soils, and minimal water during winter.",
            "duration": "3 - 5 years for grafted trees"
        },
        "grapes": {
            "category": "Fruit",
            "description": "Grapes are woody deciduous vines used for fresh consumption, raisins, and winemaking.",
            "care": "Requires warm dry weather, good drainage, deep loamy soils, and supportive trellises.",
            "duration": "2 - 3 years to harvest"
        },
        "watermelon": {
            "category": "Fruit",
            "description": "Watermelons are vining annual plants requiring warm climates and sandy soils.",
            "care": "Requires warm soils, high sunlight, low humidity to avoid fungal diseases, and regular watering.",
            "duration": "80 - 90 days"
        },
        "muskmelon": {
            "category": "Fruit",
            "description": "Muskmelons are sweet summer melons with high water content.",
            "care": "Thrives in dry, warm conditions, light well-drained soils, and moderate watering.",
            "duration": "70 - 90 days"
        },
        "apple": {
            "category": "Fruit",
            "description": "Apples are temperate fruits requiring cold winters (chilling hours) to break bud dormancy.",
            "care": "Prefers cool climates, well-drained loamy soils, and moderate rainfall.",
            "duration": "3 - 5 years to bear"
        },
        "orange": {
            "category": "Fruit",
            "description": "Oranges are citrus fruits rich in Vitamin C, grown in warm and subtropical climates.",
            "care": "Prefers well-drained soils, moderate nitrogen, and regular water without waterlogging.",
            "duration": "3 - 4 years to bear"
        },
        "papaya": {
            "category": "Fruit",
            "description": "Papaya is a fast-growing tropical fruit tree rich in proteolytic enzymes.",
            "care": "Prefers warm climate, high sunlight, rich soil, and needs protection from strong winds and frost.",
            "duration": "9 - 10 months to harvest"
        },
        "coconut": {
            "category": "Fruit / Palm",
            "description": "Coconut palms grow along coastal tropical regions and yield fruit, water, oil, and fiber.",
            "care": "Requires warm humid climate, sandy soils, high rainfall, and salt tolerance.",
            "duration": "5 - 7 years to bear"
        },
        "cotton": {
            "category": "Cash Crop / Fiber",
            "description": "Cotton is the leading natural fiber crop, highly important for textile manufacturing.",
            "care": "Requires long frost-free periods, high sun, moderate water, and deep black soils.",
            "duration": "150 - 180 days"
        },
        "jute": {
            "category": "Cash Crop / Fiber",
            "description": "Jute is a shiny vegetable fiber called the 'Golden Fiber' due to its color and value.",
            "care": "Thrives in alluvial soils, high rainfall (>1500mm), and high humidity (>80%).",
            "duration": "120 - 150 days"
        },
        "coffee": {
            "category": "Beverage Crop",
            "description": "Coffee plants produce beans that are roasted to make the popular caffeinated beverage.",
            "care": "Grows best in shaded hill slopes, cool and humid climate, and acidic rich loam soils.",
            "duration": "3 - 4 years to bear beans"
        }
    }
    
    key = crop_name.lower().replace(" ", "").replace("_", "")
    return crop_info_dict.get(key, {
        "category": "Agricultural Crop",
        "description": f"The model recommends cultivating {crop_name.capitalize()} under the analyzed soil and climate profile.",
        "care": "Observe balanced nutrient application and monitor moisture based on local extensions.",
        "duration": "Varies by local cultivar and climate"
    })

if __name__ == "__main__":
    # Get port from environment or use 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
