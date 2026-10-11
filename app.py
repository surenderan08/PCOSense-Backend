from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sqlite3
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Temporary database for serverless deployment testing.
# Data may be lost when the serverless instance is restarted.
DB_PATH = "/tmp/pcosense.db"

# Load trained files
try:
    model = joblib.load(os.path.join(MODEL_DIR, "pcos_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading PCOS model files: {e}") from e

# Median values calculated from training dataset to use as fallback defaults
DEFAULT_MEDIANS = {
    'Age (yrs)': 31.0,
    'Weight (Kg)': 59.0,
    'Height(Cm)': 156.0,
    'BMI': 24.2,
    'Blood Group': 14.0,
    'Pulse rate(bpm)': 72.0,
    'RR (breaths/min)': 18.0,
    'Hb(g/dl)': 11.0,
    'Cycle(R/I)': 2.0,
    'Cycle length(days)': 5.0,
    'Marraige Status (Yrs)': 7.0,
    'Pregnant(Y/N)': 0.0,
    'No. of aborptions': 0.0,
    'I beta-HCG(mIU/mL)': 20.0,
    'II beta-HCG(mIU/mL)': 1.99,
    'FSH(mIU/mL)': 4.85,
    'LH(mIU/mL)': 2.3,
    'FSH/LH': 2.1,
    'Hip(inch)': 38.0,
    'Waist(inch)': 34.0,
    'Waist:Hip Ratio': 0.89,
    'TSH (mIU/L)': 2.26,
    'AMH(ng/mL)': 3.7,
    'PRL(ng/mL)': 21.92,
    'Vit D3 (ng/mL)': 25.9,
    'PRG(ng/mL)': 0.32,
    'RBS(mg/dl)': 100.0,
    'Weight gain(Y/N)': 0.0,
    'hair growth(Y/N)': 0.0,
    'Skin darkening (Y/N)': 0.0,
    'Hair loss(Y/N)': 0.0,
    'Pimples(Y/N)': 0.0,
    'Fast food (Y/N)': 1.0,
    'Reg.Exercise(Y/N)': 0.0,
    'BP _Systolic (mmHg)': 110.0,
    'BP _Diastolic (mmHg)': 80.0,
    'Follicle No. (L)': 5.0,
    'Follicle No. (R)': 6.0,
    'Avg. F size (L) (mm)': 15.0,
    'Avg. F size (R) (mm)': 16.0,
    'Endometrium (mm)': 8.5
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction_result INTEGER,
            risk_status TEXT,
            probability REAL,
            inputs TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on load
init_db()

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to PCOSense API",
        "status": "Running"
    })

# ----------------------------------------------------
# Authentication Endpoints
# ----------------------------------------------------
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()

    if not username or not password or not name:
        return jsonify({"error": "All fields are required"}), 400

    password_hash = generate_password_hash(password)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash, name) VALUES (?, ?, ?)",
            (username, password_hash, name)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return jsonify({
            "message": "Signup successful",
            "user": {"id": user_id, "username": username, "name": name}
        }), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, name FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row[2], password):
            return jsonify({
                "message": "Login successful",
                "user": {"id": row[0], "username": row[1], "name": row[3]}
            }), 200
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------
# Prediction Endpoint
# ----------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    user_id = data.get("user_id", None) # Optional tracking for logged in users

    # Preprocess & dynamic calculation of basic stats
    # 1. BMI calculation
    weight = data.get("Weight (Kg)")
    height = data.get("Height(Cm)")
    if weight is not None and height is not None and height > 0:
        data["BMI"] = float(weight) / ((float(height) / 100.0) ** 2)

    # 2. Waist to Hip ratio
    waist = data.get("Waist(inch)")
    hip = data.get("Hip(inch)")
    if waist is not None and hip is not None and hip > 0:
        data["Waist:Hip Ratio"] = float(waist) / float(hip)

    # 3. FSH/LH Ratio
    fsh = data.get("FSH(mIU/mL)")
    lh = data.get("LH(mIU/mL)")
    if fsh is not None and lh is not None and lh > 0:
        data["FSH/LH"] = float(fsh) / float(lh)

    # Assemble inputs matching the 41 feature columns
    values = []
    for feature in feature_names:
        # Check if user sent a value, otherwise use dataset median
        val = data.get(feature)
        if val is None or val == "":
            val = DEFAULT_MEDIANS[feature]
        else:
            try:
                val = float(val)
            except ValueError:
                val = DEFAULT_MEDIANS[feature]
        values.append(val)

    # Convert to DataFrame
    df = pd.DataFrame([values], columns=feature_names)

    # Scale the inputs
    try:
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    prob_percentage = round(float(probability) * 100, 2)

    # Risk level classification
    if prediction == 1 or prob_percentage >= 70:
        risk_status = "High Risk of PCOS"
        risk_level = "High"
    elif prob_percentage >= 35:
        risk_status = "Moderate Risk of PCOS"
        risk_level = "Moderate"
    else:
        risk_status = "Low Risk of PCOS"
        risk_level = "Low"

    # Save to history if user_id is provided
    if user_id:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (user_id, prediction_result, risk_status, probability, inputs) VALUES (?, ?, ?, ?, ?)",
                (user_id, int(prediction), risk_status, prob_percentage, pd.Series(values, index=feature_names).to_json())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving prediction: {e}")

    # Generate custom dietary & physical recommendation content
    recommendations = get_recommendations(risk_level, data)

    return jsonify({
        "prediction": int(prediction),
        "risk_status": risk_status,
        "risk_level": risk_level,
        "probability": prob_percentage,
        "recommendations": recommendations
    })

# ----------------------------------------------------
# Prediction History Endpoint
# ----------------------------------------------------
@app.route("/api/predictions/history/<int:user_id>", methods=["GET"])
def get_history(user_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, prediction_result, risk_status, probability, inputs FROM predictions WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        history_list = []
        for row in rows:
            history_list.append({
                "id": row[0],
                "timestamp": row[1],
                "prediction_result": row[2],
                "risk_status": row[3],
                "probability": row[4],
                "inputs": json.loads(row[5]) if row[5] else {}
            })

        return jsonify({"history": history_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------
# Model Analytics Endpoint
# ----------------------------------------------------
@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    # Performance metrics extracted from train_model.py
    return jsonify({
        "models": [
            {
                "name": "Logistic Regression",
                "accuracy": 85.19,
                "precision": 86.40,
                "recall": 84.80,
                "f1": 85.59,
                "status": "Primary (Highest Accuracy)"
            },
            {
                "name": "Naive Bayes",
                "accuracy": 81.48,
                "precision": 80.20,
                "recall": 82.50,
                "f1": 81.33,
                "status": "Alternative"
            },
            {
                "name": "Decision Tree",
                "accuracy": 75.93,
                "precision": 74.50,
                "recall": 77.20,
                "f1": 75.82,
                "status": "Alternative"
            }
        ]
    })

# ----------------------------------------------------
# Recommendation Engine Helper
# ----------------------------------------------------
def get_recommendations(risk_level, inputs):
    # Base response lists
    diet = {
        "veg": [],
        "nonveg": []
    }
    exercises = {
        "home": [],
        "gym": []
    }
    medical_advice = ""

    if risk_level == "Low":
        diet["veg"] = [
            "Maintain balanced meals with healthy proteins (paneer, lentils, chickpeas).",
            "Incorporate anti-inflammatory seeds: 1 tsp pumpkin seeds and flaxseeds daily.",
            "Eat colorful seasonal vegetables (broccoli, bell peppers, carrots, leafy greens)."
        ]
        diet["nonveg"] = [
            "Incorporate grilled chicken breast or baked fish (salmon/mackerel) for high quality protein.",
            "Limit red meat; lean chicken is preferred.",
            "Egg whites are great for a high-protein breakfast."
        ]
        exercises["home"] = [
            "Aactive lifestyle: 30 minutes of brisk walking or light jogging.",
            "Surya Namaskar (Sun Salutation) - 6 to 10 rounds daily.",
            "Basic yoga stretches: Butterfly pose (Baddha Konasana) & Cobra pose (Bhujangasana)."
        ]
        exercises["gym"] = [
            "Cardio sessions: 20-30 minutes on treadmill or cross-trainer.",
            "Basic full-body compound workouts: squats, pushups, dumbbell rows.",
            "Moderate resistance training 2-3 times a week."
        ]
        medical_advice = "Your parameters indicate a low risk. Continue your active routine and consume a clean, whole-food diet to maintain hormonal balance."

    elif risk_level == "Moderate":
        diet["veg"] = [
            "Start the day with warm water and 1 cup of Spearmint Tea (helps reduce androgen/testosterone levels).",
            "Swap simple carbs with complex options: Oats, Brown Rice, Ragi, or Jowar Roti.",
            "Include high-fiber dal, boiled chana, and curd (probiotic) to support digestion and reduce insulin spikes."
        ]
        diet["nonveg"] = [
            "Incorporate 2 portions of grilled salmon or tuna weekly (high omega-3 fights inflammation).",
            "Lean chicken breast marinated in herbs, ginger-garlic paste and turmeric.",
            "Whole eggs (boiled) - 2 per day for essential healthy fats."
        ]
        exercises["home"] = [
            "Incorporate restorative yoga poses to balance hormones: Malasana (Garland pose), Butterfly pose, and Bridge pose (Setu Bandhasana).",
            "20-minute Home HIIT: Jumping jacks, spot jogging, bodyweight squats (3 rounds).",
            "Brisk walking for 45 minutes daily."
        ]
        exercises["gym"] = [
            "Strength training: Focus on large muscle groups (Legs, Back) to improve insulin sensitivity.",
            "Use resistance bands for active stretching and hip activation exercises.",
            "Low-intensity steady-state (LISS) cardio on the incline treadmill."
        ]
        medical_advice = "Your risk level is moderate. Subtle symptoms like cycle irregularity or weight changes could indicate mild hormonal changes. We recommend implementing these lifestyle changes and monitoring your cycle regularity."

    else: # High Risk
        diet["veg"] = [
            "Spearmint Tea twice a day (helps lower hirsutism/hair growth symptoms).",
            "Strict Low-Glycemic Index (GI) food: Avoid Maida, sugar, and white rice completely. Use Jowar, Bajra, and Quinoa.",
            "Incorporate anti-inflammatory foods: Turmeric milk, ginger, cinnamon water, and rich leafy greens (spinach, fenugreek/methi).",
            "Increase proteins: Sprouted sprouts salad, tofu, paneer, and double dal."
        ]
        diet["nonveg"] = [
            "Consume fatty fish (salmon, sardines, mackerel) 2-3 times a week (helps reduce inflammation and regularize cycles).",
            "Grilled skinless chicken breast with steamed broccoli.",
            "Avoid processed meats (sausages, salami) completely as they worsen insulin resistance."
        ]
        exercises["home"] = [
            "Targeted pelvic area yoga: Butterfly pose (hold for 3 mins), Supta Baddha Konasana (Reclining Butterfly), and Bhujangasana.",
            "Hiit/Cardio: 30 minutes of energetic spot jogging, jumping squats, and planks to burn visceral fat.",
            "Pranayama (Anulom Vilom and Kapalbhati) to manage cortisol/stress levels."
        ]
        exercises["gym"] = [
            "Heavy/Moderate Strength Training: Focus on building muscle mass (leg presses, deadlifts, chest presses) - building muscle is the fastest way to reverse insulin resistance.",
            "HIIT on the stationary bike: 30 seconds sprint, 1 minute slow pedaling for 15 minutes.",
            "Core conditioning to reduce abdominal weight accumulation."
        ]
        medical_advice = "Your risk score is high. This prediction strongly suggests signs of PCOS. We highly recommend booking an appointment with a gynecologist or endocrinologist for a diagnostic blood panel and ultrasound. (You can search and schedule with top specialists below using our integrated Practo appointment locator)."

    return {
        "diet": diet,
        "exercises": exercises,
        "medical_advice": medical_advice
    }


# ----------------------------------------------------
# Admin Owner Dashboard Endpoint
# ----------------------------------------------------
@app.route("/api/admin/dashboard", methods=["GET"])
def get_admin_dashboard():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # Get total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Get avg risk probability
        cursor.execute("SELECT AVG(probability) FROM predictions")
        avg_prob = cursor.fetchone()[0] or 0.0
        
        # Get high risk count
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE risk_status LIKE '%High%'")
        high_risk_count = cursor.fetchone()[0]
        
        # Get all users with prediction count
        cursor.execute("""
            SELECT u.id, u.username, u.name, u.created_at, COUNT(p.id) as test_count
            FROM users u
            LEFT JOIN predictions p ON u.id = p.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """)
        users_rows = cursor.fetchall()
        users_list = []
        for r in users_rows:
            users_list.append({
                "id": r[0],
                "username": r[1],
                "name": r[2],
                "created_at": r[3],
                "test_count": r[4]
            })
            
        # Get all prediction logs
        cursor.execute("""
            SELECT p.id, p.user_id, u.name, p.timestamp, p.risk_status, p.probability, p.inputs
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.timestamp DESC
        """)
        preds_rows = cursor.fetchall()
        preds_list = []
        for r in preds_rows:
            preds_list.append({
                "id": r[0],
                "user_id": r[1],
                "user_name": r[2] or "Anonymous",
                "timestamp": r[3],
                "risk_status": r[4],
                "probability": r[5],
                "inputs": json.loads(r[6]) if r[6] else {}
            })
            
        conn.close()
        
        return jsonify({
            "stats": {
                "total_users": total_users,
                "total_predictions": total_predictions,
                "avg_probability": round(avg_prob, 2),
                "high_risk_percentage": round((high_risk_count / total_predictions * 100), 2) if total_predictions > 0 else 0
            },
            "users": users_list,
            "predictions": preds_list
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)