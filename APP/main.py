from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained artifacts
model = joblib.load(os.path.join(BASE_DIR, "ids_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "categorical_encoders.pkl"))

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/detection", methods=["GET", "POST"])
def detection():
    result = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            data = pd.read_csv(file)

            # Drop label column if exists
            if 'labels' in data.columns:
                data = data.drop('labels', axis=1)

            # Encode categorical columns SAFELY
            categorical_cols = ['protocol_type', 'service', 'flag']
            for col in categorical_cols:
                if col in data.columns:
                    le = encoders[col]

                    # Handle unseen values
                    data[col] = data[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

            # Scale features
            data_scaled = scaler.transform(data)

            # Predict
            prediction = model.predict(data_scaled)

            if 1 in prediction:
                result = "🚨 Attack Detected"
            else:
                result = "✅ Normal Traffic"

    return render_template("detection.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
