from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("spam_model.pkl")

@app.route("/")
def home():
    return "Email Spam Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)

    if not data or "email" not in data:
        return jsonify({
            "error": "JSON body must contain 'email' field"
        }), 400

    email_text = data["email"]

    prediction = model.predict([email_text])[0]
    probability = model.predict_proba([email_text])[0][1]

    return jsonify({
        "spam": bool(prediction),
        "spam_probability": round(float(probability), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)