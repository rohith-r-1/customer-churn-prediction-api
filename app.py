from flask import Flask, request, jsonify
from src.inference import predict_churn

app = Flask(__name__)

@app.route("/predict_churn", methods=["POST"])
def predict():
    data = request.json

    required_keys = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return jsonify({"error": f"Missing keys: {missing}"}), 400

    proba, pred = predict_churn(data)

    return jsonify({
        "churn_probability": round(float(proba), 4),
        "churn_prediction": int(pred),
        "recommendation": "Target with retention offer" if pred == 1 else "No action needed"
    })

if __name__ == "__main__":
    app.run(debug=True)
