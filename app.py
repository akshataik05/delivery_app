from flask import Flask, request, render_template, jsonify
import joblib, json

app = Flask(__name__)

# --- Load model & features safely ---
try:
    model = joblib.load('rf_delivery_model.joblib')
    with open('features.json', 'r') as f:
        features = json.load(f)
    print("✅ Model & features loaded successfully.")
except Exception as e:
    print("❌ Error loading model or features:", e)
    model = None
    features = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'})

    try:
        data = request.get_json(force=True)
        print("📦 Incoming payload:", data)
        x = [data.get(f, 0) for f in features]
        pred = float(model.predict([x])[0])
        print("✅ Prediction:", pred)
        return jsonify({'eta_min': pred})
    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Flask app running locally...")
    app.run(debug=True)
