from flask import Flask, request, render_template, jsonify
import joblib, json, pandas as pd

app = Flask(__name__)

# ✅ Load model and features
try:
    model = joblib.load('rf_delivery_model.joblib')
    with open('features.json', 'r') as f:
        features = json.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    features = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'})

    try:
        data = request.get_json(force=True)

        # 🎯 Map your web form fields to model feature names
        hour = data.get('hour', 0)
        dayofweek = data.get('dayofweek', 0)
        traffic = data.get('traffic_level_enc', 1)
        distance = data.get('distance_km', 0)
        stops = data.get('num_stops', 0)

        # Approximate values for old model’s missing features
        row = {
            'market_id': 1,
            'subtotal': 500 + distance * 100 + traffic * 60,
            'num_distinct_items': stops + 1,
            'min_item_price': 100 + traffic * 10,
            'max_item_price': 300 + traffic * 20,
            'promo_item': 0,
            'promo_order': 0,
            'total_onshift_dashers': 10 - traffic,
            'total_busy_dashers': 5 + traffic,
            'total_outstanding_orders': traffic * 2,
            'hour': hour,
            'dayofweek': dayofweek
        }

        # Ensure all expected features are present
        for f in features:
            if f not in row:
                row[f] = 0

        # Build DataFrame with correct feature order
        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        pred = float(model.predict(X)[0])
        print("✅ Prediction:", pred)
        return jsonify({'eta_min': pred})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
