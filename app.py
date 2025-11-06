from flask import Flask, request, render_template, jsonify
import joblib, json, pandas as pd

app = Flask(__name__)

# ✅ Load model and features
try:
    model = joblib.load('rf_delivery_model.joblib')
    with open('features.json', 'r') as f:
        features = json.load(f)
    print("✅ Model loaded successfully with features:", features)
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

        # --- User inputs from the frontend ---
        distance = float(data.get('distance_km', 0))
        stops = int(data.get('num_stops', 0))
        hour = int(data.get('hour', 0))
        dayofweek = int(data.get('dayofweek', 0))
        traffic = int(data.get('traffic_level_enc', 1))

        # --- Improved heuristic mapping ---
        # Make ETA scale strongly with distance & traffic
        base_time = 20 + distance * (3 + traffic * 1.8) + stops * 5

        # Add realistic variation based on time of day (rush hours)
        if 7 <= hour <= 9 or 17 <= hour <= 20:
            base_time *= 1.3  # rush hour multiplier
        elif 22 <= hour <= 23 or hour < 6:
            base_time *= 0.9  # less traffic at night

        # Map to model’s expected feature columns for compatibility
        subtotal = 300 + base_time * 8
        num_distinct_items = max(1, stops + 1)
        min_item_price = 100 + traffic * 20
        max_item_price = min_item_price + distance * 30
        promo_item = 0
        promo_order = 0
        total_onshift_dashers = max(1, 10 - traffic * 2)
        total_busy_dashers = 2 + traffic * 2
        total_outstanding_orders = traffic * 3 + stops

        row = {
            'market_id': 1,
            'subtotal': subtotal,
            'num_distinct_items': num_distinct_items,
            'min_item_price': min_item_price,
            'max_item_price': max_item_price,
            'promo_item': promo_item,
            'promo_order': promo_order,
            'total_onshift_dashers': total_onshift_dashers,
            'total_busy_dashers': total_busy_dashers,
            'total_outstanding_orders': total_outstanding_orders,
            'hour': hour,
            'dayofweek': dayofweek
        }

        # Fill any missing expected features
        for f in features:
            if f not in row:
                row[f] = 0

        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        pred = float(model.predict(X)[0])

        # Apply scaling to better reflect base_time (blend ML + heuristics)
        final_eta = 0.7 * pred + 0.3 * base_time

        print("📦 Incoming payload:", data)
        print("✅ Final ETA (blended):", final_eta)

        return jsonify({'eta_min': final_eta})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
