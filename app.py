from flask import Flask, request, render_template, jsonify
import joblib, json, pandas as pd

app = Flask(__name__)

model = joblib.load('rf_delivery_model.joblib')
with open('features.json', 'r') as f:
    features = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        hour = data.get('hour', 0)
        dayofweek = data.get('dayofweek', 0)
        traffic = data.get('traffic_level_enc', 1)
        distance = data.get('distance_km', 0)
        stops = data.get('num_stops', 0)

        subtotal = 500 + distance * 100 + traffic * 50
        num_distinct_items = stops + 1
        min_item_price = 100 + traffic * 10
        max_item_price = min_item_price + 200
        promo_item = 0
        promo_order = 0
        total_onshift_dashers = 10 - traffic
        total_busy_dashers = 5 + traffic
        total_outstanding_orders = traffic * 2
        market_id = 1

        row = {
            'market_id': market_id,
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

        for f in features:
            if f not in row:
                row[f] = 0

        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        pred = float(model.predict(X)[0])
        return jsonify({'eta_min': pred})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
