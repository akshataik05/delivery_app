from flask import Flask, request, render_template, jsonify
import joblib, json, pandas as pd

app = Flask(__name__)

model = joblib.load('rf_delivery_model.joblib')
with open('features.json','r') as f:
    features = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

    
        subtotal = data.get('distance_km', 0) * 100 + 500
        total_items = data.get('num_stops', 0) + 1

        x = pd.DataFrame([{
            'hour': data.get('hour', 0),
            'dayofweek': data.get('dayofweek', 0),
            'traffic_level_enc': data.get('traffic_level_enc', 1),
            'subtotal': subtotal,
            'total_items': total_items
        }])

        pred = float(model.predict(x)[0])
        return jsonify({'eta_min': pred})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
