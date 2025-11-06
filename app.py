
from flask import Flask, request, render_template, jsonify
import joblib, json

app = Flask(__name__)
model = joblib.load('rf_delivery_model.joblib')
with open('features.json','r') as f:
    features = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    x = [data.get(f, 0) for f in features]
    pred = float(model.predict([x])[0])
    return jsonify({'eta_min': pred})

if __name__ == '__main__':
    app.run(debug=True)
