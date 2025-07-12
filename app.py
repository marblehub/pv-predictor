from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import joblib
from model import PVNet

app = Flask(__name__)

# Load model and scaler
model = PVNet()
model.load_state_dict(torch.load("pv_model.pth", map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temperature'])
        irr = float(request.form['irradiance'])
        features = np.array([[temp, irr]])
        features_scaled = scaler.transform(features)
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor).item()
        
        return render_template('index.html', prediction=f"{output:.3f} kW")
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
