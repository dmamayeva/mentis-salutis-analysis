import joblib
import pandas as pd
from flask import Flask, request, jsonify


app = Flask('mentis-analysis')

model = joblib.load('/app/xgb_model.pkl')
pipeline = joblib.load('/app/pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        
        if not data:
            return jsonify({"error": "No input data"}), 400
        
        df = pd.DataFrame([data])
        
        
        X = pipeline.transform(df)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        result = {
            'prediction': int(y_pred[0]),
            'prediction_probability': float(y_pred_proba[0])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)