from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models and preprocessing objects
logistic_regression = joblib.load('logistic_regression_model.pkl')
random_forest = joblib.load('random_forest_model.pkl')
gradient_boosting = joblib.load('gradient_boosting_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert the incoming JSON data to a DataFrame
    input_data = pd.DataFrame([data])
    
    # Preprocess the data (e.g., create dummy variables)
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    # Make predictions using the loaded models
    logistic_regression_pred = logistic_regression.predict(input_data)
    random_forest_pred = random_forest.predict(input_data)
    gradient_boosting_pred = gradient_boosting.predict(input_data)
    
    # Decode the predictions
    logistic_regression_pred_decoded = label_encoder.inverse_transform(logistic_regression_pred)
    random_forest_pred_decoded = label_encoder.inverse_transform(random_forest_pred)
    gradient_boosting_pred_decoded = label_encoder.inverse_transform(gradient_boosting_pred)
    
    # Return the predictions as JSON
    return jsonify({
        'Logistic Regression': logistic_regression_pred_decoded[0],
        'Random Forest': random_forest_pred_decoded[0],
        'Gradient Boosting': gradient_boosting_pred_decoded[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
