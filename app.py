# Diagnostic Flask App for Stroke Prediction
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os
import sys
import traceback

app = Flask(__name__)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

print("="*80)
print("DIAGNOSTIC MODE: Detailed feature analysis")
print("="*80)

# Load the model and extract expected feature names
try:
    with open('best_stroke_model.pickle', 'rb') as f:
        model = pickle.load(f)
        
    print("\nModel type:", type(model))
    
    # Extract feature names from model if available
    expected_features = []
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        print("Model explicitly expects these features:", expected_features)
    
    # For ensemble models, try to find feature names in base estimators
    elif hasattr(model, 'estimators_') or hasattr(model, 'estimator'):
        if hasattr(model, 'estimators_'):
            first_estimator = model.estimators_[0]
            if hasattr(first_estimator, 'feature_names_in_'):
                expected_features = first_estimator.feature_names_in_
                print("Found features in first estimator:", expected_features)
        elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_names_in_'):
            expected_features = model.estimator.feature_names_in_
            print("Found features in base estimator:", expected_features)
    
    if not expected_features:
        # Fallback to default expected features
        expected_features = [
            'age', 'avg_glucose_level', 'bmi', 'gender_Male', 'hypertension_1', 
            'heart_disease_1', 'ever_married_Yes', 'work_type_Never_worked', 
            'work_type_Private', 'work_type_Self-employed', 'work_type_children', 
            'Residence_type_Urban', 'smoking_status_formerly smoked', 
            'smoking_status_never smoked', 'smoking_status_smokes'
        ]
        print("Using default feature list:", expected_features)
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    traceback.print_exc()
    model = None
    expected_features = []

# Load scaler if available
try:
    with open('stroke_model_scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    scaler = None

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "="*50)
        print("NEW PREDICTION REQUEST")
        print("="*50)
        
        # Print all form data for diagnosis
        print("Form data received:")
        for key, value in request.form.items():
            print(f"  {key}: {value}")
        
        # Extract form values
        age = float(request.form.get('age'))
        avg_glucose_level = float(request.form.get('avg_glucose_level'))
        bmi = float(request.form.get('bmi'))
        gender = int(request.form.get('gender'))  # 1 for Male, 0 for Female
        hypertension = int(request.form.get('hypertension'))  # 1 for Yes, 0 for No
        heart_disease = int(request.form.get('disease'))  # 1 for Yes, 0 for No
        ever_married = int(request.form.get('married'))  # 1 for Yes, 0 for No
        residence_type = int(request.form.get('residence'))  # 1 for Urban, 0 for Rural
        work_type = int(request.form.get('work'))  # 0-4: Govt_job, Never_worked, Private, Self-employed, children
        smoking_status = int(request.form.get('smoking'))  # 0-3: Unknown, formerly smoked, never smoked, smokes
        
        # APPROACH 1: Create a dictionary with exact feature names
        input_data = {}
        
        # Add all expected features with defaults of 0
        for feature in expected_features:
            input_data[feature] = 0
        
        # Set numerical features
        if 'age' in expected_features:
            input_data['age'] = age
        if 'avg_glucose_level' in expected_features:
            input_data['avg_glucose_level'] = avg_glucose_level
        if 'bmi' in expected_features:
            input_data['bmi'] = bmi
        
        # Set categorical features based on form values
        if 'gender_Male' in expected_features:
            input_data['gender_Male'] = 1 if gender == 1 else 0
        
        if 'hypertension_1' in expected_features:
            input_data['hypertension_1'] = 1 if hypertension == 1 else 0
        
        if 'heart_disease_1' in expected_features:
            input_data['heart_disease_1'] = 1 if heart_disease == 1 else 0
        
        if 'ever_married_Yes' in expected_features:
            input_data['ever_married_Yes'] = 1 if ever_married == 1 else 0
        
        if 'Residence_type_Urban' in expected_features:
            input_data['Residence_type_Urban'] = 1 if residence_type == 1 else 0
        
        # Set work_type features
        work_type_features = [f for f in expected_features if f.startswith('work_type_')]
        for feature in work_type_features:
            if feature == 'work_type_Never_worked' and work_type == 1:
                input_data[feature] = 1
            elif feature == 'work_type_Private' and work_type == 2:
                input_data[feature] = 1
            elif feature == 'work_type_Self-employed' and work_type == 3:
                input_data[feature] = 1
            elif feature == 'work_type_children' and work_type == 4:
                input_data[feature] = 1
        
        # Set smoking_status features
        smoking_features = [f for f in expected_features if f.startswith('smoking_status_')]
        for feature in smoking_features:
            if feature == 'smoking_status_formerly smoked' and smoking_status == 1:
                input_data[feature] = 1
            elif feature == 'smoking_status_never smoked' and smoking_status == 2:
                input_data[feature] = 1
            elif feature == 'smoking_status_smokes' and smoking_status == 3:
                input_data[feature] = 1
        
        # Create DataFrame with single row
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in exact expected order
        input_df = input_df[expected_features]
        
        # Print feature data for diagnosis
        print("\nInput features (before scaling):")
        print(input_df)
        
        # Scale numerical features if scaler is available
        if scaler is not None:
            numerical_features = ['age', 'avg_glucose_level', 'bmi']
            scaled_features = []
            
            for feature in numerical_features:
                if feature in input_df.columns:
                    scaled_features.append(feature)
            
            if scaled_features:
                print("\nScaling features:", scaled_features)
                input_df[scaled_features] = scaler.transform(input_df[scaled_features])
        
        # Print final features
        print("\nFinal input features (after scaling):")
        print(input_df)
        print("\nColumn dtypes:")
        print(input_df.dtypes)
        
        # Make prediction
        if model is not None:
            try:
                prediction = model.predict(input_df)[0]
                print("\nPrediction successful:", prediction)
                
                # Try to get probability
                try:
                    prediction_proba = model.predict_proba(input_df)[0, 1]
                    print("Prediction probability:", prediction_proba)
                    has_proba = True
                except:
                    prediction_proba = None
                    has_proba = False
                    print("Could not get prediction probability")
                
                # Format result
                if prediction == '1' or prediction == 1:
                    result = f"High risk of stroke detected!"
                    if has_proba:
                        result += f" Probability: {prediction_proba:.2%}"
                else:
                    result = f"Low risk of stroke."
                    if has_proba:
                        result += f" Probability: {prediction_proba:.2%}"
            except Exception as e:
                print(f"Prediction failed: {str(e)}")
                traceback.print_exc()
                result = f"Prediction error: {str(e)}"
        else:
            result = "Error: Model not loaded properly. Please check the logs."
        
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        error_message = f"Error processing input: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return render_template('index.html', prediction_text=error_message)

if __name__ == '__main__':
    print("\nStarting Flask app in diagnostic mode...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)