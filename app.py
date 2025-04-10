# Flask App for Stroke Prediction with SHAP Visualizations
import numpy as np
import pandas as pd
import pickle
import os
import json
import shap
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, jsonify, send_file, url_for
import logging

# Create Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('static/images', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Load model and components
try:
    # Load best model
    with open('models/best_stroke_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    
    # Load model info
    with open('models/model_info.json', 'r') as f:
        model_info = json.load(f)
    logger.info(f"Model info loaded: {model_info['model_name']}")
    
    # Initialize SHAP explainer based on model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized")
    else:
        # For other models, create a small background dataset
        background_data = pd.DataFrame(np.zeros((1, model.n_features_in_)), 
                                      columns=range(model.n_features_in_))
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        logger.info("SHAP KernelExplainer initialized")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    model_info = {"model_name": "Unknown Model"}
    explainer = None

# Define feature descriptions for better interpretability
feature_descriptions = {
    'age': 'Age (years)',
    'hypertension': 'Has hypertension (0=No, 1=Yes)',
    'heart_disease': 'Has heart disease (0=No, 1=Yes)',
    'avg_glucose_level': 'Average glucose level (mg/dL)',
    'bmi': 'Body Mass Index',
    'gender_Male': 'Gender is Male',
    'ever_married_Yes': 'Ever married',
    'work_type_Never_worked': 'Never worked',
    'work_type_Private': 'Works in private sector',
    'work_type_Self-employed': 'Self-employed',
    'work_type_children': 'Child',
    'Residence_type_Urban': 'Lives in urban area',
    'smoking_status_formerly smoked': 'Formerly smoked',
    'smoking_status_never smoked': 'Never smoked',
    'smoking_status_smokes': 'Currently smokes'
}

# Create feature mapping for the form
# This maps form input fields to the expected model features
feature_mapping = {
    'age': 'age',
    'gender': {'Male': {'gender_Male': 1}, 'Female': {'gender_Male': 0}},
    'hypertension': {'Yes': 1, 'No': 0},
    'heart_disease': {'Yes': 1, 'No': 0},
    'ever_married': {'Yes': {'ever_married_Yes': 1}, 'No': {'ever_married_Yes': 0}},
    'work_type': {
        'Private': {'work_type_Never_worked': 0, 'work_type_Private': 1, 'work_type_Self-employed': 0, 'work_type_children': 0},
        'Self-employed': {'work_type_Never_worked': 0, 'work_type_Private': 0, 'work_type_Self-employed': 1, 'work_type_children': 0},
        'Govt_job': {'work_type_Never_worked': 0, 'work_type_Private': 0, 'work_type_Self-employed': 0, 'work_type_children': 0},
        'Never_worked': {'work_type_Never_worked': 1, 'work_type_Private': 0, 'work_type_Self-employed': 0, 'work_type_children': 0},
        'children': {'work_type_Never_worked': 0, 'work_type_Private': 0, 'work_type_Self-employed': 0, 'work_type_children': 1}
    },
    'residence_type': {'Urban': {'Residence_type_Urban': 1}, 'Rural': {'Residence_type_Urban': 0}},
    'avg_glucose_level': 'avg_glucose_level',
    'bmi': 'bmi',
    'smoking_status': {
        'formerly smoked': {'smoking_status_formerly smoked': 1, 'smoking_status_never smoked': 0, 'smoking_status_smokes': 0},
        'never smoked': {'smoking_status_formerly smoked': 0, 'smoking_status_never smoked': 1, 'smoking_status_smokes': 0},
        'smokes': {'smoking_status_formerly smoked': 0, 'smoking_status_never smoked': 0, 'smoking_status_smokes': 1},
        'Unknown': {'smoking_status_formerly smoked': 0, 'smoking_status_never smoked': 0, 'smoking_status_smokes': 0}
    }
}

# HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stroke Risk Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding: 20px;
        }
        .container {
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .prediction-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
        }
        .prediction-box.high-risk {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
        }
        .prediction-box.low-risk {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            font-weight: 500;
        }
        .form-text {
            color: #6c757d;
        }
        .btn-primary {
            background-color: #0d6efd;
            border: none;
        }
        .btn-primary:hover {
            background-color: #0b5ed7;
        }
        .shap-section {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        .shap-image {
            margin-top: 20px;
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #6c757d;
            font-size: 14px;
        }
        .feature-info {
            margin-top: 10px;
            font-size: 14px;
            color: #6c757d;
        }
        .model-info {
            background-color: #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Stroke Risk Prediction</h1>
            <p class="lead">Enter patient information to predict stroke risk</p>
        </div>
        
        <form action="{{ url_for('predict') }}" method="post">
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="age">Age</label>
                        <input type="number" class="form-control" id="age" name="age" min="0" max="120" step="1" required>
                        <div class="form-text">Patient's age in years</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="gender">Gender</label>
                        <select class="form-select" id="gender" name="gender" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="hypertension">Hypertension</label>
                        <select class="form-select" id="hypertension" name="hypertension" required>
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                        <div class="form-text">Does the patient have hypertension?</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="heart_disease">Heart Disease</label>
                        <select class="form-select" id="heart_disease" name="heart_disease" required>
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                        <div class="form-text">Does the patient have heart disease?</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="ever_married">Ever Married</label>
                        <select class="form-select" id="ever_married" name="ever_married" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="work_type">Work Type</label>
                        <select class="form-select" id="work_type" name="work_type" required>
                            <option value="Private">Private</option>
                            <option value="Self-employed">Self-employed</option>
                            <option value="Govt_job">Government Job</option>
                            <option value="Never_worked">Never worked</option>
                            <option value="children">Children</option>
                        </select>
                        <div class="form-text">Type of occupation</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="residence_type">Residence Type</label>
                        <select class="form-select" id="residence_type" name="residence_type" required>
                            <option value="Urban">Urban</option>
                            <option value="Rural">Rural</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="avg_glucose_level">Average Glucose Level</label>
                        <input type="number" class="form-control" id="avg_glucose_level" name="avg_glucose_level" min="50" max="300" step="0.01" required>
                        <div class="form-text">Average glucose level in blood (mg/dL)</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="bmi">BMI</label>
                        <input type="number" class="form-control" id="bmi" name="bmi" min="10" max="50" step="0.01" required>
                        <div class="form-text">Body Mass Index (weight in kg / height in m²)</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="smoking_status">Smoking Status</label>
                        <select class="form-select" id="smoking_status" name="smoking_status" required>
                            <option value="never smoked">Never Smoked</option>
                            <option value="formerly smoked">Formerly Smoked</option>
                            <option value="smokes">Currently Smokes</option>
                            <option value="Unknown">Unknown</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="d-grid gap-2">
                <button type="submit" class="btn btn-primary btn-lg">Predict Stroke Risk</button>
            </div>
        </form>
        
        {% if prediction %}
        <div class="prediction-box {{ prediction_class }}">
            <h3>Prediction Result</h3>
            <p class="fs-4">{{ prediction }}</p>
            {% if risk_factors %}
            <div class="mt-3">
                <h5>Key Risk Factors:</h5>
                <ul>
                    {% for factor in risk_factors %}
                    <li>{{ factor }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if shap_images %}
        <div class="shap-section">
            <h3>Explanation of Prediction</h3>
            <p>The charts below show which factors influenced this prediction:</p>
            
            <div class="shap-image">
                <h5>Impact of Features on Prediction</h5>
                <img src="{{ shap_images.force_plot }}" class="img-fluid" alt="SHAP Force Plot">
                <div class="feature-info">
                    <p>Features in red push toward higher stroke risk, while features in blue push toward lower risk.</p>
                </div>
            </div>
            
            {% if shap_images.waterfall_plot %}
            <div class="shap-image">
                <h5>Feature Contribution to Risk</h5>
                <img src="{{ shap_images.waterfall_plot }}" class="img-fluid" alt="SHAP Waterfall Plot">
                <div class="feature-info">
                    <p>This chart shows how each feature increased or decreased the stroke risk prediction.</p>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="model-info">
            <h5>Model Information</h5>
            <p>Prediction powered by: <strong>{{ model_name }}</strong></p>
            <p>This prediction tool was created to help identify stroke risk factors. It should be used for informational purposes only and does not replace professional medical advice.</p>
        </div>
        
        <div class="footer">
            <p>© 2025 Stroke Risk Prediction Tool</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Create template file
with open('templates/index.html', 'w') as f:
    f.write(html_template)

def process_input(form_data):
    """
    Process form data into the format expected by the model
    
    Parameters:
    -----------
    form_data : dict
        Form data from request
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features in the format expected by the model
    """
    # Start with empty feature dict with all expected features set to 0
    features = {
        'age': 0,
        'hypertension': 0,
        'heart_disease': 0,
        'avg_glucose_level': 0,
        'bmi': 0,
        'gender_Male': 0,
        'ever_married_Yes': 0,
        'work_type_Never_worked': 0,
        'work_type_Private': 0,
        'work_type_Self-employed': 0,
        'work_type_children': 0,
        'Residence_type_Urban': 0,
        'smoking_status_formerly smoked': 0,
        'smoking_status_never smoked': 0,
        'smoking_status_smokes': 0
    }
    
    # Process each form field
    for field, value in form_data.items():
        mapping = feature_mapping.get(field)
        
        if mapping is None:
            continue
            
        if isinstance(mapping, str):
            # Direct mapping (numerical features)
            try:
                features[mapping] = float(value)
            except ValueError:
                logger.warning(f"Could not convert {field}={value} to float")
        else:
            # Categorical features that need mapping
            feature_values = mapping.get(value)
            if feature_values:
                if isinstance(feature_values, dict):
                    # Update multiple features
                    for feat, val in feature_values.items():
                        features[feat] = val
                else:
                    # Update single feature
                    features[field] = feature_values
    
    # Create DataFrame with a single row
    df = pd.DataFrame([features])
    
    return df

def create_shap_plots(input_df):
    """
    Create SHAP visualizations for the prediction
    
    Parameters:
    -----------
    input_df : pd.DataFrame
        Input data
        
    Returns:
    --------
    dict
        Dictionary with paths to SHAP images
    """
    if explainer is None:
        logger.warning("SHAP explainer not available")
        return None
    
    try:
        logger.info("Generating SHAP visualizations")
        
        # Get SHAP values for the input
        shap_values = explainer.shap_values(input_df)
        
        # For tree-based models, SHAP values are returned as a list with one array per class
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # Use values for positive class (stroke = 1)
            shap_values_to_plot = shap_values[1]
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            shap_values_to_plot = shap_values
            expected_value = explainer.expected_value
        
        # Create force plot
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            expected_value, 
            shap_values_to_plot[0], 
            input_df.iloc[0], 
            feature_names=list(input_df.columns),
            matplotlib=True,
            show=False
        )
        force_plot_path = 'static/images/force_plot.png'
        plt.tight_layout()
        plt.savefig(force_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Force plot created")
        
        # Create waterfall plot
        try:
            plt.figure(figsize=(10, 8))
            shap.plots._waterfall.waterfall_legacy(
                expected_value,
                shap_values_to_plot[0],
                feature_names=list(input_df.columns),
                show=False
            )
            waterfall_plot_path = 'static/images/waterfall_plot.png'
            plt.tight_layout()
            plt.savefig(waterfall_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Waterfall plot created")
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {str(e)}")
            waterfall_plot_path = None
        
        return {
            'force_plot': force_plot_path,
            'waterfall_plot': waterfall_plot_path
        }
        
    except Exception as e:
        logger.error(f"Error creating SHAP plots: {str(e)}")
        return None

def identify_risk_factors(input_df, shap_values):
    """
    Identify key risk factors based on SHAP values
    
    Parameters:
    -----------
    input_df : pd.DataFrame
        Input data
    shap_values : np.ndarray
        SHAP values for the input
        
    Returns:
    --------
    list
        List of key risk factors
    """
    if explainer is None:
        return []
    
    try:
        # Get SHAP values for positive class
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_to_use = shap_values[1][0]
        else:
            shap_values_to_use = shap_values[0]
        
        # Create mapping of feature names to their SHAP values
        feature_contributions = {}
        for i, feature in enumerate(input_df.columns):
            # Map original feature names to their descriptions
            display_name = feature_descriptions.get(feature, feature)
            feature_contributions[display_name] = shap_values_to_use[i]
        
        # Sort by absolute contribution
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top 3 positive contributors (increasing stroke risk)
        positive_contributors = [
            f"{feature} (+{value:.4f})" 
            for feature, value in sorted_contributions 
            if value > 0
        ][:3]
        
        return positive_contributors
        
    except Exception as e:
        logger.error(f"Error identifying risk factors: {str(e)}")
        return []

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', model_name=model_info['model_name'])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Process input form and make prediction
    
    Returns:
    --------
    Rendered template with prediction and SHAP visualizations
    """
    try:
        logger.info("Processing prediction request")
        
        if model is None:
            return render_template(
                'index.html',
                prediction="Error: Model not loaded",
                prediction_class="high-risk",
                model_name=model_info['model_name']
            )
        
        # Process form data
        input_df = process_input(request.form)
        logger.info(f"Processed input: {input_df.to_dict(orient='records')[0]}")
        
        # Make prediction
        prediction_proba = model.predict_proba(input_df)[0, 1]
        prediction_binary = 1 if prediction_proba >= 0.5 else 0
        
        logger.info(f"Prediction: {prediction_binary}, Probability: {prediction_proba:.4f}")
        
        # Format prediction text
        if prediction_binary == 1:
            if prediction_proba >= 0.75:
                risk_level = "High"
            elif prediction_proba >= 0.6:
                risk_level = "Elevated"
            else:
                risk_level = "Moderate"
                
            prediction_text = f"{risk_level} risk of stroke detected ({prediction_proba:.1%} probability)"
            prediction_class = "high-risk"
        else:
            if prediction_proba <= 0.25:
                risk_level = "Very low"
            else:
                risk_level = "Low"
                
            prediction_text = f"{risk_level} risk of stroke ({prediction_proba:.1%} probability)"
            prediction_class = "low-risk"
        
        # Generate SHAP explanations
        shap_images = create_shap_plots(input_df)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_df) if explainer else None
        
        # Identify risk factors
        risk_factors = identify_risk_factors(input_df, shap_values) if shap_values is not None else []
        
        return render_template(
            'index.html',
            prediction=prediction_text,
            prediction_class=prediction_class,
            shap_images=shap_images,
            risk_factors=risk_factors,
            model_name=model_info['model_name']
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}",
            prediction_class="high-risk",
            model_name=model_info['model_name']
        )

if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(debug=True, port=5000)