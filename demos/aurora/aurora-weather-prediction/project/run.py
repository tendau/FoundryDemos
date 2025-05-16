from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
from datetime import datetime, timedelta
from utils import initialize_client, call_model, generate_colormap

app = Flask(__name__)

# Initialize the Aurora client
client_config = None
try:
    client_config = initialize_client()
    print("Successfully initialized Aurora client with Azure credentials")
except Exception as e:
    print(f"Warning: Failed to initialize Aurora client: {e}")
    # We'll check for this again in the routes

@app.route('/')
def index():
    """Render the main page of the Aurora weather prediction application."""
    variables = [
        {"id": "2t", "name": "2m Temperature (K)"},
        {"id": "10u", "name": "10m Eastward Wind (m/s)"},
        {"id": "10v", "name": "10m Southward Wind (m/s)"},
        {"id": "q", "name": "Specific Humidity (kg/kg)"}
    ]
    
    model_versions = [
        {"id": "0.25-finetuned", "name": "Aurora 0.25° Fine-tuned"},
        {"id": "0.25-pretrained", "name": "Aurora 0.25° Pretrained"},
        {"id": "0.1-finetuned", "name": "Aurora 0.1° Fine-tuned (High-Res)"}
    ]
    
    # Check environment variables availability for the UI
    env_status = {
        "azure_endpoint": bool(os.getenv("AZURE_ENDPOINT_URL")),
        "azure_token": bool(os.getenv("AZURE_ACCESS_TOKEN")),
        "storage_account": bool(os.getenv("AZURE_STORAGE_ACCOUNT_NAME")),
        "storage_container": bool(os.getenv("AZURE_STORAGE_CONTAINER")),
        "storage_sas": bool(os.getenv("AZURE_STORAGE_SAS_TOKEN"))
    }
    
    return render_template('index.html', 
                          variables=variables, 
                          model_versions=model_versions,
                          env_status=env_status)

@app.route('/guide')
def guide():
    """Render the interpretation guide for Aurora weather predictions."""
    return render_template('interpretation_guide.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Generate predictions with the Aurora model."""
    global client_config
    
    # Check if we have a valid client configuration
    if client_config is None:
        try:
            client_config = initialize_client()
        except Exception as e:
            return jsonify({
                "error": str(e),
                "message": "Failed to initialize Azure client. Please check your environment variables."
            }), 400
    
    # Get parameters from the request
    data = request.json
    variable = data.get('variable', '2t')
    num_steps = int(data.get('num_steps', 4))
    model_version = data.get('model_version', '0.25-finetuned')
    
    # Call the model to get predictions
    try:
        # Get prediction results
        result = call_model(
            client_config=client_config,
            variable=variable,
            num_steps=num_steps,
            model_version=model_version
        )
        
        # Print the result for debugging
        print(f"API result: {json.dumps(result)[:500]}...")
        
        # Make sure the result is properly structured for JSON
        if 'predictions' in result and isinstance(result['predictions'], list) and len(result['predictions']) > 0:
            # Check that predictions are not empty
            if not result['predictions'][0]:
                return jsonify({
                    "error": "Empty prediction data",
                    "message": "The model returned empty prediction data"
                }), 500
                
            # Structure the response in a consistent way
            response_data = {
                "aurora_api_connected": result.get('aurora_api_connected', False),
                "variable": variable,
                "model": f"aurora-{model_version}",
                "num_steps": num_steps,
                "lat": result.get('metadata', {}).get('lat', []),
                "lon": result.get('metadata', {}).get('lon', []),
                "predictions": []
            }
            
            # Process each prediction
            for i, pred_data in enumerate(result['predictions']):
                # Get the timestamp for this prediction
                timestamps = result.get('metadata', {}).get('time_steps', [])
                
                # Debug timestamp data
                print(f"Timestamps: {timestamps}, type: {type(timestamps)}, length: {len(timestamps) if timestamps else 0}")
                
                # Check if pred_data is a valid array/list
                if not isinstance(pred_data, (list, np.ndarray)) or len(pred_data) == 0:
                    print(f"Warning: Invalid prediction data for step {i}: {type(pred_data)}")
                    # Create an empty 2D array as placeholder
                    pred_data = [[0.0]] * 10
                
                # Safely get timestamp for this prediction
                try:
                    if timestamps and i < len(timestamps):
                        time_str = timestamps[i]
                    else:
                        # Generate a future timestamp based on the current time
                        time_str = (datetime.now() + timedelta(hours=6*i)).isoformat()
                        print(f"Generated timestamp for step {i}: {time_str}")
                except Exception as ts_error:
                    print(f"Error handling timestamp for step {i}: {ts_error}")
                    time_str = datetime.now().isoformat()
                
                # Get min/max for this prediction data safely
                try:
                    pred_array = np.array(pred_data)
                    if pred_array.size > 0:  # Check if array is not empty
                        min_val = float(np.nanmin(pred_array))
                        max_val = float(np.nanmax(pred_array))
                    else:
                        print(f"Warning: Empty prediction array for step {i}")
                        min_val = 0.0
                        max_val = 0.0
                except Exception as val_error:
                    print(f"Error calculating min/max for step {i}: {val_error}")
                    min_val = 0.0
                    max_val = 0.0
                
                # Add to response
                response_data['predictions'].append({
                    "step": i + 1,
                    "time": time_str,
                    "data": pred_data,
                    "min": min_val,
                    "max": max_val
                })
            
            # Generate colormap for visualization
            first_prediction = np.array(result['predictions'][0]) if result['predictions'] else np.zeros((10, 10))
            colormap_data = generate_colormap(variable, first_prediction)
            response_data['colormap'] = colormap_data.get('colorscale', [])
            
            # Include error message if present
            if 'error_message' in result:
                response_data['error_message'] = result['error_message']
            
            return jsonify(response_data)
        else:
            raise ValueError("Invalid prediction result structure")
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": f"An error occurred while generating predictions: {str(e)}"
        }), 500

@app.route('/check-connection', methods=['GET'])
def check_connection():
    """Check if the connection to Azure and Aurora API is working."""
    global client_config
    
    try:
        if client_config is None:
            client_config = initialize_client()
        
        # Get configuration status
        status = {
            "connected": True,
            "azure_endpoint": os.getenv("AZURE_ENDPOINT_URL", "").startswith("https://"),
            "storage_account": os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "") != "",
            "storage_container": os.getenv("AZURE_STORAGE_CONTAINER", "") != "",
            "message": "Connected to Azure services successfully."
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "connected": False,
            "error": str(e),
            "message": "Failed to connect to Azure services. Please check your configuration."
        }), 400

@app.route('/debug', methods=['GET'])
def debug_page():
    """Show the debug page for API testing."""
    return render_template('debug.html')

if __name__ == '__main__':
    # Run on localhost so it can be accessed at http://127.0.0.1:5000/
    app.run(debug=True, host='127.0.0.1')
