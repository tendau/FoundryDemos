# filepath: /workspaces/redesigned-meme/project/utils.py
import os
import json
import requests
import torch
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from urllib.parse import urlparse, parse_qs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64

# Load environment variables
load_dotenv()

# Try to import the Aurora SDK components
try:
    from aurora import Aurora, Batch, Metadata
    from aurora.foundry import FoundryClient, BlobStorageChannel, submit
    AURORA_SDK_AVAILABLE = True
    print("Aurora SDK successfully imported")
except ImportError:
    print("Warning: Aurora SDK not available. Using fallback implementation.")
    AURORA_SDK_AVAILABLE = False

def initialize_client():
    """
    Initialize the Azure clients for Aurora model.
    
    Returns:
        dict: A client configuration object with auth details
    """
    # Get environment variables
    azure_endpoint_url = os.getenv("AZURE_ENDPOINT_URL")
    azure_access_token = os.getenv("AZURE_ACCESS_TOKEN")
    azure_storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    azure_storage_container = os.getenv("AZURE_STORAGE_CONTAINER")
    azure_storage_sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    
    # Validate required environment variables
    required_vars = [
        "AZURE_ENDPOINT_URL", 
        "AZURE_ACCESS_TOKEN",
        "AZURE_STORAGE_ACCOUNT_NAME", 
        "AZURE_STORAGE_CONTAINER", 
        "AZURE_STORAGE_SAS_TOKEN"
    ]
    
    missing_vars = []
    for var_name in required_vars:
        var_value = os.getenv(var_name)
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Create a blob storage client
    try:
        # Parse SAS token from the environment variable
        sas_token = azure_storage_sas_token
        
        # Create the blob service client with account name and SAS token
        blob_service_client = BlobServiceClient(
            account_url=f"https://{azure_storage_account}.blob.core.windows.net",
            credential=sas_token
        )
        
        # Test connection by listing blobs
        container_client = blob_service_client.get_container_client(azure_storage_container)
        
        # Simply test the connection without parameters
        try:
            # Just get an iterator, don't even need to convert to list
            container_client.list_blobs()
            print(f"Successfully connected to blob storage")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Azure Blob Storage: {str(e)}")
        
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Azure Blob Storage: {str(e)}")
    
    # Create client configuration
    client_config = {
        "endpoint_url": azure_endpoint_url,
        "access_token": azure_access_token,
        "storage": {
            "account_name": azure_storage_account,
            "container": azure_storage_container,
            "blob_service_client": blob_service_client,
            "sas_token": sas_token
        },
        "model": "aurora-0.25-finetuned"  # Default model version
    }
    
    return client_config

def create_sample_batch(resolution="0.25"):
    """
    Create a sample batch for Aurora model with realistic structure
    based on the model's requirements.
    
    Args:
        resolution (str): Model resolution, either "0.25" or "0.1"
        
    Returns:
        dict: A dictionary representing an Aurora batch
    """
    # Grid size based on resolution
    if resolution == "0.25":
        h, w = 721, 1440  # Standard 0.25° resolution
    else:  # "0.1"
        h, w = 1801, 3600  # Higher 0.1° resolution
    
    # For prototype, use a smaller grid to reduce computational load
    h_small, w_small = 73, 144  # Reduced size for prototyping
    
    # Create properly structured data for Aurora
    # Surface variables with shape (batch_size, time_steps, height, width)
    surf_vars = {
        "2t": torch.randn(1, 2, h_small, w_small),    # 2m temperature in K (typical range: 200-320K)
        "10u": torch.randn(1, 2, h_small, w_small),   # 10m eastward wind in m/s (typical range: -30 to 30 m/s)
        "10v": torch.randn(1, 2, h_small, w_small),   # 10m southward wind in m/s (typical range: -30 to 30 m/s)
        "msl": torch.randn(1, 2, h_small, w_small)    # Mean sea level pressure in Pa (typical range: 95000-105000 Pa)
    }
    
    # Adjust to realistic ranges
    surf_vars["2t"] = 273.15 + torch.randn(1, 2, h_small, w_small) * 15  # Around 0°C with ±15K variation
    surf_vars["10u"] = torch.randn(1, 2, h_small, w_small) * 10  # ±10 m/s wind speed
    surf_vars["10v"] = torch.randn(1, 2, h_small, w_small) * 10  # ±10 m/s wind speed
    surf_vars["msl"] = 101325 + torch.randn(1, 2, h_small, w_small) * 2000  # Around 1013.25 hPa with ±20 hPa variation
    
    # Static variables with shape (height, width)
    static_vars = {
        "lsm": torch.rand(h_small, w_small),     # Land-sea mask (0-1)
        "z": torch.randn(h_small, w_small) * 500 + 200,  # Surface-level geopotential (typical values)
        "slt": torch.randint(0, 5, (h_small, w_small)).float()  # Soil type (categorical)
    }
    
    # Required pressure levels for Aurora as specified in the documentation
    pressure_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    num_levels = len(pressure_levels)
    
    # Atmospheric variables with shape (batch_size, time_steps, levels, height, width)
    atmos_vars = {
        "t": torch.zeros(1, 2, num_levels, h_small, w_small),    # Temperature in K
        "u": torch.zeros(1, 2, num_levels, h_small, w_small),    # Eastward wind in m/s
        "v": torch.zeros(1, 2, num_levels, h_small, w_small),    # Southward wind in m/s
        "q": torch.zeros(1, 2, num_levels, h_small, w_small),    # Specific humidity in kg/kg
        "z": torch.zeros(1, 2, num_levels, h_small, w_small)     # Geopotential in m²/s²
    }
    
    # Fill atmospheric variables with realistic values at different pressure levels
    for i, level in enumerate(pressure_levels):
        # Temperature decreases with height (higher = colder)
        temp_factor = 1.0 - (i / len(pressure_levels)) * 0.5  # Around 50% colder at highest level
        atmos_vars["t"][0, :, i, :, :] = surf_vars["2t"][0, :, :, :] * temp_factor
        
        # Wind tends to increase with height
        wind_factor = 1.0 + (i / len(pressure_levels)) * 2.0  # Up to 3x stronger at highest level
        atmos_vars["u"][0, :, i, :, :] = surf_vars["10u"][0, :, :, :] * wind_factor
        atmos_vars["v"][0, :, i, :, :] = surf_vars["10v"][0, :, :, :] * wind_factor
        
        # Humidity decreases with height
        humidity_base = 0.01 * torch.rand(1, 2, h_small, w_small)  # Base humidity level
        humidity_factor = torch.exp(torch.tensor(-i / 3.0))  # Exponential decrease with height
        atmos_vars["q"][0, :, i, :, :] = humidity_base * humidity_factor
        
        # Geopotential increases with height
        z_base = static_vars["z"].unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)  # Surface geopotential
        z_factor = ((1000 - level) / 900) ** 0.25  # Factor for height calculation
        atmos_vars["z"][0, :, i, :, :] = z_base + 9.8 * (1000 - level) * 100 * z_factor
    
    # Current time for metadata
    current_time = datetime.now()
    
    # If using Aurora SDK, return data in the format expected by the SDK
    if AURORA_SDK_AVAILABLE:
        # Create a proper Aurora Batch object with Metadata
        metadata = Metadata(
            lat=torch.linspace(90, -90, h_small),
            lon=torch.linspace(0, 360, w_small + 1)[:-1],
            time=(current_time,),  # Must be a tuple of datetime objects
            atmos_levels=pressure_levels  # Must be a tuple of pressure levels
        )
        
        # Create and return a proper Aurora Batch object
        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata
        )
    else:
        # Return a dictionary format for the fallback implementation
        return {
            "surf_vars": surf_vars,
            "static_vars": static_vars,
            "atmos_vars": atmos_vars,
            "metadata": {
                "lat": torch.linspace(90, -90, h_small).tolist(),
                "lon": torch.linspace(0, 360, w_small + 1)[:-1].tolist(),
                "time": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "atmos_levels": pressure_levels
            }
        }

def _process_aurora_predictions(predictions, batch, variable, num_steps):
    """
    Process Aurora predictions into a standardized format.
    
    Args:
        predictions (list): List of Aurora Batch objects with predictions
        batch (Batch): Original input batch
        variable (str): Variable to extract
        num_steps (int): Number of prediction steps
        
    Returns:
        dict: Processed prediction results
    """
    # Extract metadata from the original batch
    lats = batch.metadata.lat.tolist()
    lons = batch.metadata.lon.tolist()
    
    # Current time for metadata
    base_time = batch.metadata.time[0]
    
    # Prepare the result structure
    result = {
        "aurora_api_connected": True,
        "metadata": {
            "lat": lats,
            "lon": lons,
            "variable": variable,
            "units": _get_variable_units(variable),
            "time_steps": [
                (base_time + timedelta(hours=6*i)).strftime("%Y-%m-%dT%H:%M:%S")
                for i in range(num_steps)
            ]
        },
        "predictions": []
    }
    
    # Extract the requested variable from each prediction
    for i, pred in enumerate(predictions):
        # Check if variable is in surface variables
        if variable in ("2t", "10u", "10v", "msl") and variable in pred.surf_vars:
            # Extract the variable data, removing batch and time dimensions
            pred_data = pred.surf_vars[variable][0, 0].numpy()
            
            # Convert temperature from Kelvin to Celsius if needed
            if variable == "2t":
                pred_data = pred_data - 273.15
            
            # Convert pressure from Pa to hPa if needed
            if variable == "msl":
                pred_data = pred_data / 100.0
                
            # Add to result
            result["predictions"].append(pred_data.tolist())
            
        # Check if variable is in atmospheric variables (need to select a level)
        elif variable in ("t", "u", "v", "q", "z") and variable in pred.atmos_vars:
            # Extract from a middle level (e.g., 500 hPa) for visualization
            level_idx = list(pred.metadata.atmos_levels).index(500) if 500 in pred.metadata.atmos_levels else 0
            pred_data = pred.atmos_vars[variable][0, 0, level_idx].numpy()
            
            # Convert temperature from Kelvin to Celsius if needed
            if variable == "t":
                pred_data = pred_data - 273.15
                
            # Add to result
            result["predictions"].append(pred_data.tolist())
        else:
            # Variable not found, use zeros as placeholder
            shape = (len(lats), len(lons))
            result["predictions"].append(np.zeros(shape).tolist())
            print(f"Warning: Variable {variable} not found in prediction {i+1}")
    
    return result

def _generate_synthetic_predictions(batch, variable, num_steps):
    """
    Generate synthetic predictions when the API is not available.
    
    Args:
        batch: Input batch data
        variable (str): Variable to predict
        num_steps (int): Number of prediction steps
        
    Returns:
        dict: Synthetic prediction results
    """
    # Extract metadata from the batch to maintain consistency
    if AURORA_SDK_AVAILABLE and isinstance(batch, Batch):
        # Extract from Batch object
        lats = batch.metadata.lat.tolist()
        lons = batch.metadata.lon.tolist()
        base_time = batch.metadata.time[0]
    else:
        # Extract from dictionary
        lats = batch["metadata"]["lat"] if isinstance(batch["metadata"]["lat"], list) else batch["metadata"]["lat"].tolist()
        lons = batch["metadata"]["lon"] if isinstance(batch["metadata"]["lon"], list) else batch["metadata"]["lon"].tolist()
        base_time = datetime.fromisoformat(batch["metadata"]["time"].replace("Z", "")) if isinstance(batch["metadata"]["time"], str) else batch["metadata"]["time"]
    
    # Grid dimensions
    h, w = len(lats), len(lons)
    
    # Prepare the result structure
    result = {
        "aurora_api_connected": False,
        "metadata": {
            "lat": lats,
            "lon": lons,
            "variable": variable,
            "units": _get_variable_units(variable),
            "time_steps": [
                (base_time + timedelta(hours=6*i)).strftime("%Y-%m-%dT%H:%M:%S")
                for i in range(num_steps)
            ]
        },
        "predictions": []
    }
    
    # Generate realistic synthetic data
    for step in range(num_steps):
        # Create a smooth sine wave pattern that evolves over time
        y = np.linspace(-np.pi, np.pi, h)
        x = np.linspace(-np.pi, np.pi, w)
        xx, yy = np.meshgrid(x, y)
        
        # Add time evolution
        data = np.sin(xx + step * 0.2) * np.cos(yy + step * 0.3)
        
        # Add some noise
        data += np.random.normal(0, 0.1, (h, w))
        
        # Scale to realistic ranges based on variable
        if variable == "2t":
            # Temperature in Celsius (not Kelvin)
            data = data * 15 + 15  # Range from 0 to 30°C
        elif variable in ("10u", "10v", "u", "v"):
            # Wind speed in m/s
            data = data * 10  # Range from -10 to 10 m/s
        elif variable == "msl":
            # Pressure in hPa (not Pa)
            data = data * 10 + 1013  # Range from 1003 to 1023 hPa
        elif variable == "q":
            # Specific humidity in g/kg
            data = (data + 1) * 5  # Range from 0 to 10 g/kg
        elif variable == "z":
            # Geopotential in m²/s²
            data = data * 1000 + 5500  # Approximate values for 500 hPa
        
        # Add to predictions
        result["predictions"].append(data.tolist())
    
    return result

def _get_variable_units(variable):
    """
    Get the units for a specific variable.
    
    Args:
        variable (str): Variable name
        
    Returns:
        str: Unit string
    """
    units = {
        "2t": "°C",       # 2m temperature in Celsius (converted from Kelvin)
        "t": "°C",        # Temperature in Celsius (converted from Kelvin)
        "10u": "m/s",     # 10m eastward wind
        "10v": "m/s",     # 10m southward wind
        "u": "m/s",       # Eastward wind
        "v": "m/s",       # Southward wind
        "msl": "hPa",     # Mean sea level pressure in hectopascals (converted from Pa)
        "q": "g/kg",      # Specific humidity in g/kg
        "z": "m²/s²"      # Geopotential
    }
    
    return units.get(variable, "unknown")

def call_model(client_config, batch=None, variable="2t", num_steps=4, model_version="0.25-finetuned"):
    """
    Call the Aurora model API to generate weather predictions.
    
    Args:
        client_config (dict): Client configuration with auth details
        batch (dict, optional): Input batch data. If None, sample data will be created
        variable (str): Variable to predict ("2t", "10u", "10v", "msl", "q", etc.)
        num_steps (int): Number of prediction steps
        model_version (str): Aurora model version to use
        
    Returns:
        dict: Prediction results
    """
    # If no batch is provided, create a sample one
    if batch is None:
        batch = create_sample_batch(resolution="0.25" if "0.25" in model_version else "0.1")
    
    # Use the Aurora SDK if available
    if AURORA_SDK_AVAILABLE:
        try:
            # For SDK mode, batch should already be an Aurora Batch object
            # If it's not, we need to convert it (but this should not happen as create_sample_batch already creates a Batch)
            if not isinstance(batch, Batch):
                print("Warning: Converting dictionary to Aurora Batch object")
                # Convert dictionary to Aurora Batch object
                metadata = Metadata(
                    lat=torch.tensor(batch["metadata"]["lat"]) if isinstance(batch["metadata"]["lat"], list) else batch["metadata"]["lat"],
                    lon=torch.tensor(batch["metadata"]["lon"]) if isinstance(batch["metadata"]["lon"], list) else batch["metadata"]["lon"],
                    time=(datetime.fromisoformat(batch["metadata"]["time"].replace("Z", "")),) if isinstance(batch["metadata"]["time"], str) else (batch["metadata"]["time"],),
                    atmos_levels=tuple(batch["metadata"]["atmos_levels"]) if isinstance(batch["metadata"]["atmos_levels"], list) else batch["metadata"]["atmos_levels"]
                )
                
                batch = Batch(
                    surf_vars=batch["surf_vars"],
                    static_vars=batch["static_vars"],
                    atmos_vars=batch["atmos_vars"],
                    metadata=metadata
                )
            
            print(f"Using Aurora SDK to call model: aurora-{model_version}")
            print(f"Requesting {num_steps} prediction steps")
            
            # Initialize the Foundry client
            foundry_client = FoundryClient(
                endpoint=client_config["endpoint_url"],
                token=client_config["access_token"]
            )
            
            # Set up the blob storage channel
            azure_storage_account = client_config["storage"]["account_name"]
            azure_storage_container = client_config["storage"]["container"]
            azure_storage_sas_token = client_config["storage"]["sas_token"]
            
            blob_url = f"https://{azure_storage_account}.blob.core.windows.net/{azure_storage_container}?{azure_storage_sas_token}"
            channel = BlobStorageChannel(blob_url)
            
            # Submit the request to the Aurora API
            print("Submitting request to Aurora API...")
            predictions = []
            
            # The submit function returns an iterator of prediction batches
            for prediction_batch in submit(
                batch=batch,
                model_name=f"aurora-{model_version}",
                num_steps=num_steps,
                foundry_client=foundry_client,
                channel=channel
            ):
                predictions.append(prediction_batch)
                print(f"Received prediction {len(predictions)}/{num_steps}")
            
            print(f"Successfully received {len(predictions)} predictions from Aurora API")
            
            # Process the prediction results into a standardized format
            return _process_aurora_predictions(predictions, batch, variable, num_steps)
            
        except Exception as e:
            # Detailed error reporting
            error_message = f"Error using Aurora SDK: {str(e)}"
            print("\n" + "=" * 80)
            print(error_message)
            print("WARNING: Using SYNTHETIC DATA as fallback - NOT REAL AURORA PREDICTIONS")
            print("=" * 80 + "\n")
            
            # Fallback to synthetic data
            result = _generate_synthetic_predictions(batch, variable, num_steps)
            result["aurora_api_connected"] = False
            result["error_message"] = error_message
            return result
    
    else:
        # Aurora SDK not available
        error_message = "Aurora SDK not available. Using synthetic data."
        print("\n" + "=" * 80)
        print(error_message)
        print("WARNING: Using SYNTHETIC DATA as fallback - NOT REAL AURORA PREDICTIONS")
        print("=" * 80 + "\n")
        
        # Use synthetic data
        result = _generate_synthetic_predictions(batch, variable, num_steps)
        result["aurora_api_connected"] = False
        result["error_message"] = error_message
        return result

def generate_colormap(variable, data):
    """
    Generate a colormap for plotting variable data.
    
    Args:
        variable (str): Variable name
        data (ndarray): Data to plot
        
    Returns:
        dict: Colormap data
    """
    # Create different colormaps based on the variable
    if variable == "2t":  # Temperature
        colors = [
            (0, 'darkblue'),      # Cold
            (0.2, 'blue'),
            (0.4, 'cyan'),
            (0.5, 'white'),       # Neutral
            (0.6, 'yellow'),
            (0.8, 'orange'),
            (1.0, 'red')          # Hot
        ]
        cmap_name = "temperature"
    elif variable in ("10u", "u"):  # Eastward wind
        colors = [
            (0, 'darkblue'),      # Strong westerly
            (0.4, 'lightblue'),
            (0.5, 'white'),       # Calm
            (0.6, 'orange'),
            (1.0, 'darkred')      # Strong easterly
        ]
        cmap_name = "wind_u"
    elif variable in ("10v", "v"):  # Southward wind
        colors = [
            (0, 'darkgreen'),     # Strong northerly
            (0.4, 'lightgreen'),
            (0.5, 'white'),       # Calm
            (0.6, 'purple'),
            (1.0, 'darkviolet')   # Strong southerly
        ]
        cmap_name = "wind_v"
    elif variable == "msl":  # Pressure
        colors = [
            (0, 'darkblue'),      # Low pressure
            (0.3, 'lightblue'),
            (0.5, 'white'),
            (0.7, 'lightcoral'),
            (1.0, 'darkred')      # High pressure
        ]
        cmap_name = "pressure"
    elif variable == "q":  # Humidity
        colors = [
            (0, 'sandybrown'),    # Dry
            (0.3, 'khaki'),
            (0.6, 'lightblue'),
            (1.0, 'darkblue')     # Humid
        ]
        cmap_name = "humidity"
    else:  # Default
        colors = [
            (0, 'blue'),
            (0.25, 'cyan'),
            (0.5, 'lime'),
            (0.75, 'yellow'),
            (1.0, 'red')
        ]
        cmap_name = "default"
    
    # Create the custom colormap
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    
    # Normalize the data for colormap
    if variable == "2t":  # Temperature in Celsius
        vmin, vmax = -30, 40
    elif variable in ("10u", "10v", "u", "v"):  # Wind in m/s
        vmin, vmax = -20, 20
    elif variable == "msl":  # Pressure in hPa
        vmin, vmax = 980, 1040
    elif variable == "q":  # Specific humidity
        vmin, vmax = 0, 20
    else:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    
    # Get the colormap as a list of [level, R, G, B, A] entries
    levels = np.linspace(vmin, vmax, 100)
    norm = plt.Normalize(vmin, vmax)
    
    # Get RGBA colors from the colormap
    rgba_colors = cmap(norm(levels))
    
    # Convert to [level, R, G, B, A] format
    colorscale = []
    for i, level in enumerate(levels):
        r, g, b, a = rgba_colors[i]
        colorscale.append([float(level), f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"])
    
    return {
        "colorscale": colorscale,
        "vmin": vmin,
        "vmax": vmax,
        "units": _get_variable_units(variable)
    }
