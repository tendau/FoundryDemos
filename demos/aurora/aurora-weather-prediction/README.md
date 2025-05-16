# Aurora Weather Prediction Application

A Flask-based web application that utilizes the Aurora AI Foundation Model to generate and visualize weather predictions.

## Features

- **Real-time API Connection:** Direct integration with Azure's Aurora model
- **Interactive Weather Visualization:** Color-mapped representation of prediction data
- **Time Step Navigation:** Explore predictions across multiple future time points
- **Multiple Weather Variables:** Temperature, wind components, and humidity predictions
- **Interpretation Guide:** Detailed explanation of how to read and understand the visualizations

## Prerequisites

- Python 3.8+
- An Azure account with access to the Aurora model
- Azure credentials for the Aurora service

## Environment Setup

Create a `.env` file in the project root with the following variables:

```
AZURE_ENDPOINT_URL=your_azure_endpoint_url
AZURE_ACCESS_TOKEN=your_azure_access_token
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
AZURE_STORAGE_CONTAINER=your_storage_container
AZURE_STORAGE_SAS_TOKEN=your_storage_sas_token
```

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Flask application:

```bash
python run.py
```

The web interface will be available at: http://127.0.0.1:5000/

## Using the Application

1. **Configure your prediction:**
   - Select a weather variable (Temperature, Wind, or Humidity)
   - Choose the Aurora model version
   - Set the number of prediction steps (1-10)

2. **Generate predictions:**
   - Click "Generate Weather Prediction"
   - View the results on the interactive map
   - Navigate through time steps using the controls below the map

3. **Understand the results:**
   - Click the "How to interpret these results" link or visit `/guide`
   - View detailed explanations of color scales and value ranges
   - Learn how to read weather patterns in the visualizations

## Understanding the Weather Variables

### 2m Temperature (K)
Temperature at 2 meters above the surface in Kelvin. Uses a blue (cold) to red (hot) color scale.

### 10m Eastward Wind (m/s)
The eastward component of wind at 10 meters above the ground. Positive values (orange/red) indicate eastward flow, negative values (blue) indicate westward flow.

### 10m Southward Wind (m/s)
The southward component of wind at 10 meters above the ground. Positive values (purple/violet) indicate southward flow, negative values (green) indicate northward flow.

### Specific Humidity (kg/kg)
The mass of water vapor per mass of air. Uses a scale from brown (dry) to blue (humid).

## Troubleshooting

- **Connection Issues:** Verify your Azure credentials in the .env file
- **Environment Variables:** Make sure all required environment variables are correctly set
- **Dependencies:** Ensure all packages in requirements.txt are installed
- **Model Access:** Confirm you have proper authorization to access the Aurora model
