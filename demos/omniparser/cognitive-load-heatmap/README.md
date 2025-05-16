# UI Cognitive Load Analyzer

A lightweight Flask application that uses Microsoft's OmniParserV2 Foundry model to analyze UI screenshots, detect elements, and generate cognitive load heatmaps.

## Features

- **UI Element Detection**: Leverages OmniParserV2 to identify text, buttons, and interactive elements
- **Cognitive Load Analysis**: Evaluates element density, text complexity, and interactive element proximity
- **Heatmap Visualization**: Generates color-coded overlays highlighting areas of high cognitive load
- **Modern Glass UI**: Dark-themed interface with drag-and-drop functionality

## Quick Start

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file from the template:
   ```
   cp .env.template .env
   ```
4. Add your OmniParserV2 API credentials to the `.env` file
5. Run the application:
   ```
   python app.py
   ```
6. Open your browser to `http://localhost:5000`

## How It Works

1. Upload a screenshot of any UI
2. OmniParserV2 processes the image to detect UI elements
3. The cognitive load algorithm analyzes various factors that contribute to UI complexity
4. A heatmap is generated showing potential areas for UI improvement

## Extending the Prototype

This prototype can be extended in several ways:

1. **Enhanced Analysis**: Add more cognitive load metrics like color contrast or typography analysis
2. **Custom Recommendations**: Generate specific improvement suggestions based on detected issues
3. **Batch Processing**: Allow multiple UI comparisons and analysis
4. **Historical Tracking**: Save analysis results to track design improvements over time

## Requirements

- Python 3.6+
- Flask
- OpenCV
- NumPy
- Pillow
- Azure OmniParserV2 API credentials