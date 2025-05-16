import os
import base64
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import random
import math  # Added missing import for math functions

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
AZURE_DEPLOYMENT_OMNIPARSER_KEY = os.environ.get("AZURE_DEPLOYMENT_OMNIPARSER_KEY", "")
AZURE_DEPLOYMENT_OMNIPARSER_ENDPOINT = os.environ.get("AZURE_DEPLOYMENT_OMNIPARSER_ENDPOINT", "")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

def image_to_base64(image_bytes):
    """Convert image bytes to base64 encoded string"""
    return base64.b64encode(image_bytes).decode("utf-8")

def generate_enhanced_heatmap(elements, width=100, height=100):
    """
    Generate an enhanced cognitive load heatmap with better visualization properties
    
    Args:
        elements: List of UI elements with their properties
        width: Width of the heatmap array
        height: Height of the heatmap array
        
    Returns:
        Numpy array representing cognitive load heatmap (0-1 values)
    """
    # Initialize empty heatmap
    heatmap = np.zeros((height, width))
    
    # Track specific element types for specialized processing
    interactive_elements = []
    text_elements = []
    
    # Step 1: Build a base influence map for each element
    for element in elements:
        element_type = element.get("type", "")
        bbox = element.get("bbox", [0, 0, 0, 0])
        is_interactive = element.get("interactivity", False)
        content = element.get("content", "")
        
        # Skip elements with invalid bounding boxes
        if not all(0 <= coord <= 1 for coord in bbox):
            continue
            
        # Convert normalized coordinates to our heatmap dimensions
        x1, y1, x2, y2 = [int(coord * width) for coord in bbox]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Skip elements with zero area
        if x1 == x2 or y1 == y2:
            continue
        
        # Track by element type for later processing
        if is_interactive:
            interactive_elements.append((x1, y1, x2, y2))
        
        if element_type == "text":
            text_elements.append((x1, y1, x2, y2, content))
        
        # Calculate cognitive load based on element properties
        # Interactive elements have higher base cognitive load
        base_load = 0.6 if is_interactive else 0.4
        
        # Text complexity load
        text_complexity = 0
        if element_type == "text" and content:
            # More complex text = higher load
            text_complexity = min(0.3, len(content) / 400)
            
            # Special case for very high complexity text
            if len(content) > 100:
                text_complexity += 0.1
                
        # Element size adjustment
        element_area = (x2 - x1) * (y2 - y1)
        size_factor = 1.0
        
        # Small elements are harder to interact with
        if element_area < 25 and is_interactive:
            size_factor = 1.3
        
        # Apply the base load to the element's area
        total_load = base_load + text_complexity
        total_load *= size_factor
        
        # Add the load to the heatmap
        heatmap[y1:y2, x1:x2] += total_load
    
    # Step 2: Process proximity between interactive elements
    for i, elem1 in enumerate(interactive_elements):
        for j, elem2 in enumerate(interactive_elements[i+1:], i+1):
            x1_1, y1_1, x2_1, y2_1 = elem1
            x1_2, y1_2, x2_2, y2_2 = elem2
            
            # Get centers
            cx1 = (x1_1 + x2_1) // 2
            cy1 = (y1_1 + y2_1) // 2
            cx2 = (x1_2 + x2_2) // 2
            cy2 = (y1_2 + y2_2) // 2
            
            # Calculate distance between centers
            distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            
            # Create proximity load for close elements
            if distance < 20:
                proximity_factor = 0.8 * (1 - distance/20)
                
                # Draw a thick line between them to indicate the relationship
                cv2.line(
                    heatmap, 
                    (cx1, cy1), 
                    (cx2, cy2), 
                    proximity_factor,
                    thickness=3
                )
                
                # Create hotspots at the endpoints
                cv2.circle(heatmap, (cx1, cy1), 4, proximity_factor, -1)
                cv2.circle(heatmap, (cx2, cy2), 4, proximity_factor, -1)
    
    # Step 3: Apply a density kernel to highlight areas with many elements
    if interactive_elements:
        density_map = np.zeros((height, width))
        for x1, y1, x2, y2 in interactive_elements:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Create a radial falloff from the center of each element
            for y in range(max(0, cy-15), min(height, cy+15)):
                for x in range(max(0, cx-15), min(width, cx+15)):
                    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                    if dist < 15:
                        # Radial falloff influence
                        influence = 0.5 * (1 - dist/15)**2
                        density_map[y, x] += influence
        
        # Add density map to the heatmap
        heatmap += density_map
    
    # Step 4: Apply adaptive contrasting
    # Get the non-zero values for percentile calculation
    non_zero = heatmap[heatmap > 0]
    if len(non_zero) > 0:
        # Calculate appropriate percentiles for this specific heatmap
        p_low, p_high = np.percentile(non_zero, [10, 90])
        
        # Apply contrast stretching to non-zero areas
        mask = heatmap > 0
        heatmap[mask] = np.clip((heatmap[mask] - p_low) / (p_high - p_low + 1e-8), 0.1, 1.0)
    
    # Step 5: Apply Gaussian smoothing for a more natural look
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    
    # Step 6: Normalize to 0-1 range
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    
    # We want UI elements to be "high" cognitive load (red),
    # so no need to invert the values
    
    return heatmap

def generate_heatmap_image(original_img, heatmap_array):
    """Generate a heatmap overlay on the original image with enhanced visualization"""
    # Convert PIL image to numpy array if needed
    if isinstance(original_img, Image.Image):
        # Convert to RGB mode if it's not already
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        original_img = np.array(original_img)
    
    # Ensure the image is in the correct format for OpenCV
    if len(original_img.shape) == 2:
        # Convert grayscale to RGB
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    elif original_img.shape[2] == 4:
        # Convert RGBA to RGB
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
    
    # Get dimensions
    img_height, img_width = original_img.shape[:2]
    
    # Resize heatmap to match the original image
    heatmap_resized = cv2.resize(heatmap_array, (img_width, img_height))
    
    # Define areas with significant cognitive load (threshold = 0.2)
    significant_load = heatmap_resized > 0.2
    
    # Convert to uint8 for colormap application
    # IMPORTANT: Invert the values here to correctly map high cognitive load to red
    # and low cognitive load to blue in the JET colormap
    inverted_heatmap = 1.0 - heatmap_resized
    heatmap_uint8 = (inverted_heatmap * 255).astype(np.uint8)
    
    # Apply colormap (now properly mapping high cognitive load to red)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Create the result image (start with the original)
    result = original_img.copy()
    
    # Apply the heatmap with variable opacity
    for i in range(img_height):
        for j in range(img_width):
            if heatmap_resized[i, j] > 0.1:  # Only apply to areas with some cognitive load
                # Higher load = more opaque overlay (using original non-inverted values for opacity)
                alpha = min(0.75, heatmap_resized[i, j] * 0.9)
                
                # Apply color with alpha blending
                result[i, j] = cv2.addWeighted(
                    np.array([original_img[i, j]]), 1 - alpha,
                    np.array([colormap[i, j]]), alpha,
                    0
                )[0]
    
    # Optional: Add contour lines around high cognitive load areas for better visibility
    if np.max(heatmap_resized) > 0.7:
        high_load_mask = (heatmap_resized > 0.7).astype(np.uint8) * 255
        contours, _ = cv2.findContours(high_load_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 1)
    
    return Image.fromarray(result)

def analyze_cognitive_load(elements):
    """
    Analyze UI elements for cognitive overload
    
    Returns a heatmap array and metrics
    """
    # Generate enhanced heatmap using the new algorithm
    heatmap = generate_enhanced_heatmap(elements)
    
    # Calculate metrics
    metrics = {
        "total_elements": len(elements),
        "interactive_elements": sum(1 for e in elements if e.get("interactivity", False)),
        "text_elements": sum(1 for e in elements if e.get("type", "") == "text"),
        "dense_areas": np.sum(heatmap > 0.7)
    }
    
    return heatmap, metrics

def detect_ui_elements_from_image(image_data):
    """
    Detect UI elements directly from the image using basic computer vision
    This is a fallback when OmniParser API is not available
    """
    # Convert image bytes to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_height, img_width = img.shape[:2]
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to detect edges and potential UI elements
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to eliminate noise
    min_area = img_height * img_width * 0.001  # 0.1% of the image size
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Extract text-like regions using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    text_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    text_contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_contours = [c for c in text_contours if cv2.contourArea(c) > min_area]
    
    # Detect edges for potential boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Use Hough transform to detect lines (potential UI boundaries)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    # Find regions with high edge density (potential UI controls)
    edge_density = cv2.dilate(edges, None, iterations=2)
    edge_density_contours, _ = cv2.findContours(edge_density, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
    
    # Create list to store detected elements
    elements = []
    
    # Process standard contours (generic UI elements)
    for i, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small or very large contours
        if w < 10 or h < 10 or w > img_width * 0.9 or h > img_height * 0.9:
            continue
            
        # Normalize coordinates to 0-1 range
        x1 = x / img_width
        y1 = y / img_height
        x2 = (x + w) / img_width
        y2 = (y + h) / img_height
        
        # Determine if this looks like an interactive element
        # More complexity = more likely to be interactive
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 1
        is_interactive = complexity > 1.5  # Simple heuristic
        
        # Add to elements list
        element = {
            "type": "button" if is_interactive else "panel",
            "bbox": [x1, y1, x2, y2],
            "interactivity": is_interactive,
            "content": ""
        }
        elements.append(element)
    
    # Process text-like regions
    for i, contour in enumerate(text_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Normalize coordinates
        x1 = x / img_width
        y1 = y / img_height
        x2 = (x + w) / img_width
        y2 = (y + h) / img_height
        
        # Add as text element
        element = {
            "type": "text",
            "bbox": [x1, y1, x2, y2],
            "interactivity": False,
            "content": f"Text {i+1}"  # Placeholder text content
        }
        elements.append(element)
    
    # Process regions with high edge density (likely interactive controls)
    for i, contour in enumerate(edge_density_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small contours
        if w < 15 or h < 15:
            continue
            
        # Normalize coordinates
        x1 = x / img_width
        y1 = y / img_height
        x2 = (x + w) / img_width
        y2 = (y + h) / img_height
        
        # Check if this region overlaps with existing elements
        overlaps = False
        for element in elements:
            e_x1, e_y1, e_x2, e_y2 = element["bbox"]
            if not (x2 < e_x1 or x1 > e_x2 or y2 < e_y1 or y1 > e_y2):
                overlaps = True
                break
        
        if not overlaps:
            element = {
                "type": "control",
                "bbox": [x1, y1, x2, y2],
                "interactivity": True,
                "content": ""
            }
            elements.append(element)
    
    # If we still don't have enough elements, detect color regions
    if len(elements) < 10:
        # Convert to HSV and detect regions with distinct colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for h in range(0, 180, 30):  # Sample different hue ranges
            lower = np.array([h, 100, 100])
            upper = np.array([h + 30, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            
            color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in color_contours:
                if cv2.contourArea(contour) > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    x1 = x / img_width
                    y1 = y / img_height
                    x2 = (x + w) / img_width
                    y2 = (y + h) / img_height
                    
                    # Check if this region overlaps with existing elements
                    overlaps = False
                    for element in elements:
                        e_x1, e_y1, e_x2, e_y2 = element["bbox"]
                        if not (x2 < e_x1 or x1 > e_x2 or y2 < e_y1 or y1 > e_y2):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        element = {
                            "type": "colored_region",
                            "bbox": [x1, y1, x2, y2],
                            "interactivity": random.random() > 0.7,  # Some colored regions are interactive
                            "content": ""
                        }
                        elements.append(element)
    
    # Ensure we have enough elements for analysis
    if len(elements) < 5:
        # Add some basic grid elements as fallback
        # This time using image entropy to determine complex regions
        for i in range(4):
            for j in range(4):
                # Sample a region of the image
                region_x = int(img_width * (j / 4))
                region_y = int(img_height * (i / 4))
                region_w = int(img_width / 4)
                region_h = int(img_height / 4)
                
                if region_y + region_h <= img_height and region_x + region_w <= img_width:
                    region = gray[region_y:region_y+region_h, region_x:region_x+region_w]
                    
                    # Calculate entropy (measure of information/complexity)
                    histogram = cv2.calcHist([region], [0], None, [256], [0, 256])
                    histogram_norm = histogram / (region_h * region_w)
                    entropy = -np.sum(histogram_norm * np.log2(histogram_norm + 1e-7))
                    
                    # Only add elements in complex regions
                    if entropy > 4.5:  # Threshold determined empirically
                        x1 = region_x / img_width
                        y1 = region_y / img_height
                        x2 = (region_x + region_w) / img_width
                        y2 = (region_y + region_h) / img_height
                        
                        element_type = "text" if entropy > 5.5 else "panel"
                        is_interactive = entropy > 5.0  # More complex = more likely interactive
                        
                        element = {
                            "type": element_type,
                            "bbox": [x1, y1, x2, y2],
                            "interactivity": is_interactive,
                            "content": f"Element {i},{j}" if element_type == "text" else ""
                        }
                        elements.append(element)
    
    return elements

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Read the image
    image_bytes = file.read()
    
    try:
        # Convert to base64
        base64_string = image_to_base64(image_bytes)
        
        # Store elements for later use
        elements = []
        
        # Check if we can use the OmniParser API
        if AZURE_DEPLOYMENT_OMNIPARSER_KEY and AZURE_DEPLOYMENT_OMNIPARSER_ENDPOINT:
            try:
                # Call OmniParser API
                url = AZURE_DEPLOYMENT_OMNIPARSER_ENDPOINT
                headers = {
                    "Authorization": f"Bearer {AZURE_DEPLOYMENT_OMNIPARSER_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {"inputs": {"image": base64_string}}
                
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                
                # Parse the detected UI elements
                response_data = response.json()
                
                # Debug: Let's look at the actual structure
                print(f"API Response Type: {type(response_data)}")
                print(f"API Response Structure: {json.dumps(response_data, indent=2, cls=NumpyEncoder)[:500]}...")
                
                # Try to handle different response formats
                if isinstance(response_data, list):
                    # The response is already a list of elements
                    elements = response_data
                elif isinstance(response_data, dict):
                    # Try to find elements in common response structures
                    if "value" in response_data:
                        elements = response_data["value"]
                    elif "output" in response_data:
                        elements = response_data["output"]
                    elif "results" in response_data:
                        elements = response_data["results"]
                    elif "elements" in response_data:
                        elements = response_data["elements"]
            except Exception as api_error:
                print(f"Error calling OmniParser API: {api_error}")
                print("Falling back to local image processing")
                
        # If we couldn't get elements from the API, detect them locally
        if not elements:
            print("Using local image processing to detect UI elements")
            elements = detect_ui_elements_from_image(image_bytes)
            
        # Generate cognitive load heatmap
        pil_image = Image.open(BytesIO(image_bytes))
        heatmap_array, metrics = analyze_cognitive_load(elements)
        heatmap_image = generate_heatmap_image(pil_image, heatmap_array)
        
        # Convert heatmap image to base64 for display
        buffered = BytesIO()
        heatmap_image.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Convert original image to base64
        original_buffered = BytesIO()
        pil_image.save(original_buffered, format="PNG")
        original_base64 = base64.b64encode(original_buffered.getvalue()).decode('utf-8')
        
        # Convert NumPy values to Python native types to ensure JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                if isinstance(value, np.ndarray):
                    metrics_serializable[key] = value.tolist()
                else:
                    metrics_serializable[key] = value.item()
            else:
                metrics_serializable[key] = value
        
        # Return results
        return jsonify({
            'success': True,
            'original_image': original_base64,
            'heatmap_image': heatmap_base64,
            'elements': elements,
            'metrics': metrics_serializable
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)