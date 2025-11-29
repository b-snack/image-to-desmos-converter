#!/usr/bin/env python3
"""
Anime to Desmos - Proper Line Detection
Converts anime images to clean line paths for Desmos
"""

import cv2
import numpy as np
from PIL import Image
import sys
import json

def preprocess_image(image_path, target_width=300):
    """Load and preprocess the image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize maintaining aspect ratio
    height, width = img.shape[:2]
    aspect_ratio = height / width
    new_width = target_width
    new_height = int(target_width * aspect_ratio)
    
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return img, new_width, new_height

def detect_edges_advanced(img, low_threshold=50, high_threshold=150):
    """
    Advanced edge detection using Canny algorithm
    Works well for anime/manga with clear outlines
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Dilate slightly to connect nearby edges
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def find_contours(edges):
    """Find contours (continuous paths) in the edge map"""
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (noise)
    min_contour_length = 10
    filtered_contours = [c for c in contours if len(c) >= min_contour_length]
    
    return filtered_contours

def simplify_contour(contour, epsilon_factor=0.002):
    """
    Simplify contour using Douglas-Peucker algorithm
    Reduces number of points while maintaining shape
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, False)
    return simplified

def contour_to_desmos_parametric(contour, height, contour_id):
    """
    Convert a contour to Desmos parametric equations
    Returns list of Desmos expressions
    """
    # Extract points and flip Y axis for Desmos
    points = contour.reshape(-1, 2)
    xs = points[:, 0].tolist()
    ys = [height - y for y in points[:, 1]]
    
    if len(xs) < 2:
        return []
    
    expressions = []
    
    # Create the point lists
    x_var = f"x_{contour_id}"
    y_var = f"y_{contour_id}"
    t_var = f"t_{contour_id}"
    
    # X coordinates list
    x_list = f"{x_var}=[{','.join(map(str, xs))}]"
    expressions.append(x_list)
    
    # Y coordinates list
    y_list = f"{y_var}=[{','.join(map(str, ys))}]"
    expressions.append(y_list)
    
    # Parametric curve
    n = len(xs)
    # Use modular indexing to create smooth interpolation
    parametric = f"({x_var}[\\operatorname{{mod}}(\\operatorname{{round}}({t_var}),{n})+1],{y_var}[\\operatorname{{mod}}(\\operatorname{{round}}({t_var}),{n})+1])"
    param_with_domain = f"{parametric}\\left\\{{0\\le {t_var}\\le {n-1}\\right\\}}"
    expressions.append(param_with_domain)
    
    return expressions

def contour_to_desmos_simple(contour, height):
    """
    Convert contour to simple point list (alternative method)
    Simpler but works well
    """
    points = contour.reshape(-1, 2)
    xs = points[:, 0].tolist()
    ys = [height - y for y in points[:, 1]]
    
    if len(xs) < 2:
        return None
    
    return f"([{','.join(map(str, xs))}],[{','.join(map(str, ys))}])"

def process_image(image_path, output_file=None, method='parametric', 
                 edge_low=50, edge_high=150, simplify=0.002, target_width=300):
    """
    Main processing function
    
    Args:
        image_path: Path to input image
        output_file: Optional output file for expressions
        method: 'parametric' or 'simple'
        edge_low: Lower threshold for Canny edge detection
        edge_high: Upper threshold for Canny edge detection
        simplify: Simplification factor (lower = more detail)
        target_width: Target width in pixels
    """
    print(f"Loading image: {image_path}")
    img, width, height = preprocess_image(image_path, target_width)
    
    print(f"Detecting edges... (thresholds: {edge_low}, {edge_high})")
    edges = detect_edges_advanced(img, edge_low, edge_high)
    
    # Save edge preview
    cv2.imwrite('/mnt/user-data/outputs/edges_preview.png', edges)
    print("Saved edge preview to: /mnt/user-data/outputs/edges_preview.png")
    
    print("Finding contours...")
    contours = find_contours(edges)
    print(f"Found {len(contours)} contours")
    
    print("Simplifying contours...")
    simplified_contours = [simplify_contour(c, simplify) for c in contours]
    
    # Filter out very small contours after simplification
    simplified_contours = [c for c in simplified_contours if len(c) >= 3]
    print(f"After simplification: {len(simplified_contours)} contours")
    
    print(f"Generating Desmos expressions using '{method}' method...")
    all_expressions = []
    
    if method == 'parametric':
        for i, contour in enumerate(simplified_contours):
            exprs = contour_to_desmos_parametric(contour, height, i)
            all_expressions.extend(exprs)
    else:  # simple
        for contour in simplified_contours:
            expr = contour_to_desmos_simple(contour, height)
            if expr:
                all_expressions.append(expr)
    
    print(f"Generated {len(all_expressions)} Desmos expressions")
    
    # Output results
    output_text = '\n'.join(all_expressions)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"Saved expressions to: {output_file}")
    else:
        print("\n" + "="*80)
        print("DESMOS EXPRESSIONS (Copy and paste into Desmos):")
        print("="*80 + "\n")
        print(output_text)
        print("\n" + "="*80)
    
    # Create visualization
    print("Creating visualization...")
    vis = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.drawContours(vis, simplified_contours, -1, (0, 0, 0), 1)
    cv2.imwrite('/mnt/user-data/outputs/traced_lines.png', vis)
    print("Saved line visualization to: /mnt/user-data/outputs/traced_lines.png")
    
    return all_expressions

def main():
    if len(sys.argv) < 2:
        print("Usage: python anime_to_desmos.py <image_path> [options]")
        print("\nOptions:")
        print("  --method <parametric|simple>  Output method (default: simple)")
        print("  --edge-low <int>              Lower edge threshold (default: 50)")
        print("  --edge-high <int>             Upper edge threshold (default: 150)")
        print("  --simplify <float>            Simplification factor (default: 0.002)")
        print("  --width <int>                 Target width in pixels (default: 300)")
        print("  --output <file>               Save to file instead of stdout")
        print("\nExamples:")
        print("  python anime_to_desmos.py levi.jpg")
        print("  python anime_to_desmos.py levi.jpg --edge-low 30 --edge-high 120")
        print("  python anime_to_desmos.py levi.jpg --method parametric --width 400")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Parse arguments
    args = {
        'method': 'simple',
        'edge_low': 50,
        'edge_high': 150,
        'simplify': 0.002,
        'target_width': 300,
        'output_file': None
    }
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--method' and i+1 < len(sys.argv):
            args['method'] = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--edge-low' and i+1 < len(sys.argv):
            args['edge_low'] = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--edge-high' and i+1 < len(sys.argv):
            args['edge_high'] = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--simplify' and i+1 < len(sys.argv):
            args['simplify'] = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--width' and i+1 < len(sys.argv):
            args['target_width'] = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--output' and i+1 < len(sys.argv):
            args['output_file'] = sys.argv[i+1]
            i += 2
        else:
            i += 1
    
    process_image(image_path, **args)

if __name__ == "__main__":
    main()