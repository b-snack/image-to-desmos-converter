#!/usr/bin/env python3
"""
Image to Desmos - LINE EQUATIONS (y = mx + b)
Converts anime outlines to actual linear equations
"""

import cv2
import numpy as np
from PIL import Image
import sys

def detect_black_lines(img, threshold=120):
    """Detect black lines in anime images - CHARACTER OUTLINES ONLY"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edge detection to find ACTUAL OUTLINE EDGES
    # This finds boundaries between different colors, not just dark pixels
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Also detect very dark pixels (the actual black lines in anime)
    _, dark_pixels = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Combine: edges + dark pixels = character outlines
    combined = cv2.bitwise_or(edges, dark_pixels)
    
    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return combined

def find_line_segments(edges):
    """
    Use probabilistic Hough Line Transform to find actual LINE SEGMENTS
    Returns list of line segments as (x1, y1, x2, y2)
    """
    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,              # Distance resolution in pixels
        theta=np.pi/180,    # Angle resolution in radians
        threshold=20,       # Minimum number of votes
        minLineLength=10,   # Minimum line length
        maxLineGap=5        # Maximum gap between segments to treat as single line
    )
    
    if lines is None:
        return []
    
    return [line[0] for line in lines]

def line_to_equation(x1, y1, x2, y2, height):
    """
    Convert line segment to Desmos equation
    Returns equation string in form suitable for Desmos
    """
    # Flip Y coordinates for Desmos (origin at bottom-left)
    y1 = height - y1
    y2 = height - y2
    
    # Handle vertical lines
    if abs(x2 - x1) < 1:
        # Vertical line: x = constant with y domain
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        return f"x={x1}{{{y_min}<=y<={y_max}}}"
    
    # Calculate slope and intercept
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    # Determine domain (x bounds)
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    
    # Round for cleaner output
    m = round(m, 3)
    b = round(b, 2)
    
    # Build equation string
    if abs(m) < 0.01:  # Nearly horizontal line
        y_val = round((y1 + y2) / 2, 2)
        return f"y={y_val}{{{x_min}<=x<={x_max}}}"
    else:
        # Format: y = mx + b {domain}
        if b >= 0:
            eq = f"y={m}x+{b}"
        else:
            eq = f"y={m}x{b}"  # b already has negative sign
        
        return f"{eq}{{{x_min}<=x<={x_max}}}"

def merge_similar_lines(line_segments, angle_threshold=5, distance_threshold=10):
    """
    Merge line segments that are very similar (same slope, nearby)
    Returns merged line segments
    """
    if len(line_segments) == 0:
        return []
    
    def get_angle(x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    
    def get_midpoint(x1, y1, x2, y2):
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    merged = []
    used = set()
    
    for i, (x1, y1, x2, y2) in enumerate(line_segments):
        if i in used:
            continue
        
        angle1 = get_angle(x1, y1, x2, y2)
        mid1 = get_midpoint(x1, y1, x2, y2)
        
        # Find similar lines
        group = [(x1, y1, x2, y2)]
        used.add(i)
        
        for j, (x3, y3, x4, y4) in enumerate(line_segments):
            if j in used:
                continue
            
            angle2 = get_angle(x3, y3, x4, y4)
            mid2 = get_midpoint(x3, y3, x4, y4)
            
            # Check if angles are similar
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Check if close together
            dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
            
            if angle_diff < angle_threshold and dist < distance_threshold:
                group.append((x3, y3, x4, y4))
                used.add(j)
        
        # Merge group into single line
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Find endpoints of merged line
            all_points = []
            for seg in group:
                all_points.extend([(seg[0], seg[1]), (seg[2], seg[3])])
            all_points = np.array(all_points)
            
            # Fit line to all points
            if len(all_points) > 1:
                vx, vy, cx, cy = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Find extreme points along the line direction
                t_values = [(p[0] - cx) * vx + (p[1] - cy) * vy for p in all_points]
                t_min = min(t_values)
                t_max = max(t_values)
                
                x1_new = int(cx + t_min * vx)
                y1_new = int(cy + t_min * vy)
                x2_new = int(cx + t_max * vx)
                y2_new = int(cy + t_max * vy)
                
                merged.append((x1_new, y1_new, x2_new, y2_new))
    
    return merged

def filter_background_lines(line_segments, width, height, margin=5):
    """
    Filter out lines that are on the edges (background)
    Keep only lines in the center (character)
    """
    filtered = []
    for x1, y1, x2, y2 in line_segments:
        # Check if line touches the edges
        on_left_edge = (x1 <= margin or x2 <= margin)
        on_right_edge = (x1 >= width - margin or x2 >= width - margin)
        on_top_edge = (y1 <= margin or y2 <= margin)
        on_bottom_edge = (y1 >= height - margin or y2 >= height - margin)
        
        # Skip lines that touch edges (these are usually background/border)
        if on_left_edge or on_right_edge or on_top_edge or on_bottom_edge:
            continue
        
        filtered.append((x1, y1, x2, y2))
    
    return filtered

def process_image(image_path, threshold=120, target_width=300, merge_lines=True):
    """Main processing function"""
    
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize
    height, width = img.shape[:2]
    aspect_ratio = height / width
    new_width = target_width
    new_height = int(target_width * aspect_ratio)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    print(f"Resized to: {new_width}x{new_height}")
    print(f"Detecting character outlines (threshold: {threshold})...")
    
    edges = detect_black_lines(img, threshold)
    cv2.imwrite('/mnt/user-data/outputs/edges_preview.png', edges)
    
    print("Finding line segments with Hough Transform...")
    line_segments = find_line_segments(edges)
    print(f"Found {len(line_segments)} line segments")
    
    # FILTER OUT BACKGROUND LINES
    print("Filtering out background lines...")
    line_segments = filter_background_lines(line_segments, new_width, new_height, margin=10)
    print(f"After filtering: {len(line_segments)} character lines")
    
    if merge_lines:
        print("Merging similar lines...")
        line_segments = merge_similar_lines(line_segments)
        print(f"After merging: {len(line_segments)} lines")
    
    print("Converting to Desmos equations...")
    equations = []
    for x1, y1, x2, y2 in line_segments:
        eq = line_to_equation(x1, y1, x2, y2, new_height)
        equations.append(eq)
    
    # Visualize
    vis = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
    for x1, y1, x2, y2 in line_segments:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.imwrite('/mnt/user-data/outputs/line_equations_vis.png', vis)
    print("Saved visualization to: /mnt/user-data/outputs/line_equations_vis.png")
    
    return equations, new_height

def main():
    if len(sys.argv) < 2:
        print("Usage: python image_to_lines.py <image_path> [options]")
        print("\nOptions:")
        print("  --threshold <int>     Darkness threshold (default: 120)")
        print("  --width <int>         Output width (default: 300)")
        print("  --no-merge            Don't merge similar lines")
        print("  --output <file>       Save to file")
        print("\nExample:")
        print("  python image_to_lines.py levi.jpg --threshold 100 --width 400")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = 120
    width = 300
    merge = True
    output_file = '/mnt/user-data/outputs/desmos_lines.txt'
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--threshold' and i+1 < len(sys.argv):
            threshold = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--width' and i+1 < len(sys.argv):
            width = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--no-merge':
            merge = False
            i += 1
        elif sys.argv[i] == '--output' and i+1 < len(sys.argv):
            output_file = sys.argv[i+1]
            i += 2
        else:
            i += 1
    
    equations, height = process_image(image_path, threshold, width, merge)
    
    print(f"\n{'='*80}")
    print(f"Generated {len(equations)} line equations!")
    print(f"{'='*80}\n")
    
    output_text = '\n'.join(equations)
    
    with open(output_file, 'w') as f:
        f.write(output_text)
    print(f"Saved to: {output_file}\n")
    
    print("DESMOS EQUATIONS (Copy these):")
    print("="*80)
    print(output_text)
    print("="*80)

if __name__ == "__main__":
    main()