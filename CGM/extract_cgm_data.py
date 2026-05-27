"""
Extract time series data from CGM chart image.
Extracts x (data point) and y (glucose value) coordinates from a PNG chart image.

Why: Convert chart image to numerical time series data for analysis
What: Detects blue line in chart and extracts x-y coordinates
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def detect_axes_bounds(image, x_range=(0, 500), y_range=(40, 220)):
    """
    Detect axis boundaries in the chart image.
    
    Args:
        image: Input image (BGR format)
        x_range: Expected x-axis data range (min, max)
        y_range: Expected y-axis data range (min, max)
    
    Returns:
        dict with axis pixel boundaries and data ranges
    """
    h, w = image.shape[:2]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges (axis lines are typically horizontal/vertical edges)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find horizontal lines (x-axis should be near bottom)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=w//4, 
                               minLineLength=w//2, maxLineGap=10)
    
    # Find vertical lines (y-axis should be near left)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=h//4,
                               minLineLength=h//2, maxLineGap=10)
    
    # Estimate axis boundaries
    # X-axis: bottom horizontal line
    x_axis_y = h - 50  # Default: assume 50 pixels from bottom
    if h_lines is not None:
        bottom_lines = [line for line in h_lines if line[0][1] > h * 0.7]
        if bottom_lines:
            x_axis_y = int(np.mean([line[0][1] for line in bottom_lines]))
    
    # Y-axis: left vertical line
    y_axis_x = 50  # Default: assume 50 pixels from left
    if v_lines is not None:
        left_lines = [line for line in v_lines if line[0][0] < w * 0.3]
        if left_lines:
            y_axis_x = int(np.mean([line[0][0] for line in left_lines]))
    
    # Plot area boundaries (assuming some padding)
    plot_left = y_axis_x + 10
    plot_right = w - 30
    plot_top = 30
    plot_bottom = x_axis_y - 10
    
    return {
        'x_axis_y': x_axis_y,
        'y_axis_x': y_axis_x,
        'plot_left': plot_left,
        'plot_right': plot_right,
        'plot_top': plot_top,
        'plot_bottom': plot_bottom,
        'x_data_range': x_range,
        'y_data_range': y_range
    }


def detect_blue_line(image, plot_bounds):
    """
    Detect blue line in the chart using color segmentation.
    
    Args:
        image: Input image (BGR format)
        plot_bounds: Dictionary with plot boundaries
    
    Returns:
        Binary mask of detected blue line pixels
    """
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define blue color range in HSV
    # Blue typically has hue around 100-130 in OpenCV HSV
    lower_blue1 = np.array([100, 50, 50])
    upper_blue1 = np.array([130, 255, 255])
    
    # Also try RGB-based detection for blue
    mask_hsv = cv2.inRange(hsv, lower_blue1, upper_blue1)
    
    # Alternative: RGB-based blue detection
    b, g, r = cv2.split(image)
    # Blue should have high blue channel and low red/green
    mask_rgb = (b > 100) & (b > g + 20) & (b > r + 20) & (g < 150) & (r < 150)
    mask_rgb = mask_rgb.astype(np.uint8) * 255
    
    # Combine masks
    mask = cv2.bitwise_or(mask_hsv, mask_rgb)
    
    # Crop to plot area
    plot_mask = np.zeros_like(mask)
    plot_mask[plot_bounds['plot_top']:plot_bounds['plot_bottom'],
              plot_bounds['plot_left']:plot_bounds['plot_right']] = \
        mask[plot_bounds['plot_top']:plot_bounds['plot_bottom'],
             plot_bounds['plot_left']:plot_bounds['plot_right']]
    
    # Morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    plot_mask = cv2.morphologyEx(plot_mask, cv2.MORPH_CLOSE, kernel)
    plot_mask = cv2.morphologyEx(plot_mask, cv2.MORPH_OPEN, kernel)
    
    return plot_mask


def extract_line_points(mask, plot_bounds, x_range=(0, 500), y_range=(40, 220)):
    """
    Extract x-y data points from the line mask.
    
    Args:
        mask: Binary mask of line pixels
        plot_bounds: Dictionary with plot boundaries
        x_range: X-axis data range (min, max)
        y_range: Y-axis data range (min, max)
    
    Returns:
        List of (x, y) tuples in data coordinates
    """
    h, w = mask.shape
    points = []
    
    # Get plot dimensions in pixels
    plot_width = plot_bounds['plot_right'] - plot_bounds['plot_left']
    plot_height = plot_bounds['plot_bottom'] - plot_bounds['plot_top']
    
    # Data ranges
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Find all blue pixels
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0:
        print("Warning: No line pixels detected!")
        return points
    
    # Convert pixel coordinates to plot-relative coordinates
    x_plot = x_coords - plot_bounds['plot_left']
    y_plot = y_coords - plot_bounds['plot_top']
    
    # Convert to data coordinates
    # X: pixel 0 -> x_min, pixel plot_width -> x_max
    x_data = x_min + (x_plot / plot_width) * (x_max - x_min)
    
    # Y: pixel 0 -> y_max (top), pixel plot_height -> y_min (bottom)
    # Note: image y increases downward, but data y increases upward
    y_data = y_max - (y_plot / plot_height) * (y_max - y_min)
    
    # Sort by x coordinate
    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]
    
    # Remove duplicate x values by taking median y for each unique x
    # This handles cases where multiple y values exist for the same x (line thickness)
    unique_x = []
    unique_y = []
    
    current_x = x_data[0]
    current_ys = [y_data[0]]
    
    for i in range(1, len(x_data)):
        if abs(x_data[i] - current_x) < 0.1:  # Same x (within tolerance)
            current_ys.append(y_data[i])
        else:
            # Save median y for current x
            unique_x.append(current_x)
            unique_y.append(np.median(current_ys))
            # Start new x
            current_x = x_data[i]
            current_ys = [y_data[i]]
    
    # Don't forget the last point
    if len(current_ys) > 0:
        unique_x.append(current_x)
        unique_y.append(np.median(current_ys))
    
    # Create points list
    points = list(zip(unique_x, unique_y))
    
    return points


def extract_cgm_data(image_path, output_path='cgm_data.txt', 
                     x_range=(0, 500), y_range=(40, 220),
                     auto_detect=True, manual_bounds=None):
    """
    Main function to extract CGM data from chart image.
    
    Args:
        image_path: Path to input PNG image
        output_path: Path to output text file
        x_range: X-axis data range (min, max) - data points
        y_range: Y-axis data range (min, max) - glucose values
        auto_detect: Whether to auto-detect axes (if False, use manual_bounds)
        manual_bounds: Manual axis boundaries dict (if auto_detect=False)
    
    Returns:
        List of (x, y) tuples
    """
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Detect or use manual axis boundaries
    if auto_detect:
        print("Auto-detecting axis boundaries...")
        plot_bounds = detect_axes_bounds(image, x_range, y_range)
        print(f"Detected plot bounds: left={plot_bounds['plot_left']}, "
              f"right={plot_bounds['plot_right']}, "
              f"top={plot_bounds['plot_top']}, "
              f"bottom={plot_bounds['plot_bottom']}")
    else:
        if manual_bounds is None:
            raise ValueError("manual_bounds must be provided if auto_detect=False")
        plot_bounds = manual_bounds
    
    # Detect blue line
    print("Detecting blue line...")
    line_mask = detect_blue_line(image, plot_bounds)
    
    # Count detected pixels
    num_pixels = np.sum(line_mask > 0)
    print(f"Detected {num_pixels} line pixels")
    
    if num_pixels == 0:
        print("Warning: No line detected. Trying alternative method...")
        # Try RGB-based detection with different thresholds
        b, g, r = cv2.split(image)
        line_mask = ((b > 80) & (b > g) & (b > r)).astype(np.uint8) * 255
        line_mask[0:plot_bounds['plot_top'], :] = 0
        line_mask[plot_bounds['plot_bottom']:, :] = 0
        line_mask[:, 0:plot_bounds['plot_left']] = 0
        line_mask[:, plot_bounds['plot_right']:] = 0
        num_pixels = np.sum(line_mask > 0)
        print(f"Alternative method detected {num_pixels} pixels")
    
    # Extract data points
    print("Extracting data points...")
    points = extract_line_points(line_mask, plot_bounds, x_range, y_range)
    
    print(f"Extracted {len(points)} data points")
    
    if len(points) == 0:
        raise ValueError("No data points extracted. Please check image and axis ranges.")
    
    # Save to text file
    print(f"Saving data to: {output_path}")
    with open(output_path, 'w') as f:
        f.write("# CGM Time Series Data\n")
        f.write("# X: Data Point\n")
        f.write("# Y: Glucose Value (mg/dL)\n")
        f.write("# Format: x y\n")
        f.write(f"# X range: {x_range[0]} to {x_range[1]}\n")
        f.write(f"# Y range: {y_range[0]} to {y_range[1]}\n")
        f.write(f"# Total points: {len(points)}\n")
        f.write("\n")
        for x, y in points:
            f.write(f"{x:.2f} {y:.2f}\n")
    
    print(f"Successfully saved {len(points)} data points to {output_path}")
    
    # Print statistics
    if points:
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        print(f"\nData Statistics:")
        print(f"  X range: {min(x_vals):.2f} to {max(x_vals):.2f}")
        print(f"  Y range: {min(y_vals):.2f} to {max(y_vals):.2f}")
        print(f"  Mean Y: {np.mean(y_vals):.2f}")
        print(f"  Std Y: {np.std(y_vals):.2f}")
    
    return points


def visualize_extraction(image_path, points, output_viz_path='cgm_extraction_visualization.png'):
    """
    Create visualization of extracted data.
    
    Args:
        image_path: Path to original image
        points: List of (x, y) tuples
        output_viz_path: Path to save visualization
    """
    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Original image
    ax1.imshow(image_rgb)
    ax1.set_title('Original CGM Chart')
    ax1.axis('off')
    
    # Right: Extracted data
    if points:
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        ax2.plot(x_vals, y_vals, 'b-', linewidth=1.5, label='Extracted Data')
        ax2.set_xlabel('Data Point')
        ax2.set_ylabel('Glucose Value (mg/dL)')
        ax2.set_title('Extracted Time Series Data')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_viz_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    image_path = "CGMData.png"
    output_path = "cgm_data.txt"
    
    # Axis ranges (from image description)
    x_range = (0, 500)  # Data points
    y_range = (40, 220)  # Glucose values (mg/dL)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print("Please ensure CGMData.png is in the current directory.")
        exit(1)
    
    try:
        # Extract data
        points = extract_cgm_data(
            image_path=image_path,
            output_path=output_path,
            x_range=x_range,
            y_range=y_range,
            auto_detect=True
        )
        
        # Create visualization
        visualize_extraction(image_path, points)
        
        print("\n✅ Extraction completed successfully!")
        print(f"📄 Data saved to: {output_path}")
        print(f"📊 Visualization saved to: cgm_extraction_visualization.png")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Tips:")
        print("  - Ensure the image has a clear blue line")
        print("  - Check that x_range and y_range match the chart axes")
        print("  - Try adjusting color detection thresholds if line is not detected")
        print("  - You can use manual_bounds if auto-detection fails")

