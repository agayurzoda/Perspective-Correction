from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.filters import unsharp_mask
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_image(path):
    """Load an image from the specified path."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    return image

def preprocess_image(image):
    """Convert to grayscale and apply adaptive thresholding for better edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def detect_edges(binary_image):
    """Apply Canny edge detection."""
    edges = cv2.Canny(binary_image, 50, 150)
    return edges

def find_document_corners(edges):
    """Find contours and approximate them to detect the four corners of the document."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return approx
    return None

def order_points(points):
    """Order points in top-left, top-right, bottom-right, bottom-left order."""
    points = np.array(points)
    points = sorted(points, key=lambda p: p[0][0])
    left_points = sorted(points[:2], key=lambda p: p[0][1])
    right_points = sorted(points[2:], key=lambda p: p[0][1])
    return np.array([left_points[0], right_points[0], right_points[1], left_points[1]], dtype="float32")

def perspective_correction(image, corners):
    """Apply perspective transformation to the image based on the detected corners."""
    ordered_corners = order_points(corners)
    width, height = 500, 700  # Desired output dimensions
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

def enhance_image(image):
    """Enhance the image using unsharp mask."""
    image_ski = img_as_ubyte(image) / 255.0
    sharpened_image = unsharp_mask(image_ski, radius=1, amount=1.5)
    return img_as_ubyte(sharpened_image)

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    """Handle image upload and processing."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    try:
        # Load and process image
        original_image = load_image(input_path)
        binary_image = preprocess_image(original_image)
        edges = detect_edges(binary_image)
        corners = find_document_corners(edges)
        
        if corners is None:
            return jsonify({'error': 'Could not detect document corners'}), 400

        rectified_image = perspective_correction(original_image, corners)
        enhanced_image = enhance_image(rectified_image)

        # Save processed image
        output_path = os.path.join(OUTPUT_FOLDER, 'processed_' + file.filename)
        cv2.imwrite(output_path, enhanced_image)
        return jsonify({'url': f'/download/{os.path.basename(output_path)}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Serve the processed file for download."""
    path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
