from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def order_points(points):
    """Order points as top-left, top-right, bottom-right, bottom-left."""
    points = np.array(points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = points[np.argmin(s)]  # Top-left
    ordered[2] = points[np.argmax(s)]  # Bottom-right
    ordered[1] = points[np.argmin(diff)]  # Top-right
    ordered[3] = points[np.argmax(diff)]  # Bottom-left
    return ordered

def perspective_correction(image, corners):
    """Correct the perspective based on provided corners."""
    ordered_corners = order_points(corners)
    width, height = 500, 700  # Desired output dimensions
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and store it."""
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({'imagePath': f'/uploads/{file.filename}'})

@app.route('/process', methods=['POST'])
def process_image():
    """Handle perspective correction with provided corners."""
    data = request.get_json()
    image_path = os.path.join(UPLOAD_FOLDER, data['imagePath'].split('/')[-1])
    corners = np.array(data['corners'], dtype="float32")

    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Image not found'}), 400

    corrected_image = perspective_correction(image, corners)

    # Save the processed image
    output_path = os.path.join(OUTPUT_FOLDER, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, corrected_image)

    return jsonify({'processedImagePath': f'/outputs/{os.path.basename(output_path)}'})

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """Serve uploaded files."""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/outputs/<filename>')
def serve_output_file(filename):
    """Serve processed files."""
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)
