import logging
from flask import Flask, request, jsonify
import numpy as np
import cv2


# Setup logging at DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')


app = Flask(__name__)


# Load the MobileNetSSD model (Caffe format)
prototxt = 'MobileNetSSD_deploy.prototxt'
caffemodel = 'MobileNetSSD_deploy.caffemodel'
logging.info("Loading MobileNetSSD model...")
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
logging.info("Model loaded successfully.")


# Class labels MobileNetSSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


@app.route('/detect', methods=['POST'])
def detect():
    logging.debug("Received a request for detection")
    
    # Retrieve image dimensions from headers
    width_str = request.headers.get('X-Image-Width')
    height_str = request.headers.get('X-Image-Height')
    if not width_str or not height_str:
        logging.error("Missing image dimensions in headers")
        return jsonify({'error': 'Missing image dimensions in headers'}), 400
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        logging.error("Invalid image dimensions provided")
        return jsonify({'error': 'Invalid image dimensions'}), 400


    # Expect 2 bytes per pixel for RGB565
    expected_len = width * height * 2
    data = request.data
    if len(data) != expected_len:
        logging.error(f"Incorrect image data length. Expected {expected_len} bytes, got {len(data)} bytes")
        return jsonify({'error': f'Incorrect image data length. Expected {expected_len} bytes, got {len(data)} bytes'}), 400


    # Convert raw data to a NumPy array and reshape it to (height, width, 2)
    raw = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 2))
    # Convert from RGB565 to BGR (OpenCV uses BGR)
    image = cv2.cvtColor(raw, cv2.COLOR_BGR5652BGR)
    logging.info(f"Image received: {width}x{height}")


    # Prepare the image for MobileNetSSD (resize to 300x300, normalization)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 scalefactor=0.007843, size=(300, 300),
                                 mean=127.5)
    net.setInput(blob)
    detections = net.forward()
    logging.info("Object detection completed.")


    # Loop over the detections and annotate the image
    h, w = image.shape[:2]
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Use a confidence threshold (e.g., 0.2)
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx] if idx < len(CLASSES) else "unknown"
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append({
                "label": label,
                "confidence": float(confidence),
                "box": {
                    "startX": int(startX),
                    "startY": int(startY),
                    "endX": int(endX),
                    "endY": int(endY)
                }
            })
            # Draw the bounding box and label on the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.debug(f"Detection: {label} with confidence {confidence:.2f}")


    # Display the annotated image in a window
    cv2.imshow("Detections", image)
    # Wait for 1 ms for the image window to update (non-blocking)
    cv2.waitKey(1)


    return jsonify({"detections": results, "width": width, "height": height})


if __name__ == '__main__':
    logging.info("Starting Flask server...")
    # The server will keep running until interrupted (Ctrl+C)
    app.run(host='0.0.0.0', port=5000, debug=True)