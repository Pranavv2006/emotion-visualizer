from flask import Flask, Response, jsonify, render_template
import cv2
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

app = Flask(__name__)

# Load the feature extractor and the model
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# Emotion labels as per the model
emotion_labels = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']

# Function to preprocess image and perform emotion detection
def detect_emotion(frame):
    # Convert the frame to RGB and PIL Image format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    # Preprocess the image for the model
    inputs = processor(images=image_pil, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Get the predicted emotion
    predicted_emotion = emotion_labels[predicted_class_id]
    return predicted_emotion

# Real-time video capture using OpenCV
def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture video from the webcam

    while True:
        success, frame = cap.read()  # Read the frame from the webcam
        if not success:
            break

        # Perform emotion detection on the frame
        emotion = detect_emotion(frame)

        # Display the emotion on the frame
        cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('base.html');

@app.route('/contact')
def contact():
    return "<h1>Contact Us</h1><p>For inquiries, please contact us at email@example.com.</p>"


# Route for emotion detection
@app.route('/detect_emotion')
def detect_emotion_api():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    if success:
        emotion = detect_emotion(frame)
        cap.release()
        return jsonify({"emotion": emotion})
    else:
        cap.release()
        return jsonify({"error": "Failed to capture image."}), 500

if __name__ == '__main__':
    app.run(debug=True)
