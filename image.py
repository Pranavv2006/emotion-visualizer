from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch
import cv2
from PIL import Image

# Load the feature extractor and the model
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# Emotion labels as per the model
emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

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

        # Show the frame
        cv2.imshow('Emotion Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time emotion detection
generate_frames()
