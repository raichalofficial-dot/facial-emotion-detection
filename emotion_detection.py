import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('emotion_model.h5')

# Emotion labels
emotion_labels = [
    'Angry',
    'Disgusted',
    'Fearful',
    'Happy',
    'Neutral',
    'Sad',
    'Surprised'
]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# Start webcam
capture = cv2.VideoCapture(0)

while True:

    ret, frame = capture.read()

    # Resize webcam frame
    frame = cv2.resize(frame, (640,480))

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect face
    faces_rect = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7
    )

    for (x,y,w,h) in faces_rect:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        # Extract face region in COLOR
        roi_color = frame[y:y+h, x:x+w]

        # Resize to 128x128
        roi_color = cv2.resize(
            roi_color,
            (128,128)
        )

        # Normalize image
        roi = roi_color.astype('float32') / 255.0

        # Expand dimensions
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        prediction = model.predict(
            roi,
            verbose=0
        )

        # Get prediction index
        max_index = np.argmax(prediction)

        # Get label
        label = emotion_labels[max_index]

        # Display emotion
        cv2.putText(
            frame,
            label,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    # Show webcam
    cv2.imshow(
        'Real Time Emotion Detection',
        frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
capture.release()

# Close windows
cv2.destroyAllWindows()