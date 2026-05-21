import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Load trained emotion model
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

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Load facial landmark predictor
predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat'
)

# Start webcam
capture = cv2.VideoCapture(0)

while True:

    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (640,480))

    # Convert to grayscale for dlib
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces using dlib
    faces = detector(gray)

    for face in faces:

        # Face coordinates
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        # Detect landmarks
        landmarks = predictor(gray, face)

        # Draw landmark points
        for n in range(68):

            x_landmark = landmarks.part(n).x
            y_landmark = landmarks.part(n).y

            cv2.circle(
                frame,
                (x_landmark,y_landmark),
                2,
                (0,0,255),
                -1
            )

        # Extract COLOR face ROI
        # Better face extraction with padding
        padding = 30

        x1 = max(0, x-padding)
        y1 = max(0, y-padding)

        x2 = min(frame.shape[1], x+w+padding)
        y2 = min(frame.shape[0], y+h+padding)

        roi_color = frame[y1:y2, x1:x2]

        try:

            # Resize according to training
            roi_color = cv2.resize(
            roi_color,
            (128,128),
            interpolation=cv2.INTER_AREA
            )

            roi_color = cv2.GaussianBlur(
            roi_color,
            (3,3),
            0
            )

            # Normalize image
            roi = roi_color.astype('float32') / 255.0

            # Expand dimensions
            roi = np.expand_dims(
                roi,
                axis=0
            )

            # Predict emotion
            prediction = model.predict(
                roi,
                verbose=0
            )

            # Highest prediction
            max_index = prediction[0].argmax()

            # Emotion label
            emotion = emotion_labels[max_index]

            confidence = np.max(prediction) * 100

            text = f'{emotion} {confidence:.2f}%'
            # Show emotion label
            cv2.putText(
                frame,
                text,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        except Exception as e:
            print(e)

    # Show output
    cv2.imshow(
        'Dlib Emotion Detection',
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
capture.release()

# Close windows
cv2.destroyAllWindows()