import tensorflow as tf
from keras.models import load_model  # Corrected import path
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


#Sign-Language-Recognition-Using-Mediapipe\model\keras_model.h5
# Load the model (assuming model file is in the same directory)
model = load_model("Sign-Language-Recognition-Using-Mediapipe/model/keras_model.h5", compile=False)  # Updated model loading

# Load the labels
class_names = open("Sign-Language-Recognition-Using-Mediapipe/model/labels.txt", "r").readlines()

# Access camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, image = camera.read()

    # Resize and display
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)

    # Preprocess image
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Make prediction
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print results
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Exit on Esc key press
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

model.summary()
