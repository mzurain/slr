import cv2
import Track_Hand as ht     
import Classification as Classifier
import numpy as np
import math
import time


cam = cv2.VideoCapture(0)
detectHand = ht.handTracker(maximumHands=2, detConfidence=0.8)
classifier = Classifier.Classifier("C:/Users/Home PC/Desktop/miniproject/Sign-Language-Recognition-Using-Mediapipe/model/keras_model.h5", "C:/Users/Home PC/Desktop/miniproject/Sign-Language-Recognition-Using-Mediapipe/model/labels.txt")

offset = 20
imageSize = 300
counter = 0
labels = ["i love you","thank you","house","no","help"] 

while True:
    success, frame = cam.read()

    if not success:
        print("Failed to capture frame. Exiting...")
        break

    finalFrame = frame.copy()

    hand = detectHand.findAndDrawHands(finalFrame)
    lm, bbox = detectHand.findLandmarks(frame)

    
    if lm:

        x, y, w, h = bbox

    
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255
        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.size == 0:
            print("Error: imgCrop is empty.")
            continue 

        
        imgCropShape = imgCrop.shape
        aspectRatio = h / w  

        if aspectRatio > 1:     
            k = imageSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imageSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:                    
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)


        cv2.rectangle(finalFrame, (x - 20, y - 20), (x+w+20, y+h+20),
                      (255, 0, 255), 1)
        cv2.line(finalFrame, (x - 20, y - 20), (x - 20, y - 20 + 20), (255, 0, 255), 3)
        cv2.line(finalFrame, (x - 20, y - 20), (x - 20 + 20, y - 20), (255, 0, 255), 3)

        cv2.putText(finalFrame, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)



    cv2.imshow("Webcam", finalFrame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()




