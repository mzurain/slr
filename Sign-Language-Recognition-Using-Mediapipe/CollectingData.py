import cv2
import Track_Hand as ht      
import numpy as np
import math
import time

cam = cv2.VideoCapture(0)


detectHand = ht.handTracker(maximumHands=2, detConfidence=0.8)

offset = 20
imageSize = 300

folderName = "Data/N"
counter = 0

while True:
    Success, frame = cam.read()

    # find and draw the hands
    hand = detectHand.findAndDrawHands(frame)

    # find landmarks and bounding box
    lm, bbox = detectHand.findLandmarks(frame)

    # find face and draw mesh
    # mesh = detectMesh.drawFaceMesh(frame)

    # now we crop the hand image
    if lm:

        x, y, w, h = bbox

        # creating our own image for same size
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255


        # staring hight ending hight, starting width and ending width
        # imgCrop = frame[y:y + h, x:x + w]
        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        # add cropped image in to white image
        # imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        # so in order to fit our croped image on the white image we have to do
        # some calculations

        aspectRatio = h / w  # if value is above one its mean hight is greater

        if aspectRatio > 1:         # fix the hight
            k = imageSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imageSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:                       # fix the width
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize


        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("WhiteImage", imgWhite)

        cv2.rectangle(frame, (x - 20, y - 20), (x+w+20, y+h+20),
                      (0, 255, 255), 2)


    cv2.imshow("Webcam", frame)
    k = cv2.waitKey(1)
    if k == ord("s"):
        counter += 1
        cv2.imwrite(f"{folderName}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
    if k == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()




