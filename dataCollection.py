import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/D"
counter = 0

# Ensure the folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    hands, img = detector.findHands(img)  # Detect hands in the frame

    # Initialize imgWhite as a blank image
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping doesn't go out of bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        # Proceed if imgCrop is valid
        if imgCrop.size > 0:
            aspectRatio = h / w

            try:
                # Resize and center the cropped image on a white background
                if aspectRatio > 1:  # Tall image
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))

                    # Ensure correct placement
                    wGap = (imgSize - imgResize.shape[1]) // 2
                    imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize

                else:  # Wide image
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                    # Ensure correct placement
                    hGap = (imgSize - imgResize.shape[0]) // 2
                    imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                # Display the processed images
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            except cv2.error as e:
                print(f"Error during image resizing: {e}")
                continue
        else:
            print("Cropping resulted in an empty image. Check ROI dimensions.")
    else:
        print("No hands detected.")

    # Display the main image
    cv2.imshow("Image", img)

    # Save the image when 's' is pressed
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        file_path = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(file_path, imgWhite)
        print(f"Saved: {file_path}, Count: {counter}")

    # Exit on pressing 'q'
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
