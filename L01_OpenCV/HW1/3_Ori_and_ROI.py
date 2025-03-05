import cv2
import numpy as np

image_path = "./data/image.png"
image = cv2.imread(image_path)
clone = image.copy()
roi = None
selecting = False
start_x, start_y = -1, -1

def select_roi(event, x, y, flags, param):
    global start_x, start_y, roi, selecting, image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        selecting = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            temp = image.copy()
            cv2.rectangle(temp, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp)
    
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        roi = clone[start_y:y, start_x:x]
        cv2.imshow("ROI", roi)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", select_roi)

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("r"):
        image = clone.copy()
        roi = None
        cv2.destroyWindow("ROI")
    elif key == ord("s") and roi is not None:
        cv2.imwrite("selected_roi.png", roi)
        print("ROI 저장")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()