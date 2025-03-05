import cv2
import numpy as np

image_path = "./data/image.png"

image = cv2.imread(image_path)
if image is None:
    print("이미지를 불러올올 수 없습니다.")
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

combined_image = np.hstack((image, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)))

cv2.imshow("Original and Grayscale", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
