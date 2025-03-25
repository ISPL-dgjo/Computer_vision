import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('./data/edgeDetectionImage.jpg')

# 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 소벨 필터로 X, Y 방향 에지 검출
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 에지 강도 계산 (magnitude)
magnitude = cv2.magnitude(sobel_x, sobel_y)

# 에지 강도를 uint8로 변환
edge_image = cv2.convertScaleAbs(magnitude)

# 원본 이미지와 에지 강도 이미지 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edge Magnitude')
plt.imshow(edge_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
