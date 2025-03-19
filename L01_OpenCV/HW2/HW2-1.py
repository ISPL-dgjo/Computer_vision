import cv2
import matplotlib.pyplot as plt

# 이미지 불러오기
image = cv2.imread('./data/mistyroad.jpg')

# 그레이스케일 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 임계값을 이용한 이진화
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 히스토그램 계산
hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_binary = cv2.calcHist([binary_image], [0], None, [256], [0, 256])

# 결과 시각화
plt.figure(figsize=(12, 10))

# 그레이스케일 이미지
plt.subplot(2, 2, 1)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# 그레이스케일 히스토그램
plt.subplot(2, 2, 2)
plt.title('Grayscale Histogram')
plt.plot(hist_gray, color='gray')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# 이진화된 이미지
plt.subplot(2, 2, 3)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

# 이진화 이미지 히스토그램
plt.subplot(2, 2, 4)
plt.title('Binary Histogram')
plt.plot(hist_binary, color='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
