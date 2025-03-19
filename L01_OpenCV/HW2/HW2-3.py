import cv2
import matplotlib.pyplot as plt

# 이미지 불러오기
image = cv2.imread('./data/rose.png')
rows, cols = image.shape[:2]

# 회전 변환 행렬 생성 (중심 기준, 45도 회전, 1.5배 확대)
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1.5)

# 이미지 회전 및 확대 (선형 보간 적용)
rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (int(cols*1.5), int(rows*1.5)), flags=cv2.INTER_LINEAR)

# 결과 시각화
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Rotated & Scaled Image')
plt.imshow(cv2.cvtColor(rotated_scaled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()