import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('/mnt/data/image.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 초기 사각형 영역 설정 (x, y, width, height)
rect = (50, 50, img.shape[1] - 100, img.shape[0] - 100)

# 초기화
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut 수행
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 마스크 처리 (0 또는 2는 배경, 1 또는 3은 전경으로 설정)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = img_rgb * mask2[:, :, np.newaxis]

# 마스크 시각화를 위한 변환
mask_vis = (mask2 * 255).astype('uint8')

# 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask_vis, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Background Removed")
plt.imshow(result)
plt.axis("off")

plt.tight_layout()
plt.show()