# Vision Processing Basic Report

## 📁 Introduction
본 보고서는 세 가지 OpenCV 과제에 대한 구현 및 결과를 정리한 문서입니다. 각 과제의 요구사항에 따라 Python으로 작성한 코드를 설명하고, 결과 이미지를 포함하였습니다. 결과 이미지는 각 과제의 실행 결과를 시각적으로 보여줍니다.

---

## 📝 과제 1: 이진화 및 히스토그램 구하기
### 📄 설명
주어진 이미지를 사용하여 그레이스케일 변환 및 이진화 후, 히스토그램을 계산하고 시각화합니다.

### 🔧 요구사항
- OpenCV의 `cv2.cvtColor()`로 그레이스케일 변환
- `cv2.threshold()`로 임계값을 설정하여 이진화 수행
- `cv2.calcHist()`로 히스토그램 계산

### 💻 코드
```python
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./data/mistyroad.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
hist = cv2.calcHist([binary_image], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Histogram')
plt.plot(hist, color='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### 🖼️ 결과 이미지
![Result 1-1](./data/result1-1.jpg)
![Result 1-2](./data/result1-2.jpg)

---

## 📝 과제 2: 모폴로지 연산 적용하기
### 📄 설명
주어진 이진화된 이미지에 대해 팽창(Dilation), 침식(Erosion), 열림(Open), 닫힘(Close) 연산을 수행하고 결과를 비교합니다.

### 🔧 요구사항
- `cv2.getStructuringElement()`를 이용하여 5x5 사각형 커널 생성
- `cv2.morphologyEx()`로 모폴로지 연산 수행

### 💻 코드
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./data/JohnHancocksSignature.png', cv2.IMREAD_UNCHANGED)
_, b_image = cv2.threshold(image[:, :, 3], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary = b_image[b_image.shape[0] // 2:b_image.shape[0], 0:b_image.shape[0] // 2 + 1]

kernel = np.uint8([[0,0,1,0,0],
                   [0,1,1,1,0],
                   [1,1,1,1,1],
                   [0,1,1,1,0],
                   [0,0,1,0,0]])

Dilation = cv2.dilate(binary, kernel)
Erosion = cv2.erode(binary, kernel)
Close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
Open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

result = np.hstack((binary, Dilation, Erosion, Close, Open))

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 🖼️ 결과 이미지
![Result 2](./data/result2.jpg)

---

## 📝 과제 3: 기하 연산 및 선형 보간 적용하기
### 📄 설명
주어진 이미지를 45도 회전 후 1.5배 확대하고, 선형 보간을 통해 부드럽게 표현합니다.

### 🔧 요구사항
- `cv2.getRotationMatrix2D()`로 회전 행렬 생성
- `cv2.warpAffine()`를 사용하여 이미지 회전 및 확대
- 선형 보간법(cv2.INTER_LINEAR) 적용

### 💻 코드
```python
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./data/rose.png')
rows, cols = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1.5)
rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (int(cols*1.5), int(rows*1.5)), flags=cv2.INTER_LINEAR)

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
```

### 🖼️ 결과 이미지
![Result 3](./data/result3.jpg)

---

## 📌 결론
세 과제 모두 OpenCV의 다양한 연산 기법을 활용하여 이미지 처리의 기초적인 내용을 다루었습니다. 그레이스케일 변환 및 이진화를 통한 이미지 분석, 모폴로지 연산을 통한 이미지 형태 변형, 그리고 기하 연산과 선형 보간을 통한 이미지 변환 등의 기술을 실습하였습니다.
