import cv2
import matplotlib.pyplot as plt

# 1. 두 이미지 불러오기
img1 = cv2.imread('./data/mot_color70.jpg')
img2 = cv2.imread('./data/mot_color83.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 및 특징점 추출
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 3. 매칭 객체 생성 (BFMatcher 사용)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 4. 매칭 결과 이미지 생성
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# 5. 시각화
plt.figure(figsize=(15, 8))
plt.title("SIFT result")
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()