import cv2
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img = cv2.imread('./data/mot_color70.jpg')  # 파일이 같은 디렉토리에 있어야 함
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성
sift = cv2.SIFT_create()

# 3. 특징점 검출 및 기술자 계산
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 4. 특징점 시각화 (방향과 크기 포함)
img_with_keypoints = cv2.drawKeypoints(
    img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 5. 시각화
plt.figure(figsize=(10, 6))
plt.title("SIFT 특징점 시각화")
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()