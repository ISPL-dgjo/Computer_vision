import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 컬러로 불러오기
img1_color = cv.imread('./data/img1.jpg', cv.IMREAD_COLOR)
img2_color = cv.imread('./data/img2.jpg', cv.IMREAD_COLOR)

# 예외 처리
if img1_color is None or img2_color is None:
    raise IOError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")

# 그레이스케일로 변환 (SIFT는 grayscale 기준으로 동작)
img1_gray = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv.SIFT_create()

# 특징점과 기술자 계산
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFMatcher 생성 (L2 norm)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test 적용
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 충분한 매칭이 있는지 확인
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 호모그래피 계산 (RANSAC 사용)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # 이미지 정합
    h2, w2, _ = img2_color.shape
    aligned_img = cv.warpPerspective(img1_color, H, (w2, h2))

    # 매칭 결과 시각화 (good_matches만 표시)
    match_img = cv.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 결과 시각화
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.title("Matching Points")
    plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Aligned Image (Left) vs Target Image (Right)")
    combined = np.hstack((aligned_img, img2_color))
    plt.imshow(cv.cvtColor(combined, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print("매칭된 특징점이 충분하지 않습니다.")
