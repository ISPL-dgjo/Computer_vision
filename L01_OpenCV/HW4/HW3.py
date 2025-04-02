import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 (컬러)
img1 = cv.imread('./data/img1.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('./data/img2.jpg', cv.IMREAD_COLOR)

if img1 is None or img2 is None:
    raise IOError("이미지를 불러올 수 없습니다.")

# Grayscale 변환 (SIFT 용도)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 특징점 매칭 (BFMatcher + ratio test)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 호모그래피 계산
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # (1) 특징점 매칭 시각화
    match_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # (2) 이미지 정합 결과 (warpPerspective)
    h2, w2 = img2.shape[:2]
    aligned_img = cv.warpPerspective(img1, H, (w2, h2))
    comparison_img = np.hstack((aligned_img, img2))

    # (3) 파노라마 블렌딩
    h1, w1 = img1.shape[:2]
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate((warped_corners,
                                  np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-xmin, -ymin]
    H_translate = np.array([[1, 0, translation_dist[0]],
                            [0, 1, translation_dist[1]],
                            [0, 0, 1]])

    panorama = cv.warpPerspective(img1, H_translate @ H, (xmax - xmin, ymax - ymin))
    panorama[translation_dist[1]:h2+translation_dist[1], translation_dist[0]:w2+translation_dist[0]] = img2

    # 결과 출력
    plt.figure(figsize=(24, 14))

    plt.subplot(1, 3, 1)
    plt.title("Matching Keypoints")
    plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Warped Image vs Original")
    plt.imshow(cv.cvtColor(comparison_img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Panorama Blending Result")
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print("호모그래피를 계산하기 위한 충분한 매칭쌍이 없습니다.")
