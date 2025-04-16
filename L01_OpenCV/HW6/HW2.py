import cv2
import mediapipe as mp

# Mediapipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,     # 실시간 영상 처리를 위한 설정
    max_num_faces=1,             # 검출할 최대 얼굴 수 (필요에 따라 조정)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 영상 캡처
cap = cv2.VideoCapture(0)

# 화면 출력을 위한 윈도우 이름
window_name = "Face Mesh"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Mediapipe는 RGB 이미지를 필요로 함.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 검출 수행
    results = face_mesh.process(rgb_frame)
    
    # 얼굴 랜드마크가 검출되었을 경우
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 각 랜드마크에 대해 점을 그립니다.
            for landmark in face_landmarks.landmark:
                # 정규화 좌표를 실제 픽셀 좌표로 변환
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    # 결과 영상 출력
    cv2.imshow(window_name, frame)
    
    # ESC 키를 누르면 종료 (ASCII 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
