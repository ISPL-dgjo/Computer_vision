import cv2
import numpy as np
from sort import Sort

# YOLOv4 설정
cfg_file = "./data/yolov4.cfg"
weights_file = "./data/yolov4.weights"
names_file = "./data/coco.names"

# 클래스 이름 로드
with open(names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
person_idx = classes.index("person")  # 사람 클래스 인덱스

# YOLO 네트워크 불러오기
net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# SORT 추적기 초기화
tracker = Sort()

# 비디오 읽기
cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# YOLOv4의 출력 레이어 이름 얻기
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv4 입력 준비
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    # 객체 검출 결과 처리
    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == person_idx and confidence > 0.5:
                box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                center_x, center_y, w, h = box.astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detections.append([x, y, x + int(w), y + int(h), confidence])

    # numpy로 변환
    dets = np.array(detections)
    
    # SORT 추적
    tracks = tracker.update(dets)

    # 결과 시각화
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 디스플레이
    cv2.imshow('SORT Tracking', frame)
    if cv2.waitKey(1) == 27:  # ESC 키 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()