import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import pathlib

# PosixPath 오류 해결
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 모델 설정
weights = 'best.pt'  # 사용자 모델 가중치 파일
device = 'cpu'       # 'cpu' 또는 'cuda:0'

# 모델 로드
device = select_device(device)
model = DetectMultiBackend(weights, device=device)
stride = model.stride
names = model.names  # 클래스 이름들 (['iris', 'eye'] 등)
img_size = (640, 640)  # 입력 이미지 크기

# 웹캠 설정
cap = cv2.VideoCapture(0)  # 0번 카메라 사용
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 이미지 전처리
    img = cv2.resize(frame, img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float()  # uint8 to fp32
    img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 추론
    pred = model(img_tensor)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # 결과 처리
    for i, det in enumerate(pred):  # detections per image
        im0 = frame.copy()
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], im0.shape).round()

            # 결과 그리기
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                # 바운딩 박스 그리기
                cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                # 레이블 텍스트 그리기
                cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 출력
        cv2.imshow('Real-Time Detection', im0)

    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
