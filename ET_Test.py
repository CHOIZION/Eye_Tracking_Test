import pathlib
from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH
import torch
import cv2
import os
from pathlib import Path

# PosixPath 문제 해결 코드
pathlib.PosixPath = pathlib.WindowsPath

def yolo_process(img):
    yolo_results = model(img)
    df = yolo_results.pandas().xyxy[0]
    obj_list = []
    for i in range(len(df)):
        obj_confi = round(df['confidence'][i], 2)
        obj_name = df['name'][i]
        x_min = int(df['xmin'][i])
        y_min = int(df['ymin'][i])
        x_max = int(df['xmax'][i])
        y_max = int(df['ymax'][i])
        obj_dict = {
            'class': obj_name,
            'confidence': obj_confi,
            'xmin': x_min,
            'ymin': y_min,
            'xmax': x_max,
            'ymax': y_max
        }
        obj_list.append(obj_dict)
    return obj_list

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.25  # confidence threshold 조정 (0.25로 설정)
model.iou = 0.45  # IoU threshold 설정

# 모델 클래스 확인 (디버깅 용도)
print("Model classes:", model.names)

resize_rate = 1
iris_x_threshold, iris_y_threshold = 0.15, 0.26  # 눈동자가 중앙에서 벗어나는 정도

# 모니터 해상도
monitor_width = 1920
monitor_height = 1080

cap = cv2.VideoCapture(0)
cap.set(CAP_PROP_FRAME_WIDTH, 1920)  # 해상도 설정
cap.set(CAP_PROP_FRAME_HEIGHT, 1440)

iris_status = 'Center'
left_x_per = 'None'

# 기본값 초기화
iris_x_position = 0
iris_y_position = 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    # 좌우 반전 추가 (flipCode=1)
    img = cv2.flip(img, 1)
    
    imgS = cv2.resize(img, (0, 0), None, resize_rate, resize_rate)
    results = yolo_process(imgS)

    eye_list = []
    iris_list = []

    # bbox 그리기 및 디버깅용 출력
    for result in results:
        xmin_resize = int(result['xmin'] / resize_rate)
        ymin_resize = int(result['ymin'] / resize_rate)
        xmax_resize = int(result['xmax'] / resize_rate)
        ymax_resize = int(result['ymax'] / resize_rate)
        
        print(f"Detected {result['class']} at [{xmin_resize}, {ymin_resize}, {xmax_resize}, {ymax_resize}] with confidence {result['confidence']}")

        if result['class'] == 'eyes':
            cv2.rectangle(img, (xmin_resize, ymin_resize), (xmax_resize, ymax_resize), (255, 255, 255), 1)
            eye_list.append(result)
        elif result['class'] == 'iris':
            x_length = xmax_resize - xmin_resize
            y_length = ymax_resize - ymin_resize
            circle_r = int((x_length + y_length) / 4)
            x_center = int((xmin_resize + xmax_resize) / 2)
            y_center = int((ymin_resize + ymax_resize) / 2)
            cv2.circle(img, (x_center, y_center), circle_r, (255, 255, 255), 1)
            iris_list.append(result)

    # 왼쪽 눈과 오른쪽 눈 나누기
    if len(eye_list) == 2 and len(iris_list) == 2:
        left_part = []
        right_part = []

        if eye_list[0]['xmin'] > eye_list[1]['xmin']:
            right_part.append(eye_list[0])
            left_part.append(eye_list[1])
        else:
            right_part.append(eye_list[1])
            left_part.append(eye_list[0])

        if iris_list[0]['xmin'] > iris_list[1]['xmin']:
            right_part.append(iris_list[0])
            left_part.append(iris_list[1])
        else:
            right_part.append(iris_list[1])
            left_part.append(iris_list[0])

        # 왼쪽 눈동자 위치 비율 계산
        left_x_iris_center = (left_part[1]['xmin'] + left_part[1]['xmax']) / 2
        left_x_per = (left_x_iris_center - left_part[0]['xmin']) / (left_part[0]['xmax'] - left_part[0]['xmin'])
        left_y_iris_center = (left_part[1]['ymin'] + left_part[1]['ymax']) / 2
        left_y_per = (left_y_iris_center - left_part[0]['ymin']) / (left_part[0]['ymax'] - left_part[0]['ymin'])

        # 오른쪽 눈동자 위치 비율 계산
        right_x_iris_center = (right_part[1]['xmin'] + right_part[1]['xmax']) / 2
        right_x_per = (right_x_iris_center - right_part[0]['xmin']) / (right_part[0]['xmax'] - right_part[0]['xmin'])
        right_y_iris_center = (right_part[1]['ymin'] + right_part[1]['ymax']) / 2
        right_y_per = (right_y_iris_center - right_part[0]['ymin']) / (right_part[0]['ymax'] - right_part[0]['ymin'])

        # 눈동자 위치 비율 평균
        avr_x_iris_per = (left_x_per + right_x_per) / 2
        avr_y_iris_per = (left_y_per + right_y_per) / 2

        # 모니터 상의 눈동자 위치 계산 (1920x1080 기준)
        iris_x_position = int(avr_x_iris_per * monitor_width)
        iris_y_position = int(avr_y_iris_per * monitor_height)

        # Threshold 기준으로 눈동자 위치 계산
        if avr_x_iris_per < (0.5 - iris_x_threshold):
            iris_status = 'Left'
        elif avr_x_iris_per > (0.5 + iris_x_threshold):
            iris_status = 'Right'
        elif avr_y_iris_per < (0.5 - iris_y_threshold):
            iris_status = 'Up'
        else:
            iris_status = 'Center'
    elif len(eye_list) == 2 and len(iris_list) == 0:
        iris_status = 'Down'

    # 눈동자 모니터 위치 표시 (빨간 점 그리기)
    if 'iris_x_position' in locals() and 'iris_y_position' in locals():
        # 눈동자 위치에 빨간 점을 그립니다
        cv2.circle(img, (iris_x_position, iris_y_position), 10, (0, 0, 255), -1)

        # 디버깅용으로 눈동자 위치 정보 출력
        cv2.putText(img, f'Iris Position: ({iris_x_position}, {iris_y_position})', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (30, 30, 30), 2)
    
    # 눈동자 방향 출력
    cv2.putText(img, f'Iris Direction: {iris_status}', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (30, 30, 30), 2)
    
    # 이미지를 화면에 보여줌
    cv2.imshow('img', img)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 모든 창 닫기
cv2.destroyAllWindows()
