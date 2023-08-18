import time
import cv2
import torch
import numpy as np
import requests

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


WEIGHTS = 'camdetection.pt'
IMG_SIZE = 64
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

IpAddress = 'http://10.10.183.213:81/stream'

# Webcam
cap = cv2.VideoCapture(IpAddress)

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

# Load model
model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


# Detect function
def detect(frame):
    # Load image
    img0 = frame

    # Padded resize
    imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size
    img = letterbox(img0, imgsz, stride=stride)[0]


    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    # Inference
    t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]

    # det : list of data

    # s : size of box size
    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            # n : number of detected objects
            n = (det[:, -1] == c).sum()  # detections per class

            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # 아두이노의 IP 주소와 포트 번호
            arduino_ip = "127.0.0.1"
            arduino_port = "5000"

            print(cls)

            # 아두이노에 보낼 데이터 (문자열 형식)
            if (cls == 0):
                data = "Celurit"
            elif (cls == 1):
                data = "Machete"
            elif (cls == 2):
                data = "Person"

            # 아두이노에 보낼 데이터 (JSON 형식)
            # data = {
            #     "key1": "value1",
            #     "key2": "value2"
            # }

            # HTTP POST 요청을 보낼 URL
            url = f"http://{arduino_ip}:{arduino_port}"

            try:
                # HTTP POST 요청 보내기
                response = requests.post(url, data=data)

                # 응답 확인
                if response.status_code == 200:
                    # 요청이 성공한 경우
                    print("HTTP POST 요청 성공")
                    print("응답 데이터:", response.text)
                else:
                    print(f"HTTP POST 요청 실패. 응답 코드: {response.status_code}")
            except Exception as e:
                print(f"HTTP POST 요청 중 오류 발생: {e}")


            # cls : index of object (0 : Celurit, 1 : Machete, 2 : Person)
            
            # if (cls == 0):
            #     URL = ''
            # elif (cls == 1):
            #     URL = ''
            # elif (cls == 2):
            #     URL = ''

            # only send requests when detected object is changed
            # if (oldURL != URL):
            #     try:
            #         response = requests.get(URL)
            #         print("request sent!")
            #     except IncompleteRead:
            #         print("request couldn't be sent!")

            # oldURL = URL


            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

        print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

    # return results
    return img0




# main
check_requirements(exclude=('pycocotools', 'thop'))
with torch.no_grad():
    while(True):
        ret, frame = cap.read()
        result = detect(frame)
        cv2.imshow('pred_image', result)

        if cv2.waitKey(1) == ord('q'):
            break
            