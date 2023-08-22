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

from datetime import datetime

import firebase_admin
from firebase_admin import db, credentials
from firebase_admin import storage

cred = credentials.Certificate("esp32-b8e97-firebase-adminsdk-gtaqd-34a681d2f0.json")
firebase_admin.initialize_app(cred, {
    "storageBucket" : "esp32-b8e97.appspot.com",
    "databaseURL" : "https://esp32-b8e97-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

WEIGHTS = 'best (1).pt'
IMG_SIZE = 32
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False


ArduinoIp = '10.10.180.100'
IpAddress = f'http://{ArduinoIp}:81/stream'


# Webcam
cap = cv2.VideoCapture(IpAddress)

bucket = storage.bucket()


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

    '''
    // Import the functions you need from the SDKs you need
    import { initializeApp } from "firebase/app";
    // TODO: Add SDKs for Firebase products that you want to use
    // https://firebase.google.com/docs/web/setup#available-libraries

    // Your web app's Firebase configuration
    

    // Initialize Firebase
    const app = initializeApp(firebaseConfig);
    '''
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
            # cls : index of object (0 : Celurit, 1 : Machete, 2 : Person)
            print(int(cls))

            # 아두이노에 보낼 데이터 (문자열 형식)
            if (int(cls) == 0 or int(cls) == 1):
                data = "danger"
                current_time = datetime.now()
                
                ret, frame = cap.read()

                blob = bucket.blob("Screen.png")
                blob.upload_from_filename("Screen.png")
                image_path = f"detect_image/{current_time}.png"
                cv2.imwrite(image_path, frame)

                data = {
                    current_time:
                        {
                            "info" : "ESP32-CAM 1",
                            "image": image_path, 
                            "time" : current_time 
                        }
                    }
                ref = db.reference('/data')
            elif (int(cls) == 2):
                data = "person"

            # HTTP GET 요청을 보낼 URL
            url = f"http://{ArduinoIp}/{data}"
            # url = f"http://{ArduinoIp}"
            print(url)

            try:
                # HTTP POST 요청 보내기
                response = requests.get(url)

                # 응답 확인
                if response.status_code == 200:
                    # 요청이 성공한 경우
                    print("HTTP GET 요청 성공")
                    # print("응답 데이터:", response.text)
                else:
                    print(f"HTTP GET 요청 실패. 응답 코드: {response.status_code}")
            except Exception as e:
                print(f"HTTP GET 요청 중 오류 발생: {e}")


            # 이 부분 DB 안 터지게 하려면 추가해야 할 듯
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
    

# Load all models
model1 = attempt_load('best (1).pt', map_location=device)
model2 = attempt_load('best(2).pt', map_location=device)
model3 = attempt_load('best1.pt', map_location=device)
model4 = attempt_load('best2.pt', map_location=device)

# Inference and Ensemble
with torch.no_grad():
    while True:
        ret, frame = cap.read()

        # Detect using all models
        result1 = detect(frame, model1)
        result2 = detect(frame, model2)
        result3 = detect(frame, model3)
        result4 = detect(frame, model4)

        # Combine results from all models
        final_result = combine_results(result1, result2, result3, result4)

        cv2.imshow('Ensemble Result', final_result)

        if cv2.waitKey(1) == ord('q'):
            break

# Combine Results Logic
def combine_results(result1, result2, result3, result4):
    # Implement your logic here to combine results from all models
    # For example, you can average bounding box coordinates, or choose the more confident prediction, etc.
    # This logic should depend on your specific use case and desired behavior.

    # For demonstration purposes, let's just use the result from model1
    return result1

