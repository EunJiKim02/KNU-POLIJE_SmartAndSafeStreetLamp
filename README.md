# KNU-POLIJE_SmartAndSafeStreetLamp


## Project Summary
In order to improve some areas in Indonesia where security is not good, a street
light with adjustable brightness and a reporting function was devised. When a
person is recognized using the YOLOv7 model, the brightness of the street light
is adjusted with the illuminance sensor, and when a dangerous object other than
a person is recognized, the final goal is to perform a reporting function.

<br>

## Overall Structure 
![image](https://github.com/EunJiKim02/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/95c9aa53-e53d-4d4c-a27a-ed055182fd42)


<br>

## Technical Description

### Circuit components and Circuit Diagram

- Arduino Uno
- ESP32-CAM
- NeoPixel (SW2812)
- Photoresistor
- some wires and resistors


### AI Model Training
We will use YOLOv7 for object detections. YOLO stands for You Only
Look Once, which is a deep learning model commonly used for object
detection in videos or images. We will take the model already provided
in the YOLO v7 paper and further learn threat objects (Machete, Celurit) to increase accuracy.

### Real-time information management system
When AI detects a dangerous situation, it automatically takes pictures of the streets and sends them to administrators. Store the results detected through YOLO in the Firebase Realtime database and Firebase Storage.<br>
Stored information displays content on the web created using the Laravel Framework. It contains photos, dates, and module information (location) designed to help administrators obtain information efficiently.

<br>

## Functional Description

### Automatic brightness adjustment
The light sensor in the streetlight recognizes
the ambient brightness and brightens the street accordingly as the ambient
brightness decreases from the reference brightness.


### Object detection
The camera module in the streetlight recognizes objects near
the streetlight. If a person passes near the lamppost, it recognizes the person and
increases the brightness from a level adapted to the ambient brightness to the
maximum brightness to ensure pedestrian safety. Once the person is out of the
camera module's field of view, it returns to ambient brightness after a few seconds
to maximize energy efficiency. If a weapon is recognized by the camera ( Machete, Celurit ), a report function is triggered in case of an accident.


### Reporting function
Considering Indonesia's lack of security, AI automatically detects dangerous situations and sends information to the police. It sends information containing the names, photos, and times of photos and streetlights, and updates to the website in real time. Photos are kept on the website for three days, and can then be found on Firebase Storage.

<br>

## Result
### Object Detection
![image](https://github.com/EunJiKim02/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/a9598da9-c111-4717-bded-6ecb91a5a240)


<br>

### Manager (Police) Web
![image](https://github.com/EunJiKim02/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/74cb0710-5c94-46b4-bacb-03ff8567abd1)

<br>

### Demonstration



## Team Info.
We are a team of KNU students from Korea and POLIJE students from Indonesia.
#### KNU Students
- Eunji Kim
- SeongHee Gu
- Dongje Park

#### POLIJE Students
- Athiyah
- Johardio Eka
- Dimas Raditya


