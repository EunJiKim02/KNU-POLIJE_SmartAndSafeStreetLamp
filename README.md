# KNU-POLIJE_SmartAndSafeStreetLamp


## Project Summary
In order to improve some areas in Indonesia where security is not good, a street
light with adjustable brightness and a reporting function was devised. When a
person is recognized using the YOLOv7 model, the brightness of the street light
is adjusted with the illuminance sensor, and when a dangerous object other than
a person is recognized, the final goal is to perform a reporting function.

<br>

## Overall Structure 
![image](https://github.com/mobuktodae/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/551bd8a9-7582-48f1-ba62-26654508938c)

<br>

## Technical Description

### Circuit components and Circuit Diagram

- Arduino Uno
- ESP32-CAM
- NeoPixel (SW2812)
- Photoresistor
- piezoelectric buzzer
- tect switch
- some wires and resistors


![image](https://github.com/mobuktodae/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/1314e050-7ae3-4406-ae99-7577a1bcb251)

### AI Model Training
We will use YOLOv7 for object detections. YOLO stands for You Only
Look Once, which is a deep learning model commonly used for object
detection in videos or images. We will take the model already provided
in the YOLO v7 paper and further learn threat objects such as knives and
hammers to increase accuracy.

### Message send function using STMP
We’ll use ESP-Mail-Client Library file. Using this, we will make it possible
to report using buttons and through detection of dangerous objects. See
the link in the reference for details on implementation

<br>

## Functional Description

### Automatic brightness adjustment
The light sensor in the streetlight recognizes
the ambient brightness and brightens the street accordingly as the ambient
brightness decreases from the reference brightness.

![image](https://github.com/mobuktodae/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/adf0566b-fd23-4e96-8e61-f9556f786751)

### Object detection
The camera module in the streetlight recognizes objects near
the streetlight. If a person passes near the lamppost, it recognizes the person and
increases the brightness from a level adapted to the ambient brightness to the
maximum brightness to ensure pedestrian safety. Once the person is out of the
camera module's field of view, it returns to ambient brightness after a few seconds
to maximize energy efficiency. If a weapon is recognized by the camera (e.g. knife,
baseball bat, etc.), a report function is triggered in case of an accident.

![image](https://github.com/mobuktodae/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/ac1ae49b-e047-4033-a162-cb42557e6ab2)

### Reporting function
Considering Indonesia's lack of security, the street lights are
equipped with a button for reporting incidents. When a pedestrian presses the
button, a photo from the camera is immediately sent to the administrator's email.
The name of the light and photo at that time are sent together, and the name of
the camera can be set to the location of the streetlight to identify which location
needs help. Additionally, the AI model also serves as a security CCTV, including the
ability to automatically report street people if they have dangerous objects.

![image](https://github.com/mobuktodae/KNU-POLIJE_SmartAndSafeStreetLamp/assets/87495422/5774b107-efd1-4cba-a0c4-ad37a3aba356)


<br>

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


