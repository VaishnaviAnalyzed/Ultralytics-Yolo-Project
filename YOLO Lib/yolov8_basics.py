from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolo11n.pt", "v11")  

# predict on an image
detection_output = model.predict(source=r"D:\Deep learning\YOLO Lib\young-thoughtful-african-female-employee-looking-out-window-at-workplace-and-dreaming-about-vacation-2J488B0.jpg", conf=0.25, save=True) 

# Display tensor array
print(detection_output)

# Display numpy array
# print(detection_output[0].numpy())