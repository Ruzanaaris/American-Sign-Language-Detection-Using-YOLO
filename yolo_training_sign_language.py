#%%
#1. Setup - load packages
from ultralytics import YOLO

#2. Load the pretrained yolo model parameter
model = YOLO('yolov8n.pt')
#%%
#3. Train the yolo model in the main method
if __name__ =="__main__":
    results = model.train(data='C:/Users/ruzan/Documents/Ruzana/SHRDC/DL/Hands_On/dlcv/object_detection/datasets/American Sign Language Letters.v1-asl-test.yolov8/data.yaml', epochs=3, batch=4)

# %%
