#%%
#1. Setup -load packages
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%%
#2. Load your trained model
model = YOLO(r"C:\Users\ruzan\Documents\Ruzana\SHRDC\DL\Hands_On\runs\detect\train7\weights\best.pt")

#%%
#3. Prediction 1 - From image
#Note: This image can be from a URL or from a file path as well
image_path = r'C:\Users\ruzan\Documents\Ruzana\SHRDC\DL\Hands_On\dlcv\object_detection\datasets\a_sign.jpg'
results = model.predict(image_path)

#%%
#Put the prediction in a numpy array
img = results[0].plot()
#Display the reult with matplotlib
img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

#%%
#4. Deployment with webcam stream
results =model.predict(source="0", show=True, stream=True)

#%%
for i in results:
    continue
cv2.destroyAllWindows()

