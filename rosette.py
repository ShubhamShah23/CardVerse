import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

x,y,w,h = 1035,842,1269,868
data = []
folder_path = r"training_set\Real text"

for filename in os.listdir(folder_path):
    file = os.path.join(folder_path, filename)
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = img[y:y+h,x:x+w]
    img = cv2.GaussianBlur(img, (1, 1), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_length = sum([cv2.arcLength(cnt, True) for cnt in contours])
    data.append(total_length)

data2 = []
folder_path = r"training_set\Fake text"

for filename in os.listdir(folder_path):
    file = os.path.join(folder_path, filename)
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = img[y:y+h,x:x+w]
    img = cv2.GaussianBlur(img, (1, 1), 0)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_length = sum([cv2.arcLength(cnt, True) for cnt in contours])
    data2.append(total_length)

threshold = (np.median(data)+np.median(data2))/2
counter = 0
for i in data:
    if i >= threshold:
        counter +=1
for j in data2:
    if j < threshold:
        counter += 1
print("Misclassified = ", counter)

x = [i for i in range(0,15)]
y1 = (data)
y2 = (data2+[np.mean(data2)]*3)
y3 = [threshold]*15
fig, ax = plt.subplots()
ax.plot(x, y1, 'b-', label='Real')
ax.plot(x, y2, 'r-', label='Fake')
ax.plot(x, y3, 'g-', label='Threshold')
ax.set_xlabel('Image')
ax.set_ylabel('Number of Contours')
ax.set_title("Rosette Pattern Test")
ax.legend()
plt.show()