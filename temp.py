import json
import numpy as np
import cv2
import torch

img2mse = lambda x, y : torch.mean((x - y) ** 2)

img = cv2.imread("nerf_screenshot.png")
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
W = img.shape[1]
H = img.shape[0]
coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1), dtype=int)
print(img.shape)
print(coords.shape)

mesh = np.array([[2,10],[40,20]])
print("MSEH",mesh[:,0])
img1points = img[mesh[:,0],mesh[:,1]]

tensor1 = torch.tensor(img1points)
img2points = np.array([img[10,40],img[24,45]],dtype = float)
img2points.reshape(2,3)
tensor2 = torch.tensor(img2points)
print("img1",img1points)
print("img2",img2points)


print(img2mse(tensor1,tensor2))
# batch = 
# print(w,h)
# i = 1
# rgb = img[i*len(batch): i*len(batch) + len(batch)]
