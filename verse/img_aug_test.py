# pip install git+https://github.com/lyg1597/imgaug.git
import imgaug.augmenters as iaa

import cv2 
import matplotlib.pyplot as plt 
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(os.path.join(script_dir, './imgs/img_58.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def set_fog_properties(img, val):
    aug = iaa.Fog(
        intensity_mean = (255, 255),
        density_multiplier = (val, val)
    )
    image_fog = aug(image = img)
    return image_fog

def set_dark_properties(img, val):
    aug = iaa.Add(val*100)
    image_dark = aug(image=img)
    return image_dark

# Value between 0 and 1
image_fog = set_fog_properties(image, 0.5)
plt.figure(1)
plt.imshow(image_fog)

# Value between -1 and 1
image_dark = set_dark_properties(image, 0)
plt.figure(2)
plt.imshow(image_dark)
plt.show()