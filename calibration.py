import os
import random
from PIL import Image, ImageOps
import cv2
import glob
import numpy as np

root = "C:\\Users\\C23080\\Downloads\\archive\\coco2017\\val2017"

image_list = os.listdir(root)
image_list = [os.path.join(root, f) for f in image_list]
random.shuffle(image_list)

img_datas = []
for idx, file in enumerate(image_list[:100]):
    image = Image.open(file).convert('RGB')
    # Get sample input data as a numpy array in a method of your choosing.
    img_width, img_height = image.size
    size = max(img_width, img_height)
    image = ImageOps.pad(image, (size, size), method=Image.BILINEAR)
    image = image.resize((640, 640), Image.BILINEAR)
    tensor_image = np.asarray(image).astype(np.float32)
    tensor_image /= 255.0
    tensor_image = np.expand_dims(tensor_image, axis=0)
    img_datas.append(tensor_image)

calib_datas = np.vstack(img_datas)
print(f'calib_datas.shape: {calib_datas.shape}')
np.save(file='tflite_calibration_data_100_images_640.npy', arr=calib_datas)
