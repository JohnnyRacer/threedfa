from threedfa import *
import cv2
from PIL import Image
from skimage import io as skio

nn = Instance()
loaded_img = skio.imread('https://images.unsplash.com/photo-1646061633551-6e8a1e09b72b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=830&q=80')
landmarks = nn.detect_lms(loaded_img, ret_dense=True,round_int=False)
print(f"Found {len(landmarks)} number of faces ")
mesh = depth(loaded_img, landmarks, nn.tddfa.tri)
Image.fromarray(mesh).save('meshoutput_depth.png')
print(mesh)

