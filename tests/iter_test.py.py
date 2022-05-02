from threedfa import *
import cv2
from PIL import Image
from skimage import io as skio
from subprocess import run

nn = Instance()
loaded_img = skio.imread('')
landmarks = nn.detect_lms(loaded_img, ret_dense=True,round_int=False)
print(f"Found {len(landmarks)} number of faces ")
mesh = depth(loaded_img, landmarks, nn.tddfa.tri)
Image.fromarray(mesh).save('meshoutput_depth.png')
print(mesh)

