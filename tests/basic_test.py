from threedfa import *
import cv2
from PIL import Image
from skimage import io as skio

nn = Instance()
images_url = lambda img_idx : f'https://raw.githubusercontent.com/JohnnyRacer/threedfa/citest/tests/media/test_images/{img_idx}.jpeg'
loaded_img = skio.imread(images_url(1))
landmarks = nn.detect_lms(loaded_img, ret_dense=True,round_int=False)
print(f"Found {len(landmarks)} number of faces ")
mesh = depth(loaded_img, landmarks, nn.tddfa.tri)
Image.fromarray(mesh).save('meshoutput_depth.png')


