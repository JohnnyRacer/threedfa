from threedfa import *
import cv2
from PIL import Image
from skimage import io as skio

nn = Instance()
loaded_img = skio.imread(f'./media/test_images/1.jpeg ')
landmarks = nn.detect_lms(loaded_img, ret_dense=True,round_int=False)
print(f"Found {len(landmarks)} number of faces ")
mesh = depth(loaded_img, landmarks, nn.tddfa.tri)
Image.fromarray(mesh).save('meshoutput_depth.png')


