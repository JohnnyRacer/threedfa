from threedfa import *
import cv2
from PIL import Image
from skimage import io as skio

nn = Instance()

loaded_img = skio.imread('https://bibliogram.pussthecat.org/imageproxy?url=https%3A%2F%2Fscontent-bos3-1.cdninstagram.com%2Fv%2Ft51.2885-15%2F274536839_332749138780209_3662527826268926236_n.jpg%3Fstp%3Ddst-jpg_e35_p1080x1080%26_nc_ht%3Dscontent-bos3-1.cdninstagram.com%26_nc_cat%3D109%26_nc_ohc%3D5-6tl3CNvYIAX9JMpAZ%26edm%3DAAuNW_gBAAAA%26ccb%3D7-4%26oh%3D00_AT9ZJg9vX1shPoC7q1pWqN6NQe7nYOqGc2LDKWz9h5jZcw%26oe%3D6257F2AF%26_nc_sid%3D498da5')
boxes = nn.detect_face(loaded_img)

ver_lst = nn.vlist(loaded_img, boxes, ret_dense=True)

mesh = depth(loaded_img, ver_lst, nn.tddfa.tri)

Image.fromarray(mesh).save('meshoutput_depth.png')

print(mesh)

print(len(ver_lst))
