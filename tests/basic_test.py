from threedfa import *
import cv2
from PIL import Image
from skimage import io as skio

nn = Instance()
loaded_img = skio.imread('https://images.unsplash.com/photo-1646061633551-6e8a1e09b72b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=830&q=80')
boxes = nn.detect_face(loaded_img)
ver_lst = nn.vlist(loaded_img, boxes, ret_dense=True)
mesh = depth(loaded_img, ver_lst, nn.tddfa.tri)
Image.fromarray(mesh).save('meshoutput_depth.png')
print(mesh)
print(len(ver_lst))
