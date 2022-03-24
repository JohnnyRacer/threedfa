
# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
from pathlib import Path
from re import I
import cv2
import yaml
import os
from .FaceBoxes import FaceBoxes
from .TDDFA import TDDFA
from . import configs
from .utils.functions import draw_landmarks
from .utils.render import render
from .utils.depth import depth
from .utils.pncc import pncc
from .utils.pose import viz_pose
from .utils.serialization import ser_to_ply, ser_to_obj
from .utils.uv import uv_tex
from skimage import io as skio

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

from .FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from .TDDFA_ONNX import TDDFA_ONNX
import numpy as np
abs_cfg_path =Path(configs.__file__).parent.resolve()
CFG_FILEPATH = f'{abs_cfg_path}/default_120x120.yml'




class ImgUtils:

    rgb2bgr = lambda in_img : in_img[..., ::-1] #Converts an image from RGB to BGR

    path_imload = lambda img_dir, conv_bgr=False: cv2.cvtColor(skio.imread(img_dir), cv2.COLOR_BGR2RGB) if conv_bgr else cv2.imread(img_dir) 

    at = lambda in_npd, d_type='int': in_npd.astype(d_type)

class FaceUtils:
 
    ret_lmpts = lambda in_pts : [ (int(round(in_pts[0, i])), int(round(in_pts[1, i]))) for i in range(in_pts.shape[1])   ]

    def cv_draw_landmark(loaded_im : np.ndarray, pts : list, box=None, dot_color=(255,196,96),ln_color=(255,160,255),s_size=4, c_size=1): #ret_plm flag enables return of 2d landmarks, the z points are truncated
        img = loaded_im.copy()
        n = pts.shape[1]
        if n <= 106:
            for i in range(n):
                x = int(round(pts[0, i]))
                y = int(round(pts[1, i]))
                xy =  (x, y)
                cv2.circle(img,xy, s_size, dot_color, 1)
        else:
            sep = 1
            for i in range(0, n, sep):
                x = int(round(pts[0, i]))
                y = int(round(pts[1, i]))
                xy =  (x, y)
                cv2.circle(img,xy, c_size, dot_color, 1)

        if box is not None:
            left, top, right, bottom = np.round(box).astype(np.int32)
            left_top = (left, top)
            right_top = (right, top)
            right_bottom = (right, bottom)
            left_bottom = (left, bottom)
            cv2.line(img, left_top, right_top, ln_color, 1, cv2.LINE_AA)
            cv2.line(img, right_top, right_bottom, ln_color, 1, cv2.LINE_AA)
            cv2.line(img, right_bottom, left_bottom, ln_color, 1, cv2.LINE_AA)
            cv2.line(img, left_bottom, left_top, ln_color, 1, cv2.LINE_AA)

        return img

class Instance:

    def __init__(self, config_path=CFG_FILEPATH) -> None:
        self.cfg = yaml.load(open(CFG_FILEPATH), Loader=yaml.SafeLoader)
        self.tddfa = TDDFA_ONNX(**self.cfg)
        self.face_boxes = FaceBoxes_ONNX()
    
    detect_face = lambda self, loaded_im, filter_score=0.85, trunc_score=True : [e[:-1] if trunc_score else e for e in self.face_boxes(FaceUtils.rgb2bgr(loaded_im)) if e[-1] >= filter_score] #Detects human faces and returns the bounding box with the score as the last element

    detect_lms = lambda self, loaded_im, in_bboxes,ret_dense=False : self.tddfa.recon_vers(*self.tddfa(loaded_im, in_bboxes) , dense_flag=ret_dense)

    def overlay(self, loaded_im : np.ndarray, vert_list : list,mode='pncc',show=False, alpha=0.6, with_bg=True):
        
        if mode == 'base':
            overlay = render(loaded_im, vert_list, self.tddfa.tri, alpha=alpha, show_flag=show)
        elif mode == 'depth':
            overlay = depth(loaded_im, vert_list, self.tddfa.tri, show_flag=show,with_bg_flag=with_bg)
        elif mode == 'pncc':
            overlay = pncc(loaded_im, vert_list, self.tddfa.tri, show_flag=show,with_bg_flag=with_bg)
        elif mode == 'uvtex':
            overlay = uv_tex(loaded_im, vert_list, self.tddfa.tri, show_flag=show)
        return overlay
    
    pose = lambda self, loaded_im, param_list, ver_list,show=False :  viz_pose(loaded_im, param_list, ver_list, show_flag=show)

    ser_to_mesh = lambda self,loaded_im, ver_list, mode='obj' :  ser_to_obj(loaded_im, ver_list, self.tddfa.tri, height=loaded_im.shape[0]) if mode == 'obj' else ser_to_ply(ver_list, self.tddfa.tri, height=loaded_im.shape[0])