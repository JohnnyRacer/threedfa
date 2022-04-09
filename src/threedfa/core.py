
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
import gdown #Change this to using requests to get files from a CDN or FTP server if Drive stops hosting files for whatever reason.
from hashlib import md5
import numpy as np
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # So ONNX and numpy operations won't have strange compute errors.
os.environ['OMP_NUM_THREADS'] = '4' # ... ^

from .FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from .TDDFA_ONNX import TDDFA_ONNX
import numpy as np

abs_cfg_path = str(Path(configs.__file__).parent.resolve()).split('/configs')[:-1][0] #We'll use the empty configs module's file attribute to determine the absolute path of our module.
CFG_FILEPATH = f'{abs_cfg_path}/configs/default_120x120.yml'

class FileHandler: #Utilies to take care of pathing and the aquisition of necessary files.

    base_internal_paths = ['configs', 'weights', 'FaceBoxes/weights']

    required_files = [['param_mean_std_62d_120x120.pkl', 'bfm_noneck_v3.pkl','bfm_noneck_v3.onnx', 'tri.pkl'], ['mb1_120x120.pth', 'mb1_120x120.onnx', 'mb05_120x120.pth', 'mb05_120x120.onnx'],['FaceBoxesProd.pth','FaceBoxesProd.onnx'] ] #Each nested array corresponds to the files that should be contained in each in the base paths list.

    file_ids = [['1m8VPhpYkXIek9kO71SirL9F0mhlk7yYM', '1En1iwXZQMXQ65LEzue4qIsHVvKrHwfe1', '1IHA_i9ITOiEVgTTbmg9_gb3-PVXAgG2C', '19XpIMji1ENHT2YiDyDXEbw7jufwW7Vcq'], ['1Ysw90k3Q_Mj-nUbmAglPm43wCaYn7dgl', '1THbYS8R-KTFk8rWyYBrBih1MBJu-2rBC', '1FpKjd2B5DP7Y3MOJ6RjPQYMqNjm_NlAf', '1bgwi36agDW89qZnY_TG2BEJGB43Tg_5O'], ['18-a_EsknrOSHVAwsjf-QVye1dzh5nmAM', '1iopmrRtAr7CpX0KaR_5tR7tXrWA2-j0Q'] ]

    file_hashes = [['fbb4bec8aeff07bbe034d9cd174256b5', 'b01fc91ed6f5c6b2600b51d4040f0aaa', '14bf6688c7a099ca0ac6efa784663687', 'a9e3b48c325ee719e18b4463605a65e0'],['9e3ca4bdba2cd0e01c8bf9ee4af99f09', '5a11ce627337a9c92a98c1f98dced869', '05000fb422a8ce4a6dff7bf2618e6fc7', 'c327e57b27e159abb4c0c258c1253a4e'], ['29a0018584a22991f119aaa078a462fc', '33a70f92febea5447667c073d4a7283f'] ]

    base_config = lambda rdir=abs_cfg_path ,arch='mobilenet',mdl_type="mb05_120x120" :  {'arch': arch, 'widen_factor': 0.5, 'checkpoint_fp': f'{rdir}/weights/{mdl_type}.pth', 'bfm_fp': f'{rdir}/configs/bfm_noneck_v3.pkl', 'size': 120, 'num_params': 62}

    gd_dlfile = lambda file_id,fp_output , cached_dl=False, show_progress=True: gdown.cached_download(id=file_id,output=fp_output, quiet=show_progress) if cached_dl else gdown.download(id=file_id,output=fp_output, quiet=show_progress)

    def hash_verif(in_fname:str, chunk:int=4096): #We read the file with chunks to prevent OOM on large files, see https://www.quickprogrammingtips.com/python/how-to-calculate-md5-hash-of-a-file-in-python.html for more details.
        md5_hash = md5()
        with open(in_fname,"rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(chunk),b""):
                md5_hash.update(byte_block)
        return md5_hash.hexdigest()

    file_chk = lambda : [[os.path.isfile(os.path.join(rdir , efp)) for efp in (FileHandler.required_files[i]) ] for i, rdir in enumerate([ os.path.join(abs_cfg_path, e) for e in FileHandler.base_internal_paths ]) ]

    def setup_dlfiles(replace_existing:bool=True):

        coalesce_bool = lambda in_iterable : np.multiply.reduce(np.array(in_iterable).astype('int'))
        bool_chk = FileHandler.file_chk()
        if not bool(coalesce_bool([coalesce_bool(el) for el in bool_chk])):
            fdl_list = [[FileHandler.gd_dlfile(FileHandler.file_ids[i][idx], os.path.join(abs_cfg_path, os.path.join(FileHandler.base_internal_paths[i], FileHandler.required_files[i][idx]) ) ) for idx in range(len(ems)) if ems[idx] == False ] for i, ems in enumerate(bool_chk)]  
            print(fdl_list)
            print(f"Downloaded {len(fdl_list)} files " )
    
    def yaml_chk(ifp=CFG_FILEPATH):
        if not os.path.isfile(ifp):
            with open(ifp, "w") as wfb:
                yaml.dump(FileHandler.base_config(), wfb)

class ImgUtils:

    rgb2bgr = lambda in_img : in_img[..., ::-1] #Converts an image from RGB to BGR

    path_imload = lambda img_dir, conv_bgr=False: cv2.cvtColor(skio.imread(img_dir), cv2.COLOR_BGR2RGB) if conv_bgr else cv2.imread(img_dir) 

    at = lambda in_npd, d_type='int': in_npd.astype(d_type)

class FaceUtils:
    
    bbox_shape = lambda bbox :np.abs(np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]]).astype('int'))

    ret_lmpts = lambda in_pts, lm_type='2d' : [(int(round(in_pts[0, i])), int(round(in_pts[1, i]))) if lm_type == '2d' else (int(round(in_pts[0, i])), int(round(in_pts[1, i])), int(round(in_pts[2, i]))) for i in range(in_pts.shape[1])   ]

    def cv_draw_landmark(loaded_im : np.ndarray, pts:list, box:list=None, dot_color=(255,196,96),ln_color=(255,160,255),s_size=4, c_size=1): #ret_plm flag enables return of 2d landmarks, the z points are truncated
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
    
    def split_uv (loaded_im:np.ndarray,n_faces=0):
        if loaded_im.shape[0] == loaded_im.shape[1] or n_faces == 0:
            return [loaded_im]
        offset_index = 0
        width = np.max(loaded_im.shape[:-1] )
        height = np.min(loaded_im.shape[:-1])
        splitted_ims = []
        offset_width = np.trunc(width/n_faces)
        for i in range(0, n_faces):
            splitted_im = loaded_im[:height,i*height:(1+i)*height ] 
            print(i*height,(1+i)*height)
            splitted_ims.append(splitted_im)
        return splitted_ims

class Instance:

    def __init__(self, config_path=CFG_FILEPATH) -> None:
        FileHandler.yaml_chk(config_path)
        self.cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        FileHandler.setup_dlfiles()
        self.tddfa = TDDFA_ONNX(**self.cfg)
        self.face_boxes = FaceBoxes_ONNX()
    
    detect_face = lambda self, loaded_im, filter_score=0.85, trunc_score=True : [e[:-1] if trunc_score else e for e in self.face_boxes(loaded_im) if e[-1] >= filter_score] #Detects human faces and returns the bounding box with the score as the last element

    detect_lms = lambda self, loaded_im, in_bboxes,ret_dense=False, lm_type='2d' :  FaceUtils.ret_lmpts(self.tddfa.recon_vers(*self.tddfa(loaded_im, in_bboxes) , dense_flag=ret_dense), lm_type=lm_type)

    vlist = lambda self, loaded_im, in_bboxes,ret_dense=False : self.tddfa.recon_vers(*self.tddfa(loaded_im, in_bboxes) , dense_flag=ret_dense)

    def overlay(self, loaded_im : np.ndarray, vert_list : list,mode='pncc',show=False, alpha=0.6, with_bg=True):
        
        if mode == 'base':
            ovly_func = render
        elif mode == 'depth':
            ovly_func = depth
        elif mode == 'pncc':
            ovly_func = pncc
        elif mode == 'uvtex':
            ovly_func = uv_tex
        overlay = ovly_func(loaded_im, vert_list, self.tddfa.tri, show_flag=show)
        return overlay
    
    pose = lambda self, loaded_im, param_list, ver_list,show=False :  viz_pose(loaded_im, param_list, ver_list, show_flag=show)

    ser_to_mesh = lambda self,loaded_im, ver_list, mode='obj', sv_fp=None :  ser_to_obj(loaded_im, ver_list, self.tddfa.tri, height=loaded_im.shape[0], wfp=sv_fp if sv_fp is not None else f'faceobjs_{len(ver_list)}_{time.time().__trunc__()}.obj') if mode == 'obj' else ser_to_ply(ver_list, self.tddfa.tri, height=loaded_im.shape[0],wfp=sv_fp if sv_fp is not None else f'faceobjs_{len(ver_list)}_{time.time().__trunc__()}.ply')