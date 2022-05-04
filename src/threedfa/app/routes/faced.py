from __future__ import annotations
from typing import Union
import os
from time import time
from fastapi import APIRouter, Depends, File, HTTPException,Form, UploadFile,Response
from fastapi.responses import ORJSONResponse
import numpy as np
from pydantic import BaseModel
from typing import Optional
from http import HTTPStatus
from app.utils.handler import ImageHandler
from app.core import config 
import base64 as b64 
import io
from PIL import Image
from threedfa import *
from app.routes.images import save_posted_b64img,save_posted_binimg
from app.utils.landmarks import bbox_dists, crop_bbox
import os
import zipfile
import io


def zipfiles(filenames,contents, zip_filename=f"{time().__trunc__()} .zip"):
    

    with io.BytesIO() as wb:
        zf = zipfile.ZipFile(wb, "w")
        for fp, data in zip(filenames,contents):
            zf.writestr(fp, data)
        wb.seek(0)
        zip_bytes = wb.read()
        zf.close() # Must close zip for all contents to be written

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(zip_bytes, media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp

router = APIRouter()

fa = Instance()

class FaceDetect(BaseModel):
    timestamp : int
    faces_detected : int
    bounding_boxes: list
    landmarks : list
    original_img: Optional[bytes]
    filename: Optional[str]

def resolve_lms(posted:dict,convert_to_int:bool,return_original:bool=True,size_thresh:str="64,64",return_dense=False) -> Union[int, int, list,list]:

    image_thresh = size_thresh.split(',')
    np_image = bytes_img_tonp(posted['image_bytes'])
    bbox = fa.detect_face(np_image)
    bbox = [[(e.astype('int') if convert_to_int else e.astype('float')).tolist() for e in bb] for bb in bbox ]
    dist = [FaceUtils.bbox_shape(bb) for bb in bbox]
    bbox = [bb for i, bb in enumerate(bbox) if not (dist[i][0] > int(image_thresh[0]) or int(image_thresh[1]) > dist[i][1])]
    
    
    print(f"bbox dists :{dist} ")
    landmarks = fa.detect_lms(np_image, ret_dense=return_dense,round_int=convert_to_int)
    landmarks = [lm.tolist() for lm in landmarks]
    ret_dict = {'timestamp':time().__trunc__(), 'faces_detected':len(bbox),'bounding_boxes':bbox , 'landmarks': landmarks, 'filename':posted['filename'] }

    if return_original:
        ret_dict['image_bytes'] = posted['image_bytes']

    return ret_dict



@router.post("/detect",tags=["face_detection_landmarks"],response_model=FaceDetect, response_class=ORJSONResponse)
async def face_detect(image_handle:Depends=Depends(save_posted_binimg),size_threshold="64,64",return_original:bool=True,convert_to_int:bool=False) -> FaceDetect:
    
    try:
        ret_dict = resolve_lms(image_handle, convert_to_int,return_original,size_thresh=size_threshold)
    except Exception as exp:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail= f'Cannot perform face detection on image due to an error : {exp} '
        )

    return ret_dict

@router.post("/b64/detect",tags=["face_detection_landmarks"],response_class=ORJSONResponse, response_model=FaceDetect)
async def b64_face_detect(image_handle:Depends=Depends(save_posted_b64img),size_threshold="64,64",return_original:bool=True,convert_to_int:bool=False) -> FaceDetect: # Enable convert_to_int flag to return integer 2D image coordinates of landmarks
    
    try:
        ret_dict = resolve_lms(image_handle, convert_to_int,return_original,size_thresh=size_threshold)
    except Exception as exp:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail= f'Cannot perform face detection on image due to an error : {exp} '
        )
    return ret_dict

bytes_img_tonp = lambda bytes_im : np.array(Image.open(io.BytesIO(bytes_im)))

np_img_tobytes = lambda np_im : ImageHandler.dump_pil(Image.fromarray(np_im))

@router.post("/detect/crop",response_class=Response,tags=["face_crop"])
async def face_detect_crop(image_handle:Depends=Depends(face_detect)):
    
    try:
        np_image = bytes_img_tonp(image_handle['image_bytes']) # Using pillow as an intermediary to prevent dealing with reshaping 1D array
        crops = crop_bbox(np_image,image_handle['bounding_boxes'])
        bytes_crops = [np_img_tobytes(im) for im in crops]
        crops_fp = [ f'{idx}.png' for idx in range(len(bytes_crops))]
        
    except Exception as exp:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail= f'Cannot perform crop on detected faces of the image due to an error : {exp} '
        )
    
    return zipfiles(crops_fp, bytes_crops) 

hex_to_rgb = lambda hex : tuple(int(hex.strip('#')[i:i+2], 16) for i in (0, 2, 4))

@router.post("/detect/overlay",response_class=Response,tags=["face_overlay"])
async def overlay_face_landmarks(dot_color:str='#00FF80',image_handle:Depends=Depends(face_detect)):
    if int('81D8D0',16) > 16777215:
         dot_color = '#FFFFFF'
    try:
        np_image = bytes_img_tonp(image_handle['image_bytes'])
        landmarks = np.array(image_handle['landmarks'] )
        for lm in landmarks:
            plotted = FaceUtils.cv_draw_landmark(np_image, np.array(lm),dot_color=hex_to_rgb(dot_color),c_size=-1)
            

        return Response(content=np_img_tobytes(plotted),media_type="image/png")
    except Exception as exp:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail= f'Cannot perform crop on detected faces of the image due to an error : {exp} '
        )


@router.post("/detect/dense/{mode}",response_class=Response,tags=["face_dense_mesh"])
async def face_dense_mesh_overlay():
    pass

