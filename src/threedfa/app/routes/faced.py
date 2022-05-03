from __future__ import annotations
from ctypes import Union
import os
from time import time
from fastapi import APIRouter, Depends, File, HTTPException,Form, UploadFile
import numpy as np
from pydantic import BaseModel
from typing import Optional
from http import HTTPStatus
from hashlib import sha256
from app.utils.handler import ImageHandler
from app.core import config 
import base64 as b64 
import io
from PIL import Image
from threedfa import *
from app.routes.images import save_posted_b64img,save_posted_binimg

router = APIRouter()

fa = Instance()

class FaceDetect(BaseModel):
    timestamp : int
    faces_detected : int
    bounding_boxes: list
    landmarks : list


@router.post("/detect",tags=["face_detection"])
async def face_detect(image_handle:Depends=Depends(save_posted_binimg), convert_to_int:bool=True) -> FaceDetect:
    image_handle['filename']
    np_image = np.array(Image.open(io.BytesIO(image_handle['image_bytes'])))
    bbox = fa.detect_face(np_image)
    landmarks = fa.detect_lms(np_image)
    
    print(np_image)
    return {'timestamp':time().__trunc__(), 'faces_detected':len(bbox), 'landmarks':[e.astype('int') for e in landmarks]}

@router.post("b64/detect",tags=["face_detection"])
async def b64_face_detect() -> FaceDetect:
    pass