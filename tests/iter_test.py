from hashlib import md5
import io
import os
import shutil
from threedfa import *
import cv2
from PIL import Image
from skimage import io as skio
from subprocess import run
nn = Instance()
import json
import numpy as np

images_url = lambda img_idx : f'https://raw.githubusercontent.com/JohnnyRacer/threedfa/citest/tests/media/test_images/{img_idx}.jpeg'
test_types = ['base', 'depth', 'pncc', 'uvtex']

hash_dict = {}

for test_t in test_types:
    os.makedirs('./tests/media/test_results/lms', exist_ok=True)
    for i in range(0,5):
        in_fp = f'./tests/media/test_images/{i+1}.jpeg'
        loaded_img = skio.imread(images_url(i+1))
        landmarks = nn.detect_lms(loaded_img, ret_dense=True,round_int=False)
        print(f"Found {len(landmarks)} number of faces ")
        mesh = nn.overlay(loaded_img, landmarks, mode=test_t)
        out_fp = f'./tests/media/test_results/{test_t}_{i+1}.png'
        hash_dict[test_t] = {}
        
        hash_dict[test_t]['in_fp'] = in_fp
        hash_dict[test_t]['out_fp'] = out_fp
        

        landmarks = nn.detect_lms(loaded_img, ret_dense=False,round_int=False)
        plotted_lms = [FaceUtils.cv_draw_landmark(loaded_img,lm) for lm in landmarks]
        for idx, lm in enumerate(landmarks):
            print(f"Landmark shape is {lm.shape} ")
            with open(f'./tests/media/test_results/lms/sparse_{i}_{idx}.npy', 'wb') as wb:
                np.save(wb ,lm)

        for idx, im in enumerate(plotted_lms):
            plotted_im = f'./tests/media/test_results/lmsparse_{i}_{idx}.png'
            Image.fromarray(im).save(plotted_im)

hash_dict['base_hashes'] = {"./tests/media/test_images/1.jpeg": "007941a9befa67541921074ae3718113", "./tests/media/test_images/2.jpeg": "ae0b40dddba5d8666017ea45243102e5", "./tests/media/test_images/3.jpeg": "618c8c96fafc1cbf964f205f9a115ee1", "./tests/media/test_images/4.jpeg": "61ab934ad44f079a2c13826a4fad93f7", "./tests/media/test_images/5.jpeg": "9e467cf7fd0e543521c647dd4d8e372f"}

print(hash_dict)

with open('./tests/test_hashesRef.json', 'r') as fp:
    loaded_hashes = json.load(fp)

print(f"Hashes are the same" if hash_dict == loaded_hashes else "Hashes don't match")
