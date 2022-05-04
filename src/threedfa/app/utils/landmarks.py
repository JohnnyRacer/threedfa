import numpy as np

def crop_bbox(loaded_img:np.ndarray,bbox:list) -> list:
    bbox = np.array(bbox)
    bbox = np.where(bbox > 0, bbox, 0) # Remove all negative values by converting them to zero
    cropped = [loaded_img[bb[1]:bb[3],bb[0]:bb[2]] for bb in bbox]
    return cropped

def bbox_dists(bbox, return_hw=False):
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    center_dist = (h+w)*0.25 #Since we are dealing with an diagonal of the rect, we only need 1/4 the distance instead of 1/2
    center_point = np.array([bbox[0]+center_dist, bbox[1]+center_dist]).astype('int')
    if return_hw:
        return [h,w], center_point
    return center_dist, center_point, 

def realign_lm(in_lm : np.ndarray, bbox : np.ndarray, scale=0.4,offset_center_dist=True) -> np.ndarray:
    assert len(in_lm) == 68 and len(bbox) == 4, "Invalid number of points for landmarks or bounding box, landmarks needs to be 68 points and the bounding box needs to be 2 points "
    center_dist = 1
    if offset_center_dist:
        center_dist, center_point = bbox_dists(bbox)
    return np.array(list(map((lambda lm : [lm[0]- bbox[0]+(scale*center_dist), lm[1] - bbox[1]+(scale*center_dist) ]),in_lm))) #Is this the closure you were looking for? 
