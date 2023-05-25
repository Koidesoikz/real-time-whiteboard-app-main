# The functions in this script are heavily inspired from various functions in the Ultralytics library. They have here been rewritten to function with tensorflow.

import time
import torch
import torchvision
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0, # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    ):

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    '''
    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    '''

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = K.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [tf.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction): # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0 # width-height
        x = tf.transpose(x, perm=[1, 0])[xc[xi]] # confidence
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = tf.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = tf.concat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = tf.split(x, (4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = tf.where(cls > conf_thres)[0], tf.where(cls > conf_thres)[1]
            x = tf.concat((box[i], x[i, 4 + j, None], tf.cast(j[:, None], dtype=tf.float32), mask[i]), 1)
        else:  # best class only
            conf = tf.math.reduce_max(cls, axis=1, keepdims=True)
            j = tf.math.argmax(cls, axis=1)
            j = tf.expand_dims(j, axis=1)
            x = tf.concat((box, conf, tf.cast(j, dtype=tf.float32), mask), 1)[tf.reshape(conf, (-1,)) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == tf.constant(classes)).any(1)]

        # Apply finite constraint
        # if not tf.math.is_finite(x).all():
        #     x = x[tf.math.is_finite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # !_ MÅSKE FORKERT
        xConfSortInd = tf.argsort(x[:, 4], direction='DESCENDING')[:max_nms]
        x = tf.gather(x, xConfSortInd)
        # x = x[tf.argsort(x[:, 4], direction='DESCENDING')[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        _, i = tf.image.non_max_suppression_with_scores(boxes, scores, max_det, iou_thres)  # NMS
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes)
            weights = iou * scores[None]  # box weights

            x[i, :4] = tf.cast(tf.linalg.matmul(weights, x[:, :4]), dtype=tf.float32) / tf.math.reduce_sum(weights, axis=1, keepdims=True) # merged boxes
            # x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = tf.gather(x, tf.cast(i, dtype=tf.int32))
    
    return output

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape

    # masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
    
    # mask1 = masks_in @ protos.float().view(c, -1)


    # mask1 = masks_in @ tf.reshape(protos, [c, -1])
    # mask2 = K.softmax
    # mask3 = tf.reshape(mask2, [-1, mh, mw])

    masks = tf.reshape(K.sigmoid((masks_in @ tf.reshape(tf.cast(protos, dtype=tf.float32), [c, -1]))), [-1, mh, mw]) # CHW

    # downsampled_bboxes = tf.identity(bboxes)
    # downsampled_bboxes[:, 0] *= mw / iw
    # downsampled_bboxes[:, 2] *= mw / iw
    # downsampled_bboxes[:, 3] *= mh / ih
    # downsampled_bboxes[:, 1] *= mh / ih

    # masks = crop_mask(masks, downsampled_bboxes)  # CHW
    # if upsample:
    #     masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    if upsample:
        masks = tf.transpose(masks, perm=[1, 2, 0])
        masks = tf.image.resize(masks[:], (ih, iw), method="bilinear", preserve_aspect_ratio=False)
        masks = tf.transpose(masks, perm=[2, 0, 1])
    return tf.math.greater(masks, 0.5)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
        eps

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)