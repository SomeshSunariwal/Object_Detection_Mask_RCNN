"""
@@@@@@

This is a program to detect the object in the video or live feed.
This Program uses Tensorflow, Keras=2.1.6, cython, scikit-image

### For Further more information go throgh readme.txt file

Program is written by "Somesh Sunariwal" and "Diksha Rani" students of NIT Srinagar at IISc Bangalore.

@@@@@@
"""

import cv2
import numpy as np


def random_color(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=.4):
    "apply mask to image"

    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask==1, image[:, :, n] * (1 - alpha) + alpha * c, image[:, :, n])

    return image


def display_instances(image, boxes, masks, ids, names, scores):

    n_instances = boxes.shape[0]
    if not n_instances:
        print("NO INSTANCES TO DISPLAY")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_color(n_instances)
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{}{:2f}'.format(label, score) if score else label
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image


if __name__ == "__main__":
    import os
    import sys
    import random
    import math
    import time
    import coco
    import utils
    import model as modellib

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)


    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    """
    If you want to ...
    1.  Take a live Video feed from your camera then replace the "Video_1.mp4" to 0.
    2.  Take a diffrent video from your directory then give the full directory file path the file name
        at the place of Video_1.mp4.
    3.  Record the video then you must change the video resolution accordingly at the
        place of (1280, 720) otherwise your output video file will be corrupted.
    4.  Record the video then uncomment the "fourcc" and "out"
    5.  Resize the output video resolution then uncommnet the cv2.resize.

    frames is only to check the number of frames processed but for this you have to uncommet the frames.
    
    """
    capture = cv2.VideoCapture("Video_1.mp4")
    #capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter("Output_5.mp4", fourcc, 24, (1280, 720))
    frames = 0
    while 1:
        ret, frame = capture.read()
        result = model.detect([frame])
        r = result[0]
        frame = display_instances(
            frame, r["rois"], r["masks"], r["class_ids"], class_names, r['scores'])
        #frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("frame", frame)
        out.write(frame)
        print("Processed frame :", frames)
        frames += 1
        if cv2.waitKey(1) & 0xff == ord("q"):
            out.release()
            break
    print("Done")
    capture.release()
    out.release()
    cv2.destroyAllWindows()

