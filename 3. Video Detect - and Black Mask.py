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
    colors = [tuple([255, 255, 255]) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=1, pera=0):
    "apply mask to image"
    if pera == 1:
        im = np.zeros((720, 1280, 3))
        for n, c in enumerate(color):
            image[:, :, n] = np.where(mask == 1, image[:, :, n] * (1 - alpha) + alpha * c, im[:, :, n])
    if pera == 0:
        for n, c in enumerate(color):
            image[:, :, n] = np.where(mask == 1, image[:, :, n] * (1 - alpha) + alpha * c, image[:, :, n])
    else:
        pass
    return image


def display_instances_color(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    if not n_instances:
        print("NO INSTANCES TO DISPLAY")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_color(n_instances)
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        #image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = names[ids[i]]
        caption = '{}'.format(label)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def display_instances_black(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    if not n_instances:
        print("NO INSTANCES TO DISPLAY")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_color(n_instances)
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color, pera=1)
        #image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = names[ids[i]]
        caption = '{}'.format(label)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


if __name__ == "__main__":
    import os
    import random
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

    frames = 0
    ## put the video file name at the place of the Video_1.mp4
    capture = cv2.VideoCapture("Video_1.mp4")

    ## If you want to record the output video then uncomment the below two lines.
    ## Change the video resolution at "(1280, 720)" and also change the output file name accordingly (if wish).
    1
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter("Output_black_4.mp4", fourcc, 24, (1280, 720))
    
    while 1:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (1280, 720))
        result = model.detect([frame])
        r = result[0]
        frame_1 = display_instances_black(frame, r["rois"], r["masks"], r["class_ids"], class_names, r['scores'])
        frame = display_instances_color(frame_1, r["rois"], r["masks"], r["class_ids"], class_names, r['scores'])
        #out.write(frame)
        #print("Processed frame :", frames)
        #frames += 1
        cv2.imshow("Black Window", frame_1)
        if cv2.waitKey(25) & 0xff == ord("q"):
            #out.release()
            capture.release()
            break

    print("Done")
    #out.release()
    capture.release()
    cv2.destroyAllWindows()

