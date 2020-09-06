# imports
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from helper import sliding_window
from helper import image_pyramid
import numpy as np
import imutils
import argparse
import time
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image')
ap.add_argument('-s', '--size', type=str, default="(200,150)",
                help='ROI size (in pixels)')
ap.add_argument('-c', '--min-conf', type=float, default=0.9,
                help='min probability to filter weak detections')
ap.add_argument('-v', '--visualize', type=int, default=-1,
                help='wether or not to show the extra visualization for debugging')

args = vars(ap.parse_args())

# constants
WIDTH = 600  # ( setting the starting width of image )
PYR_SCALE = 1.5  # ( pyramid scale factor, this value controls that how much
# the image is resized at every layer, smaller the value more
# image pyramid are generated )
WIN_STEP = 16
ROI_SIZE = eval(args['size'])  # min size of the roi
INPUT_SIZE = (224, 224)  # input size of the model


# loading network
print('Loading network')
model = ResNet50(weights='imagenet', include_top=True)

# loading the image, resizing it and grabbing the original size
origImage = cv2.imread(args["image"])
origImage = imutils.resize(origImage, WIDTH)
(H, W) = origImage.shape[:2]

# initialize the image pyramid
pyramid = image_pyramid(origImage.copy(), scale=PYR_SCALE, minsize=ROI_SIZE)

# list to store the roi generated from the image pyramid and sliding window
rois = []
# list to store the x, y corrs of the generated roi in the original image
locs = []

startTime = time.time()

for image in pyramid:
    # detemine the scale of the roi in the original image
    # scale is used to upscale the values acc to the original image
    scale = W / float(image.shape[1])

    for (x, y, roi) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # scale the corrs with respect to the original image
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        # if roi.shape[0] != 0 and roi.shape[1] != 0:
        roi = cv2.resize(roi, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        if args["visualize"] > 0:
            clone = origImage.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("visualization", clone)
            cv2.imshow("ROI", image)
            cv2.waitKey(0)

endTime = time.time()

print(
    'looping over pyramid/sliding window took {:.5f} seconds'.format(endTime - startTime))

# converting the rois to numpy array
rois = np.array(rois, dtype="float32")
print('Classifying rois')
startTime = time.time()
# predicting the rois
preds = model.predict(rois)
endTime = time.time()
print('classifying preds took {:.5f} seconds'.format(endTime - startTime))
# decode the predictions which returns a tuple (class_name, class_desc, score)
preds = imagenet_utils.decode_predictions(preds, top=1)
# dict to map class name as key with the roi
# label:[(roi, confidance)], this is how the dict will hold the information
labels = {}


# start filling the labels dictionary
for (i, p) in enumerate(preds):
    # grabbing the params of the prediction
    (imageNetId, label, confidance) = p[0]

    # if confidance of the prediction is greater than
    # the min confidance, then we store it in our dictionay
    if confidance >= args["min_conf"]:
        box = locs[i]
        # getting the list of the label
        L = labels.get(label, [])
        # adding a tupple containing the roi and the confidance
        L.append((box, confidance))
        # mapping the list with the label in the dict
        labels[label] = L


# looping over the labels for each detected objects in the image
for label in labels.keys():
    print('Showing results for "{}"'.format(label))
    # cloning the image to show the bounding boxes
    clone = origImage.copy()
    # looping over the particular label list
    for (box, confidance) in labels[label]:
        # grabbing the start and end's of the roi
        (startX, startY, endX, endY) = box
        # drawing an rectangle on the cloned image
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # showing the results before applying the non-maxima suppression
    cv2.imshow('before non-maxima suppression', clone)

    # cloning the image again to apply non maxima suppression
    clone = origImage.copy()

    # extracting the boxes and the confidance and storing it in a numpy array
    boxes = np.array([p[0] for p in labels[label]])
    confidances = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, confidances)

    # looping through the rois
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # calculating corrdinates for placing the label
        y = startY - 10 if (startY - 10) > 10 else startY + 10
        cv2.putText(clone, label, (startX, y),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("After non-maxima suppression", clone)
    cv2.waitKey(0)
