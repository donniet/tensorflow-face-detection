#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import argparse
import grpc

from utils import label_map_util
from utils import visualization_utils_color as vis_util

from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from google.protobuf import text_format

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
    def __init__(self, args):
        self.channel = grpc.insecure_channel(args.addr)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.model_name = args.model_name
        self.signature_name = args.signature_name
        self.version = args.version

    def run(self, image, width, height):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        request.model_spec.version.value = int(self.version)

        request.inputs['image'].CopyFrom(make_tensor_proto(image, shape=[1, width, height, 3]))
        
        result = self.stub.Predict(request)

        boxes, scores, classes, num_detections = result.outputs['boxes'].float_val, result.outputs['scores'].float_val, result.outputs['classes'].float_val, result.outputs['num_detections'].float_val

        num_detections = int(num_detections[0])
        boxes = np.reshape(boxes, [1, num_detections, 4])
        scores = np.reshape(scores, [1, num_detections])
        classes = np.reshape(classes, [1, num_detections])

        return boxes, scores, classes, num_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tensorflow mtcnn')
    parser.add_argument('--addr', default="localhost:8500", help="address of tensorflow serving")
    parser.add_argument('--model_name', default="tensorflow-face-detection", help="model name")
    parser.add_argument('--signature_name', default="serving_default", help="signature name of the tensorflow serving model")
    parser.add_argument('--version', default=1, help="version number")
    
    args = parser.parse_args()
    
    camID = 0
    width = 1280
    height = 720
    
    tDetector = TensoflowFaceDector(args)

    cap = cv2.VideoCapture(camID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    windowNotSet = True
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        print (h, w)
        image = cv2.flip(image, 1)

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        (boxes, scores, classes, num_detections) = tDetector.run(image_np, h, w)

        # print(boxes)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        if windowNotSet is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
