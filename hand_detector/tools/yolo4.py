#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a YOLOv3/YOLOv2 style detection model on test images.
"""

import os
import onnxruntime as rt
from tensorflow.keras import backend as K
from tensorflow_model_optimization.sparsity import keras as sparsity

from tools.model import get_yolo4_model
from tools.postprocess_np import yolo3_postprocess_np
from tools.utils import get_classes, get_anchors, get_colors, my_preprocess_image


default_config = {
        "weights_path": os.path.join('logs', 'trained_model.h5'),
        "pruning_model": False,
        "anchors_path": os.path.join('config', 'yolov4-tiny-anchors.txt'),
        "classes_path": os.path.join('config', 'hand_class.txt'),
        "score": 0.1,
        "iou": 0.4,
        "model_image_size": (416, 416),
        "elim_grid_sense": False,
        "gpu_num": 0,
    }


class YOLO_np(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # def __init__(self, **kwargs):
    def __init__(self, my_args=None):
        super(YOLO_np, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        # self.__dict__.update(kwargs) # and update with user overrides
        self.__dict__.update(my_args) # set up default values
        # else:
        #     self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)

        sess_options = rt.SessionOptions()
        self.yolo_model = rt.InferenceSession(self.weights_path,
                                              sess_options,
                                              providers=['CPUExecutionProvider'])
        self.input_name = self.yolo_model.get_inputs()[0].name

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        # assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        num_feature_layers = num_anchors//3

        try:
            # YOLOv2 entrance
            yolo_model, _ = get_yolo4_model(num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)

            yolo_model.load_weights(weights_path) # make sure model, anchors and classes match
            if self.pruning_model:
                yolo_model = sparsity.strip_pruning(yolo_model)
            yolo_model.summary()
        except Exception as e:
            print(repr(e))
            assert yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(weights_path))

        return yolo_model


    def detect_image(self, image, draw_box=True):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = my_preprocess_image(image, self.model_image_size)
        #origin image shape, in (height, width) format
        image_shape = (image.shape[0], image.shape[1])

        # start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # end = time.time()
        # print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        out_classnames = [self.class_names[c] for c in out_classes]
        return out_boxes, out_scores, out_classnames

    def getONNXOutput(self, input_data):
        # st = time.time()
        pred = self.yolo_model.run(None, {self.input_name: input_data})
        # print("Elapse", time.time() - st)
        return pred

    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        output = self.getONNXOutput(image_data)
        # print('Output', output)
        out_boxes, out_classes, out_scores = yolo3_postprocess_np(output, image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)

        return out_boxes, out_classes, out_scores

    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)


