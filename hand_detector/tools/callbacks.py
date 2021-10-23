"""custom model callbacks."""
import os
import sys
import random
import tempfile
import numpy as np
from tensorflow.keras.callbacks import Callback
from tools.model import get_yolo4_model
from tools.evaluation import eval_AP

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..'))


class DatasetShuffleCallBack(Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        np.random.shuffle(self.dataset)


class EvalCallBack(Callback):
    def __init__(self, annotation_lines, anchors, class_names,
                 model_image_size, model_pruning,
                 log_dir, weights_path,
                 eval_epoch_interval=10,
                 save_eval_checkpoint=False):
        self.annotation_lines = annotation_lines
        self.weights_path = weights_path
        self.anchors = anchors
        self.class_names = class_names
        self.model_image_size = model_image_size
        self.model_pruning = model_pruning
        self.log_dir = log_dir
        self.eval_epoch_interval = eval_epoch_interval
        self.save_eval_checkpoint = save_eval_checkpoint
        self.best_mAP = 0.0
        self.eval_model = self.get_eval_model()

    def get_eval_model(self):
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        num_feature_layers = num_anchors//3
        eval_model, _ = get_yolo4_model(num_feature_layers,
                                        num_anchors,
                                        num_classes,
                                        input_shape=self.model_image_size + (3,),
                                        model_pruning=self.model_pruning)

        self.v5_decode = False

        return eval_model

    def update_eval_model(self, train_model):
        # create a temp weights file to save training result
        tmp_weights_path = os.path.join(tempfile.gettempdir(),
                                        str(random.randint(10, 1000000)) + '.h5')
        train_model.save_weights(tmp_weights_path)

        # load the temp weights to eval model
        self.eval_model.load_weights(tmp_weights_path)
        os.remove(tmp_weights_path)

        eval_model = self.eval_model

        return eval_model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.eval_epoch_interval == 0:
            # Do eval every eval_epoch_interval epochs
            eval_model = self.update_eval_model(self.model)
            mAP = eval_AP(eval_model, 'H5', self.annotation_lines,
                          self.anchors, self.class_names,
                          self.model_image_size,
                          eval_type='VOC', iou_threshold=0.5,
                          conf_threshold=0.001,
                          v5_decode=self.v5_decode, save_result=False)

            if self.save_eval_checkpoint and mAP > self.best_mAP:
                # Save best mAP value and model checkpoint
                self.best_mAP = mAP
                self.model.save(os.path.join(self.log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-mAP{mAP:.3f}.h5'.format(epoch=(epoch+1), loss=logs.get('loss'), val_loss=logs.get('val_loss'), mAP=mAP)))

