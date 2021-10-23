from tools.yolo4 import YOLO_np
import numpy as np
import os
import cv2
from sort import Sort


class HandDetector:
    def __init__(self, confidence_threshold=0.5, iou_score=0.5):
        my_config = {
            "weights_path": os.path.join('hand_detector/model', 'yolo4_model.onnx'),
            "pruning_model": False,
            "anchors_path": os.path.join('hand_detector/config', 'yolov4-tiny-anchors.txt'),
            "classes_path": os.path.join('hand_detector/config', 'hand_class.txt'),
            "score": confidence_threshold,
            "iou": iou_score,
            "model_image_size": (416, 416),
            "elim_grid_sense": False,
            "gpu_num": 0,
        }

        self.yolo = YOLO_np(my_args=my_config)
        self.tracker = Sort(
            max_age=5,
            min_hits=3,
            iou_threshold=0.3)

        # detect bbox after this many frame
        self.detect_delay = 30
        self.delay_count = 0
        self.enable_detect = True

    def detect_img(self, img):
        boxes, scores, classes = self.yolo.detect_image(img[:, :, ::-1],
                                                        draw_box=False)
        return boxes, scores

    def detect_video(self, img, old_bboxes=None, multi_hand=False):
        if multi_hand:
            return self.detect_video_multi(img, old_bboxes)
        return self.detect_video_single(img, old_bboxes)

    def detect_video_single(self, img, old_bboxes=None):
        '''
        img: BGR image
        '''
        if old_bboxes is None:
            boxes, scores, classes = self.yolo.detect_image(img[:, :, ::-1],
                                                            draw_box=False)
            boxes, scores, ids = self.sort_tracking(img, boxes,
                                                    scores)
        else:
            fake_scores = np.ones(len(old_bboxes))
            boxes, scores, ids = self.sort_tracking(img, old_bboxes,
                                                    fake_scores)

        return boxes, scores, ids

    def detect_video_multi(self, img, old_bboxes=None):
        '''
        img: BGR image
        '''
        if self.delay_count > self.detect_delay:
            self.delay_count = 0
            self.enable_detect = True

        if old_bboxes is None or self.enable_detect:
            boxes, scores, classes = self.yolo.detect_image(img[:, :, ::-1],
                                                            draw_box=False)
            boxes, scores, ids = self.sort_tracking(img, boxes,
                                                    scores)
            self.delay_count += 1
            if self.delay_count > 3:
                self.delay_count = 0
                self.enable_detect = False
            else:
                self.enable_detect = True
        else:
            fake_scores = np.ones(len(old_bboxes))
            boxes, scores, ids = self.sort_tracking(img, old_bboxes,
                                                    fake_scores)
            self.delay_count += 1

        return boxes, scores, ids

    def sort_tracking(self, img, boxes, scores):
        detections = []
        for box, score in zip(boxes, scores):
            det = box + score
            detections.append(det)

        if len(boxes) > 0:
            track_bbs_ids = self.tracker.update(np.array(detections))
        else:
            track_bbs_ids = self.tracker.update(np.empty((0, 5)))
            if len(track_bbs_ids) > 0:
                print(f"Got 0 boxes and {len(track_bbs_ids)} tracks")

        r_boxes = []
        r_ids = []
        for track in track_bbs_ids:
            r_boxes.append([track[0], track[1], track[2], track[3]])
            r_ids.append(track[4])

        return r_boxes, scores, r_ids
