import os
import sys
import cv2
import numpy as np


class SKD:
    def __init__(self, confidence_threshold=0.5,
                 iou_score=0.5,
                 keypoint_confidence=0.65):
        sys.path.append(os.path.abspath("./hand_detector"))
        sys.path.append(os.path.abspath("./keypoint_detector"))
        from detector import HandDetector
        from keypoint import KeypointDetector, KeypointDrawer

        self.hand_detector = HandDetector(confidence_threshold, iou_score)
        self.kp_detector = KeypointDetector("keypoint_detector/model/keypoint_model.onnx")
        self.keypoint_drawer = KeypointDrawer.draw_landmark

        self.box_color = (0, 255, 255)

        # box tracking
        self.old_bboxes = None
        self.keypoint_confidence = keypoint_confidence

    def detect_img(self, img):
        boxes, scores = self.hand_detector.detect_img(img)
        if len(boxes) > 0:
            return self.draw_box(img, boxes, scores)
        return img

    def detect_video(self, img, multi_hand=True):
        boxes, scores, ids = self.hand_detector.detect_video(img, self.old_bboxes, multi_hand)
        if len(boxes) > 0:
            # keypoint detection
            if not multi_hand:
                index = np.argmax(scores)
                boxes = [boxes[index]]

            adjust_bbox = []
            for box in boxes:
                new_box = self.kp_adjust_box(img, box)
                adjust_bbox.append(new_box)

            kp_list_list, kp_boxes, kp_scores = self.kp_detector.predict(img, adjust_bbox,
                                                                         ids, return_bbox=True)

            min_score = np.min(kp_scores)
            if min_score > self.keypoint_confidence:
                self.old_bboxes = kp_boxes
            else:
                # print("Fail", min_score)
                self.old_bboxes = None

            img = self.keypoint_drawer(img, kp_list_list)
            return img
            # return self.draw_box(img, boxes, scores, ids)
        else:
            self.old_bboxes = None

        return img

    def draw_box(self, img, boxes, scores=None, ids=None):
        index = 0
        total = len(boxes)
        for index in range(total):
            box = boxes[index]
            score = 1
            if scores is not None:
                score = scores[index]
            xmin, ymin, xmax, ymax = list(map(int, box))

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          self.box_color, 2)

            i = ''
            if ids is not None:
                i = ids[index]
            score = int(score * 100)
            text = f"{score}% id: {i}"
            cv2.putText(img, text, (xmin + 5, ymin + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, self.box_color, 2)

        return img

    def kp_adjust_box(self, img, box):
        xmin, ymin, xmax, ymax = list(map(int, box))
        w = abs(xmax - xmin)
        h = abs(ymax - ymin)

        factor = 0.2
        xmin -= factor * w
        ymin -= factor * h
        xmax += factor * w
        ymax += factor * h

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img.shape[1], xmax)
        ymax = min(img.shape[0], ymax)
        return np.array([xmin, ymin, xmax, ymax], dtype='int32')
