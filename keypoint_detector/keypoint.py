import tensorflow as tf
import cv2
import numpy as np
# import time
import onnxruntime as rt
from kp_tracker import KeypointTracker


def argmax_2d(tensor, return_score=False):

    # flatten the Tensor along the height and width axes
    flat_tensor = np.reshape(tensor, (len(tensor), -1, 16))

    # argmax of the flat tensor
    argmax = np.argmax(flat_tensor, axis=1).astype('int32')

    # convert indexes into 2D coordinates
    argmax_y = argmax // 28
    argmax_x = argmax % 28

    # stack and return 2D coordinates
    if return_score:
        score_list = np.max(flat_tensor, axis=1)
        scores = np.mean(score_list, axis=-1)
        return np.stack((argmax_x, argmax_y), axis=-1), scores
    return np.stack((argmax_x, argmax_y), axis=-1)


class KeypointDrawer:
    @classmethod
    def draw_landmark(cls, img, kp_list_list):
        for kp_list in kp_list_list:
            kp_list.draw_landmark(img)
        return img


class KeypointDetector:
    def __init__(self, model_path):
        sess_options = rt.SessionOptions()
        self.model = rt.InferenceSession(model_path,
                                         sess_options,
                                         providers=['CPUExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name

        self.order_indices = [0, 1, 4, 7, 10, 13,
                              2, 5, 8, 11, 14,
                              3, 6, 9, 12, 15]
        self.join_pairs = [(0, 1), (1, 2), (2, 3),  # thumb
                           (0, 4), (4, 5), (5, 6),  # index
                           (0, 7), (7, 8), (8, 9),  # middle
                           (0, 10), (10, 11), (11, 12),  # ring
                           (0, 13), (13, 14), (14, 15),  # pinky
                           (1, 4), (4, 7), (7, 10), (10, 13)]
        self.grid_size = 28
        self.cell_size = 224 // self.grid_size

        # tracker
        self.trackers = {}
        # self.tracker = KeypointTracker()

    def predict(self, org_img, bboxes, ids, return_bbox=False):
        # convert to int
        bboxes = np.array(bboxes, dtype='int32')
        raw_img_list = self.get_crop_img_list(org_img, bboxes)

        img_list = []
        process_img_list = []
        for raw_img in raw_img_list:
            img = cv2.cvtColor(raw_img.copy(), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_list.append(img)
            p_img = tf.keras.applications.mobilenet.preprocess_input(img)
            process_img_list.append(p_img)
        img_list = np.array(img_list)
        process_img_list = np.array(process_img_list)

        # st = time.time()
        # _, pred = self.model.predict(process_img_list)
        _, pred = self.model.run(None, {self.input_name: process_img_list})
        pred = pred[:, :, :, 1:]

        # g_coords_pred = argmax_2d(pred)
        g_coords_pred, kp_scores = argmax_2d(pred, return_score=True)

        kp_list_list = []
        if return_bbox:
            kp_bboxes = []

        for g_keypoints, raw_img, img, box, i in zip(g_coords_pred, raw_img_list,
                                                     img_list, bboxes, ids):
            org_indices = self.chaos_to_normal(g_keypoints, self.order_indices)
            g_keypoints = g_keypoints[org_indices]
            p_keypoints = self.convert_to_pixel_coord(g_keypoints,
                                                      self.cell_size)

            # add new tracker
            if i not in self.trackers or self.trackers[i] is None:
                self.trackers[i] = KeypointTracker()
            tracker = self.trackers[i]
            p_keypoints = tracker.update(p_keypoints)
            tracker.predict()
            # self.tracker.predict()

            adjusted_kp = self.rescale_kp(raw_img, img, box, p_keypoints)
            if return_bbox:
                x, y, w, h = cv2.boundingRect(adjusted_kp)
                xmax = x + w
                ymax = y + h
                kp_bboxes.append(np.array([x, y, xmax, ymax]).astype('int32'))
            kp_list = self.convert_to_keypoint_list(box, adjusted_kp)
            kp_list_list.append(kp_list)

        # remove unsed trackers
        for i in self.trackers:
            if i not in ids:
                self.trackers[i] = None

        if return_bbox:
            return kp_list_list, kp_bboxes, kp_scores

        return kp_list_list

    def get_crop_img_list(self, img, bboxes):
        crop_images = []
        for box in bboxes:
            crop_img = self.crop_img(img, box)
            crop_images.append(crop_img)
        return crop_images

    def crop_img(self, img, box):
        xmin, ymin, xmax, ymax = box
        img_crop = img[ymin:ymax, xmin:xmax]
        return img_crop

    def chaos_to_normal(self, keypoints, reorder_indices):
        org = np.zeros(len(reorder_indices))
        for i, index in enumerate(reorder_indices):
            org[index] = i
        org = org.astype('int32')
        return org

    def convert_to_pixel_coord(self, g_keypoints, cell_size):
        half_cell_size = cell_size // 2
        p_keypoints = g_keypoints.astype('int32') * cell_size + half_cell_size
        return np.clip(p_keypoints.astype("int32"), 0, 223)

    def rescale_kp(self, org_img, img, box, keypoints):
        '''
        org_img: original cropped img
        img: resized img
        '''
        # resize to original cropped img
        factor_w = org_img.shape[1] / img.shape[1]
        factor_h = org_img.shape[0] / img.shape[0]

        keypoints[:, 0] = (keypoints[:, 0] * factor_w).astype('int32')
        keypoints[:, 1] = (keypoints[:, 1] * factor_h).astype('int32')

        # resize to original full img
        offset_w = box[0]
        offset_h = box[1]

        keypoints[:, 0] = keypoints[:, 0] + offset_w
        keypoints[:, 1] = keypoints[:, 1] + offset_h

        return keypoints

    def convert_to_keypoint_list(self, box, keypoints):
        kp_list = []

        for kp in keypoints:
            kp_list.append(KeyPoint(kp[0], kp[1]))

        return KeyPointList(kp_list)


class KeyPoint():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_list(self):
        return [self.x, self.y]

    def to_tuple(self):
        return (self.x, self.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"


class KeyPointList():
    def __init__(self, keypoints):
        self.keypoints = keypoints
        self.connect_points = [(0, 1), (1, 2), (2, 3),  # thumb
                               (0, 4), (4, 5), (5, 6),  # index
                               (0, 7), (7, 8), (8, 9),  # middle
                               (0, 10), (10, 11), (11, 12),  # ring
                               (0, 13), (13, 14), (14, 15),  # pinky
                               (1, 4), (4, 7), (7, 10), (10, 13)]

        # eliminated vectors
        self.width_elim = [(10, 13), (7, 10), (0, 10), (1, 2),
                           (2, 3), (0, 13), (4, 7)]
        self.angle_elim = [(10, 13), (7, 10), (0, 10), (0, 13)]
        self.point_elim  = [4, 7, 10]
        self.point_elim_width = 13

    # in radian
    def get_angle(self, vector_1, vector_2):
        dot_product = np.dot(vector_1, vector_2)
        length_product = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        if length_product <= 0:
            return 0
        try:
            angle = np.arccos(dot_product / (length_product + 0.0001))
            return angle
        except RuntimeWarning:
            return 0

    def compute_cross_factor(self, vec1, vec2):
        '''
        compute the cross product of 2 vector and
        return the sign of the product 1 or -1
        '''
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return 1 if cross_product >= 0 else -1

    def get_undirected_feature(self):
        feature = []

        # base vector 0 -> 7
        pt1 = self.keypoints[0]
        pt2 = self.keypoints[7]
        base_vec = (pt2.x - pt1.x, pt2.y - pt1.y)
        base_width = math.dist(pt1.to_list(), pt2.to_list())

        # vector to decide angle direction 0 -> 10
        pt1 = self.keypoints[0]
        pt2 = self.keypoints[10]
        direction_vec = (pt2.x - pt1.x, pt2.y - pt1.y)

        # compute cross product to determine direction
        # standardize the direction is either 1 or -1
        direction_factor = self.compute_cross_factor(base_vec, direction_vec)

        # print("Start")
        for pair in self.connect_points:
            # eliminate unimportant vectors
            if pair[0] == 0 and pair[1] == 7:
                continue
            if pair in self.width_elim and pair in self.angle_elim:
                continue

            pt1 = self.keypoints[pair[0]]
            pt2 = self.keypoints[pair[1]]

            if pair not in self.width_elim:
                width = math.dist(pt1.to_list(), pt2.to_list())
                width = width / base_width
                feature.append(width)

            if pair not in self.angle_elim:
                # angle must not be none
                vec = (pt2.x - pt1.x, pt2.y - pt1.y)
                angle = self.get_angle(base_vec, vec)

                # the direction of the angle
                direction = self.compute_cross_factor(base_vec, vec)\
                    * direction_factor
                angle = angle * direction
                feature.append(angle)

        return np.array(feature)

    def get_directed_feature(self):
        feature = []

        # base vector 0 -> 7
        pt1 = self.keypoints[0]
        pt2 = self.keypoints[7]
        base_width = math.dist(pt1.to_list(), pt2.to_list())

        # vector to decide angle direction 0 -> 10
        base_y = np.array([0, -1])
        base_x = np.array([1, 0])

        # print("Start")
        for pair in self.connect_points:
            # eliminate unimportant vectors
            if pair[0] == 0 and pair[1] == 7:
                continue
            if pair in self.width_elim and pair in self.angle_elim:
                continue

            pt1 = self.keypoints[pair[0]]
            pt2 = self.keypoints[pair[1]]

            if pair not in self.width_elim:
                width = math.dist(pt1.to_list(), pt2.to_list())
                width = width / base_width
                feature.append(width)

            if pair not in self.angle_elim:
                # angle must not be none
                vec = (pt2.x - pt1.x, pt2.y - pt1.y)
                angle_x = self.get_angle(base_x, vec)
                comp = np.pi / 2
                # if angle_x > comp:
                #     angle_x = angle_x - np.pi
                angle_y = self.get_angle(base_y, vec)
                if angle_x > comp:
                    angle = angle_y
                else:
                    angle = -angle_y
                feature.append(angle)

        return np.array(feature)

    def get_transition_feature(self, prev_keypointList):
        feature = []

        # base vector 0 -> 7
        pt1 = self.keypoints[0]
        pt2 = self.keypoints[7]
        base_width = math.dist(pt1.to_list(), pt2.to_list())

        # vector to decide angle direction 0 -> 10
        base_y = np.array([0, -1])
        base_x = np.array([1, 0])

        prev_keypoints = prev_keypointList.keypoints

        # only consider the wrist point
        current_pt = self.keypoints[0]
        prev_pt = prev_keypoints[0]

        vec = np.array([current_pt.x - prev_pt.x, current_pt.y - prev_pt.y])
        width = math.dist(current_pt.to_list(), prev_pt.to_list())
        width = width / base_width
        feature.append(width)

        vec = (current_pt.x - prev_pt.x, current_pt.y - prev_pt.y)

        angle_x = self.get_angle(base_x, vec)
        comp = np.pi / 2
        angle_y = self.get_angle(base_y, vec)
        if angle_x > comp:
            angle = angle_y
        else:
            angle = -angle_y
        feature.append(angle)

        return feature

    def draw_point(self, img, pt, color=(0, 0, 255)):
        cv2.circle(img, (pt.x, pt.y), 5, color, -1)

    def connect_point(self, img, pt1, pt2, color=(0, 255, 255)):
        cv2.line(img, (pt1.x, pt1.y), (pt2.x, pt2.y), color, 2)

    def draw_landmark(self, img):
        for index, pair in enumerate(self.connect_points):
            self.connect_point(img, self.keypoints[pair[0]],
                               self.keypoints[pair[1]])

        for index, pt in enumerate(self.keypoints):
            self.draw_point(img, pt)
