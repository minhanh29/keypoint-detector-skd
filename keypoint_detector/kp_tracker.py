import numpy as np
from kalman_filter import KalmanFilter


class KeypointTracker:
    def __init__(self):
        self._kf_list = []
        finger_tips = set([3, 6, 9, 12, 15])
        middle_nuckles = set([2, 5, 8, 11, 14])
        for i in range(16):
            if i in finger_tips:
                dt = 0.26
            elif i in middle_nuckles:
                dt = 0.23
            elif i == 0:
                dt = 0.15
            else:
                dt = 0.18
            self._kf_list.append(KalmanFilter(dt, 1, 1, 1, 0.1, 0.1))

    def predict(self):
        predicted_kp = np.empty((16, 2), dtype='int32')
        for i, kf in enumerate(self._kf_list):
            x, y = kf.predict()
            predicted_kp[i,:] = np.squeeze([x, y]).astype('int32')
        return predicted_kp

    def update(self, kp_list):
        '''
        kp_list: ndarray of shape (16, 2)
        '''
        updated_kp = np.empty((16, 2), dtype='int32')
        i = 0
        for pt, kf in zip(kp_list, self._kf_list):
            x, y = kf.update(np.expand_dims(pt, axis=-1))
            updated_kp[i,:] = np.squeeze([x, y]).astype('int32')
            i += 1
        return updated_kp
