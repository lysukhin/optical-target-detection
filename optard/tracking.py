import numpy as np
from filterpy.kalman import KalmanFilter


def get_F():
    """
    State transition matrix F: 
    State_{i+1} = F @ State_{i} + w_{i+1}, w ~ N(0, Q).
    """
    return np.asarray([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def get_Q():
    """
    State transition error covariance matrix Q: 
    State_{i+1} = F @ State_{i} + w_{i+1}, w ~ N(0, Q). 
    """
    return np.asarray([
        [1., 0, 0, 0],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ], dtype=np.float32)


def get_H():
    """
    Measurement matrix H:
    z_{i} = H @ Measurement_{i} + v, v ~ N(0, R).
    """
    return np.asarray([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)


def get_R():
    """
    Measurement error covarianve matrix R:
    z_{i} = H @ Measurement_{i} + v, v ~ N(0, R).
    """
    return np.asarray([
        [1., 0],
        [0, 1.]
    ], dtype=np.float32)


def get_P():
    """
    Initial state covariance matrix P.
    """
    return np.asarray([
        [10, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 1000, 0],
        [0, 0, 0, 1000]
    ], dtype=np.float32)


def init_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # Measurement is [x, y] (len = 2)
    # State is [x, y, x', y'] (len = 4)
 
    kf.F = get_F()
    kf.Q = get_Q()
    kf.H = get_H()
    kf.R = get_R()
    kf.P = get_P()
    return kf

class PointKalmanFilter:
    def __init__(self):
        self.kf = init_kalman_filter()

    def update(self, z=None):
        self.kf.predict()
        if z is not None:
            z = np.asarray(z)
            self.kf.update(z)
        new_z = self.kf.measurement_of_state(self.kf.x)
        return new_z.ravel().tolist()


if __name__ == "__main__":
    pkf = PointKalmanFilter()
    print(pkf)


