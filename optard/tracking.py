import numpy as np
from filterpy.kalman import KalmanFilter


# FIXME: values in get_* require tuning.


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


def get_Q(std_pos=1., std_vel=10.):
    """
    State transition error covariance matrix Q: 
    State_{i+1} = F @ State_{i} + w_{i+1}, w ~ N(0, Q). 
    """
    return np.asarray([
        [std_pos, 0, 0, 0],
        [0, std_pos, 0, 0],
        [0, 0, std_vel, 0],
        [0, 0, 0, std_vel]
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


def get_R(std_pos=1.):
    """
    Measurement error covariance matrix R:
    z_{i} = H @ Measurement_{i} + v, v ~ N(0, R).
    """
    return np.asarray([
        [std_pos, 0],
        [0, std_pos]
    ], dtype=np.float32)


def get_P(std_pos=10., std_vel=1000.):
    """
    Initial state covariance matrix P.
    """
    return np.asarray([
        [std_pos, 0, 0, 0],
        [0, std_pos, 0, 0],
        [0, 0, std_vel, 0],
        [0, 0, 0, std_vel]
    ], dtype=np.float32)


class PointKalmanFilter:
    def __init__(self,
                 trans_err_std_pos=1., trans_err_std_vel=10.,
                 measure_err_std_pos=.1,
                 init_state_std_pos=10., init_state_std_vel=1000.):
        # Measurement is [x, y] (len = 2)
        # State is [x, y, x', y'] (len = 4)
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = get_F()
        kf.Q = get_Q(trans_err_std_pos, trans_err_std_vel)
        kf.H = get_H()
        kf.R = get_R(measure_err_std_pos)
        kf.P = get_P(init_state_std_pos, init_state_std_vel)
        self.kf = kf

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


