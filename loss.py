from haversine import haversine
import numpy as np
import logging
import tensorflow as tf

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def tf_deg2rad(deg):
    return deg * (np.pi/180)


def tf_rad2deg(rad):
    return rad / (np.pi/180)


# haversine loss function for regression
def errors_mean(latlon_true, latlon_pred):
    distances = []
    for i in range(0, len(latlon_true)):
        lat_true, lon_true = latlon_true[i]
        lat_pred, lon_pred = latlon_pred[i]
        distance = haversine((lat_true, lon_true), (lat_pred, lon_pred))
        distances.append(distance)
    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info("Mean: " + str(int(np.mean(distances))) + " Median: " +
                 str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
    # return np.mean(distances), np.median(distances), acc_at_161
    return np.mean(distances)


# A helper method that computes distance between two points on the surface of earth according to their coordinates.
# Inputs are tensors.
def dist(y_pred, y):
    y_pred_ra = tf_deg2rad(tf.convert_to_tensor(y_pred))
    y_ra = tf_deg2rad(tf.convert_to_tensor(y))
    lat1 = y_pred_ra[:, 0]
    lat2 = y_ra[:, 0]
    dlon = (y_pred_ra - y_ra)[:, 1]
    EARTH_R = 6372.8
    y = tf.sqrt((tf.cos(lat2) * tf.sin(dlon)) ** 2 +
                (tf.cos(lat1) * tf.sin(lat2) - tf.sin(lat1) * tf.cos(lat2) * tf.cos(dlon)) ** 2)
    x = tf.sin(lat1) * tf.sin(lat2) + tf.cos(lat1) * tf.cos(lat2) * tf.cos(dlon)
    c = tf.atan2(y, x)
    return EARTH_R * c


def errors_mean(y_true, y_pred):
    if y_true.shape.ndims != y_pred.shape.ndims:
        raise TypeError('y should have the same shape as self.y_pred', ('y', y_true.type, 'y_pred', y_pred.type))
    print("y_true.dtype", y_true.dtype)
    if str(y_true.dtype).__contains__('float'):
        dists = dist(y_pred, y_true)
        return tf.reduce_mean(dists)
    else:
        raise NotImplementedError()
