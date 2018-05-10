from PyQt5.QtGui import QImage, qRgb
import cv2
import numpy as np
import math


def toQImage(im):
    """
    Utility method to convert a numpy array to a QImage object.
    Args:
        im          numpy array to be converted. It can be a 2D (BW) image or a color image (3 channels + alpha)
    Returns:
        QImage      The image created converting the numpy array
    """
    gray_color_table = [qRgb(i, i, i) for i in range(256)]
    if im is None:
        return QImage()
    if len(im.shape) == 2:  # 1 channel image
        qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
        qim.setColorTable(gray_color_table)
        return qim
    elif len(im.shape) == 3:
        if im.shape[2] == 3:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)       # im.strides[0]
            return qim
        elif im.shape[2] == 4:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
            return qim

def resizeImage(img, value):
    img1 = img
    if isinstance(img1, str):
        img = cv2.imread(img)
    img = cv2.resize(img, dsize=(value, value), interpolation=cv2.INTER_CUBIC)
    if isinstance(img1, str):
        img = np.require(img, np.uint8, 'C')
        img = toQImage(img)
    return img

def center_image(img):
    tmp_img = img[:, :, 0]
    index = np.nonzero(tmp_img)
    size = index[0].size - 1
    first_row = index[0][0]
    first_column = min(index[1])
    last_row = index[0][size]
    last_column = max(index[1])
    margin_diff_row = tmp_img.shape[0] - last_row
    margin_diff_col = tmp_img.shape[0] - last_column
    diff_row = first_row - margin_diff_row
    diff_col = first_column - margin_diff_col

    if diff_row != 0:
        if first_row > margin_diff_row:
            img = np.pad(img, ((0, math.ceil(abs(diff_row)/2)), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
            vector_to_delete = np.arange(math.ceil(abs(diff_row)/2.0))
            img = np.delete(img, vector_to_delete, 0)

        else:
            img = np.pad(img, ((math.ceil(abs(diff_row)/2), 0), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
            vector_to_delete = np.arange(tmp_img.shape[0]-math.ceil(abs(diff_row)/2.0), tmp_img.shape[0])
            img = np.delete(img, vector_to_delete, 0)
    if diff_col != 0:
        if first_column > margin_diff_col:
            img = np.pad(img, ((0, 0), (0, math.ceil(abs(diff_col)/2)), (0, 0)), 'constant', constant_values=(0, 0))
            vector_to_delete = np.arange(math.ceil(abs(diff_col)/2.0))
            img = np.delete(img, vector_to_delete, 1)
        else:
            img = np.pad(img, ((0, 0), ((math.ceil(abs(diff_col)/2)), 0), (0, 0)), 'constant', constant_values=(0, 0))
            vector_to_delete = np.arange(tmp_img.shape[1]-(math.ceil(abs(diff_col)/2.0)), tmp_img.shape[1])
            img = np.delete(img, vector_to_delete, 1)
    return img
