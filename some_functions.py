from PyQt5.QtGui import QImage, qRgb
import cv2
import numpy as np


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
