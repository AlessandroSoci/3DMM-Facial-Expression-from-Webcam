import h5py
import time
import sys
sys.path.append('expression_code/')
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import scipy.misc
from scipy import misc
from scipy.misc.pilutil import Image
import scipy.ndimage as ndimage
from PyQt5.QtCore import pyqtSignal, QThread
from landmark_detect import Detector
from _3DMM import _3DMM
from RP import RP
from util_for_graphic import graphic_tools
from Matrix_operations import Matrix_op
from Matrix_operations import Vector_op
from create_expression import predict_expr

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class Model(QThread):

    updated = pyqtSignal()  # in order to work it has to be defined out of the constructor
    progress_bar = pyqtSignal()

    def __init__(self, image, expression, main_window):
        super().__init__()
        self.image_path = image
        self.image = None
        self.expression = expression
        self.dictionary = None
        self.take_photo = True
        self.onclick = None
        self.mw = main_window

    def get_image(self):
        return self.image

    def get_dictionary(self):
        return self.dictionary

    def set_image(self, image):
        self.image_path = image

    def deactivate(self):
        if self.isRunning():
            self.terminate()

    def run(self):
        self.expression = self.mw.get_expression()
        self.onclick = self.mw.get_bool_onclick()
        if self.onclick:
            self.image = self.apply_expression(self.image_path, expression=self.expression)
        else:
            self.image = self.apply_expression_modelPreloaded(expression=self.expression)
        self.updated.emit()

    def apply_expression(self, original_image_path, path_log='log/', expression='angry'):
        # start from the 2D IMAGE. Compute the landmarks on the face within the image and compute
        # the 3DMM, then apply the selected expresison and return the 2D image midifed.

        remove_boundary = False
        # lambda_opt = 0.01
        # vars for 3DMM pose and fitting
        _lambda = 0.007
        rounds = 1
        r = 3
        C_dist = 700

        # _path_log = str(path_log) + str('log_file.h5')

        # get the image
        image = Image.open(original_image_path)
        #image = Image.open('expression_code/imgs/img_4.jpg')
        # image.convert('RGB')
        # image = image.resize((250,250))
        image = np.asarray(image)
        original_image = image
        detect_landmarks = Detector()
        # landmark of the input image
        landImage = detect_landmarks.eval_landmarks(original_image_path,
                                                    'expression_code/shape_predictor_68_face_landmarks.dat')

        # remove first 19 elements
        if remove_boundary:
            index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 63, 64]
            landImage = np.delete(landImage, index, axis=0)
        else:
            index = [65, 61]
            landImage = np.delete(landImage, index, axis=0)

        # load components and weights file
        components_DL_300 = h5py.File('expression_code/data/components_DL_300.mat', 'r')

        # SHAPE
        Components = np.transpose(np.array(components_DL_300["Components"]))
        Weights = np.transpose(np.array(components_DL_300["Weights"]))
        Components_res = np.transpose(np.array(components_DL_300["Components_res"]))
        m_X_obj = Matrix_op(Components, None)
        m_X_obj.reshape(Components)  # define Components_res
        v_weights_obj = Vector_op(Weights)
        #v_weights_obj.scale(1, 0.1)

        # load AVGmodel SHAPE (it contains avgModel, id landmarks 3D, landmarks 3D)
        avgModel_file = h5py.File('expression_code/data/avgModel_bh_1779_NE.mat', 'r')
        avg_model_data = np.transpose(np.array(avgModel_file["avgModel"]))
        id_landmarks_3D = np.transpose(np.array(avgModel_file["idxLandmarks3D"]))
        landmarks_3D = np.transpose(np.array(avgModel_file["landmarks3D"]))

        # remove first 17 elements
        if remove_boundary:
            index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            id_landmarks_3D = np.delete(id_landmarks_3D, index, axis=1)
            id_landmarks_3D = id_landmarks_3D - 1
            landmarks_3D = np.delete(landmarks_3D, index, axis=0)
            id_landmarks_3D_texture = np.delete(id_landmarks_3D_texture, index, axis=1)
            id_landmarks_3D_texture = id_landmarks_3D_texture - 1
            landmarks_3D_texture = np.delete(landmarks_3D_texture, index, axis=0)

        # define RP values
        RP_obj = RP()
        # Fitting and pose estimation are within the 3DMM obj
        _3DMM_obj = _3DMM()
        _graph_tools_obj = graphic_tools(_3DMM_obj)

        pos_est = _3DMM_obj.opt_3DMM_fast(v_weights_obj.V, m_X_obj.X_after_training, m_X_obj.X_res,
                                          landmarks_3D, id_landmarks_3D, landImage, avg_model_data, _lambda, rounds, r,
                                          C_dist)
        shape_neutral_model = pos_est["defShape"]

        # create texture for neutral model
        projShape_neutral = np.transpose(_3DMM_obj.getProjectedVertex(shape_neutral_model, pos_est["S"], pos_est["R"], pos_est["T"]))
        texture_neutral_model = (_graph_tools_obj.getRGBtexture(projShape_neutral, image)) * 255

        self.dictionary = {
            'original_image': original_image,
            'shape_neutral': shape_neutral_model,
            'projShape_neutral': projShape_neutral,
            'texture_neutral': texture_neutral_model,
            'components': Components,
            'pos_est_S': pos_est["S"],
            'pos_est_R': pos_est["R"],
            'pos_est_T': pos_est["T"],
            'visIdx': pos_est["visIdx"]
        }
        # neutral is the default expression
        if expression == 'neutral':
            if 'image_neutral' in self.dictionary:
                return self.dictionary['image_neutral']
            else:
                image = _graph_tools_obj.render3DMM(projShape_neutral[:, 0], projShape_neutral[:, 1], texture_neutral_model, 256, 256)
                image = ndimage.gaussian_filter(image, sigma=(1, 1, 0), order=0)
                image = Image.fromarray(image)

                # brightness
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.3)

                # color
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(0.7)

                # Sharpness
                #enhancer = ImageEnhance.Sharpness(image)
                #image = enhancer.enhance(0.0)

                image = np.asarray(image)

        # Sharpness
        #enhancer = ImageEnhance.Sharpness(image)
        #image = enhancer.enhance(2)

        # color
        #enhancer = ImageEnhance.Color(image)
        #image = enhancer.enhance(0.8)
                self.dictionary['image_neutral'] = image
                return image
                # add expression to neutral face
        '''
        else:
            print('SECONDO ELSE')
            h5f = h5py.File('log.h5', 'w')
            h5f.create_dataset('image', data=np.transpose(original_image))
            h5f.close()
            self.progress_bar.emit()
            exprObj = predict_expr()
            vect, nameExpr = predict_expr.create_expr(expression)
            new_expr = _3DMM_obj.deform_3D_shape_fast(np.transpose(pos_est["defShape"]), Components, vect)

            shape_expressional_model = np.transpose(new_expr)
            self.progress_bar.emit()
            # create texture for expressional model
            projShape_expr = np.transpose(
                _3DMM_obj.getProjectedVertex(np.transpose(shape_expressional_model), pos_est["S"], pos_est["R"],
                                         pos_est["T"]))
            image = _graph_tools_obj.render3DMM(projShape[:, 0], projShape[:, 1], texture_neutral_model, 512, 512)
            self.progress_bar.emit()
        '''

    def apply_expression_modelPreloaded(self, path_log='log/', expression='angry'):
        if expression == 'neutral':
            return self.dictionary['image_neutral']
        self.progress_bar.emit()
        # define RP values
        RP_obj = RP()
        # Fitting and pose estimation are within the 3DMM obj
        _3DMM_obj = _3DMM()
        _graph_tools_obj = graphic_tools(_3DMM_obj)

        # add expression to neutral face
        exprObj = predict_expr()
        vect, nameExpr = predict_expr.create_expr(expression)
        print(vect.shape)
        #print(nameExpr)
        shape_expressional_model = np.transpose(
            _3DMM_obj.deform_3D_shape_fast(np.transpose(self.dictionary['shape_neutral']), self.dictionary['components'], vect))
        self.progress_bar.emit()
        # create texture for expressional model
        projShape_expressional_model = np.transpose(
            _3DMM_obj.getProjectedVertex(shape_expressional_model, self.dictionary['pos_est_S'], self.dictionary['pos_est_R'],
                                         self.dictionary['pos_est_T']))
        
        texture_expressional_model = (_graph_tools_obj.getRGBtexture(projShape_expressional_model, self.dictionary['original_image']))*255
        # [frontalView, colors, mod3d] = _graph_tools_obj.renderFaceLossLess(shape_expressional_model, projShape, image,  dict_['pos_est_S'], dict_['pos_est_R'], dict_['pos_est_T'], 1, dict_['visIdx'])
        image = _graph_tools_obj.render3DMM(projShape_expressional_model[:, 0], projShape_expressional_model[:, 1], self.dictionary['texture_neutral'], 256, 256)
        self.progress_bar.emit()

        # post preocessing on the image
        image = ndimage.gaussian_filter(image, sigma=(0.8, 0.8, 0), order=0)
        image = Image.fromarray(image)

        # brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)

        # color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.7)
        image = np.asarray(image)

        # scale values from in range [0,1]
        projShape_expressional_model_norm = self.scale(projShape_expressional_model, 1, 0)
        projShape_neutral_model_norm = self.scale(self.dictionary['projShape_neutral'], 1, 0)

        h5f = h5py.File('log.h5', 'w')
        h5f.create_dataset('image', data=np.transpose(self.dictionary['original_image']))
        h5f.create_dataset('projShape_expr', data=np.transpose(projShape_expressional_model))
        h5f.create_dataset('projShape_expr_norm', data=np.transpose(projShape_expressional_model_norm))
        h5f.create_dataset('projShape_ne', data=np.transpose(self.dictionary['projShape_neutral']))
        h5f.create_dataset('projShape_ne_norm', data=np.transpose(projShape_neutral_model_norm))
        h5f.create_dataset('shape_expr', data=np.transpose(shape_expressional_model))
        h5f.create_dataset('shape_ne', data=np.transpose(self.dictionary['shape_neutral']))
        h5f.create_dataset('texture_expr', data=np.transpose(texture_expressional_model))
        h5f.create_dataset('rendered_image', data=np.transpose(image))
        h5f.close()

        

        return image


    def scale(self,V, mx,mn):
        min_w = np.amin(V)
        max_w = np.amax(V)
        V = (((V-min_w)*(mx-mn))/(max_w-min_w)) + mn
        return V
