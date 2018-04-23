import h5py
import time
from PIL import Image
import numpy as np
from landmark_detect import Detector
from _3DMM import _3DMM
from RP import RP
from util_for_graphic import graphic_tools
from Matrix_operations import Matrix_op
from Matrix_operations import Vector_op
from create_expression import predict_expr

def apply_expression(original_image_path, path_log = 'log/', expression='angry'):
	remove_boundary = False
	#lambda_opt = 0.01
	# vars for 3DMM pose and fitting
	_lambda = 0.01
	rounds = 1
	r = 3
	C_dist = 700

	#_path_log = str(path_log) + str('log_file.h5')
	fExp = h5py.File('log_file.h5', "w")

	# get the image
	image = Image.open(original_image_path)
	#image.convert('RGB')
	#image = image.resize((250,250))
	image = np.asarray(image)

	detect_landmarks = Detector()
	# also save the landmarks coordinates into landmarks_coord var into the class
	landImage = detect_landmarks.eval_landmarks(original_image_path, 'expression_code/shape_predictor_68_face_landmarks.dat')
	
	# remove first 19 elements
	if remove_boundary:
		index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 63, 64]
		landImage = np.delete(landImage, index, axis=0)
	else:
		index = [65,61]
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
	v_weights_obj.scale(1, 0.1)

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
									landmarks_3D, id_landmarks_3D, landImage, avg_model_data, _lambda, rounds, r, C_dist)
	# add expression to neutral face
	exprObj = predict_expr()
	vect, nameExpr = predict_expr.create_expr(expression)
	print(nameExpr)
	new_expr = _3DMM_obj.deform_3D_shape_fast(np.transpose(pos_est["defShape"]), Components, vect)

	shape_neutral_model = pos_est["defShape"]
	shape_expressional_model = np.transpose(new_expr)

	# create texture for neutral model
	projShape = np.transpose(_3DMM_obj.getProjectedVertex(np.transpose(shape_neutral_model), pos_est["S"], pos_est["R"], pos_est["T"]))
	texture_neutral_model = (_graph_tools_obj.getRGBtexture(projShape, image))*255
	projShape_neutral = projShape

	# create texture for expressional model
	projShape = np.transpose(_3DMM_obj.getProjectedVertex(np.transpose(shape_expressional_model), pos_est["S"], pos_est["R"], pos_est["T"]))
	texture_expressional_model = (_graph_tools_obj.getRGBtexture(projShape, image))*255
	[frontalView, colors, mod3d] = _graph_tools_obj.renderFaceLossLess(shape_expressional_model, projShape, image, pos_est["S"], pos_est["R"], pos_est["T"], 1, pos_est["visIdx"])
	# posso prendere il modello con maggiore risoluzione solo del modello neutro

	#projShape = np.transpose(_3DMM_obj.getProjectedVertex(np.transpose(shape_expressional_model), pos_est["S"], pos_est["R"], pos_est["T"]))
	image = _graph_tools_obj.render3DMM(projShape[:,0], projShape[:,1], texture_neutral_model, 512, 512)

	# save date for re create model withou re doing neutral model
	dictionary_data = {
			'shape_neutral': shape_neutral_model,
			'projShape_neutral': projShape_neutral,
			'texture_neutral': texture_neutral_model,
			'components': Components,
			'pos_est_S': pos_est["S"],
			'pos_est_R': pos_est["R"],
			'pos_est_T': pos_est["T"],
			'visIdx': pos_est["visIdx"]
	}	
	
	# to create 2D image
	fExp.create_dataset("expressional_shape", data=shape_expressional_model)
	fExp.create_dataset("neutral_text", data=np.transpose(texture_neutral_model))
	fExp.create_dataset("projShape", data=np.transpose(projShape))

	fExp.create_dataset("expressional_text", data=np.transpose(texture_expressional_model))
	fExp.create_dataset("frontalView", data=np.transpose(frontalView))

	fExp.create_dataset("image", data=np.transpose(image))

	fExp.close()
	print('SAVED')

	return image, dictionary_data
	

def apply_expression_modelPreloaded(dict_, path_log = 'log/', expression='angry'):
	# define RP values
	RP_obj = RP()
	# Fitting and pose estimation are within the 3DMM obj
	_3DMM_obj = _3DMM()
	_graph_tools_obj = graphic_tools(_3DMM_obj)

	# add expression to neutral face
	exprObj = predict_expr()
	vect, nameExpr = predict_expr.create_expr(expression)
	print(nameExpr)
	shape_expressional_model = np.transpose(_3DMM_obj.deform_3D_shape_fast(np.transpose(dict_['shape_neutral']), dict_['components'], vect))

	# create texture for expressional model
	projShape = np.transpose(_3DMM_obj.getProjectedVertex(np.transpose(shape_expressional_model), dict_['pos_est_S'], dict_['pos_est_R'], dict_['pos_est_T']))
	#texture_expressional_model = (_graph_tools_obj.getRGBtexture(projShape, image))*255
	#[frontalView, colors, mod3d] = _graph_tools_obj.renderFaceLossLess(shape_expressional_model, projShape, image,  dict_['pos_est_S'], dict_['pos_est_R'], dict_['pos_est_T'], 1, dict_['visIdx'])
	image = _graph_tools_obj.render3DMM(projShape[:,0], projShape[:,1], dict_['texture_neutral'], 512, 512)


	return image, dict_














