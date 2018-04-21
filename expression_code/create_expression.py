import h5py
import numpy as np
from py_script import prediction
import random

expressions = ["disgust", "surprise", "angry", "sadness", "fear", "contempt", "happy"]
# Specify your expression
expr = "happy"

class predict_expr():
	def create_expr(expr):
		#index = random.randint(0,len(expressions)-1)
		#expr = expressions[index]
		#expr = 'surprise'
		pred_vector = prediction.m_prediction(expr, tec="mode")
		pred_vector = np.asarray(np.transpose(pred_vector))
		#regr = prediction.regressor(expr, tec = "svr")
		#pred_vector = regr.prediction(faccia neutra)
		if expr == 'disgust' or expr == 'angry' or expr == 'sadness' or expr == 'contempt':
			pred_vector = pred_vector * 2
		return pred_vector, expr
