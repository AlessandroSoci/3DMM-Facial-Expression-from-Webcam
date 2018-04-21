import sys
sys.path.append('expression_code/')
import main_expression
import scipy.misc

image = main_expression.apply_expression('expression_code/imgs/outfile.jpg')
scipy.misc.imsave('outfile.jpg', image)




