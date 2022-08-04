from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def get_model(args):
	model_type = args['model']['model_type']
	model_class = get_model_class_from_model_type(model_type)
	return model_class(**args['model'].get('model_args', {}))
	
def get_model_class_from_model_type(model_type):
	if model_type == 'linear_regression':
		return LinearRegression
	elif model_type == 'decision_tree':
		return DecisionTreeRegressor
	else:
		assert False, f"Model type {model_type} not found."