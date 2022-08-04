from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def get_metrics(args):
	METRIC_NAME_TO_METRIC = {
		'r_squared': r2_score,
		'mean_error': mean_absolute_error,
		'l2': mean_squared_error,
	}
	return {metric_name: METRIC_NAME_TO_METRIC[metric_name] for metric_name in args['metrics']}
	