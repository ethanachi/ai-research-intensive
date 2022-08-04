import argparse
from collections import defaultdict
import random
import string
import os
import shutil
import yaml

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def generate_label():
	return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def make_experiment_folder(base_path, yaml_args):
	label = generate_label()
	experiment_dir = os.path.join(base_path, label) + '/'
	os.makedirs(experiment_dir)
	with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
		f.write(yaml.dump(yaml_args))
	shutil.copytree('.', os.path.join(experiment_dir, 'code'))
	print(f"Logging to: {experiment_dir}")
	return experiment_dir
	
def load_dataset(args):
	data_path = args['data']['path']
	df = pd.read_csv(data_path, sep='\t')
	X = df[[col for col in df.columns if col != 'y']]
	y = df['y']
	return X, y

def get_model_class_from_model_type(model_type):
	if model_type == 'linear_regression':
		return LinearRegression
	elif model_type == 'decision_tree':
		return DecisionTreeRegressor
	else:
		assert False, f"Model type {model_type} not found."

def get_model(args):
	model_type = args['model']['model_type']
	model_class = get_model_class_from_model_type(model_type)
	return model_class(**args['model'].get('model_args', {}))
	
def get_metrics(args):
	METRIC_NAME_TO_METRIC = {
		'r_squared': r2_score,
		'mean_error': mean_absolute_error,
		'l2': mean_squared_error,
	}
	return {metric_name: METRIC_NAME_TO_METRIC[metric_name] for metric_name in args['metrics']}
		
	
def train(args, dataset, model, metrics):
	kf = KFold(n_splits=args['data']['kfold_splits'])
	X, y = dataset
	metrics_to_results = defaultdict(list)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		for metric_name, metric_func in metrics.items():
			metrics_to_results[metric_name].append(metric_func(y_test, y_pred))
	return metrics_to_results
			
def log(args, results):
	for result_name, result_val in results.items():
		with open(os.path.join(args['experiment_path'], result_name), 'w') as f:
			f.write(f'{np.mean(result_val):.5f}')
	
def run_experiment(args):
	dataset = load_dataset(args)
	model = get_model(args)
	metrics = get_metrics(args)
	results = train(args, dataset, model, metrics)
	log(args, results)
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("config_path", type=str,
						help="path to a YAML config file")
	parser.add_argument("base_path", type=str,
						help="path where experimental logs should be dumped")
	args = parser.parse_args()
	
	with open(args.config_path, 'r') as f:
		yaml_args = yaml.safe_load(f)

	experiment_folder = make_experiment_folder(args.base_path, yaml_args)
	yaml_args['experiment_path'] = experiment_folder
	run_experiment(yaml_args)
	
if __name__ == '__main__':
	main()