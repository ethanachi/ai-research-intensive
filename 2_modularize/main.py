import argparse
import random
import string
import os
import yaml

import data 
import logs
import reporting 
import models
import training

def generate_label():
	return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def make_experiment_folder(base_path, yaml_args):
	label = generate_label()
	experiment_dir = os.path.join(base_path, label) + '/'
	os.makedirs(experiment_dir)
	with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
		f.write(yaml.dump(yaml_args))
	return experiment_dir
	
def run_experiment(args):
	dataset = data.load_dataset(args)
	model = models.get_model(args)
	metrics = reporting.get_metrics(args)
	results = training.train(args, dataset, model, metrics)
	logs.log(args, results)
	
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