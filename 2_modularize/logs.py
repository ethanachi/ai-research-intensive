import os
import numpy as np

def log(args, results):
	for result_name, result_val in results.items():
		with open(os.path.join(args['experiment_path'], result_name), 'w') as f:
			f.write(f'{np.mean(result_val):.5f}')