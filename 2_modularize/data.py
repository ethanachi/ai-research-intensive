import pandas as pd 

def load_dataset(args):
	data_path = args['data']['path']
	df = pd.read_csv(data_path, sep='\t')
	X = df[[col for col in df.columns if col != 'y']]
	y = df['y']
	return X, y