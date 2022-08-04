from sklearn.model_selection import KFold
from collections import defaultdict

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