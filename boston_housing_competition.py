"""
File: boston_housing_competition.py
Name: 
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TRAIN_PATH = 'boston_housing/train.csv'
TEST_PATH = 'boston_housing/test.csv'


def main():
	# train_data = pd.read_csv(TRAIN_PATH)
	# y = train_data.pop('medv')
	# x_train = np.array(train_data.rm).reshape(-1, 1)
	#
	# # y = wx + b
	# w, b, c = 0, 0, 0.6
	# alpha = 0.01
	# num_epoch = 20
	# history = []
	# for epoch in range(num_epoch):
	# 	total = 0
	# 	for i in range(len(x_train)):
	# 		x, labels = x_train[i], y[i]
	# 		h = w*x+b
	# 		loss = (h - labels)**2*(sign(h-labels)-c)**2
	# 		total += loss
	# 		w = w - alpha * 2 * (h - labels) * x*(sign(h-labels)-c)**2
	# 		b = b - alpha * 2 * (h - labels) * 1*(sign(h-labels)-c)**2
	# 	history.append(total/len(train_data))
	#
	# predictions = []
	# for x in x_train:
	# 	predictions.append(w*x+b)
	# print(sum(predictions)/len(x_train))

	# Load the training and test data
	train_data = pd.read_csv(TRAIN_PATH)
	test_data = pd.read_csv(TEST_PATH)
	train_data.pop('ID')
	id = test_data.pop('ID')
	print(id)
	# Separate target variable
	y = train_data.pop('medv')

	# Normalize the training data
	normalizer = preprocessing.MinMaxScaler()
	train_data = normalizer.fit_transform(train_data)
	test_data_normalized = normalizer.transform(test_data)

	# Split the training data into train and validation sets
	X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.3)

	# Train the model on the original features
	h = linear_model.LinearRegression()
	regressor = h.fit(X_train, y_train)
	training_acc = regressor.score(X_train, y_train)
	prediction1 = regressor.predict(X_train)
	print(f'Training Accuracy: {training_acc}')

	# Generate polynomial features
	poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
	X_train_poly = poly_phi_extractor.fit_transform(X_train)
	regressor_poly = h.fit(X_train_poly, y_train)
	training_acc_poly = regressor_poly.score(X_train_poly, y_train)
	print(f'Training Accuracy with Polynomial Features: {training_acc_poly}')

	# Apply the polynomial transformation to the normalized test data
	X_test_poly = poly_phi_extractor.transform(test_data_normalized)
	prediction2 = regressor_poly.predict(X_test_poly)


	# out_file(predictions, id, 'boston_housing.csv')


def sign(x):
	if x > 0:
		return 1
	elif x == 0:
		return 0
	else:
		return -1


def out_file(predictions, id, filename):
    """
    :param predictions: numpy.array, a list-like data structure that stores 0's and 1's
    :param filename: str, the filename you would like to write the results to
    """
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        i = 0
        for ans in predictions:
            out.write(f"{id[i]},{ans:.8f}\n")
            i += 1
    print('===============================================')


if __name__ == "__main__":
    main()