"""
File: titanic_random_forest.py
Name:
---------------------------
This file shows how to use pandas and sklearn
packages to build a random forest, which enables
students to improve decision tree classifier by
reducing overfitting.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Constants - filenames for data set
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'

# Global Variable - cache for nan data in training data
nan_cache = {}


def main():
    try:
        # Data cleaning
        train_data = pd.read_csv(TRAIN_PATH)
        test_data = pd.read_csv(TEST_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Extract true labels (target variable)
    y = train_data['medv']
    train_data.pop('ID')
    id = test_data.pop('ID')

    # Extract features
    feature_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    x_train = train_data[feature_names]

    # Construct Forest
    forest = RandomForestRegressor(max_depth=5, min_samples_leaf=5)
    forest_regressor = forest.fit(x_train, y)
    print(f"Training R^2 Score: {forest_regressor.score(x_train, y)}")

    # Test Data
    x_test = test_data[feature_names]

    # Make predictions
    predictions = forest_regressor.predict(x_test)
    out_file(predictions, id, 'forest.csv')


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


if __name__ == '__main__':
    main()
