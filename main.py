# Assignmnet 1
# Liam Duncan - 301476562
# February 16, 2024


from cgi import test
import pandas as pd
import argparse
from random_forest import RandomForest

def parse_args():
    parser = argparse.ArgumentParser(description='Run random forrest with specified input arguments')
    parser.add_argument('--n-classifiers', type=int,
                        help='number of features to use in a tree',
                        default=1)
    parser.add_argument('--train-data', type=str, default='data/train.csv',
                        help='train data path')
    parser.add_argument('--test-data', type=str, default='data/test.csv',
                        help='test data path')
    parser.add_argument('--criterion', type=str, default='entropy',
                        help='criterion to use to split nodes. Should be either gini or entropy.')
    parser.add_argument('--maxdepth', type=int, help='maximum depth of the tree',
                        default=5)
    parser.add_argument('--min-sample-split', type=int, help='The minimum number of samples required to be at a leaf node',
                        default=20)
    parser.add_argument('--max-features', type=int,
                        help='number of features to use in a tree',
                        default=12)
    a = parser.parse_args()
    return(a.n_classifiers, a.train_data, a.test_data, a.criterion, a.maxdepth, a.min_sample_split, a.max_features)


def read_data(path):
    data = pd.read_csv(path)
    return data

def main():
    n_classifiers, train_data_path, test_data_path, criterion, max_depth, min_sample_split, max_features = parse_args()
    train_data = read_data(train_data_path)
    test_data = read_data(test_data_path)

    train_data_no_missing = train_data.loc[(train_data['native-country'] != ' ?') & (train_data['occupation'] != ' ?') & (train_data['workclass'] != ' ?')]
    test_data_no_missing = test_data.loc[(test_data['native-country'] != ' ?') & (test_data['occupation'] != ' ?') & (test_data['workclass'] != ' ?')]
    test_data_no_missing = test_data_no_missing.copy()
    test_data_no_missing['income'] = test_data_no_missing['income'].str.strip('.')

    # YOU NEED TO HANDLE MISSING VALUES HERE
    # ...
    
    random_forest = RandomForest(n_classifiers=10,
                  criterion = 'gini',
                  max_depth= 10,
                  min_samples_split = 20 ,
                  max_features = 11 )

    print(random_forest.fit(train_data_no_missing, 'income'))
    print(random_forest.evaluate(train_data_no_missing, 'income'))
    print(random_forest.evaluate(test_data_no_missing, 'income'))
    

if __name__ == '__main__':
    main()

