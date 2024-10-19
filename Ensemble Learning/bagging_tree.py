from decision_tree import ID3, predict_all
import numpy as np
import pandas as pd

# build a bagging tree using ID3 simple decision tree
def bagging_tree(df, label, n_trees, n_samples, method='entropy', max_depth=17):
    # n_samples=1000
    trees = []
    for i in range(n_trees):
        sampled_df = df.sample(n=n_samples, replace=False)
        decision_tree = ID3(sampled_df, label, method, max_depth=max_depth)
        trees.append(decision_tree)
    return trees


def bagging_predict(trees, test_data, default_value):
    # trees: tree lists, test_data: data frame
    y_test_predict = pd.DataFrame(index=range(len(trees)), columns=range(len(test_data)))
    y_test_predict = []
    for i, decision_tree in enumerate(trees):
        y_test_predict.append(predict_all(decision_tree, test_data, default_value))
    # return the most voted
    y_test_predict = pd.DataFrame(y_test_predict)
    return list(y_test_predict.mode().iloc[0])







