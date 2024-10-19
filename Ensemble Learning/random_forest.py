from decision_tree import ID3, predict_all
import numpy as np
import pandas as pd

def select_random_features_with_label(sampled_df, label, feature_size):
    # Exclude the last column (assumed to be the label)
    feature_columns = sampled_df.drop(columns=[label]).columns
    # Randomly select two feature columns
    selected_features = np.random.choice(feature_columns, size=feature_size, replace=False)
    #print("features:", selected_features)
    # Create a new DataFrame with the selected features and the label
    new_df = sampled_df[list(selected_features) + [label]]
    return new_df

# build a bagging tree using ID3 simple decision tree
def random_forest_tree(df, label, n_trees, n_samples, feature_size, method='entropy', max_depth=17):
    # n_samples=1000
    trees = []
    for i in range(n_trees):
        sampled_df = df.sample(n=n_samples, replace=False)
        sampled_df = select_random_features_with_label(sampled_df, label, feature_size)
        decision_tree = ID3(sampled_df, label, method, max_depth=max_depth)
        trees.append(decision_tree)
    return trees


def random_forest_predict(trees, test_data, default_value):
    # trees: tree lists, test_data: data frame
    y_test_predict = []
    for i, decision_tree in enumerate(trees):
        #print("i=",i)
        y_test_predict.append(predict_all(decision_tree, test_data, default_value))
    # return the most voted
    y_test_predict = pd.DataFrame(y_test_predict)
    return list(y_test_predict.mode().iloc[0])
