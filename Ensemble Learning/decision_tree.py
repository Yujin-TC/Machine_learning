import pandas as pd
import numpy as np

def ID3_summary(train, test, label, methods, depths):
    # initialize the dataframe to store the error of the test data and method and depth
    model_summary = pd.DataFrame(columns=["method", "depth", "error_train", "error_test"])
    for method in methods:
        for depth in depths:
            #print("depth=",depth)
            tree = ID3(train, label, method=method, max_depth=depth)
            # how to store the error of the test data and method and depth in dataframe
            error_train = cal_errors(predict_all(tree, train), train, label)
            error_test = cal_errors(predict_all(tree, test), test, label)
            model_summary = pd.DataFrame(model_summary._append({"method": method, "depth": depth, "error_train": error_train, "error_test": error_test}, ignore_index=True))
            #model_summary = pd.DataFrame(model_summary,columns=['method', 'depth', 'accuracy_train', 'accuracy_test'])
        print("method=%s, depth=%s", method, depth)
        print("tree=%s", tree)
    return model_summary

def interpolate_missing(train, col):
    # interpolate missing value in col with the majority of other values of the same attribute in the training set
    # col = 'poutcome'
    # filter the train data by the col value != "unknown"
    train_no_missing = train[train[col] != "unknown"]
    majority_class = train_no_missing[col].mode()[0]
    # replace the col value == "unknown" with the majority of other values of the same attribute in the training set
    train[col] = train[col].replace("unknown", majority_class)
    return train

# 2.model training
# calculate the total entropy of the training data
# train_data is a matrix, each row is a sample, each column is a feature,the last column is the label
# class_list is a list of all the result of the label
def total_entropy(train_data, label, class_list):
    total_count = len(train_data)
    total_entropy = 0
    for c in class_list:
        # print("label:",label)
        # print("train_data:",train_data[label])
        # print("prob:",len(train_data[train_data[label] == c]))
        p = len(train_data[train_data[label] == c]) / total_count
        if p != 0:
            total_entropy -= p * np.log2(p)
    return total_entropy


def total_majority_error(train_data, label, class_list):
    # calculate the majority error of the training data
    '''
    train_data = train
    label = label
    class_list = train[label].unique()

    '''
    total_count = len(train_data)
    majority_error = 0

    for c in class_list:
        p = len(train_data[train_data[label] == c]) / total_count
        if majority_error < p:
            majority_error = p
    # print("majority_error:", 1-majority_error)
    return 1 - majority_error


def total_gini_index(train_data, label, class_list):
    # calculate the majority error of the training data
    '''
    train_data = train
    label = label
    class_list = train[label].unique()

    '''
    total_count = len(train_data)
    gini_index = 1.0

    for c in class_list:
        p = len(train_data[train_data[label] == c]) / total_count
        gini_index -= p ** 2
    # print("majority_error:", 1-majority_error)
    return gini_index


# calculate the information gain of the training data
def information_gain(feature_name, train_data, label, class_list, method):
    '''
    feature_name = 'maint'
    label = 'label'
    class_list = train_data[label].unique()

    '''
    feature_value_list = train_data[feature_name].unique()  # values of the feature
    total_row = len(train_data)
    feature_information = 0.0

    for feature_value in feature_value_list:
        feature_data = train_data[
            train_data[feature_name] == feature_value
            ]  # filtering rows with that feature_value
        feature_value_count = len(feature_data)
        if method == "entropy":
            feature_value_method = total_entropy(
                feature_data, label, class_list
            )  # calculate entropy for the feature value
        if method == "majority_error":
            feature_value_method = total_majority_error(
                feature_data, label, class_list
            )  # calculate majority error for the feature value
        if method == "gini_index":
            feature_value_method = total_gini_index(
                feature_data, label, class_list
            )  # calculate gini index for the feature value
        # calculate entropy for the feature value
        feature_value_probability = feature_value_count / total_row
        feature_information += feature_value_probability * feature_value_method
        # calculate information of the feature value
        # print(feature_name, feature_value, feature_value_probability, feature_value_entropy)
    # calculate information gain by subtracting
    return (total_entropy(train_data, label, class_list) - feature_information)


def find_best_feature(train_data, label, class_list, method):
    # find the feature names
    #print("train dim-----", train_data.shape)
    feature_names = train_data.columns.drop(label)
    #print("feature_names-----", feature_names)
    # print("finding best feature, train data dim: ",train_data.shape)
    best_feature = None
    max_information_gain = 0

    for feature in feature_names:
        feature_information_gain_value = information_gain(
            feature, train_data, label, class_list, method
        )
        #print("feature, info gain=", feature, feature_information_gain_value)
        # select the feature with the highest information gain
        if feature_information_gain_value > max_information_gain:
            max_information_gain = feature_information_gain_value
            # print("max_information_gain:", max_information_gain)
            best_feature = feature

    return best_feature


# generate the sub decision tree
def generate_tree(feature_name, train_data, label, class_list, max_depth):
    # get the feature values
    '''
    feature_name = colnames[0]
    train_data = train
    label = 'label'
    class_list =train['label'].unique()
    '''
    feature_value_dict = train_data[feature_name].value_counts(sort=False)
    sub_tree = {}

    for feature_value, count in feature_value_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        majority_class = feature_value_data[label].value_counts().idxmax()
        isPure = False

        if max_depth == 0:
            sub_tree[feature_value] = majority_class
        else:
            for c in class_list:
                class_count = feature_value_data[feature_value_data[label] == c].shape[0]
                if class_count == count:
                    sub_tree[feature_value] = c
                    train_data = train_data[train_data[feature_name] != feature_value]
                    isPure = True
            if not isPure:
                sub_tree[feature_value] = "#"

    return sub_tree, train_data

    # generate the total tree


def generate_total_tree(root, prev_feature, train_data, label, class_list, method, max_depth):
    '''
    root=tree
    prev_feature =  None
    train_data = train_data
    label = label
    class_list = class_list
    '''
    # get the feature names
    # if len(train_data) == 0:
    #    return prev_feature
    if max_depth <= 0:
        return
    # print("train_data.columns[:-1]: ", train_data.columns[:-1])

    for feature in train_data.columns[:-1]:
        column_data = train_data[feature]
        # print("feature:", feature)
        # print("column_data:", column_data)
        if column_data.unique().shape[0] == 1:
            train_data = train_data.drop(feature, axis=1)
    # print("train_data after removing:", train_data)
    if len(train_data) == 1:
        root[prev_feature] = train_data[label].value_count().idxmax()

    if len(train_data) != 0:
        best_feature = find_best_feature(train_data, label, class_list, method)
        #print("best_feature:", best_feature)
        if best_feature is None:
            return
        sub_tree, train_data = generate_tree(best_feature, train_data, label, class_list, max_depth - 1)
        # print("sub_tree for best feature:", sub_tree)
        # print("train_data:", train_data)
        next_root = None

        if prev_feature != None:
            root[prev_feature] = dict()
            root[prev_feature][best_feature] = sub_tree
            next_root = root[prev_feature][best_feature]
        else:  # add to root of the tree
            root[best_feature] = sub_tree
            next_root = root[best_feature]
        for key, value in list(next_root.items()):
            if value == "#":
                feature_data = train_data[train_data[best_feature] == key]
                # print("feature_data:", feature_data)
                # update class_list for updated data
                class_list = feature_data[label].unique()
                generate_total_tree(next_root, key, feature_data, label, class_list, method, max_depth-1)


# 3.model testing
# implement ID3 algorithm
def ID3(train, label, method, max_depth=None):
    train_data = train.copy()
    tree = {}
    class_list = train_data[label].unique()
    generate_total_tree(tree, None, train_data, label, class_list, method, max_depth)
    return tree


# 4.predicting model
def predict(tree, instance, default_value=None):
    try:
        if not isinstance(tree, dict):  # if it is leaf node
            return tree  # return the value
        else:
            root_node = next(iter(tree))  # getting first key/feature name of the dictionary
            feature_value = instance[root_node]  # value of the feature
            if feature_value in tree[root_node]:  # checking the feature value in current tree node
                return predict(tree[root_node][feature_value], instance, default_value)  # goto next feature
            else:
                return default_value
    except:
        return default_value


def predict_all(tree, test_data, default_value=None):
    predictions = []
    for index, instance in test_data.iterrows():
        prediction = predict(tree, instance, default_value=default_value)
        predictions.append(prediction)
    return predictions


def cal_errors(predictions, test_data, label):
    error = 0
    for i in range(len(predictions)):
        if predictions[i] != test_data.iloc[i][label]:
            error += 1
    return error / len(predictions)

def weighted_ID3(df, label, sample_wight, max_depth):
    from sklearn.tree import DecisionTreeClassifier
    y = df[[label]]
    x = df.drop(columns=['label'])
    pass


