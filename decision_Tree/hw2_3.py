from xxlimited_35 import error

import pandas as pd  # for maniputing the csv file data
import numpy as np  # for mathmatical calculation
import matplotlib.pyplot as plt


def main():
    # 1. load training data
    label_car = 'label'
    label_bank = 'y'
    max_depth = 3

    # load the csv file and save it in pandas dataframe, column name is none, give new column names
    colnames_car = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
    # class_list_car = ["unacc", "acc", "good", "vgood"]
    train_car = pd.read_csv("car/train.csv", names=colnames_car, header=None, index_col=False)
    test_car = pd.read_csv("car/test.csv", names=colnames_car, header=None, index_col=False)
    colnames_bank = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
    # class_list = ["unacc", "acc", "good", "vgood"]
    train_bank = pd.read_csv("bank/train.csv", names=colnames_bank, header=None, index_col=False)
    test_bank = pd.read_csv("bank/test.csv", names=colnames_bank, header=None, index_col=False)
    # check the type of the columns in the csv file
    #print(train.dtypes)
    # select the integer columns in the csv file
    integer_columns_bank = train_bank.select_dtypes(include=['int64']).columns
    #print(integer_columns)
    # convert the integer columns to boolean
    # calculate the median of every column_value
    for column_value in integer_columns_bank:
        #print("column_value and mediam:", column_value, train[column_value].median())
        train_bank[column_value] = train_bank[column_value].apply(lambda x: 1 if x > train_bank[column_value].median() else 0)
        test_bank[column_value] = test_bank[column_value].apply(lambda x: 1 if x > test_bank[column_value].median() else 0)
        #print("colume_data", train[column_value])
    methods = ["entropy", "majority_error", "gini_index"]
    # generate array of depth from 1 to 16
    depths_bank = np.arange(1,17)
    depths_car = [1,2,3,4,5,6]
    model_summary_car = ID3_summary(train_car, test_car, label_car, methods, depths_car)
    model_summary_bank = ID3_summary(train_bank, test_bank, label_bank, methods, depths_bank)
    #print(model_summary)
    # update missing
    train = interpolate_missing(train_bank, "poutcome")
    test = interpolate_missing(test_bank, "poutcome")
    model_summary_updated_bank = ID3_summary(train, test, label_bank, methods, depths_bank)
    print("model_summary_car is:", model_summary_car)
    print("model_summary_bank is:", model_summary_bank)
    print("model_summary_updated_bank is:", model_summary_updated_bank)
    return model_summary_car, model_summary_bank, model_summary_updated_bank




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
def predict(tree, instance):
    if not isinstance(tree, dict):  # if it is leaf node
        return tree  # return the value
    else:
        root_node = next(iter(tree))  # getting first key/feature name of the dictionary
        feature_value = instance[root_node]  # value of the feature
        if feature_value in tree[root_node]:  # checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance)  # goto next feature
        else:
            return None


def predict_all(tree, test_data):
    predictions = []
    for index, instance in test_data.iterrows():
        prediction = predict(tree, instance)
        predictions.append(prediction)
    return predictions


def cal_errors(predictions, test_data, label):
    error = 0
    for i in range(len(predictions)):
        if predictions[i] != test_data.iloc[i][label]:
            error += 1
    return error / len(predictions)


if __name__ == '__main__':
    #pass
    model_summary_car, model_summary_bank, model_summary_updated_bank = main()
    # save the model summary to a csv file using pandas
    #print((model_summary_bank, model_summary_updated_bank))
    model_summary_car.to_csv('model_summary_car.csv', index=False)
    model_summary_bank.to_csv('model_summary_bank.csv', index=False)
    model_summary_updated_bank.to_csv('model_summary_update_bank.csv', index=False)
    # for each method, plot the accuracy_train and accuracy_test vs depth graph
    '''
    plt.figure(figsize=(10, 5))
    plt.title('Accuracy vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    for method in model_summary["method"].unique():
        method_summary = model_summary[model_summary["method"] == method]
        plt.plot(method_summary["depth"], method_summary["accuracy_train"], label=f"{method} train")
        plt.plot(method_summary["depth"], method_summary["accuracy_test"], label=f"{method} test")
    #pass
    model_summary = pd.DataFrame(columns=["method", "depth", "accuracy_train", "accuracy_test"])
    '''
