import pandas as pd
import numpy as np
from bagging_tree import bagging_tree, bagging_predict
from decision_tree import cal_errors, predict_all
import time
import matplotlib.pyplot as plt

from random_forest import random_forest_tree, random_forest_predict

# 1.load bank train and test dataset
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
           'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data = pd.read_csv("bank/train.csv", names=columns, header=None, index_col=False)
test_data = pd.read_csv("bank/test.csv", names=columns, header=None, index_col=False)
print("train data shape:", train_data.shape) # 5000 rows
print("test data shape:", test_data.shape) # 5000 rows
print(train_data.dtypes)

# 2.data cleaning and formating
# numeric columns to binary
numeric_columns = train_data.select_dtypes(include=['number']).columns
print(numeric_columns)
# for train
for col in numeric_columns:
    median = train_data[col].median()
    train_data[col] = (train_data[col] >= median).astype(int)
# for test
for col in numeric_columns:
    median = test_data[col].median()
    test_data[col] = (test_data[col] >= median).astype(int)
# no need to deal with unknown, treat unknown as attributes
# convert response to 1 and 0, no as 0,
# 3. build bagging trees
iteration = 2 # code is slow, so set a small iteration number
n_trees = 500
n_samples = 1000
label = 'y'
default_value = train_data[label].mode().iloc[0]
tree_train_error = []
tree_test_error = []
tree_number = []
# number of trees iteration to check the impact of number trees on error
#for n_tree in range(1, n_trees, 10):

#for n_tree in range(0, n_trees, 150):
for n_tree in [1, 20, 50, 100, 150, 300, 450, 500]:
    start_time = time.time()
    # build bagging tree
    bagging_model = bagging_tree(train_data, label, n_tree, n_samples)
    end_time = time.time()
    print("it takes:", (end_time-start_time)/60, " minutes to run for n_tree=", n_tree)
    # predict
    bagging_train_predict = bagging_predict(bagging_model, train_data, default_value)
    # error
    tree_train_error.append(cal_errors(bagging_train_predict, train_data, label))
    # test
    bagging_test_predict = bagging_predict(bagging_model, test_data, default_value)
    tree_test_error.append(cal_errors(bagging_test_predict, test_data, label))
    tree_number.append(n_tree)
# summary
tree_number_impact_summary = pd.DataFrame({
    'n_trees': tree_number,
    'train_error': tree_train_error,
    'test_error': tree_test_error
})
tree_number_impact_summary.to_csv("bagging_tree_number_summary.csv", index=False)

plt.figure(figsize=(10, 6))
# Plotting train_error and test_error
plt.plot(tree_number_impact_summary['n_trees'], tree_number_impact_summary['train_error'], marker='o', label='Train Error', color='blue')
plt.plot(tree_number_impact_summary['n_trees'], tree_number_impact_summary['test_error'], marker='o', label='Test Error', color='orange')

# Adding labels and title
plt.xlabel('number of trees')
plt.ylabel('Error')
plt.title('Train and Test Error vs number of trees')
plt.legend()  # Show the legend
plt.grid()    # Show grid for better readability
# Display the plot
plt.show()
plt.savefig('bagging_tree_number.png')
# iteration summary for bagging vs simple decision tree
iteration = iteration
bagging_tree_predicts = []
single_tree_predicts = []
for i in range(iteration):
    start_time = time.time()
    # build bagging tree
    bagging_model = bagging_tree(train_data, label, 500, n_samples)
    end_time = time.time()
    print("it takes:", (end_time-start_time)/60, " minutes to run")
    simple_tree = bagging_model[0]
    # test
    bagging_test_predict = bagging_predict(bagging_model, test_data, default_value)
    simple_test_predict = predict_all(simple_tree, test_data, default_value)
    bagging_tree_predicts.append(bagging_test_predict)
    single_tree_predicts.append(simple_test_predict)
    pd.DataFrame(bagging_tree_predicts).to_csv("bagging_tree_predicts.csv", index=False)
    pd.DataFrame(single_tree_predicts).to_csv("single_tree_predicts.csv", index=False)

bagging_tree_predicts_df = pd.DataFrame(bagging_tree_predicts)
single_tree_predicts_df = pd.DataFrame(single_tree_predicts)
single_tree_predicts_df = single_tree_predicts_df.replace({'#': default_value})

#bagging_tree_predicts_df = pd.read_csv("bagging_tree_predicts.csv")
#single_tree_predicts_df = pd.read_csv("single_tree_predicts.csv")
def cal_mean_variance(test_data, all_predictions, label):
    bias_list = []
    variance_list = []
    n_samples = len(test_data)
    y_test = test_data[[label]]
    # format y_test to 1 or 0
    y_test = y_test.replace({'yes': 1, 'no': 0})
    y_test = y_test[label].to_list()
    # default value
    all_predictions = all_predictions.replace({'yes': 1, 'no': 0})
    for i in range(n_samples):
        # Ground truth label for the test example
        true_value = y_test[i]
        # Predictions for the current test example by 100 trees
        predictions_for_example = all_predictions.iloc[:, i]
        # Compute the average prediction for the test example
        average_prediction = np.mean(predictions_for_example)
        # Compute the bias term: (average prediction - true label)^2
        bias = (average_prediction - true_value) ** 2
        bias_list.append(bias)
        # Compute the variance term: sample variance of the predictions
        variance = np.var(predictions_for_example, ddof=1)  # ddof=1 gives sample variance
        variance_list.append(variance)
    # Take the average of bias and variance across all test examples
    mean_bias = np.mean(bias_list)
    mean_variance = np.mean(variance_list)
    general_error = mean_bias+mean_variance
    # Output the bias and variance estimates
    print(f"Average Bias: {mean_bias}")
    print(f"Average Variance: {mean_variance}")
    print(f"General Error: {general_error}")
    return mean_bias, mean_variance, general_error

# for question 2.2 C
print("bagging tree:")
mean_bias_bagging, mean_var_bagging, general_error_bagging = cal_mean_variance(test_data.copy(), bagging_tree_predicts_df.copy(), label)
print("single tree:")
mean_bias_single, mean_var_single, general_error_single = cal_mean_variance(test_data.copy(), single_tree_predicts_df.copy(), label)

# for question 2.2 D for random forest
#for n_tree in range(0, n_trees, 150):
rf_train_error = []
rf_test_error = []
rf_tree_number = []
feature_number = []
#for n_tree in [1, 20, 50, 100, 450, 500]:
for n_tree in [1, 20, 50, 100, 500]:
    for feature_size in [2, 4, 6]:
        print("trees number, feature_size:", n_tree, feature_size)
        start_time = time.time()
        # build bagging tree
        rf_model = random_forest_tree(train_data, label, n_tree, n_samples, feature_size, max_depth=feature_size-1)
        end_time = time.time()
        print("it takes:", (end_time-start_time)/60, " minutes to run for n_tree=", n_tree)
        # predict
        rf_train_predict = random_forest_predict(rf_model, train_data, default_value)
        # error
        rf_train_error.append(cal_errors(rf_train_predict, train_data, label))
        # test
        rf_test_predict = random_forest_predict(rf_model, test_data, default_value)
        rf_test_error.append(cal_errors(rf_test_predict, test_data, label))
        feature_number.append(feature_size)
        rf_tree_number.append(n_tree)
        print("it takes:", (time.time()-end_time)/60, " minutes to predict for n_tree=", n_tree)

# summary
rf_number_impact_summary = pd.DataFrame({
    'n_trees': rf_tree_number,
    'n_features': feature_number,
    'train_error': rf_train_error,
    'test_error': rf_test_error
})
rf_number_impact_summary.to_csv("random_forest_tree_number_summary.csv", index=False)
'''
plt.figure(figsize=(10, 6))
# Plotting train_error and test_error
plt.plot(tree_number_impact_summary['n_trees'], tree_number_impact_summary['train_error'], marker='o', label='Train Error', color='blue')
plt.plot(tree_number_impact_summary['n_trees'], tree_number_impact_summary['test_error'], marker='o', label='Test Error', color='orange')

# Adding labels and title
plt.xlabel('number of trees')
plt.ylabel('Error')
plt.title('Train and Test Error vs number of trees')
plt.legend()  # Show the legend
plt.grid()    # Show grid for better readability
# Display the plot
plt.show()
plt.savefig('bagging_tree_number.png')
'''

# for question 2.2 e for random forest iteration summary for random vs simple decision tree
iteration = iteration
rf_tree_predicts = []
single_tree_predicts = []
feature_size = 6
for i in range(iteration):
    start_time = time.time()
    # build bagging tree
    rf_model = random_forest_tree(train_data, label, 500, n_samples, feature_size, max_depth=feature_size-1)
    end_time = time.time()
    print("iteration=", i, "it takes:", (end_time-start_time)/60, " minutes to run")
    simple_tree = rf_model[0]
    # test
    rf_test_predict = random_forest_predict(rf_model, test_data, default_value)
    simple_test_predict = predict_all(simple_tree, test_data, default_value)
    rf_tree_predicts.append(rf_test_predict)
    single_tree_predicts.append(simple_test_predict)
    pd.DataFrame(rf_tree_predicts).to_csv("rf_tree_predicts.csv", index=False)
    pd.DataFrame(single_tree_predicts).to_csv("single_rf_tree_predicts.csv", index=False)
    print("it takes:", (time.time()-end_time)/60, " minutes to predict")


rf_tree_predicts_df = pd.DataFrame(rf_tree_predicts)
single_tree_predicts_df = pd.DataFrame(single_tree_predicts)
#single_tree_predicts_df = single_tree_predicts_df.replace({'#': default_value})
# for question 2.2 C
print("rf tree:")
mean_bias_rf, mean_var_rf, general_error_rf = cal_mean_variance(test_data.copy(), rf_tree_predicts_df.copy(), label)
print("single tree:")
mean_bias_single, mean_var_single, general_error_single = cal_mean_variance(test_data.copy(), single_tree_predicts_df.copy(), label)










