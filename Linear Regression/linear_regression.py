import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Tolerance = 0.000001
Max_iter = 10000
Learning_rate_list = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

def main():
    data_folder = f"./data/concrete"
    
    # Get train data and test data
    train_data = pd.read_csv(data_folder + "concrete/train.csv", header = None, sep = ",")
    test_data = pd.read_csv(data_folder + "concrete/test.csv", header = None, sep = ",")
    
    # Get features and ouput
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Add a column of ones for the bias term
    X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
    X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)

    # analytical form solution
    analytic_weights = analytic_solution(X_train, y_train)
    analytic_cost = cost_function(X_train, y_train, analytic_weights)
    print("analytic solution weights: ", analytic_weights)
    print("analytics solution cost: ", analytic_cost)
    print("analytic solution test data cost value: ", cost_function(X_test, y_test, analytic_weights))
    print("-----------------------------------------------------------------------------------------------------------------------")
    n, m = X_train.shape
    # initial weight is set to be 0
    initial_weights = np.zeros(m)

    # gradient descent
    # find the first converged learning rate and plot the cost history
    GD_weight, GD_cost_hist, GD_first_lr = find_first_converging_lr(X_train, y_train, initial_weights, Tolerance, Max_iter, Learning_rate_list, batch_GD)

    print("First converging learning rate for gradient descent is ", GD_first_lr)
    print("The final weight for gradient descent is ", GD_weight)
    print("the final cost for gradient descent is ", GD_cost_hist[-1])
    print("the total steps to converge for gradient descent is ", len(GD_cost_hist))
    print("Gradient Descent test data cost value: ", cost_function(X_test, y_test, GD_weight))
    fig, ax = plt.subplots()
    ax.plot(range(len(GD_cost_hist)), GD_cost_hist, label='cost value', color='blue')
    ax.set_title('Gradient Descent cost vs interations')
    ax.set_xlabel('interations')
    ax.set_ylabel('cost')
    ax.legend()
    plt.show()
    print("-----------------------------------------------------------------------------------------------------------------------")

    # stochastic gradient descent
    # find the first converged learning rate and plot the cost history
    SGD_weight, SGD_cost_hist, SGD_first_lr = find_first_converging_lr(X_train, y_train, initial_weights, Tolerance, Max_iter, Learning_rate_list, stochastic_GD)

    print("First converging learning rate for stochastic gradient descent is ", SGD_first_lr)
    print("The final weight for stochastic gradient descent is ", SGD_weight)
    print("the final cost for stochastic gradient descent is ", SGD_cost_hist[-1])
    print("the total steps to converge for stochastic gradient descent is ", len(SGD_cost_hist))
    print("Stochastic Gradient Descent test data cost value: ", cost_function(X_test, y_test, SGD_weight))
    fig, ax = plt.subplots()
    ax.plot(range(len(SGD_cost_hist)), SGD_cost_hist, label='cost value', color='blue')
    ax.set_title('Stochastic Gradient Descent cost vs interations')
    ax.set_xlabel('interations')
    ax.set_ylabel('cost')
    ax.legend()
    plt.show()



def find_first_converging_lr(X, y, w, tol, max_iter, lr_list, method):
    weight_list = []
    converging_steps = []
    cost_hist_list = []
    for lr in lr_list:
        weights, cost_hist = method(X, y, w, tol, max_iter, lr)
        weight_list.append(weights)
        cost_hist_list.append(cost_hist)
        converging_steps.append(len(cost_hist))
    # fastest_lr_idx = converging_steps.index(min(converging_steps))
    first_converge_lr = next(i for i,v in enumerate(converging_steps) if v < max_iter)
    return weight_list[first_converge_lr], cost_hist_list[first_converge_lr], lr_list[first_converge_lr]



def analytic_solution(X, y):
    t1 = np.linalg.inv(np.matmul(X.T, X))
    t2 = np.matmul(X.T, y)
    return np.matmul(t1,t2)

def batch_GD(X, y, w, tol, max_iter, lr):
    # w is the initial weights
    weights = w
    # training size
    n, _ = X.shape
    cost_hist = [cost_function(X, y, weights)]
    iter = 0
    diff = float("inf")
    while iter <= max_iter and diff > tol:
        gradients = X.T.dot(X.dot(weights) - y) * (1 / n)
        next_weights = weights - lr * gradients
        # use the weight change as the converging criteria
        diff = np.linalg.norm(weights - next_weights)
        weights = next_weights
        cost = cost_function(X, y, weights)
        cost_hist.append(cost_function(X, y, weights))
        iter += 1
        
    return weights, cost_hist

def stochastic_GD(X, y, w, tol, max_iter, lr):
    # w is the initial weights
    weights = w
    # training size
    n, _ = X.shape
    cost_hist = [cost_function(X, y, weights)]
    iter = 0
    diff = float("inf")
    while iter <= max_iter and diff > tol:
        rand_idx = np.random.randint(0, n)
        rand_x, rand_y = X[rand_idx, :].flatten(), y[rand_idx]
        gradient = rand_x.T.dot(rand_x.dot(weights) - rand_y) / n
        next_weights = weights - lr * gradient
        # use the weight change as the converging criteria
        diff = np.linalg.norm(weights - next_weights)
        weights = next_weights
        cost = cost_function(X, y, weights)
        cost_hist.append(cost)
        iter += 1
    return weights, cost_hist

def cost_function(X, y, w):
    return np.sum((X.dot(w) - y) ** 2) / (2 * len(y))

if __name__ == "__main__":
    main()
