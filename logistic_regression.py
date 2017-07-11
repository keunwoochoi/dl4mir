"""A numpy-free logistic regression"""
from collections import Iterable


def _sigmoid_scalar(x):
    """sigmoid mapping function for a scalar value"""
    return 1. / (1 + 2.7182818284 ** (-1 * x))


def sigmoid(x):
    """sigmoid for a list input"""
    if isinstance(x, Iterable):
        return [_sigmoid_scalar(x_ele) for x_ele in x]
    else:
        return _sigmoid_scalar(x)


def _d_sigmoid_scalar(x):
    """derivative of signoid for a scalar input"""
    return sigmoid(x) * (1 - sigmoid(x))


def d_sigmoid(x):
    """derivative of sigmoid for a list"""
    if isinstance(x, Iterable):
        return [_d_sigmoid_scalar(x_ele) for x_ele in x]
    else:
        return _d_sigmoid_scalar(x)


def cost(y_est, y_true):
    """compute mean square error"""
    return sum([(y_e - y_t) ** 2 for y_e, y_t in zip(y_est, y_true)]) / float(len(y_est))


def get_y_est(x, w_est):
    return [dot(x_each, w_est) for x_each in x]


def dot(vec_x, vec_y):
    """dot product"""
    return sum([x * y for x, y in zip(vec_x, vec_y)])


def _minus_scalar(x, y):
    return x - y


def minus(x, y):
    if isinstance(x, Iterable):
        return [_minus_scalar(x_ele, y_ele) for x_ele, y_ele in zip(x, y)]
    else:
        return _minus_scalar(x, y)


# input data (2D)
x = [[1, 3], [2, 6], [3, 1], [10, 3]]
for x_ele in x:
    x_ele.append(1)  # append 1 for bias

n_data = len(x)
n_dim = len(x[0])
w_est = [0.1, -0.1, 0.03]  # initial weight values
w_true = [2., 3., 4.]  # the true model weight
y_true = get_y_est(x, w_true)  # groundtruth

learning_rate = 1e-2
tolerance = 1e-5

err = cost(get_y_est(x, w_est), y_true)  # compute initial error

iter_idx = 0
while err >= tolerance:  # while the error is bigger than our tolerance
    w_dot_x = [dot(x_ele, w_est) for x_ele in x]  # shape: (4, )
    dist_pred_true = [wx_ele - y_ele for wx_ele, y_ele in zip(w_dot_x, y_true)]  # (4, )
    d_cost = []  # (3, )
    for dim_idx in xrange(n_dim):
        d_cost_onedim = []
        for data_idx in xrange(n_data):
            d_cost_onedim.append(dist_pred_true[data_idx] * sigmoid(w_dot_x[data_idx]) * x[data_idx][dim_idx])

        d_cost.append(sum(d_cost_onedim) / float(n_data))

    w_est = [w_ele - learning_rate * d_cost_ele for w_ele, d_cost_ele in zip(w_est, d_cost)]  # (3, )
    err = cost(get_y_est(x, w_est), y_true)
    iter_idx += 1
    if iter_idx % 100 == 0:
        print iter_idx, ': ', err, w_est

print 'w_true: ', w_true
print 'w_est: ', w_est
print 'y_true: ', y_true
print 'y_est: ', get_y_est(x, w_est)
