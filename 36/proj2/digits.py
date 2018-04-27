from pylab import *
import numpy as np
from scipy.io import loadmat

M = loadmat("mnist_all.mat")


################################ PART 1 ####################################


def showDigits():
    '''
    Load the dataset from mnist_all.mat. Randomly choose 100 pictures, 10 for each digit
    Combines the 100 pictures to 1 picture and show the picture.
    '''
    M = loadmat("mnist_all.mat")

    f, axarr = plt.subplots(10, 10)
    np.random.seed(1)
    for i in range(10):
        digit = "train" + str(i)
        shape = M[digit].shape
        for j in range(10):
            rand = np.random.randint(0, shape[0])
            axarr[i, j].imshow(M[digit][rand].reshape((28, 28)), cmap=cm.gray)
            plt.setp([axarr[i, j].get_xticklabels()], visible=False)
            plt.setp([axarr[i, j].get_yticklabels()], visible=False)

    plt.show()


################################ PART 2 ####################################


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))


def forward_p2(x, w, b):
    '''
    the input x should be 784 x n, the input w should be 784 x 10, the input b
    should be 10 x 1 in our case
    '''
    Os = dot(w.T, x) + b
    result = softmax(Os)
    return result


################################ PART 3 ####################################


def NLL(y, y_):
    return -sum(y_ * log(y))


def grad_p3(x, w, b, y):
    '''
    return the gradient of weights and the gradient of biases
    '''
    p = forward_p2(x, w, b)
    dc_do = p - y
    return np.dot(dc_do, x.T).T, np.dot(dc_do, np.ones((x.shape[1], 1)))


def cost(p, y):
    cost = 0
    for i in range(p.shape[1]):
        cost += NLL(p[:, i], y[:, i])
    return cost


def check_grad_w(xn, wn, b, y, grad_w):
    h = 0.0001
    wh = np.zeros((784, 10))
    difference_matrix = np.ones((784, 10))
    error = 0
    for j in range(wn.shape[0]):
        for k in range(wn.shape[1]):
            wh[j, k] = h / 2

            wn_m = wn - wh
            p_m = forward_p2(xn, wn_m, b)
            # cost0 = 0
            # for i in range(25):
            #     cost0 += NLL(p_m[:, i], y[:, i])
            cost0 = cost(p_m, y)

            wn_p = wn + wh
            p_p = forward_p2(xn, wn_p, b)
            cost1 = cost(p_p, y)

            difference_matrix[j, k] = (cost1 - cost0) / h - grad_w[j, k]
            error += abs((cost1 - cost0) / h - grad_w[j, k])
            wh[j, k] = 0
    # print(difference_matrix)
    print(
        "The sum of the absolute value of difference for weight is " + str(
            error))
    print(
        "The average of the absolute value of difference for weight is " + str(
            error / 7840))


def check_grad_b(xn, wn, b, y, grad_b):
    h = 0.0001
    bh = np.zeros(b.shape)
    error = 0
    for j in range(b.shape[0]):
        bh[j, 0] = h / 2
        b_p = b + bh
        b_m = b - bh
        p_p = forward_p2(xn, wn, b_p)
        p_m = forward_p2(xn, wn, b_m)

        cost0 = cost(p_m, y)

        cost1 = cost(p_p, y)

        error += abs((cost1 - cost0) / h - grad_b[j, 0])
        bh[j, 0] = 0
    print(
        "The sum of the absolute value of difference for bias is " + str(
            error))
    print(
        "The average of the absolute value of difference for bias is " + str(
            error / 10))


def part3():
    # generating random numbers for test purpose
    np.random.seed(0)
    xn = np.random.rand(784, 25)
    b = np.random.rand(10, 1)
    y = np.zeros((10, 25))
    j = 0
    for i in range(25):
        y[j, i] = 1
        j = (j + 1) % 10
    wn = np.random.rand(784, 10)
    print("Using a random 784 * 25 matrix as x")
    # calculate the gradient value for weights and b
    grad_w, grad_b = grad_p3(xn, wn, b, y)

    # verifying the gradient for weights using finite differences
    check_grad_w(xn, wn, b, y, grad_w)

    # verifying the gradient for biases using finite differences
    check_grad_b(xn, wn, b, y, grad_b)


################################ PART 4 ####################################


def perform(data, weight, bias, answer):
    result = forward_p2(data, weight, bias)
    # correct = 0
    # for i in range(result.shape[1]):
    #     if argmax(result[:, i]) == argmax(answer[:, i]):
    #         correct += 1
    # return correct / result.shape[1]
    return np.mean(np.argmax(answer, 0) == np.argmax(result, 0))


def grad_descent(f, df, x, y, init_t, init_b, alpha, max_iter, test, t_answer,
                 validation, v_answer, plotCurve=True):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    prev_b = init_b - 10 * EPS
    b = init_b.copy()
    iter = 0
    perfomance_x = [perform(x, t, b, y)]
    perfomance_t = [perform(test, t, b, t_answer)]
    perfomance_v = [perform(validation, t, b, v_answer)]
    iterations = [0]
    print('Doing gradient Descent')
    while (np.linalg.norm(t - prev_t) + np.linalg.norm(
                b - prev_b)) > EPS and iter < max_iter:
        prev_t = t.copy()
        prev_b = b.copy()
        grad_t, grad_b = df(x, t, b, y)
        t -= alpha * grad_t
        b -= alpha * grad_b
        if iter % 100 == 0:
            print("Iter", iter)
            print("f(x) = %.2f" % (f(forward_p2(x, t, b), y)))
            # print("Gradient: ", df(x, t, b, y), "\n")
        if plotCurve:
            perfomance_x.append(perform(x, t, b, y))
            perfomance_t.append(perform(test, t, b, t_answer))
            perfomance_v.append(perform(validation, t, b, v_answer))
            iterations.append(iter + 1)
        iter += 1
    if plotCurve:
        plt.plot(iterations, perfomance_x)
        plt.plot(iterations, perfomance_t)
        plt.plot(iterations, perfomance_v)
        plt.legend(['Training Set', 'Test Set', 'Validation Set'],
                   loc='upper right')
        plt.axis([-30, max_iter, 0, 1])
        plt.ylabel('Prop. of correct guesses')
        plt.xlabel('Iterations')
        plt.show()
    print('Final Cost is ' + format(f(forward_p2(x, t, b), y), '.2f'))
    return t, b


def get_sets():
    x = np.zeros((784, 4000 * 10))
    y = np.zeros((10, 4000 * 10))
    test_set = np.zeros((784, 850 * 10))
    answer_set = np.zeros((10, 850 * 10))
    validation_set = np.zeros((784, 1000 * 10))
    v_answer_set = np.zeros((10, 1000 * 10))
    np.random.seed(0)
    for i in range(10):
        train = "train" + str(i)
        test = "test" + str(i)
        train_a = M[train].copy()
        test_a = M[train].copy()
        np.random.shuffle(train_a)
        np.random.shuffle(test_a)

        one_hot = [0] * 10
        one_hot[i] = 1
        x[:, 4000 * i:4000 * (i + 1)] = (train_a[0:4000]).T / 255.0
        y[:, 4000 * i:4000 * (i + 1)] = np.array([one_hot] * 4000).T
        test_set[:, 850 * i:850 * (i + 1)] = (test_a[0:850]).T / 255.0
        answer_set[:, 850 * i:850 * (i + 1)] = np.array([one_hot] * 850).T
        validation_set[:, 1000 * i:1000 * (i + 1)] = (train_a[
                                                      4000:5000]).T / 255.0
        v_answer_set[:, 1000 * i:1000 * (i + 1)] = np.array([one_hot] * 1000).T
    return x, y, test_set, answer_set, validation_set, v_answer_set


def part4(iter, graph=True):
    x, y, test_set, answer_set, validation_set, v_answer_set = get_sets()
    # np.random.seed(0)
    init_t = np.zeros((784, 10))
    init_b = np.zeros((10, 1))
    t, b = grad_descent(cost, grad_p3, x, y, init_t, init_b, 0.00003, iter,
                        test_set, answer_set, validation_set,
                        v_answer_set, graph)  # 3000 has highest percentage
    # np.savetxt('weight.txt', t)
    # np.savetxt('bias.txt', b)
    f, axarr = plt.subplots(2, 5)
    for i in range(10):
        axarr[i // 5, i % 5].imshow(t[:, i].reshape((28, 28)), cmap="RdBu")
        axarr[i // 5, i % 5].set_title('Digit' + str(i))
    plt.show()


################################ PART 5 ####################################


def grad_descent_m(f, df, x, y, init_t, init_b, alpha, gamma, max_iter, test,
                   t_answer, validation, v_answer, plotCurve=True):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    prev_b = init_b - 10 * EPS
    b = init_b.copy()
    vt = 0
    vb = 0
    iter = 0
    perfomance_x = [perform(x, t, b, y)]
    perfomance_t = [perform(test, t, b, t_answer)]
    perfomance_v = [perform(validation, t, b, v_answer)]
    iterations = [0]
    print('Doing gradient Descent')
    while (np.linalg.norm(t - prev_t) + np.linalg.norm(
                b - prev_b)) > EPS and iter < max_iter:
        prev_t = t.copy()
        prev_b = b.copy()
        grad_t, grad_b = df(x, t, b, y)
        vt = gamma * vt + alpha * grad_t
        vb = gamma * vb + alpha * grad_b
        t -= vt
        b -= vb
        if iter % 100 == 0:
            print("Iter", iter)
            print("f(x) = %.2f" % (f(forward_p2(x, t, b), y)))
            # print("Gradient: ", df(x, t, b, y), "\n")
        if plotCurve:
            perfomance_x.append(perform(x, t, b, y))
            perfomance_t.append(perform(test, t, b, t_answer))
            perfomance_v.append(perform(validation, t, b, v_answer))
            iterations.append(iter + 1)
        iter += 1
    if plotCurve:
        plt.plot(iterations, perfomance_x)
        plt.plot(iterations, perfomance_t)
        plt.plot(iterations, perfomance_v)
        plt.legend(['Training Set', 'Test Set', 'Validation Set'],
                   loc='upper right')
        plt.axis([-30, max_iter, 0, 1])
        plt.ylabel('Prop. of correct guesses')
        plt.xlabel('Iterations')
        plt.show()
    print('Final Cost is ' + format(f(forward_p2(x, t, b), y), '.2f'))
    return t, b


def part5(iter, graph=True):
    x, y, test_set, answer_set, validation_set, v_answer_set = get_sets()
    init_t = np.zeros((784, 10))
    init_b = np.zeros((10, 1))
    t, b = grad_descent_m(cost, grad_p3, x, y, init_t, init_b, 0.00003, 0.9,
                          iter,
                          test_set, answer_set, validation_set,
                          v_answer_set, graph)  # 3000 has highest percentage
    np.savetxt('weight_m.txt', t)
    np.savetxt('bias_m.txt', b)


################################ PART 6 ####################################

def part6_grad_descent_vanilla(x, y, b, init_w, alpha, max_iter, weight1, weight2, digit1, digit2):
    gd_traj = []

    w_adj = np.zeros([784, 10])
    w_adj[weight1][digit1] = 1
    w_adj[weight2][digit2] = 1

    EPS = 1e-5
    w = init_w.copy()
    prev_W = w - 10 * EPS
    iter = 0

    while norm(w - prev_W) > EPS and iter < max_iter:
        if iter % 1 == 0:
            # print(((w * w_adj)[weight1][digit1], (w * w_adj)[weight2][digit2]))
            gd_traj.append(((w * w_adj)[weight1][digit1], (w * w_adj)[weight2][digit2]))
        prev_W = w.copy()
        w -= alpha * grad_p3(x, w, b, y)[0] * w_adj
        iter += 1

    return gd_traj


def part6_grad_descent_momentum(x, y, b, init_w, alpha, gamma, max_iter, weight1, weight2, digit1, digit2):
    mo_traj = []

    w_adj = np.zeros([784, 10])
    w_adj[weight1][digit1] = 1
    w_adj[weight2][digit2] = 1

    EPS = 1e-5
    w = init_w.copy()
    prev_W = w - 10 * EPS
    iter = 0
    v = 0

    while norm(w - prev_W) > EPS and iter < max_iter:
        if iter % 1 == 0:
            # print(((w * w_adj)[weight1][digit1], (w * w_adj)[weight2][digit2]))
            mo_traj.append(((w * w_adj)[weight1][digit1], (w * w_adj)[weight2][digit2]))
        prev_W = w.copy()
        v = gamma * v + alpha * grad_p3(x, w, b, y)[0] * w_adj
        w -= v
        iter += 1

    return mo_traj


def part6a():
    w = np.loadtxt('weight_m.txt')
    b = np.loadtxt('bias_m.txt')
    b = b.reshape((10, 1))
    x = get_sets()[0]
    y = get_sets()[1]
    weight1 = 11 * 28 + 11
    weight2 = 17 * 28 + 17

    w1s = np.arange(-1.2, 1.1, 0.2)
    w2s = np.arange(-1.2, 1.1, 0.2)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    k = 0
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            w[weight1][2] = w1
            w[weight2][2] = w2
            p = forward_p2(x, w, b)
            C[i, j] = cost(p, y)
            k += 1
            # print(k)

    CS = plt.contour(w1z, w2z, C, 15)
    plt.legend(loc='upper left')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour plot')
    plt.show()


def part6bc():
    w = np.loadtxt('weight_m.txt')
    b = np.loadtxt('bias_m.txt')
    b = b.reshape((10, 1))
    x = get_sets()[0]
    y = get_sets()[1]
    weight1 = 11 * 28 + 11
    weight2 = 17 * 28 + 17

    w[weight1][2] = 0.8
    w[weight2][2] = 0.8

    alpha = 0.0003
    gamma = 0.7

    gd_traj = part6_grad_descent_vanilla(x, y, b, w, alpha, 20, weight1, weight2, 2, 2)
    mo_traj = part6_grad_descent_momentum(x, y, b, w, alpha, gamma, 20, weight1, weight2, 2, 2)

    w1s = np.arange(-1.2, 1.1, 0.2)
    w2s = np.arange(-1.2, 1.1, 0.2)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    k = 0
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            w[weight1][2] = w1
            w[weight2][2] = w2
            p = forward_p2(x, w, b)
            C[i, j] = cost(p, y)
            k += 1
            # print(k)

    CS = plt.contour(w1z, w2z, C, 15)
    plt.plot([a for a, b in gd_traj], [b for a, b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a, b in mo_traj], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour plot')
    plt.show()


def part6e():
    w = np.loadtxt('weight_m.txt')
    b = np.loadtxt('bias_m.txt')
    b = b.reshape((10, 1))
    x = get_sets()[0]
    y = get_sets()[1]
    weight1 = 0
    weight2 = 783

    w[weight1][2] = 0.8
    w[weight2][2] = 0.8

    alpha = 0.0003
    gamma = 0.7

    gd_traj = part6_grad_descent_vanilla(x, y, b, w, alpha, 20, weight1, weight2, 2, 2)
    mo_traj = part6_grad_descent_momentum(x, y, b, w, alpha, gamma, 20, weight1, weight2, 2, 2)

    w1s = np.arange(-1.2, 1.1, 0.2)
    w2s = np.arange(-1.2, 1.1, 0.2)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    k = 0
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            w[weight1][2] = w1
            w[weight2][2] = w2
            p = forward_p2(x, w, b)
            C[i, j] = cost(p, y)
            k += 1
            # print(k)

    CS = plt.contour(w1z, w2z, C, 15)
    plt.plot([a for a, b in gd_traj], [b for a, b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a, b in mo_traj], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour plot')
    plt.show()

if __name__ == "__main__":
    print("========================= Part 1 =========================")
    showDigits()

    # M = loadmat("mnist_all.mat")
    # x = M["train5"][148:149].T / 255.0
    # result = forward_p2(x, np.random.rand(784, 10), 0.2)
    # print(result)
    print("========================= Part 3 =========================")
    part3()
    print("========================= Part 4 =========================")
    part4(1500)
    print("========================= Part 5 =========================")
    part5(1500)
    print("========================= Part 6 =========================")
    part6a()

    part6bc()

    part6e()
