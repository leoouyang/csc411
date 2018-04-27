import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.misc import imread

actors = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']

y_p3 = np.zeros(0)
x_p3 = np.zeros(0)

male = ['baldwin', 'carell', 'hader', 'radcliffe', 'butler', 'vartan']
female = ['bracco', 'gilpin', 'harmon', 'chenoweth', 'drescher', 'ferrera']
male_p5 = np.zeros(0)
female_p5 = np.zeros(0)
test_p5 = np.zeros(0)
answer_p5 = np.zeros(0)

y_p7 = np.zeros(0)
x_p7 = np.zeros(0)
actor_pattern = {}

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def seperate_sets(directory):
    counter = {}
    for a in actors:
        counter[a] = 0

    make_dir('sets')
    make_dir('sets/training_set')
    make_dir('sets/test_set')
    make_dir('sets/validation_set')

    images = os.listdir(directory)
    images.sort()
    np.random.seed(0)
    np.random.shuffle(images)
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)
        if cur_name in counter:
            counter[cur_name] += 1
            cur_num = counter[cur_name]
            if cur_num <= 10:
                shutil.copy(directory + '/' + image, 'sets/test_set')
            elif cur_num <= 20:
                shutil.copy(directory + '/' + image, 'sets/validation_set')
            elif cur_num <= 90:
                shutil.copy(directory + '/' + image, 'sets/training_set')

def init_grad_descent_p3(directory):
    images = os.listdir(directory)
    images.sort()
    x_temp = []
    y_temp = []
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)
        flag = -1
        if cur_name == actors[3]:
            flag = 1
        elif cur_name == actors[5]:
            flag = 0

        if flag != -1:
            im = imread(directory+"/"+image)
            im = im.reshape(1024)
            im = im.astype(float)
            im = im/255
            x_temp.append(im)
            y_temp.append(flag)
    global y_p3
    y_p3 = np.array(y_temp)
    global x_p3
    x_p3 = np.array(x_temp)


def f_p3(x, y, theta):
    return np.sum((y - np.dot(theta.T, x)) ** 2)

def df_p3(x, y, theta):
    return -2 * np.sum((y - np.dot(theta.T, x)) * x, axis = 1)

def grad_descent(f, df, x, y, init_t, alpha, max_iter):
    EPS = 1e-6   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    x = np.vstack((np.ones((1, x.T.shape[1])), x.T))
    print 'Doing gradient Descent'
    while np.linalg.norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 5000 == 0:
            print "Iter", iter
        #     print "theta = ", t, ", f(x) = %.2f" % (f(x, y, t))
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    print 'Final Cost is ' + format(f(x, y, t), '.2f')
    return t

def classifier_p3(x, theta):
    x = np.hstack((np.ones(1), x))
    h_x = np.dot(theta.T, x)
    if h_x > 0.5:
        return actors[3]
    else:
        return actors[5]

def tester_p3(directory, theta):
    images = os.listdir(directory)
    images.sort()
    correct = 0.0
    total = 0.0
    x_temp = []
    y_temp = []
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)
        flag = -1
        if cur_name == actors[3]:
            flag = 1
        elif cur_name == actors[5]:
            flag = 0

        if flag != -1:
            im = imread(directory+"/"+image)
            im = im.reshape(1024)
            im = im.astype(float)
            im = im/255
            x_temp.append(im)
            y_temp.append(flag)
            total += 1
            result = classifier_p3(im, theta)
            if result == cur_name:
                correct += 1

    y_temp = np.array(y_temp)
    x_temp = np.array(x_temp)
    x_temp = np.vstack((np.ones((1, x_temp.T.shape[1])), x_temp.T))
    cost = f_p3(x_temp,y_temp,theta)
    print directory + ' tested'
    print "Cost function's value is " + str(cost)
    print str(correct) + "/" + str(total) + " is correct. The accuracy rate is " + str(correct/total*100) + "%."

def part_4_a(theta_p3):
    theta_1 = np.delete(theta_p3, 0).reshape((32, 32))
    plt.imshow(theta_1, cmap="RdBu")
    # plt.show()
    plt.imsave('p4_1.jpg', theta_1, cmap="RdBu")
    x_p4 = np.vstack((x_p3[0:2], x_p3[70:72]))
    y_p4 = np.hstack((y_p3[0:2], y_p3[70:72]))
    init_t = np.zeros(1025)
    theta_2 = grad_descent(f_p3,df_p3,x_p4,y_p4,init_t,0.00001,50000)
    theta_2 = np.delete(theta_2, 0).reshape((32, 32))
    plt.imshow(theta_2, cmap="RdBu")
    # plt.show()
    plt.imsave('p4_2.jpg', theta_2, cmap="RdBu")

def part_4_b():
    init_t = np.full(1025, 0, float)
    theta = grad_descent(f_p3, df_p3, x_p3, y_p3, init_t, 0.00001,1000)
    theta = np.delete(theta, 0).reshape((32, 32))
    plt.imshow(theta, cmap="RdBu")
    # plt.show()
    plt.imsave('p4_3.jpg', theta, cmap="RdBu")

def init_p5(directory):
    images = os.listdir(directory)
    images.sort()
    male_temp = []
    female_temp = []
    test_temp = []
    answer_temp = []
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)
        flag = -1
        if cur_name in male:
            flag = 1
        elif cur_name in female:
            flag = 0

        if flag != -1:
            im = imread(directory + "/" + image)
            im = im.reshape(1024)
            im = im.astype(float)
            im = im / 255
            if cur_name in actors:
                if flag == 1:
                    male_temp.append(im)
                else:
                    female_temp.append(im)
            else:
                test_temp.append(im)
                answer_temp.append(flag)

    global male_p5, female_p5, test_p5, answer_p5
    male_p5 = np.array(male_temp)
    female_p5 = np.array(female_temp)
    test_p5 = np.array(test_temp)
    answer_p5 = np.array(answer_temp)

def tester_p5(theta, test, answer):
    correct = 0.0
    total = len(test)
    for i in range(len(test)):
        x = np.hstack((np.ones(1), test[i]))
        h_x = np.dot(theta.T, x)
        result = -1
        if h_x > 0.5:
            result = 1
        else:
            result = 0
        if result == answer[i]:
            correct += 1

    x = np.vstack((np.ones((1, test.T.shape[1])), test.T))
    cost = f_p3(x,answer,theta)
    print "Cost function's value is " + str(cost)
    print str(correct) + "/" + str(total) + " is correct. The accuracy rate is " + str(correct/total*100) + "%."
    return correct/total*100

def part_5():
    init_p5('cropped')
    size_lst = []
    perf_valid_lst = []
    perf_train_lst = []
    for size in range(50, 251, 50):
        cur_x = np.vstack((male_p5[:size],female_p5[:size]))
        cur_y = np.hstack((np.ones(size),np.zeros(size)))
        init_t = np.zeros(1025)
        if size < 150:
            alpha = 0.00001
        else:
            alpha = 0.000005
        theta = grad_descent(f_p3, df_p3, cur_x, cur_y, init_t, alpha, 100000)
        print 'size '+ str(size*2) + ' tested'
        perf_training = tester_p5(theta, cur_x, cur_y)
        perf_train_lst.append(perf_training)
        perf_validation = tester_p5(theta, test_p5, answer_p5)
        perf_valid_lst.append(perf_validation)
        size_lst.append(size*2)

    print perf_train_lst
    print perf_valid_lst
    print size_lst
    plt.clf()
    plt.plot(size_lst, perf_train_lst)
    plt.plot(size_lst, perf_valid_lst)
    for a, b in zip(size_lst, perf_valid_lst):
        plt.text(a, b, format(b, '.2f'))
    plt.legend(['Training Set', 'Validation Set'], loc='upper left')
    plt.axis([0, 540, 0, 110])
    plt.ylabel('percentage')
    plt.xlabel('training set size')
    plt.savefig('p5.jpg')
    plt.clf()



def f_p6(x, y, theta):
    return np.sum((np.dot(theta.T, x)- y) ** 2)

def df_p6(x, y, theta):
    return 2 * (np.dot(x, (np.dot(theta.T, x)- y).T))

def part_6():
    np.random.seed(0)
    x = np.random.rand(5,10)
    y = np.random.rand(3,10)
    theta = np.random.rand(5,3)
    h = 0.0000001
    h_array = np.zeros((5,3))
    for i in range(5):
        h_array[i][1] = h
        print 'changing theta at [' + str(i)+', ' +  '1]'
        print (f_p6(x, y, theta+np.array(h_array)) - f_p6(x, y, theta-np.array(h_array)))/(2*h)
        h_array[i][1] = 0.0
    print df_p6(x, y, theta)


def init_part_7(directory):
    actor_num = len(actors)
    for i in range(actor_num):
        cur_y = [0]*actor_num
        cur_y[i] = 1
        actor_pattern[actors[i]] = cur_y

    images = os.listdir(directory)
    images.sort()
    x_temp = []
    y_temp = []
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)

        im = imread(directory + "/" + image)
        im = im.reshape(1024)
        im = im.astype(float)
        im = im / 255

        x_temp.append(im)
        y_temp.append(actor_pattern[cur_name])

    global x_p7,y_p7
    x_p7 = np.array(x_temp)
    y_p7 = np.array(y_temp).T

def tester_p7(directory, theta):
    images = os.listdir(directory)
    images.sort()
    correct = 0.0
    total = 0.0
    x_temp = []
    y_temp = []
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)

        im = imread(directory + "/" + image)
        im = im.reshape(1024)
        im = im.astype(float)
        im = im / 255
        x_temp.append(im)
        y_temp.append(actor_pattern[cur_name])
        total += 1
        x = np.hstack((np.ones(1), im))
        h_x = np.dot(theta.T, x)
        if actors[np.argmax(h_x)] == cur_name:
            correct += 1

    y_temp = np.array(y_temp).T
    x_temp = np.array(x_temp)
    x_temp = np.vstack((np.ones((1, x_temp.T.shape[1])), x_temp.T))
    cost = f_p6(x_temp,y_temp,theta)
    print directory + ' tested'
    print "Cost function's value is " + str(cost)
    print str(correct) + "/" + str(total) + " is correct. The accuracy rate is " + str(correct/total*100) + "%."

def part_7():
    init_part_7('sets/training_set')
    init_t = np.zeros((1025,len(actors)))
    theta = grad_descent(f_p6, df_p6, x_p7, y_p7, init_t, 0.000001, 20000)
    # np.savetxt('theta_P7.txt', theta)
    # theta = np.loadtxt('theta_P7.txt')
    tester_p7('sets/training_set', theta)
    tester_p7('sets/validation_set', theta)
    return theta

def part_8(theta):
    for i in range(theta.shape[1]):
        cur_theta = theta[:,i]
        cur_theta = np.delete(cur_theta, 0).reshape((32, 32))
        plt.imshow(cur_theta, cmap="RdBu")
        # plt.show()
        plt.imsave('p8_'+str(i)+'.jpg', cur_theta, cmap="RdBu")





if __name__ == "__main__":
    # print 'Seperating Sets'
    # seperate_sets('cropped')
    # print 'Doing part3'
    # init_grad_descent_p3('sets/training_set')
    # init_t = np.zeros(1025)
    # theta_p3 = grad_descent(f_p3,df_p3,x_p3,y_p3,init_t,0.00002, 20000)
    # # np.savetxt('theta_P3.txt', theta_p3)
    # #theta_p3 = np.loadtxt('theta_P3.txt')
    # tester_p3('sets/training_set', theta_p3)
    # tester_p3('sets/validation_set', theta_p3)
    # print '==============================================================='
    # print 'Doing part4'
    # part_4_a(theta_p3)
    # part_4_b()
    # print '==============================================================='
    # print 'Doing part5'
    # part_5()
    # print '==============================================================='
    # print 'Doing part6'
    # part_6()
    # print '==============================================================='
    print 'Doing part7'
    theta = part_7()
    print '==============================================================='
    print 'Doing part8'
    part_8(theta)
    print '==============================================================='

