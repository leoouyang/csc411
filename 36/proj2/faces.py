from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
from scipy.misc import imread
from scipy.misc import imresize
import re

import time
# def gray2rgb(im):
#     result = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
#     result[:, :, 0] = im
#     result[:, :, 1] = im
#     result[:, :, 2] = im
#     return result

def get_sets(directory, im_size = 32):
    actors = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
    actor_pattern = {}
    counter = {}
    for a in actors:
        counter[a] = 0

    actor_num = len(actors)
    for i in range(actor_num):
        cur_y = np.zeros(actor_num)
        cur_y[i] = 1
        actor_pattern[actors[i]] = cur_y

    images = os.listdir(directory)
    images.sort()

    training = np.zeros((0,im_size*im_size*3))
    training_ans = np.zeros((0,6))
    test = np.zeros((0,im_size*im_size*3))
    test_ans = np.zeros((0,6))
    validation = np.zeros((0,im_size*im_size*3))
    validation_ans = np.zeros((0,6))

    np.random.seed(0)
    np.random.shuffle(images)
    test_file = []
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)

        try:
            im = imread(directory + "/" + image, mode="RGB")
        except OSError:
            continue
        im = imresize(im, (im_size, im_size))
        im = im.reshape(im_size*im_size*3)
        im = im / 255.0

        counter[cur_name] += 1
        cur_num = counter[cur_name]
        if cur_name == 'gilpin':
            if cur_num <= 18:
                test = np.vstack((test,im))
                test_ans = np.vstack((test_ans,actor_pattern[cur_name]))
                test_file.append(image)
            elif cur_num <= 27:
                validation = np.vstack((validation, im))
                validation_ans = np.vstack((validation_ans, actor_pattern[cur_name]))
            elif cur_num <= 87:
                training = np.vstack((training, im))
                training_ans = np.vstack((training_ans, actor_pattern[cur_name]))
        else:
            if cur_num <= 20:
                test = np.vstack((test,im))
                test_ans = np.vstack((test_ans,actor_pattern[cur_name]))
                test_file.append(image)
            elif cur_num <= 30:
                validation = np.vstack((validation, im))
                validation_ans = np.vstack((validation_ans, actor_pattern[cur_name]))
            elif cur_num <= 100:
                training = np.vstack((training, im))
                training_ans = np.vstack((training_ans, actor_pattern[cur_name]))
    test_file.sort()
    # print(test_file)
    return training, training_ans, test, test_ans, validation, validation_ans


def perform(model, x, y):
    return np.mean(np.argmax(model(x).data.numpy(), 1) == np.argmax(y, 1))


def train(data, max_iter, batch_size = 100, activation = torch.nn.ReLU(), hidden_unit = 12, init_v = 0.01, init_m = 0, bias = 1,  plotCurve = True):
    train_x, train_y, test_x, test_y, validation_x, validation_y = data

    dim_x = train_x.shape[1]
    dim_h = hidden_unit
    dim_out = 6
    sample_size = train_x.shape[0]

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_V = Variable(torch.from_numpy(train_x), requires_grad=False).type(
        dtype_float)
    test_V = Variable(torch.from_numpy(test_x), requires_grad=False).type(
        dtype_float)
    validation_V = Variable(torch.from_numpy(validation_x), requires_grad=False).type(
        dtype_float)


    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        activation,
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()
#do the mini-batch
    torch.manual_seed(0)
    model[0].weight.data.normal_(init_m, init_v)
    model[0].bias.data.fill_(bias)
    model[2].weight.data.uniform_(init_m, init_v)
    model[2].bias.data.fill_(bias)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    epoch = [0]
    train_perform = [perform(model, train_V, train_y)]
    validation_perform = [perform(model, validation_V, validation_y)]

    np.random.seed(0)
    for t in range(max_iter):
        idx = np.random.permutation(range(train_x.shape[0]))
        cur_pos = 0
        while cur_pos != sample_size:
            next_pos = min(cur_pos + batch_size, sample_size)
            mini_batch_x = train_x[idx[cur_pos:next_pos]]
            mini_batch_y = train_y[idx[cur_pos:next_pos]]
            cur_pos = next_pos

            x = Variable(torch.from_numpy(mini_batch_x), requires_grad=False).type(
                dtype_float)
            y_classes = Variable(torch.from_numpy(np.argmax(mini_batch_y, 1)),
                                 requires_grad=False).type(dtype_long)
            y_pred = model(x)
            loss = loss_fn(y_pred, y_classes)
            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()  # Compute the gradient
            optimizer.step()  # Use the gradient information to  make a step

        if plotCurve:
            epoch.append(t+1)
            train_perform.append(perform(model, train_V, train_y))
            validation_perform.append(perform(model, validation_V, validation_y))
        if t%100 == 0:
            print("Epochs: "+ str(t))


    print("Test Set performance : " + str(perform(model, test_V, test_y)))
    print("Training Set performance : " + str(perform(model, train_V, train_y)))
    print("Validation Set performance : " + str(perform(model, validation_V, validation_y)))

    if plotCurve:
        plt.plot(epoch, train_perform)
        plt.plot(epoch, validation_perform)
        plt.legend(['Training Set', 'Validation Set'],
                   loc='upper right')
        plt.axis([-30, max_iter, 0, 1.1])
        plt.ylabel('Prop. of correct guesses')
        plt.xlabel('Epochs')
        plt.show()
    return model

def part9(model):
    bracco_w = model[2].weight.data.numpy()[1, :].copy()
    f, axarr = plt.subplots(1, 5)
    plt.suptitle("Five most useful weights for Bracco")
    for i in range(5):
        index = np.argmax(bracco_w)
        bracco_w[index] = float('-inf')
        weight = model[0].weight.data.numpy()[index, :].reshape((32,32,3))
        weight = weight[:,:,0] + weight[:,:,1] + weight[:,:,2]
        axarr[i].imshow(weight, cmap = "coolwarm")
        axarr[i].set_title(str(i))
    plt.show()

    baldwin_w = model[2].weight.data.numpy()[3, :].copy()
    f, axarr = plt.subplots(1, 5)
    plt.suptitle("Five most useful weights for Baldwin")
    for i in range(5):
        index = np.argmax(baldwin_w)
        baldwin_w[index] = float('-inf')
        weight = model[0].weight.data.numpy()[index, :].reshape((32,32,3))
        weight = weight[:,:,0] + weight[:,:,1] + weight[:,:,2]
        axarr[i].imshow(weight, cmap = "coolwarm")
        axarr[i].set_title(str(i))
    plt.show()

if __name__ == "__main__":
    data = get_sets("cropped")
    start = time.time()
    model = train(data, 10000, plotCurve=False)
    end = time.time()
    print(end - start)
    # model = train(data, 900)
    # print("Using Tanh")
    # train(data, 900, bias = 0, init_v= 0.01, activation=torch.nn.Tanh(),plotCurve=False)
    # print("Using batch size 50")
    # train(data, 900, batch_size=50, plotCurve=False)
    # for i in range(12,25,3):
    #     print("Using "+ str(i) + " neurons in hidden layer")
    #     train(data, 900, hidden_unit=i, plotCurve=False)
    # # torch.save(model, "model.txt")
    # # model = torch.load("model.txt")
    # print("========================= Part 9 =========================")
    # part9(model)



    # f, axarr = plt.subplots(3, 6)
    # for i in range(12):
    #     weight = model[0].weight.data.numpy()[i, :].reshape((32, 32, 3))
    #     final = weight[:,:,0] + weight[:,:,1] + weight[:,:,2]
    #     axarr[i//6, i%6].imshow(final, cmap = "coolwarm")
    # plt.show()
