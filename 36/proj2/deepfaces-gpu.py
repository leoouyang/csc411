import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

import torch.nn as nn
import os
import re

import time

class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x


def get_sets(directory, im_size = 227):
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

    # training = np.zeros((410,3 ,im_size, im_size))
    # training_ans = np.zeros((410,6))
    # test = np.zeros((118,3 ,im_size, im_size))
    # test_ans = np.zeros((118,6))
    # validation = np.zeros((59,3, im_size, im_size))
    # validation_ans = np.zeros((59,6))

    training = []
    training_ans = []
    test = []
    test_ans = []
    validation = []
    validation_ans = []

    np.random.seed(0)
    np.random.shuffle(images)
    for image in images:
        cur_name = re.search('[a-z]+', image).group(0)

        try:
            im = imread(directory + "/" + image, mode="RGB")
        except OSError:
            continue
        im = imresize(im, (im_size, im_size))
        im = im - np.mean(im.flatten())
        im = im / np.max(np.abs(im.flatten()))

        im = np.rollaxis(im, -1).astype(np.float32)

        counter[cur_name] += 1
        cur_num = counter[cur_name]
        if cur_name == 'gilpin':
            if cur_num <= 18:
                test.append(im)
                test_ans.append(actor_pattern[cur_name])
            elif cur_num <= 27:
                validation.append(im)
                validation_ans.append(actor_pattern[cur_name])
            elif cur_num <= 87:
                training.append(im)
                training_ans.append(actor_pattern[cur_name])
        else:
            if cur_num <= 20:
                test.append(im)
                test_ans.append(actor_pattern[cur_name])
            elif cur_num <= 30:
                validation.append(im)
                validation_ans.append(actor_pattern[cur_name])
            elif cur_num <= 100:
                training.append(im)
                training_ans.append(actor_pattern[cur_name])
    return np.array(training), np.array(training_ans), np.array(test), np.array(
        test_ans), np.array(validation), np.array(validation_ans)


def perform(model, x, y):
    return np.mean(np.argmax(model(x).data.cpu().numpy(), 1) == np.argmax(y, 1))


def train(data, max_iter, batch_size=100, activation=torch.nn.ReLU(),
          hidden_unit=12, init_v=0.01, bias = 1):
    train_x, train_y, test_x, test_y, validation_x, validation_y = data

    dim_x = train_x.shape[1]
    dim_h = hidden_unit
    dim_out = 6
    sample_size = train_x.shape[0]

    dtype_float = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor

    train_V = Variable(torch.from_numpy(train_x), requires_grad=False).type(
        dtype_float)
    test_V = Variable(torch.from_numpy(test_x), requires_grad=False).type(
        dtype_float)
    validation_V = Variable(torch.from_numpy(validation_x),
                            requires_grad=False).type(
        dtype_float)

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        activation,
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    torch.manual_seed(0)
    model[0].weight.data.normal_(0.0, init_v)
    model[0].bias.data.fill_(bias)
    model[2].weight.data.normal_(0.0, init_v)
    model[2].bias.data.fill_(bias)

    model.cuda()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    np.random.seed(0)
    x = Variable(torch.from_numpy(train_x),
                 requires_grad=False).type(
        dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)),
                         requires_grad=False).type(dtype_long)
    for t in range(max_iter):
        idx = np.random.permutation(range(train_x.shape[0]))
        cur_pos = 0
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to  make a step
        if t % 1000 == 0:
            print("Epochs: " + str(t))

    print("Test Set performance : " + str(perform(model, test_V, test_y)))
    print("Training Set performance : " + str(perform(model, train_V, train_y)))
    print("Validation Set performance : " + str(
        perform(model, validation_V, validation_y)))
    return model


def preprocess(data, model):
    sample_size = data.shape[0]
    result = np.zeros((0, 9216))
    cur_pos = 0
    while cur_pos != sample_size:
        next_pos = min(cur_pos + 40, sample_size)
        set_v = Variable(torch.from_numpy(data[cur_pos:next_pos]),
                         requires_grad=False)
        result = np.vstack((result, model.forward(set_v).data.numpy()))
        cur_pos = next_pos
    return result


if __name__ == "__main__":
    model = MyAlexNet()
    model.eval()

    print("Preparing data")
    train_x, train_y, test_x, test_y, validation_x, validation_y = get_sets(
        "cropped")
    train_x_in = preprocess(train_x, model)
    test_x_in = preprocess(test_x, model)
    validation_x_in = preprocess(validation_x, model)
    start = time.time()
    mymodel = train(
        (train_x_in, train_y, test_x_in, test_y, validation_x_in, validation_y),
        30000)
    end = time.time()
    print(end-start)
