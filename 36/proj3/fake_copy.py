import random
import math
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# import graphviz
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

Data = {}
Sets = {}
Count = {}
word2num = {}

FAKE = "fake"
REAL = "real"
TRAINING_FAKE = "training_fake"
TRAINING_REAL = "training_real"
TEST_FAKE = "test_fake"
TEST_REAL = "test_real"
VALIDATION_FAKE = "validation_fake"
VALIDATION_REAL = "validation_real"


def read_data():
    fake = open("clean_fake.txt", "r")
    fake_l = fake.read().splitlines()
    Data[FAKE] = fake_l
    real = open("clean_real.txt", "r")
    real_l = real.read().splitlines()
    Data[REAL] = real_l
    fake.close()
    real.close()


def separate_sets():
    random.seed(0)
    fake = Data[FAKE].copy()
    real = Data[REAL].copy()

    random.shuffle(fake)
    random.shuffle(real)

    training_fake, test_fake, validation_fake = fake[0:int(0.7 * len(fake))], fake[int(0.7 * len(fake)):int(
        0.85 * len(fake))], fake[int(0.85 * len(fake)):]
    training_real, test_real, validation_real = real[0:int(0.7 * len(real))], real[int(0.7 * len(real)):int(
        0.85 * len(real))], real[int(0.85 * len(real)):]

    Sets[TRAINING_FAKE], Sets[TEST_FAKE], Sets[VALIDATION_FAKE] = training_fake, test_fake, validation_fake
    Sets[TRAINING_REAL], Sets[TEST_REAL], Sets[VALIDATION_REAL] = training_real, test_real, validation_real


def words_counts(real, fake):
    count = {}
    for headline in real:
        for word in set(headline.split()):
            if word in count:
                count[word][REAL] += 1
            else:
                count[word] = {REAL: 1, FAKE: 0}

    for headline in fake:
        for word in set(headline.split()):
            if word in count:
                count[word][FAKE] += 1
            else:
                count[word] = {REAL: 0, FAKE: 1}

    return count


def find_sensitive_word(count, real_count, fake_count, key=3):
    result = []
    for word in count:
        if count[word][REAL] == 0:
            rcount = 1
        else:
            rcount = count[word][REAL]
        if count[word][FAKE] == 0:
            fcount = 1
        else:
            fcount = count[word][FAKE]

        real_ratio = rcount / real_count
        fake_ratio = fcount / fake_count
        if real_ratio / fake_ratio > key or fake_ratio / real_ratio > key:
            result.append((word, real_ratio, fake_ratio))
    return result


def how_often(word):
    # real_count = len(Data[REAL])
    # fake_count = len(Data[FAKE])
    # count = words_counts(Data[REAL], Data[FAKE])
    # result = find_sensitive_word(count, real_count, fake_count, 40)
    # print(result)  # ban, korea, black, breaking

    total = 0
    count = 0
    for headline in Data[REAL]:
        if word in headline:
            count += 1
        total += 1
    real_ratio = count / total * 100

    total = 0
    count = 0
    for headline in Data[FAKE]:
        if word in headline:
            count += 1
        total += 1
    fake_ratio = count / total * 100

    return real_ratio, fake_ratio


def part1(words):
    print("========================== Part 1 ==========================")
    for word in words:
        real_ratio, fake_ratio = how_often(word)
        print('Word \"' + word + '\" appears ' + str(format(real_ratio, '.3f')) +
              '% in real news and ' + str(format(fake_ratio, '.3f')) + '% in fake news.')


def multi(nums):
    result = math.exp(sum(map(lambda x: math.log(x), nums)))
    return result


def init_naive_bayes_classifier(m, p):
    fake_len = len(Sets[TRAINING_FAKE])
    real_len = len(Sets[TRAINING_REAL])

    p_fake = fake_len / (fake_len + real_len)
    p_real = 1 - p_fake

    p_words = {}
    for key in Count:
        cur_dict = Count[key]
        p_words[key] = {}
        p_words[key][REAL] = (cur_dict[REAL] + m * p) / (real_len + m)
        p_words[key][FAKE] = (cur_dict[FAKE] + m * p) / (fake_len + m)
    return p_fake, p_real, p_words


def naive_bayes_classifier(p_fake, p_real, p_words, headline):
    """
    return true is the news is real, return false if it is fake
    """
    probs_fake = []
    probs_real = []
    words = headline.split()

    for word, cur_dict in p_words.items():
        if word in words:
            probs_fake.append(cur_dict[FAKE])
            probs_real.append(cur_dict[REAL])
        else:
            probs_fake.append(1 - cur_dict[FAKE])
            probs_real.append(1 - cur_dict[REAL])

    p_words_fake = multi(probs_fake)
    p_fake_words = p_words_fake * p_fake

    p_words_real = multi(probs_real)
    p_real_words = p_words_real * p_real

    return p_fake_words < p_real_words


def p2_helper(key, p_fake, p_real, p_words, TF):
    """
    if the set is for real news, TF is True, if the set if for
    fake news, TF is False.
    """
    correct = 0
    for line in Sets[key]:
        if naive_bayes_classifier(p_fake, p_real, p_words, line) is TF:
            correct += 1
    return len(Sets[key]), correct


def tune_func_p2(m, p):
    p_fake, p_real, p_words = init_naive_bayes_classifier(m, p)

    validation_real_total, validation_real_corr = p2_helper(VALIDATION_REAL, p_fake, p_real, p_words, True)
    validation_fake_total, validation_fake_corr = p2_helper(VALIDATION_FAKE, p_fake, p_real, p_words, False)
    performance = (validation_real_corr + validation_fake_corr) / (validation_real_total + validation_fake_total)
    print("Value for m, p is:" + str((m, p)))
    print("Performance on validation set is " + str(performance) + "%")
    return performance


def tune_p2():
    result = {}
    for m in range(1, 4):
        for p in range(1, 11):
            performance = tune_func_p2(m, p / 10)
            result[(m, p / 10)] = performance
    return max(result, key=result.get)


def part2(m, p):
    p_fake, p_real, p_words = init_naive_bayes_classifier(m, p)

    print("Value for m, p is:" + str((m, p)))

    train_real_total, train_real_corr = p2_helper(TRAINING_REAL, p_fake, p_real, p_words, True)
    train_fake_total, train_fake_corr = p2_helper(TRAINING_FAKE, p_fake, p_real, p_words, False)
    print("Performance on training set is " + str(
        (train_real_corr + train_fake_corr) / (train_real_total + train_fake_total)) + "%")

    test_real_total, test_real_corr = p2_helper(TEST_REAL, p_fake, p_real, p_words, True)
    test_fake_total, test_fake_corr = p2_helper(TEST_FAKE, p_fake, p_real, p_words, False)
    print("Performance on test set is " + str(
        (test_real_corr + test_fake_corr) / (test_real_total + test_fake_total)) + "%")

    validation_real_total, validation_real_corr = p2_helper(VALIDATION_REAL, p_fake, p_real, p_words, True)
    validation_fake_total, validation_fake_corr = p2_helper(VALIDATION_FAKE, p_fake, p_real, p_words, False)
    print("Performance on validation set is " + str(
        (validation_real_corr + validation_fake_corr) / (validation_real_total + validation_fake_total)) + "%")


def part3(m, p):
    # P(label|word) = P(word & label)/P(word) = count(word in label set) + mp/count(word in both set) + m

    p_label_real_word_1 = {}
    p_label_fake_word_1 = {}
    p_label_real_word_0 = {}
    p_label_fake_word_0 = {}

    fake_count = len(Sets[TRAINING_FAKE])
    real_count = len(Sets[TRAINING_REAL])
    t_count = fake_count + real_count
    for key in Count:
        cur_dict = Count[key]
        total_count = cur_dict[REAL] + cur_dict[FAKE]

        p_label_real_word_1[key] = (cur_dict[REAL] + m * p) / (total_count + m)
        p_label_fake_word_1[key] = (cur_dict[FAKE] + m * p) / (total_count + m)
        p_label_real_word_0[key] = (real_count - cur_dict[REAL] + m * p) / (t_count - total_count + m)
        p_label_fake_word_0[key] = (fake_count - cur_dict[FAKE] + m * p) / (t_count - total_count + m)

    sorted_real_1 = sorted(p_label_real_word_1, key=p_label_real_word_1.get, reverse=True)
    sorted_fake_1 = sorted(p_label_fake_word_1, key=p_label_fake_word_1.get, reverse=True)
    sorted_real_0 = sorted(p_label_real_word_0, key=p_label_real_word_0.get, reverse=True)
    sorted_fake_0 = sorted(p_label_fake_word_0, key=p_label_fake_word_0.get, reverse=True)
    print("========================== Part 3a ==========================")
    print("10 words whose presence most strongly predicts that the news is real")
    result = []
    for word in sorted_real_1[:10]:
        result.append(word)
    print(result)

    print("10 words whose absence most strongly predicts that the news is real")
    result = []
    for word in sorted_real_0[:10]:
        result.append(word)
    print(result)

    print("10 words whose presence most strongly predicts that the news is fake")
    result = []
    for word in sorted_fake_1[:10]:
        result.append(word)
    print(result)

    print("10 words whose absence most strongly predicts that the news is fake")
    result = []
    for word in sorted_fake_0[:10]:
        result.append(word)
    print(result)

    print("========================== Part 3b ==========================")
    print("10 words whose presence most strongly predicts that the news is real")
    result = []
    i = 0
    while len(result) < 10:
        if not sorted_real_1[i] in ENGLISH_STOP_WORDS:
            result.append(sorted_real_1[i])
        i += 1
    print(result)

    print("10 words whose absence most strongly predicts that the news is real")
    result = []
    i = 0
    while len(result) < 10:
        if not sorted_real_0[i] in ENGLISH_STOP_WORDS:
            result.append(sorted_real_0[i])
        i += 1
    print(result)

    print("10 words whose presence most strongly predicts that the news is fake")
    result = []
    i = 0
    while len(result) < 10:
        if not sorted_fake_1[i] in ENGLISH_STOP_WORDS:
            result.append(sorted_fake_1[i])
        i += 1
    print(result)

    print("10 words whose absence most strongly predicts that the news is fake")
    result = []
    i = 0
    while len(result) < 10:
        if not sorted_fake_0[i] in ENGLISH_STOP_WORDS:
            result.append(sorted_fake_0[i])
        i += 1
    print(result)


def perform(model, x, y):
    y_pred = np.argmax(model(x).data.numpy(), 1)
    return np.mean(y_pred == y)


def train_LR(max_iter=500, plotCurve=False, init_v=0.01, init_m=0, bias=1):
    train_fake_x, train_fake_y = vectorize_set(TRAINING_FAKE, False)
    train_real_x, train_real_y = vectorize_set(TRAINING_REAL, True)
    train_x = np.vstack((train_real_x, train_fake_x))
    train_v = Variable(torch.from_numpy(train_x), requires_grad=False).type(torch.FloatTensor)
    train_y = np.hstack((train_real_y, train_fake_y))

    test_fake_x, test_fake_y = vectorize_set(TEST_FAKE, False)
    test_real_x, test_real_y = vectorize_set(TEST_REAL, True)
    test_x = np.vstack((test_real_x, test_fake_x))
    test_v = Variable(torch.from_numpy(test_x), requires_grad=False).type(torch.FloatTensor)
    test_y = np.hstack((test_real_y, test_fake_y))

    valid_fake_x, valid_fake_y = vectorize_set(VALIDATION_FAKE, False)
    valid_real_x, valid_real_y = vectorize_set(VALIDATION_REAL, True)
    valid_x = np.vstack((valid_real_x, valid_fake_x))
    valid_v = Variable(torch.from_numpy(valid_x), requires_grad=False).type(torch.FloatTensor)
    valid_y = np.hstack((valid_real_y, valid_fake_y))

    x = Variable(torch.from_numpy(train_x), requires_grad=False).type(torch.FloatTensor)
    y_classes = Variable(torch.from_numpy(train_y), requires_grad=False).type(torch.LongTensor)

    model = nn.Linear(len(word2num), 2)
    torch.manual_seed(0)
    model.weight.data.normal_(init_m, init_v)
    model.bias.data.fill_(bias)
    learning_rate = 7e-5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)

    iter = []
    train_perform = []
    validation_perform = []

    iter_plt = []
    train_perform_plt = []
    validation_perform_plt = []

    for t in range(max_iter + 1):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to make a step

        perform_t = perform(model, train_v, train_y)
        perform_v = perform(model, valid_v, valid_y)

        train_perform.append(perform_t)
        validation_perform.append(perform_v)
        iter.append(t)

        if t % (max_iter/10) == 0:
            train_perform_plt.append(perform_t)
            validation_perform_plt.append(perform_v)
            iter_plt.append(t)
            print('Epochs: ' + str(t))
            print("Training Set performance : " + str(perform_t))
            print("Validation Set performance : " + str(perform_v))
            print("\n")

    print("Test Set performance : " + str(perform(model, test_v, test_y)))

    if plotCurve:
        plt.plot(iter, train_perform)
        plt.plot(iter, validation_perform)
        for a, b in zip(iter_plt, train_perform_plt):
            plt.text(a, b, format(b, ".3f"))
        for a, b in zip(iter_plt, validation_perform_plt):
            plt.text(a, b, format(b, ".3f"))
        plt.legend(["Training Set", "Validation Set"],
                   loc="lower right")
        plt.axis([0, max_iter, 0.4, 1.1])
        plt.ylabel("Prop. of correct predictions")
        plt.xlabel("Iteration")
        plt.show()

    return model


def part4():
    model = train_LR(2000, True)
    np.save("LR_model.npy", model.weight.data.numpy())
    return 0


def part6():
    weight = np.load("LR_model.npy")
    thetas = weight[1] - weight[0]

    top_positive = []
    top_negative = []
    top_theta_pos = []
    top_theta_neg = []

    if not word2num:
        words = []
        for headline in Sets[TRAINING_FAKE]:
            for word in set(headline.split()):
                if word not in words:
                    words.append(word)

        for headline in Sets[TRAINING_REAL]:
            for word in set(headline.split()):
                if word not in words:
                    words.append(word)
        words.sort()
        random.seed(0)
        random.shuffle(words)
        for i in range(len(words)):
            word2num[words[i]] = i

    for i in range(len(ENGLISH_STOP_WORDS) + 10):
        max = np.argmax(thetas)
        top_positive.append(max)
        top_theta_pos.append(thetas[max])
        thetas[max] = 0
        min = np.argmin(thetas)
        top_negative.append(min)
        top_theta_neg.append(thetas[min])
        thetas[min] = 0

    print("========================== Part 6a ==========================")

    print("TOP 10 positive thetas with stop words: ")
    words = []
    count = 1
    for i, theta in enumerate(top_positive):
        for word, index in word2num.items():
            if index == theta:
                words.append(word)
                print("\t" + str(count) + " : \"" + word + "\"   with theta " + str(format(top_theta_pos[i], '.4f')))
                count += 1
            if count > 10:
                break

    print("\nTOP 10 negative thetas with stop words: ")
    words = []
    count = 1
    for i, theta in enumerate(top_negative):
        for word, index in word2num.items():
            if index == theta:
                words.append(word)
                print("\t" + str(count) + " : \"" + word + "\"   with theta " + str(format(top_theta_neg[i], '.4f')))
                count += 1
            if count > 10:
                break

    print("\n========================== Part 6b ==========================")

    print("TOP 10 positive thetas without stop words: ")
    words = []
    count = 1
    for i, theta in enumerate(top_positive):
        for word, index in word2num.items():
            if index == theta and word not in ENGLISH_STOP_WORDS:
                words.append(word)
                print("\t" + str(count) + " : \"" + word + "\"   with theta " + str(format(top_theta_pos[i], '.4f')))
                count += 1
            if count > 10:
                break

    print("\nTOP 10 negative thetas without stop words: ")
    words = []
    count = 1
    for i, theta in enumerate(top_negative):
        for word, index in word2num.items():
            if index == theta and word not in ENGLISH_STOP_WORDS:
                words.append(word)
                print("\t" + str(count) + " : \"" + word + "\"   with theta " + str(format(top_theta_neg[i], '.4f')))
                count += 1
            if count > 10:
                break


def vectorize_set(set_key, TF):
    """

    :param set_key: the key of the set we want to vectorize
    :param TF: False for fake set, True for real set
    """
    if not word2num:
        words = []
        for headline in Sets[TRAINING_FAKE]:
            for word in set(headline.split()):
                if not word in words:
                    words.append(word)

        for headline in Sets[TRAINING_REAL]:
            for word in set(headline.split()):
                if not word in words:
                    words.append(word)
        words.sort()
        random.seed(0)
        random.shuffle(words)
        for i in range(len(words)):
            word2num[words[i]] = i
    x = []
    y = np.full(len(Sets[set_key]), int(TF))
    for headline in Sets[set_key]:
        cur_x = [0] * len(word2num)
        for word in headline.split():
            if word in word2num:
                cur_x[word2num[word]] = 1
        x.append(cur_x)
    x = np.array(x)
    return x, y


def decisiontree(max_depth, x, y):
    dt = DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=max_depth, min_samples_split=10,
                                max_features=0.1, random_state=0)
    # dt = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=max_depth, min_samples_split=5, max_features="log2", random_state=0)
    dt.fit(x, y)
    return dt


def part7a(max_depth, step, plotCurve=False):
    train_fake_x, train_fake_y = vectorize_set(TRAINING_FAKE, False)
    train_real_x, train_real_y = vectorize_set(TRAINING_REAL, True)
    train_x = np.vstack((train_real_x, train_fake_x))
    train_y = np.hstack((train_real_y, train_fake_y))

    test_fake_x, test_fake_y = vectorize_set(TEST_FAKE, False)
    test_real_x, test_real_y = vectorize_set(TEST_REAL, True)
    test_x = np.vstack((test_real_x, test_fake_x))
    test_y = np.hstack((test_real_y, test_fake_y))

    valid_fake_x, valid_fake_y = vectorize_set(VALIDATION_FAKE, False)
    valid_real_x, valid_real_y = vectorize_set(VALIDATION_REAL, True)
    valid_x = np.vstack((valid_real_x, valid_fake_x))
    valid_y = np.hstack((valid_real_y, valid_fake_y))

    depths = []
    validation_perform = []
    train_perform = []
    for depth in range(0, max_depth, step):
        depth = max(depth, 1)
        print("max_depth:", depth)
        depths.append(depth)
        dt = decisiontree(depth, train_x, train_y)
        train_perform.append(dt.score(train_x, train_y))
        validation_perform.append(dt.score(valid_x, valid_y))

    if plotCurve:
        plt.plot(depths, train_perform)
        plt.plot(depths, validation_perform)
        for a, b in zip(depths, validation_perform):
            plt.text(a, b, format(b, ".3f"))
        plt.legend(["Training Set", "Validation Set"],
                   loc="upper right")
        plt.axis([0, max_depth, 0, 1.1])
        plt.ylabel("Prop. of correct predictions")
        plt.xlabel("Max Depth")
        plt.show()

    best_depth = depths[validation_perform.index(max(validation_perform))]
    print("Max Depth =", best_depth, "return the best result of", max(validation_perform), "on validation set")
    dt = decisiontree(best_depth, train_x, train_y)
    train_performance = dt.score(train_x, train_y)
    test_performance = dt.score(test_x, test_y)
    print("The performance on training set is", train_performance)
    print("The performance on test set is", test_performance)
    return dt


def part7b(dt):
    words = sorted(word2num, key=word2num.get, reverse=True)
    words.reverse()
    # dot_data = tree.export_graphviz(dt, out_file=None, feature_names=words,
    #                      filled=True, rounded=True, class_names=[FAKE, REAL],
    #                      special_characters=True, max_depth=2, )
    # graph = graphviz.Source(dot_data, format="png")
    # graph.render("DT", view=True)


def PL(num):
    if num != 0:
        return (-num) * math.log(num, 2)
    else:
        return 0  # to account for the case where num is 0


def IY_x(word):
    m, p = 1, 0.4
    # H(Y) = Sum(-P(Y=y)logP(Y=y))for y=real,fake
    fake_count = len(Sets[TRAINING_FAKE])
    real_count = len(Sets[TRAINING_REAL])
    total_count = fake_count + real_count
    P_y_fake = fake_count / (fake_count + real_count)
    P_y_real = real_count / (fake_count + real_count)
    H_Y = PL(P_y_fake) + PL(P_y_real)
    # H(Y|xi) = P(xi=0)[-P(y=real|xi=0)logP(y=real|xi=0)+-P(y=fake|xi=0)logP(y=fake|xi=0)]+
    #           P(xi=1)[-P(y=real|xi=1)logP(y=real|xi=1)+-P(y=fake|xi=1)logP(y=fake|xi=1)]
    xi_count = Count[word][FAKE] + Count[word][REAL]
    p_xi_1 = xi_count / total_count
    p_xi_0 = 1 - p_xi_1
    p_y_fake_xi_1 = Count[word][FAKE] / xi_count
    p_y_real_xi_1 = Count[word][REAL] / xi_count
    if xi_count == total_count:
        p_y_fake_xi_0, p_y_real_xi_0 = 1, 1
    else:
        p_y_fake_xi_0 = (fake_count - Count[word][FAKE]) / (total_count - xi_count)
        p_y_real_xi_0 = (real_count - Count[word][REAL]) / (total_count - xi_count)
    H_Y_xi = p_xi_0 * (PL(p_y_real_xi_0) + PL(p_y_fake_xi_0)) + p_xi_1 * (PL(p_y_real_xi_1) + PL(p_y_fake_xi_1))
    # print(H_Y_xi)

    return H_Y - H_Y_xi


def part8():
    print("H(Y|'korea'):", IY_x("korea"))
    print("H(Y|'wall'):", IY_x("wall"))
    # result = {}
    # for key in Count:
    #     result[key] = IY_x(key)
    # result2 = []
    # sorted_l = sorted(result, key=result.get,
    #                      reverse=True)
    # for word in sorted_l[:10]:
    #     # result.append((word, p_label_word_real[word]))
    #     result2.append((word, result[word], Count[word]))
    # print(result2)


if __name__ == "__main__":
    read_data()
    separate_sets()
    Count = words_counts(Sets[TRAINING_REAL], Sets[TRAINING_FAKE])
    words = ['korea', 'black', 'ban']
    part1(words)
    m, p = tune_p2()
    m, p = 1, 0.4
    print("best performance with m = "+ str(m) + ", p = "+ str(p))
    part2(m, p)
    part3(m, p)
    part4()
    part6()
    dt = part7a(251, 10, True)
    part7b(dt)
    part8()
