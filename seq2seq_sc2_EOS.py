from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import json
import stream
import gflags as flags
import multiprocessing as mp
import subprocess
import signal
import numpy as np
from scipy import sparse
import gflags as flags
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
# use_cuda = False
print("torch.cuda.is_available():" + str(use_cuda))
MAX_LENGTH = 200


def unitRange(races):
    unitsinfo = {"PvP_P": [71, 317, 563], "ZvP_P": [71, 317, 1031], "ZvP_Z": [71, 785, 1031], "ZvZ_Z": [71, 785, 1499],
                 "PvT_P": [71, 317, 653], "PvT_T": [81, 417, 663]
        , "TvZ_Z": [71, 785, 1121], "TvZ_T": [81, 417, 1131], "TvT_T": [81, 417, 753]}
    return unitsinfo[races]


FLAGS = flags.FLAGS
flags.DEFINE_string(name='PATH',
                    default='/home/zd/MSC/parsed_replays/GlobalFeatureVector/Protoss_vs_Terran/Protoss/',
                    help='GlobalFeatureVector path')
flags.DEFINE_string(name='races', default='PvT_P',
                    help='own & enemy race: like PvT_P')
flags.DEFINE_string(name='model_path', default='./sc2/',
                    help='model_path')

FLAGS(sys.argv)


def prepareDataPVP():
    PATH = FLAGS.PATH
    races = FLAGS.races
    print(PATH)
    print(races)
    parents = os.listdir(PATH)
    AllUnits = []

    friend_and_enemy_range = unitRange(races)
    friend_unit_start_index = friend_and_enemy_range[0]
    enemy_unit_start_index = friend_and_enemy_range[1]
    frameslide = 200
    for p in parents:
        # for p in parents[:1000]:
        child = os.path.join(PATH, p)
        print(child)
        try:
            F = np.asarray(sparse.load_npz(child).todense())
        except:
            print("bad zip file: " + child)
            continue
        if F.shape[0] < frameslide:
            continue
        currF = F[:, friend_and_enemy_range[0]:friend_and_enemy_range[2]:6]
        temp = np.ndarray.tolist(currF)
        #EOS is the end tag
        EOS = [-1 for i in range(int((friend_and_enemy_range[2]-friend_and_enemy_range[0])/6))]
        EOS = np.array(EOS).reshape(1,int((friend_and_enemy_range[2]-friend_and_enemy_range[0])/6))

        temp = np.vstack((temp,EOS))
        temp = Variable(torch.FloatTensor(temp))
        temp = temp.cuda() if use_cuda else temp
        AllUnits.append((temp, temp))
    return AllUnits, races, friend_and_enemy_range


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        print("input_size:" + str(input_size))

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        embedded = input.view(1, 1, -1)
        output = embedded
        # output = input
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        # result = Variable(torch.zeros(1, 1, self.hidden_size))
        # result = Variable(torch.randn(1, 1, self.hidden_size))
        result = Variable(torch.zeros(1, 1, self.hidden_size).float())
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        #output = self.embedding(input).view(1, 1, -1)
        output = input.view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        # result = Variable(torch.zeros(1, 1, self.hidden_size))

        result = Variable(torch.zeros(1, 1, self.hidden_size).float())
        if use_cuda:
            return result.cuda()
        else:
            return result


teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    # print(input_length)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)

    #decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = Variable(torch.zeros(encoder.hidden_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # topv, topi = decoder_output.data.topk(1)
            # ni = topi[0][0]

            # decoder_input = Variable(torch.FloatTensor([[ni]]))
            decoder_input = decoder_output.data
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            ni = decoder_output.data[0]
            loss += criterion(decoder_output, target_variable[di])
            sum = 0
            for i in range(len(ni)):
                sum += (ni[i]-EOS_token[i])
            if abs(sum)<0.1:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.

def trainIters_sc2(enemyUnits, encoder, decoder, n_iters, print_every=10, plot_every=10,
                   learning_rate=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    lr_base = learning_rate
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr_base)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr_base)
    # training_pairs = [variablesFromPair(input_lang, output_lang, random.choice(pairs))
    #                  for i in range(n_iters)]
    training_pairs = enemyUnits
    criterion = nn.L1Loss()
    # criterion = nn.BCELoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        input_variable = input_variable.cuda() if use_cuda else input_variable
        target_variable = target_variable.cuda() if use_cuda else target_variable

        # print(input_variable)
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            # torch.save(encoder1, encode_model_path+"_"+str(iter))
            # torch.save(decoder1, decode_model_path+"_"+str(iter))
            torch.save(encoder1.state_dict(), encode_model_path[:-4] + "_" + str(iter) + ".pkl")
            torch.save(decoder1.state_dict(), decode_model_path[:-4] + "_" + str(iter) + ".pkl")

        if iter % (int(n_iters+1 / 100)) == 0 and lr_base > 0.01:
            lr_base -= 0.1
            if lr_base < 0.01:
                lr_base = 0.01
            print("lr: " + str(lr_base))
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr_base)
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr_base)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append((int(iter), plot_loss_avg))
            plot_loss_total = 0

    showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

def showPlot(points):
    plt.figure()
    x = []
    y = []
    for x_, y_ in points:
        x.append(x_)
        y.append(y_)
    plt.plot(x, y)
    plt.xlabel("iter")
    plt.ylabel("loss(%)")
    races = FLAGS.races
    # plt.show()
    plt.savefig(races + "_plot_loss.png")
    plt.close()


def encoder_sc2(enemyUnit, encoder):
    input_variable = enemyUnit
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
    return encoder_hidden


def encoderContext(Units, encoder):
    contextVector = []
    for unit in Units:
        # print(unit[0])
        encoder_output = encoder_sc2(unit[0], encoder)
        encoder_output = encoder_output.cpu()
        contextVector.append(encoder_output[0][0].data.numpy())
        # print(encoder_output)
    return contextVector


def decoder_sc2(enemyUnit, decoder):
    decoder_input = Variable(torch.zeros(decoder.hidden_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = enemyUnit
    input_length = decoder_hidden.size()[0]
    decoder_hiddens = []
    for di in range(200):
        # decoder_output, decoder_hidden, decoder_attention = decoder(
        #    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_hidden = decoder_hidden.view(1, 1, -1)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_input = decoder_output.data
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hiddens.append(decoder_hidden[0][0].data.numpy())
    return decoder_hiddens


def decoderContext(cents, decoder):
    contextVector = []
    for cent in cents:
        cent = Variable(torch.FloatTensor(cent))
        decoder_outputs = decoder_sc2(cent, decoder)
        # contextVector.append(decoder_output[0][0].data.numpy())
        contextVector.append(decoder_outputs)
        # print(encoder_output)
    return contextVector


# plot units

def unit2index(list, race):
    path = "/home/zd/MSC/parsed_replays/Stat/{race}.json".format(race=race)
    with open(path) as f:
        stat = json.load(f)
    units_name = stat["units_name"]
    units_type = stat["units_type"]

    unit_index = {}
    unit_list = {}
    for unit in list:
        for key in units_name:
            if units_name[key] == unit:
                unit_index[unit] = int(units_type[key])
                temp = []
                unit_list[unit] = temp
                break
    return unit_index, unit_list


def unit_color(unit_list):
    color = ['blue', 'red', 'brown', 'gold', 'green', 'pink', 'black']
    color_set = {}
    for i in range(len(unit_list)):
        color_set[unit_list[i]] = color[i]
    return color_set


def plotSingleCenter(Data, index, races, friend_and_enemy_range, name):
    x = [i for i in range(len(Data[0]))]
    plt.figure('data' + str(index))
    offset = int((friend_and_enemy_range[1] - friend_and_enemy_range[0]) / 6)

    race_fullname = {"P": "Protoss", "T": "Terran", "Z": "Zerg"}
    race_list = {"P": ["Zealot", "Stalker", "Sentry", "Phoenix", "Immortal"],
                 "T": ["Marine", "Marauder", "Reaper", "Hellion", "VikingFighter"]
        , "Z": ["Zergling", "Roach", "Baneling" "Hydralisk", "Infestor", "Mutalisk"]}
    ownrace = races[-1]
    enemyrace = races[0] if races[0] != ownrace else races[2]

    y_friend_index, y_friend = unit2index(race_list[ownrace], race_fullname[ownrace])
    # y_enemy_index, y_enemy = unit2index(race_list[enemyrace], race_fullname[enemyrace])
    y_friend_color = unit_color(race_list[ownrace])
    # y_enemy_color = unit_color(race_list[enemyrace])

    for j in range(len(Data[index])):
        frame_len = Data.shape[1]
        for key, val in y_friend_index.items():
            y_friend[key].append(Data[index][frame_len - j - 1][val]) if Data[index][frame_len - j - 1][val] > 0 else \
            y_friend[key].append(0)
            # for key, val in y_enemy_index.items():
            #   y_enemy[key].append(Data[index][j][val + offset])
    for key, val in y_friend.items():
        plt.plot(x, val, color=y_friend_color[key], label=key)
    # for key, val in y_enemy.items():
    #    plt.plot(x, val, color=y_enemy_color[key], ls="--")

    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("unit num")
    # plt.show()
    filename = "{races}_{name}_units_frame_{index}.png".format(races=races, name=name, index=index)
    plt.savefig(filename)
    plt.close()

def latestModel(races,model_path):
    max = 0
    filePre = '{0}_encode_model_sc2_'.format(races)
    hasModel = False
    for filename in os.listdir(model_path):
        if filename.startswith(races):
            hasModel = True
            num = int(filename[len(filePre):-4])
            max = num if num > max else max
    if hasModel == False:
        return "",""
    return encode_model_path[:-4]+"_"+str(max)+".pkl",decode_model_path[:-4]+"_"+str(max)+".pkl"

# Training and Evaluating

import matplotlib.ticker as ticker
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Units, races, friend_and_enemy_range = prepareDataPVP()

testset = [random.choice(Units) for i in range(2)]
testset = [x[0] for x in testset]

EOS_token = [-1 for i in range(int((friend_and_enemy_range[2]-friend_and_enemy_range[0])/6))]
SOS_token = [-2 for i in range(int((friend_and_enemy_range[2]-friend_and_enemy_range[0])/6))]

size = int((friend_and_enemy_range[2] - friend_and_enemy_range[0]) / 6)
hidden_size = size
pvp_feature_size = size
model_path = FLAGS.model_path
races = FLAGS.races
#encode_model_path = '{0}/{1}_encode_model_sc2'.format(model_path, races)
#decode_model_path = '{0}/{1}_decode_model_sc2'.format(model_path, races)
encode_model_path = '{0}/{1}_encode_model_sc2.pkl'.format(model_path, races)
decode_model_path = '{0}/{1}_decode_model_sc2.pkl'.format(model_path, races)

encoder1 = EncoderRNN(pvp_feature_size, hidden_size)
decoder1 = DecoderRNN(hidden_size, pvp_feature_size, 1)

latest_file_encode,latest_file_decode = latestModel(races,model_path)

#if not os.path.exists(encode_model_path):
if latest_file_encode == "" or latest_file_decode == "":
    print("not available model, start training:")
    if use_cuda:
        print("encoder1 = encoder1.cuda()")
        encoder1 = encoder1.cuda()
        print("decoder1 = decoder1.cuda()")
        decoder1 = decoder1.cuda()
    n_iters = len(Units) * 2
    training_pairs = []
    print("unit = unit.cuda()")
    for i in range(n_iters):
        unit = random.choice(Units)
        training_pairs.append(unit)

    trainIters_sc2(training_pairs, encoder1, decoder1, n_iters)
    ######################################################################
    print("save model:")
    torch.save(encoder1.state_dict(), encode_model_path[:-4] + "_" + str(n_iters) + ".pkl")
    torch.save(decoder1.state_dict(), decode_model_path[:-4] + "_" + str(n_iters) + ".pkl")
    latest_file_encode = encode_model_path[:-4] + "_" + str(n_iters) + ".pkl"
    latest_file_decode = decode_model_path[:-4] + "_" + str(n_iters) + ".pkl"
else:
    print("already exist, load model:")
    encoder1.load_state_dict(torch.load(latest_file_encode))
    encoder1 = encoder1.cpu()
    print("load loaded:")
    # if use_cuda:
    #    print("encoder1 = encoder1.cuda()")
    #    encoder1 = encoder1.cuda()


def plotTestSet(testset, i, races, friend_and_enemy_range, encoder, decoder):
    contextVector = encoderContext(testset, encoder)
    # print(contextVector)
    Units_np = np.asarray(contextVector)
    decoderVector = decoderContext(Units_np, decoder1)
    decoder_out_np = np.asarray(decoderVector)
    input_np = np.asarray(testset)
    for i in range(len(testset)):
        plotSingleCenter(input_np, i, races, friend_and_enemy_range, "testset")
        plotSingleCenter(decoder_out_np, i, races, friend_and_enemy_range, "test-result")


contextVector = encoderContext(Units, encoder1)
# print(contextVector)
Units_np = np.asarray(contextVector)
print(Units_np.shape[0])
print(Units_np.shape[1])
n_clusters = 20

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(Units_np)
kmeans_label = kmeans.predict(Units_np)
print(kmeans_label)
cents = kmeans.cluster_centers_
# print(cents)
decoder1.load_state_dict(torch.load(latest_file_decode))
decoder1 = decoder1.cpu()
# if use_cuda:
#    decoder1 = decoder1.cuda()

decoderVector = decoderContext(cents, decoder1)
decoder_out_np = np.asarray(decoderVector)

for i in range(n_clusters):
    plotSingleCenter(decoder_out_np, i, races, friend_and_enemy_range, "center")
'''
cents = TSNE(n_components=2,learning_rate=100).fit_transform(cents)

X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(Units_np)

colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868','b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
y = [i for i in range(Units_np.shape[0])]
y =np.array(y)

for i in range(n_clusters):
    index = np.nonzero(kmeans_label==i)[0]
    x0 = X_tsne[index,0]
    x1 = X_tsne[index,1]
    y_i = y[index]
    for j in range(len(x0)):
        #plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i])
        plt.scatter(x0[j], x1[j], color=colors[i])
    plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i])
#plt.axis([-30,30,-30,30])
plt.show()

'''
