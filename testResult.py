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

#use_cuda = torch.cuda.is_available()
use_cuda = False
print("torch.cuda.is_available():"+str(use_cuda))
MAX_LENGTH = 200

def unitRange(races):
    unitsinfo = {"PvP_P":[71,317,563],"ZvP_P":[71,317,1031],"ZvP_Z":[71,785,1031],"ZvZ_Z":[71,785,1499],"PvT_P":[71,317,653],"PvT_T":[81,417,663]
        ,"TvZ_Z":[71,785,1121],"TvZ_T":[81,417,1131],"TvT_T":[81,417,753]}
    return unitsinfo[races]

FLAGS = flags.FLAGS
flags.DEFINE_string(name='PATH', default='/home/zhandong/mycode/seq2seq/GlobalFeatureVector_noNormal/Protoss_vs_Terran/Protoss/',
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
    AllUnits=[]
    originalUnit=[]
    friend_and_enemy_range = unitRange(races)
    friend_unit_start_index = friend_and_enemy_range[0]
    enemy_unit_start_index = friend_and_enemy_range[1]
    frameslide = 200
    for p in parents:
    #for p in parents[:1000]:
        child = os.path.join(PATH,p)
        print(child)
        try :
            F = np.asarray(sparse.load_npz(child).todense())
        except:
            print("bad zip file: "+child)
            continue
        if F.shape[0] < frameslide:
            continue
        currF = F[0:200,friend_and_enemy_range[0]:friend_and_enemy_range[2]:6]
        temp =np.ndarray.tolist(currF)
        temp = Variable(torch.FloatTensor(temp))
        #temp = temp.cuda() if use_cuda else temp
        AllUnits.append((temp,temp))
        originalUnit.append(currF)
    return AllUnits,races,friend_and_enemy_range,originalUnit

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        print("input_size:" + str(input_size))

    def forward(self, input, hidden):
        embedded = input.view(1, 1, -1)
        output = embedded
        #output = input
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        #result = Variable(torch.zeros(1, 1, self.hidden_size))
        #result = Variable(torch.randn(1, 1, self.hidden_size))
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
        output = input.view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        #result = Variable(torch.zeros(1, 1, self.hidden_size))

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
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

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
            #topv, topi = decoder_output.data.topk(1)
            #ni = topi[0][0]

            #decoder_input = Variable(torch.FloatTensor([[ni]]))
            decoder_input = decoder_output.data
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            #if ni == EOS_token:
            #    break

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

def trainIters_sc2(enemyUnits, encoder, decoder, n_iters, print_every=1000, plot_every=1000,
               learning_rate=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    lr_base = learning_rate
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr_base)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr_base)
    #training_pairs = [variablesFromPair(input_lang, output_lang, random.choice(pairs))
    #                  for i in range(n_iters)]
    training_pairs = enemyUnits
    criterion = nn.L1Loss()
    #criterion = nn.BCELoss()

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
            torch.save(encoder1, encode_model_path+"_"+str(iter))
            torch.save(decoder1, decode_model_path+"_"+str(iter))
        if iter % (int(n_iters / 100)) == 0 and lr_base > 0.01:
            lr_base -= 0.1
            if lr_base < 0.01:
                lr_base = 0.01
            print("lr: "+str(lr_base)) 
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr_base)
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr_base)
            
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append((int(iter),plot_loss_avg))
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
    for x_,y_ in points:
        x.append(x_)
        y.append(y_)
    plt.plot(x, y)
    plt.xlabel("iter")
    plt.ylabel("loss(%)")
    races = FLAGS.races
    #plt.show()
    plt.savefig(races+"_plot_loss.png")
    plt.close()

def encoder_sc2(enemyUnit, encoder):
    input_variable = enemyUnit
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_hidden =encoder_hidden.cpu()
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
    return encoder_hidden

def encoderContext(Units, encoder):
    contextVector = []
    for unit in Units:
        #print(unit[0])
        encoder_output = encoder_sc2(unit[0], encoder)
        encoder_output= encoder_output.cpu()
        contextVector.append(encoder_output[0][0].data.numpy())
        #print(encoder_output)
    return contextVector


def decoder_sc2(enemyUnit, decoder):

    decoder_input = Variable(torch.zeros(decoder.hidden_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = enemyUnit
    input_length = decoder_hidden.size()[0]
    decoder_hiddens=[]
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
        #contextVector.append(decoder_output[0][0].data.numpy())
        contextVector.append(decoder_outputs)
        #print(encoder_output)
    return contextVector

# plot units

def unit2index(list,race):
    path = "/home/data1/zhandong/MSC/parsed_replays/Stat/{race}.json".format(race=race)
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
    return unit_index,unit_list

def unit_color(unit_list):
    color = ['blue', 'red', 'brown', 'gold', 'green', 'pink','black']
    color_set={}
    for i in range(len(unit_list)):
        color_set[unit_list[i]]=color[i]
    return color_set

def plotSingleCenter(Data,index,races,friend_and_enemy_range,name):
    x = [i for i in range(len(Data[0]))]
    plt.figure('data'+str(index))
    offset = int((friend_and_enemy_range[1] - friend_and_enemy_range[0]) / 6)

    race_fullname = {"P":"Protoss","T":"Terran","Z":"Zerg"}
    race_list = {"P":["Zealot","Stalker","Sentry", "Phoenix", "Immortal"],"T":["Marine","Marauder","Reaper","Hellion","VikingFighter"]
        ,"Z":["Zergling", "Roach", "Baneling" "Hydralisk", "Infestor","Mutalisk"]}
    ownrace = races[-1]
    enemyrace = races[0] if races[0] != ownrace else races[2]

    y_friend_index, y_friend = unit2index(race_list[ownrace], race_fullname[ownrace])
    #y_enemy_index, y_enemy = unit2index(race_list[enemyrace], race_fullname[enemyrace])
    y_friend_color = unit_color(race_list[ownrace])
    #y_enemy_color = unit_color(race_list[enemyrace])

    for j in range(len(Data[index])):
        #normal order
        for key, val in y_friend_index.items():
            y_friend[key].append(Data[index][j][val]) if Data[index][j][val] >0 else y_friend[key].append(0)
        #for key, val in y_enemy_index.items():
        #   y_enemy[key].append(Data[index][j][val + offset])
    for key,val in y_friend.items():
        plt.plot(x, val, color=y_friend_color[key],label = key)
    #for key, val in y_enemy.items():
    #    plt.plot(x, val, color=y_enemy_color[key], ls="--")

    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("unit num")
    #plt.show()
    filename = "{races}_{name}_units_frame_{index}.png".format(races=races,name=name,index=index)
    plt.savefig(filename)
    plt.close()

def plotSingleCenterReverse(Data,index,races,friend_and_enemy_range,name):
    x = [i for i in range(len(Data[0]))]
    plt.figure('data'+str(index))
    offset = int((friend_and_enemy_range[1] - friend_and_enemy_range[0]) / 6)

    race_fullname = {"P":"Protoss","T":"Terran","Z":"Zerg"}
    race_list = {"P":["Zealot","Stalker","Sentry", "Phoenix", "Immortal"],"T":["Marine","Marauder","Reaper","Hellion","VikingFighter"]
        ,"Z":["Zergling", "Roach", "Baneling" "Hydralisk", "Infestor","Mutalisk"]}
    ownrace = races[-1]
    enemyrace = races[0] if races[0] != ownrace else races[2]

    y_friend_index, y_friend = unit2index(race_list[ownrace], race_fullname[ownrace])
    #y_enemy_index, y_enemy = unit2index(race_list[enemyrace], race_fullname[enemyrace])
    y_friend_color = unit_color(race_list[ownrace])
    #y_enemy_color = unit_color(race_list[enemyrace])

    for j in range(len(Data[index])):
        frame_len =Data.shape[1]
        for key, val in y_friend_index.items():
            y_friend[key].append(Data[index][frame_len-j-1][val]) if Data[index][frame_len-j-1][val] >0 else y_friend[key].append(0)
        #for key, val in y_enemy_index.items():
        #   y_enemy[key].append(Data[index][j][val + offset])
    for key,val in y_friend.items():
        plt.plot(x, val, color=y_friend_color[key],label = key)
    #for key, val in y_enemy.items():
    #    plt.plot(x, val, color=y_enemy_color[key], ls="--")

    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("unit num")
    #plt.show()
    filename = "{races}_{name}_units_frame_{index}.png".format(races=races,name=name,index=index)
    plt.savefig(filename)
    plt.close()

def latestModel(races,model_path):
    max = 0
    filePre = '{0}_encode_model_sc2_'.format(races)
    for filename in os.listdir(model_path):
        if filename.startswith(races):
            num = int(filename[len(filePre):])
            max = num if num > max else max
    return encode_model_path+"_"+str(max),decode_model_path+"_"+str(max)

def plotTestSet(testset, races, friend_and_enemy_range, encoder, decoder):
    contextVector = encoderContext(testset,encoder)
    #print(contextVector)
    Units_np = np.asarray(contextVector)
    decoderVector = decoderContext(Units_np,decoder1)
    decoder_out_np = np.asarray(decoderVector)
    input_np = [x[0].data.numpy() for x in testset]
    input_np = np.asarray(input_np)
    for i in range(len(testset)):
        plotSingleCenter(input_np, i, races, friend_and_enemy_range,"testset")
        plotSingleCenterReverse(decoder_out_np, i, races, friend_and_enemy_range,"test-result")
# Training and Evaluating

import matplotlib.ticker as ticker
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Units,races,friend_and_enemy_range,originalUnit = prepareDataPVP()

testset = [random.choice(Units) for i in range(10)]

originalUnit = [random.choice(originalUnit) for i in range(10)]

for i in range(len(originalUnit)):
    plotSingleCenter(originalUnit, i, races, friend_and_enemy_range,"originalUnit")

size= int((friend_and_enemy_range[2] - friend_and_enemy_range[0])/6)

hidden_size = size
pvp_feature_size = size
model_path = FLAGS.model_path
races = FLAGS.races
encode_model_path = '{0}/{1}_encode_model_sc2'.format(model_path,races)
decode_model_path = '{0}/{1}_decode_model_sc2'.format(model_path,races)
encoder1 = EncoderRNN(pvp_feature_size, hidden_size)
decoder1 = DecoderRNN(hidden_size, pvp_feature_size,1)


latest_file_encode,latest_file_decode = latestModel(races,model_path)
print("latest_file_encode: "+ latest_file_encode)
print("latest_file_decode: "+ latest_file_decode)
if not os.path.exists(latest_file_encode):
    print("not available model, start training:")
    sys.exit()
print("already exist, load model:")
encoder1 = torch.load(latest_file_encode)
encoder1 = encoder1.cpu()
print("encoder1 loaded :")
decoder1 = torch.load(latest_file_decode)
decoder1 = decoder1.cpu()
print("decoder1 loaded:")
plotTestSet(testset, races, friend_and_enemy_range, encoder1, decoder1)
