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
        #temp =np.ndarray.tolist(currF)
        #temp = Variable(torch.FloatTensor(temp))
        #temp = temp.cuda() if use_cuda else temp
        #AllUnits.append((temp,temp))
        originalUnit.append(currF)
    #return AllUnits,races,friend_and_enemy_range,originalUnit
    return originalUnit,races,friend_and_enemy_range

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


import matplotlib.ticker as ticker
import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

originalUnit,races,friend_and_enemy_range = prepareDataPVP()


originalUnit = [random.choice(originalUnit) for i in range(10)]

for i in range(len(originalUnit)):
    plotSingleCenter(originalUnit, i, races, friend_and_enemy_range,"originalUnit")
