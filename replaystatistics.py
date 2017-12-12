import os
import json
import stream
import gflags as flags
import multiprocessing as mp
import subprocess
import signal
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gflags as flags
import sys
from sklearn.manifold import TSNE


def jsonload(path):
    with open(path) as f:
        stat = json.load(f)

def unit2index(list,race):
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
    return unit_index,unit_list

def feature2Unit(Units,length):
    units_list=[]
    for units in Units:
        real_unit = [x for x in (units[k: k + length]
              for k in range(0, len(units), length))]
        units_list.append(real_unit)
        print(len(real_unit))
    print(len(units_list), len(units_list[0]), len(units_list[0][0]))
    return units_list

def plotData1(Data):
    x = [i for i in range(len(Data[0]))]
    plt.figure('data')
    color = ['blue','black','brown','gold','green','pink']

    for i in range(len(Data)):
        y = []
        for j in range(len(Data[i])):
            y.append(Data[i][j][6])
        plt.subplot(6,1,i+1)
        plt.plot(x, y, color=color[i],label = "Center"+str(i))
        plt.legend()
    plt.xlabel("frame")
    plt.ylabel("unit num")
    #plt.show()
    plt.savefig("pvp_unit_vs_frame.png")
    plt.close()

def unit_color(unit_list):
    color = ['blue', 'red', 'brown', 'gold', 'green', 'pink','black']
    color_set={}
    for i in range(len(unit_list)):
        color_set[unit_list[i]]=color[i]
    return color_set

def plotCenter(Data,races,friend_and_enemy_range):
    print(len(Data), len(Data[0]), len(Data[0][0]))
    x = [i for i in range(len(Data[0]))]
    plt.figure('data')

    #y = {'Zealot': 1, 'Stalker': 30, "Sentry": 16, 'Phoenix': 8, 'Immortal': 2}
    race_fullname = {"P":"Protoss","T":"Terran","Z":"Zerg"}
    race_list = {"P":["Zealot","Stalker","Sentry", "Phoenix", "Immortal"],"T":["Marine","Marauder","Reaper","Hellion","VikingFighter"]
        ,"Z":["Zergling", "Roach", "Baneling" "Hydralisk", "Infestor","Mutalisk"]}
    ownrace = races[-1]
    enemyrace = races[0] if races[0] != ownrace else races[2]

    offset = int((friend_and_enemy_range[1]-friend_and_enemy_range[0])/6)
    print("offset"+str(offset))
    print(len(Data))
    for i in range(len(Data)):
        #y = {'Probe':13,'Zealot':1,'Gateway':10,'Stalker':30,'Stargate':33,'Phoenix':8}
        count =0
        y_friend_index, y_friend = unit2index(race_list[ownrace], race_fullname[ownrace])
        y_enemy_index, y_enemy = unit2index(race_list[enemyrace], race_fullname[enemyrace])
        y_friend_color = unit_color(race_list[ownrace])
        y_enemy_color = unit_color(race_list[enemyrace])
        #y_friend = {'Zealot': [], 'Stalker': [], "Sentry": [], 'Phoenix': [], 'Immortal': []}
        #y_color = {'Zealot': "black", 'Stalker': "gold", "Sentry": 'red', 'Phoenix': "pink", 'Immortal': 'blue'}
        #y_enemy = {'Zealot': [], 'Stalker': [], "Sentry": [], 'Phoenix': [], 'Immortal': []}
        for j in range(len(Data[i])):
            count+=1
            for key,val in y_friend_index.items():
                y_friend[key].append(Data[i][j][val])
            for key, val in y_enemy_index.items():
                y_enemy[key].append(Data[i][j][val+offset])
        plt.subplot(6, 1, i + 1)
        for key,val in y_friend.items():
            plt.plot(x, val, color=y_friend_color[key],label = key)
        for key,val in y_enemy.items():
            plt.plot(x, val, color=y_enemy_color[key],ls="--")
    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("unit num")
    #plt.show()
    plt.savefig(races+"_units_frame2.png")
    plt.close()

def plotSingleCenter(Data,index,races,friend_and_enemy_range):
    x = [i for i in range(len(Data[0]))]
    plt.figure('data'+str(index))
    offset = int((friend_and_enemy_range[1] - friend_and_enemy_range[0]) / 6)

    race_fullname = {"P":"Protoss","T":"Terran","Z":"Zerg"}
    race_list = {"P":["Zealot","Stalker","Sentry", "Phoenix", "Immortal"],"T":["Marine","Marauder","Reaper","Hellion","VikingFighter"]
        ,"Z":["Zergling", "Roach", "Baneling" "Hydralisk", "Infestor","Mutalisk"]}
    ownrace = races[-1]
    enemyrace = races[0] if races[0] != ownrace else races[2]

    y_friend_index, y_friend = unit2index(race_list[ownrace], race_fullname[ownrace])
    y_enemy_index, y_enemy = unit2index(race_list[enemyrace], race_fullname[enemyrace])
    y_friend_color = unit_color(race_list[ownrace])
    y_enemy_color = unit_color(race_list[enemyrace])

    for j in range(len(Data[index])):
        for key, val in y_friend_index.items():
            y_friend[key].append(Data[index][j][val])
        for key, val in y_enemy_index.items():
            y_enemy[key].append(Data[index][j][val + offset])
    for key,val in y_friend.items():
        plt.plot(x, val, color=y_friend_color[key],label = key)
    for key, val in y_enemy.items():
        plt.plot(x, val, color=y_enemy_color[key], ls="--")

    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("unit num")
    #plt.show()
    filename = "{races}_units_frame_{index}.png".format(races=races,index=index)
    plt.savefig(filename)
    plt.close()

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

def plotCenterDistribution(AllUnits,kmeans_label):

    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(AllUnits)

    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868','b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    y = [i for i in range(kmeans_label)]

    for i in range(kmeans_label):
        index = np.nonzero(kmeans_label == i)[0]
        x0 = X_tsne[index, 0]
        x1 = X_tsne[index, 1]
        y_i = y[index]
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i], \
                     fontdict={'weight': 'bold', 'size': 9})
        #plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i], linewidths=12)
    plt.axis([-30, 30, -30, 30])
    plt.show()

def plotLengthDistribution(framelength,races,count,minT,maxT,aveT):
    x = [i for i in range(len(framelength))]

    title = "min_{0}_ave_{1}_max_{2}".format(minT, int(aveT / count), maxT)
    plt.figure("frame distribution")
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("replay num")
    plt.plot(x, framelength)
    plt.savefig(races + "_frame_distribution.png")
    plt.close()

def plotReplayVersion(key,value):
    title = "Replay Version"
    plt.figure("Replay Version")
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("replay num")
    plt.bar(range(len(value)), value,tick_label=key)
    plt.savefig("replay_version.png")
    plt.close()

def replayVersion(info_path):
    path = info_path
    versions = {}
    parents = os.listdir(path)
    for p in parents:
        child = os.path.join(path, p)
        try:
            with open(child) as f:
                print(child)
                state = json.load(f)
                version = state['info']
                version = eval(version)
                version = version['gameVersion']
                if version in versions:
                    versions[version] += 1
                else:
                    versions[version] = 1
        except Exception:
            print(child+" failed")
            continue
    key = list(versions.keys())
    value = list(versions.values())
    print(key)
    print(value)
    plotReplayVersion(key, value)

def lengthDistribution():
    PATH = FLAGS.PATH
    races = FLAGS.races
    print(PATH)
    print(races)
    #PATH = "/home/zd/MSC/parsed_replays/GlobalFeatureVector/Protoss_vs_Terran/Protoss/"
    #races = "PvT:P"
    parents = os.listdir(PATH)
    minF = 10000
    maxF = 0
    aveF = 0
    count = 0
    minT = 10000
    aveT = 0
    maxT = 0
    countT = 0
    AllUnits=[]
    framelength = [0 for i in range(2000)]
    friend_and_enemy_range = unitRange(races)
    friend_unit_start_index = friend_and_enemy_range[0]
    enemy_unit_start_index = friend_and_enemy_range[1]
    frameslide = 200
    for p in parents:
    #for p in parents[:1000]:
        child = os.path.join(PATH,p)
        print(child)
    #PATH = "/home/zd/MSC/parsed_replays/GlobalFeatureVector/Protoss_vs_Terran/Protoss/1@0ae1567d951717f72fa6d666b7782762b2386bf77339cffb8c23c2db9851f95b.SC2Replay.npz"
        try :
            F = np.asarray(sparse.load_npz(child).todense())
        except:
            print("bad zip file: "+child)
            continue
        #print(F.shape)
        #X = F[:,71:563]
        #X = F[:, friend_and_enemy_range[0]:friend_and_enemy_range[2]:6]
        #print(X[0,0].dtpye)
        #zeros = np.zeros(len(X[0]),'float64')
        length = F.shape[0]
        print(length)
        if length >= 2000:
            length = 1999
        framelength[length] +=1
        maxT = max(maxT,length)
        minT = min(minT,length)
        aveT+=length
        count+=1
        #if F.shape[0] < frameslide:
        #    continue
    plotLengthDistribution(framelength,races,count,minT,maxT,aveT)
if __name__ == '__main__':
    #lengthDistribution()
    replayVersion("/home/zhandong/sc2_pack1/MSC/replays_infos/")

