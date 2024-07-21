import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
arg = "FOPPA"
#NOMDATA = "results/ResultatsShapleyBons/"+str(arg)+str(arg)+"ResultsShapley.csv"
#NOMDATA = "results/ResultatsShapleyBons/FOPPAresultsShapleyInduced33.csv"
NOMDATA = "../results/"+str(arg)+"/resultsShapley.csv"
NAMESORTIERBO= "../results/"+str(arg)+"/"+ str(arg)+"ShapleyRBO"
NAMESORTIEF1= "../results/"+str(arg)+"/"+ str(arg)+"ShapleyF1"

dicoNumeroNom = {
    0 : "Acc",
    1 : "Brins",
    2 : "CConf",
    3 : "CFactor",
    4 : "ColStr",
    5 : "Cole",
    6 : "Conf",
    7 : "Cos",
    8 : "Cover",
    9 : "Dep",
    10 : "Excex",
    11 : "FPR",
    12 : "GR",
    13 : "Gain",
    14 : "InfGain",
    15 : "Jacc",
    16 : "Klos",
    17 : "Lap",
    18 : "Lever",
    19 : "Lift",
    20 : "MDisc",
    21 : "MutInf",
    22 : "NetConf",
    23 : "OddsR",
    24 : "Pearson",
    25 : "RelRisk",
    26 : "Sebag",
    27 : "Spec",
    28 : "Str",
    29 : "Sup",
    30 : "SupDif",
    31 : "AbsSupDif",
    32 : "WRACC",
    33 : "Zhang",
    34 : "chiTwo",

}

datas = pd.read_csv(NOMDATA)
print(datas)
print(datas.columns)


#Print le RBO
#Regarder le nombre de valeur unique dans la colonne Score2
nBScore = datas['Score2'].nunique()
plt.figure()
TOKEEP = [0,12,10,15,21,29,31,33]
TODELETE = []
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        res.append(ds['RBO'].mean())
    if i in TOKEEP:
        plt.plot(res, label=dicoNumeroNom[i])
    elif i not in TODELETE:
        plt.plot(res,alpha=0.1)
plt.xlabel("Number of representatives (%)")
plt.ylabel("RBO")
plt.legend()
plt.savefig(NAMESORTIERBO)
print(datas)


#TOKEEP1 = [0,4,7,9,10,12,13,15]
TOKEEP1 = [0,4,10,12,21,27,29,31]
TOKEEP2 = [16,17,18,20,21,22,23,24]
TOKEEP3 = [25,27,28,29,31,33,34]

TOKEEPA = [0,4,7,9,10]
TOKEEPB = [12,13,15,16,17]
TOKEEPC = [20,21,22,23,24]
TOKEEPD = [25,27,29,31,34]

#TOKEEPA = [0,4,7,9,10,12,13]
#TOKEEPB = [15,16,17,20,21,22,23]
#TOKEEPC = [24,25,27,29,31,34]
#TOKEEPD = [0,12]

TOKEEPA = [0,4,7,9,10,12,13,15]
TOKEEPB = [16,17,18,20,21,22,23,24]
TOKEEPC = [25,27,28,29,31,33,34]
#TOKEEPD = [0,12]

#Put all figures on the same values of Y. 
# Min will always be 0
# Max will always be the max value of the RBO
maxVal = 0
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if ds['RBO'].mean() > maxVal:
            maxVal = ds['RBO'].mean()

#Arrondi le maxVal au dixieme superieur
maxVal = round(maxVal,1)
if maxVal >1:
    maxVal = 1
#Print le Fscore
import random
plt.figure()
resShapley = []
for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            resShapley.append(ds['RBO'].mean())
            oldVal = ds['RBO'].mean()
plt.plot(resShapley, label="SAGE")

plt.figure()
# Representer les F1Score
#Sepaarer les donnees en 2 figures
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['RBO'].mean())
            oldVal = ds['RBO'].mean()
            GoldenTruth.append(ds['RBO'].mean())
    if i in TOKEEP1:
        plt.plot(res, label=dicoNumeroNom[i])
plt.xlabel("Number of representatives (%)")
plt.ylabel("RBO")
plt.ylim(0,maxVal)
plt.legend()
plt.savefig(NAMESORTIERBO+"_main.pdf")

plt.figure()
# Representer les F1Score
#Sepaarer les donnees en 2 figures
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['RBO'].mean())
            oldVal = ds['RBO'].mean()
            GoldenTruth.append(ds['RBO'].mean())
    if i in TOKEEPA:
        plt.plot(res, label=dicoNumeroNom[i])
plt.xlabel("Number of representatives (%)")
plt.ylabel("RBO")
plt.ylim(0,maxVal)
plt.legend()
plt.savefig(NAMESORTIERBO+"_1.pdf")
plt.figure()
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['RBO'].mean())
            oldVal = ds['RBO'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEPB:
        plt.plot(res, label=dicoNumeroNom[i])
plt.xlabel("Number of representatives (%)")
plt.ylabel("RBO")
plt.ylim(0,maxVal)
plt.legend()
plt.savefig(NAMESORTIERBO+"_2.pdf")

plt.figure()
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['RBO'].mean())
            oldVal = ds['RBO'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEPC:
        plt.plot(res, label=dicoNumeroNom[i])
plt.xlabel("Number of representatives (%)")
plt.ylabel("RBO")
plt.ylim(0,maxVal)
plt.legend()
plt.savefig(NAMESORTIERBO+"_3.pdf")

plt.figure()
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['RBO'].mean())
            oldVal = ds['RBO'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEPD:
        plt.plot(res, label=dicoNumeroNom[i])
plt.xlabel("Number of representatives (%)")
plt.ylabel("RBO")
plt.ylim(0,maxVal)
plt.legend()
plt.savefig(NAMESORTIERBO+"_4.pdf")


TODELETE = []


##################

#Put all figures on the same values of Y. 
# Min will always be 0
# Max will always be the max value of the RBO
maxVal = 0
minVal = 1
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if ds['F1SCORE'].mean() > maxVal:
            maxVal = ds['F1SCORE'].mean()
        if ds['F1SCORE'].mean() < minVal:
            minVal = ds['F1SCORE'].mean()

#Arrondi le maxVal au dixieme superieur
maxVal = round(maxVal,1)
maxVal = maxVal + 0.1
if maxVal >1:
    maxVal = 1
#Arrondi le minVal au dixieme inferieur
minVal = round(minVal,1)
minVal = minVal - 0.1
if minVal < 0:
    minVal = 0

plt.figure()
# Representer les F1Score
#Sepaarer les donnees en 2 figures
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,101):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['F1SCORE'].mean())
            oldVal = ds['F1SCORE'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEP1:
        plt.plot(res, label=dicoNumeroNom[i])
    if i==34:
        #plot the golden truth in black dotted line
        plt.plot(GoldenTruth, label="Gold Standard", color='black', linestyle='dotted')
plt.xlabel("Number of representatives (%)")
plt.ylabel("F1-Score")
#plt.ylim(minVal,maxVal)
#change the limit of the axe Y
#Fixer les limites de l'axe Y
#plt.ylim(0.6,1)
plt.legend()
plt.savefig(NAMESORTIEF1+"_main.pdf")

plt.figure()
# Representer les F1Score
#Sepaarer les donnees en 2 figures
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,101):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['F1SCORE'].mean())
            oldVal = ds['F1SCORE'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEPA:
        plt.plot(res, label=dicoNumeroNom[i])
    if i==34:
        #plot the golden truth in black dotted line
        plt.plot(GoldenTruth, label="Gold Standard", color='black', linestyle='dotted')
plt.xlabel("Number of representatives (%)")
plt.ylabel("F1-Score")
plt.ylim(minVal,maxVal)
#log for the y axis
#Fixer les limites de l'axe Y
#plt.ylim(0.6,1)
plt.legend()
plt.savefig(NAMESORTIEF1+"_1.pdf")
plt.figure()
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['F1SCORE'].mean())
            oldVal = ds['F1SCORE'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEPB:
        plt.plot(res, label=dicoNumeroNom[i])
    if i==32:
        #plot the golden truth in black dotted line
        plt.plot(GoldenTruth, label="Gold Standard", color='black', linestyle='dotted')
plt.xlabel("Number of representatives (%)")
plt.ylabel("F1-Score")
plt.ylim(minVal,maxVal)
#plt.ylim(0.6,1)
plt.legend()
plt.savefig(NAMESORTIEF1+"_2.pdf")

plt.figure()
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['F1SCORE'].mean())
            oldVal = ds['F1SCORE'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEPC:
        plt.plot(res, label=dicoNumeroNom[i])
    if i==34:
        #plot the golden truth in black dotted line
        plt.plot(GoldenTruth, label="Gold Standard", color='black', linestyle='dotted')
plt.xlabel("Number of representatives (%)")
plt.ylabel("F1-Score")
plt.ylim(minVal,maxVal)
#plt.ylim(0.6,1)
plt.legend()
plt.savefig(NAMESORTIEF1+"_3.pdf")

plt.figure()
for i in range(0, nBScore):
    Keep=datas[datas['Score2'] == i]
    res = []
    GoldenTruth = []
    for percent in range(0,100):
        ds = Keep[Keep['PercentK'] == percent]
        ds.reset_index(drop=True, inplace=True)
        if len(ds)>0:
            res.append(ds['F1SCORE'].mean())
            oldVal = ds['F1SCORE'].mean()
            GoldenTruth.append(ds['F1SHAPLEY'].mean())
    if i in TOKEEPD:
        plt.plot(res, label=dicoNumeroNom[i])
    if i==34:
        #plot the golden truth in black dotted line
        plt.plot(GoldenTruth, label="Gold Standard", color='black', linestyle='dotted')
plt.xlabel("Number of representatives (%)")
plt.ylabel("F1-Score")
plt.ylim(minVal,maxVal)
#plt.ylim(0.6,1)
plt.legend()
plt.savefig(NAMESORTIEF1+"_4.pdf")
