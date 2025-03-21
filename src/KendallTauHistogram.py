from networkx.algorithms.isomorphism import ISMAGS
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import tqdm
from matplotlib.colors import ListedColormap
import rbo
import scipy as sp
import re
import copy

def read_Sizegraph(fileName):
    """Read the number of graphs in a file
    Input: fileName (string) : the name of the file
    Ouptut: TAILLE (int) : the number of graphs in the file"""
    
    file = open(fileName, "r")
    nbGraph=0
    for line in file:
       if line[0]=="t":
            nbGraph=nbGraph+1
    return nbGraph

def load_graphs(fileName,TAILLE):
    """Load graphs from a file.
    args: fileName (string) : the name of the file)
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of graphs
    numbers (list of list of int) : the list of occurences of each graph
    nom (list of string) : the list of names of each graph)"""
    
    nbV=[]
    nbE=[]
    numbers = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
                nbV.append(len(labelVertices[compteur-1]))
                nbE.append(len(labelEdges[compteur-1]))
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            for j in range(1,len(b)):
                if not(b[j]=="#"):
                    val = b[j]
                    val = re.sub("\n","",val)
                    val = int(val)
                    temp.append(val)
            numbers[numero]=temp  
    noms.append(temptre)
    nbV.append(len(labelVertices[compteur-1]))
    nbE.append(len(labelEdges[compteur-1]))
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            #tempDictionnaireNodes = {"color":labelVertices[i][j]}
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,noms

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV



def RedondanceMotif(motifs,NbOccurences,NBGRAPHES):
    results=[]
    for motif in motifs:
        print(motif)
        results.append(np.zeros(NBGRAPHES))
        for i in range(len(NbOccurences[motif])):
            results[-1][NbOccurences[motif][i]]=1
    return results

from sklearn import metrics
def compute_results(features_train, labels_train, features_test, labels_test):
	clf = SVC(C=1000)
	clf.fit(features_train,labels_train)
	res = clf.predict(features_test)
	ok = 0
	fp, fn = 0, 0
	tp, tn = 0, 0
	nAbuses = 0
	f1 = 0
	for r in range(len(res)):
		if res[r] == labels_test[r]:
			ok += 1
		if labels_test[r] == 1:
			if res[r] == 1:
				tp += 1
			if res[r] == 0:
				fn += 1
			nAbuses += 1
		if labels_test[r] == 0:
			if res[r] == 1:
				fp += 1
			if res[r] == 0:
				tn += 1

	rec = tp / float(nAbuses)
	pre = tp / max(float(tp + fp),1)
	if pre + rec > 0:
		f1 = 2 * (pre * rec) / float(pre + rec)
	return metrics.f1_score(res,labels_test, average=None)[1]


def load_patterns(fileName,TAILLE):
    """ This function loads the post-processed patterns, i.e with occurences.
    fileName (string) : the name of the file
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of patterns
            numbers (list of list of int) : the list of occurences of each graph
            numberoccurences (list of list of int) : the list of occurences of each pattern
    """
    numbers = []
    numberoccurences = []
    numbercoverage = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
        numberoccurences.append([])
        numbercoverage.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            tempOccu = []
            tempCoverage = []
            for j in range(1,len(b)-1):
                val = b[j]
                val = re.sub("\n","",val)
                if not(val=="#" or val==""):
                    val = str(val).split("/")
                    numeroGraph = int(val[0])
                    val = str(val[1]).split(":")
                    coverage=1
                    if len(val)>1:
                        coverage = float(val[1])
                    occurences = int(float(val[0]))
                    temp.append(numeroGraph)
                    tempOccu.append(occurences)
                    tempCoverage.append(coverage)
            numbers[numero]=temp 
            numberoccurences[numero]=tempOccu
            numbercoverage[numero]=tempCoverage
    noms.append(temptre)
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,numberoccurences

def computeCorrelation(patterns,id_graphs,numberoccurences,TAILLEGRAPHE,NAMEFILE):
    MatriceCorrelation = np.zeros((len(patterns),TAILLEGRAPHE))
    for j in range(len(patterns)):
        nbPattern = patterns[j]
        for graphe in range(TAILLEGRAPHE):
            if graphe in id_graphs[nbPattern]:
                if numberoccurences==1:
                    MatriceCorrelation[j][graphe]=1
                else:
                    MatriceCorrelation[j][graphe]=numberoccurences[nbPattern][id_graphs[nbPattern].index(graphe)]
    alpha = np.corrcoef(MatriceCorrelation)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(alpha, interpolation='nearest',label='Correlation')
    plt.colorbar()
    plt.savefig(NAMEFILE+".pdf")
    np.savetxt(NAMEFILE+".txt",MatriceCorrelation)



#### Function for pattern measures
    
def patternMeasures(keep,labels,id_graphs,TAILLEPATTERN):
    lenC = 0
    lennotC = 0
    for i in range(len(keep)):
        if labels[keep[i]]==1:
            lenC = lenC+1
        else:
            lennotC = lennotC+1
    lenALL = lenC+lennotC

    pC = lenC/lenALL
    print("pC",pC)
    pnotC = 1 - pC
    print("pnotC",pnotC)
    ####Initialisation
    pP = np.zeros(TAILLEPATTERN)
    pnotP = np.zeros(TAILLEPATTERN)
    pPC = np.zeros(TAILLEPATTERN)
    pPnotC = np.zeros(TAILLEPATTERN)
    pnotPC = np.zeros(TAILLEPATTERN)
    pnotPnotC = np.zeros(TAILLEPATTERN)
    pPassumingC = np.zeros(TAILLEPATTERN)
    pPassumingnotC = np.zeros(TAILLEPATTERN)
    pnotPassumingC = np.zeros(TAILLEPATTERN)
    pnotPassumingnotC = np.zeros(TAILLEPATTERN)
    pCassumingP = np.zeros(TAILLEPATTERN)
    pCassumingnotP = np.zeros(TAILLEPATTERN)
    pnotCassumingP = np.zeros(TAILLEPATTERN)
    pnotPnotC = np.zeros(TAILLEPATTERN)
    pnotCassumingnotP = np.zeros(TAILLEPATTERN)
    toConsider = np.zeros(TAILLEPATTERN)
    t11 = np.zeros(TAILLEPATTERN)
    t12 = np.zeros(TAILLEPATTERN)
    t21 = np.zeros(TAILLEPATTERN)
    t22 = np.zeros(TAILLEPATTERN)
    for i in range(TAILLEPATTERN):
        t_Pos = 0
        t_Neg = 0
        for j in range(len(id_graphs[i])):
            if j in keep:
                if labels[id_graphs[i][j]] == 1:
                    t_Pos += 1
                else:
                    t_Neg += 1
        toConsider[i]=1
        if t_Pos+t_Neg==0:
            toConsider[i]=0
            #print("Pattern ",i," is not in the dataset")
        t11[i] = t_Pos
        t12[i] = t_Neg
        t21[i] = lenC-t_Pos
        t22[i] = lennotC-t_Neg


        ## pP = Nombre de graphes contenant le pattern / Nombre de graphes
        pP[i] = (t_Pos+t_Neg)/lenALL

        ## pnotP = 1-pP
        pnotP[i] = 1-pP[i]

        ## pPC = Nombre de graphes contenant le pattern et de classe C / Nombre de graphes

        if lenALL == 0:
            pPC[i] = 0
            pPnotC[i] = 0
        else:
            pPC[i] = t_Pos/lenALL
            pPnotC[i] = (t_Neg)/lenALL

        pnotPC[i] = (lenC-t_Pos)/lenALL
        pnotPnotC[i] = (lennotC-t_Neg)/lenALL
        pPassumingC[i]= t_Pos/lenC
        pPassumingnotC[i]= t_Neg/lennotC
        pnotPassumingC[i]= (lenC-t_Pos)/lenC
        pnotPassumingnotC[i]= (lennotC-t_Neg)/lennotC

        if t_Pos+t_Neg==0:
            pCassumingP[i]= 0
            pnotCassumingP[i]= 0
        else:
            pCassumingP[i]= t_Pos/(t_Pos+t_Neg)
            pnotCassumingP[i]= t_Neg/(t_Pos+t_Neg)
        
        if t_Pos+t_Neg==lenALL:
            pCassumingnotP[i]= 0
            pnotCassumingnotP[i]= 0
        else:
            pCassumingnotP[i]= (lenC-t_Pos)/(lenALL-t_Pos-t_Neg)
            pnotCassumingnotP[i]= (lennotC-t_Neg)/(lenALL-t_Pos-t_Neg)
    
    ds = DiscriminationScores(toConsider,lenALL,pP,pnotP,pC,pnotC,pPC,pPnotC,pnotPC,pnotPnotC,pPassumingC,pPassumingnotC,pnotPassumingC,pnotPassumingnotC,pCassumingP,pCassumingnotP,pnotCassumingP,pnotCassumingnotP,t11,t12,t21,t22)
    
    return ds


#### Function for discrimination scores

class DiscriminationScores:
    def __init__(self,toConsider,lenALL,pP,pnotP,pC,pnotC,pPC,pPnotC,pnotPC,pnotPnotC,pPassumingC,pPassumingnotC,pnotPassumingC,pnotPassumingnotC,pCassumingP,pCassumingnotP,pnotCassumingP,pnotCassumingnotP,t11,t12,t21,t22):
        self.toConsider = toConsider
        self.lenALL = lenALL
        self.pP = pP
        self.pnotP = pnotP
        self.pC = pC
        self.pnotC = pnotC
        self.pPC = pPC
        self.pPnotC = pPnotC
        self.pnotPC = pnotPC
        self.pnotPnotC = pnotPnotC
        self.pPassumingC = pPassumingC
        self.pPassumingnotC = pPassumingnotC
        self.pnotPassumingC = pnotPassumingC
        self.pnotPassumingnotC = pnotPassumingnotC
        self.pCassumingP = pCassumingP
        self.pCassumingnotP = pCassumingnotP
        self.pnotCassumingP = pnotCassumingP
        self.pnotCassumingnotP = pnotCassumingnotP
        self.t11 = t11
        self.t12 = t12
        self.t21 = t21
        self.t22 = t22
    


def Acc(discriminationScore):
    return discriminationScore.pPC + discriminationScore.pnotPnotC

def Brins(discriminationScore):
    numerator = discriminationScore.pP * discriminationScore.pnotC
    denominator = discriminationScore.pPnotC 

    # Use np.where to handle the two special cases
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result
def CConf(discriminationScore):
    return discriminationScore.pCassumingP - discriminationScore.pC

def Cole(discriminationScore):
    return (discriminationScore.pCassumingP - discriminationScore.pC)/(1-discriminationScore.pC)

def ColStr(discriminationScore):
    term1num= (discriminationScore.pPC +discriminationScore.pnotPnotC)
    term1denom = (discriminationScore.pP*discriminationScore.pC+discriminationScore.pnotP*discriminationScore.pnotC)
    term2num = 1 - discriminationScore.pP*discriminationScore.pC - discriminationScore.pnotC*discriminationScore.pnotP
    term2denom = 1 - discriminationScore.pPC - discriminationScore.pnotCassumingnotP
    # Use np.where to handle the two special cases for term1 and term2
    term1 = np.where(term1denom == 0, float('inf'), term1num / term1denom)
    term1 = np.where(term1num == 0, 0, term1)
    term2 = np.where(term2denom == 0, float('inf'), term2num / term2denom)
    term2 = np.where(term2num == 0, 0, term2)
    return term1 * term2

def Conf(discriminationScore):
    return discriminationScore.pCassumingP

def Cos(discriminationScore):
    product = discriminationScore.pCassumingP * discriminationScore.pPassumingC
    # Use np.where to handle the special case
    result = np.where(product == 0, 0, np.sqrt(product))
    return result

def Cover(discriminationScore):
    return discriminationScore.pPassumingC

def Dep(discriminationScore):
    return np.abs(discriminationScore.pnotC - discriminationScore.pnotCassumingP)

def Excex(discriminationScore):
    numerator = discriminationScore.pnotCassumingP
    denominator = discriminationScore.pCassumingnotP

    # Use np.where to handle the special case
    result = np.where(denominator == 0, -100000, 1 - (numerator / (denominator)))
    result = np.where(numerator == 0, 1, result)
    return result

def Gain(discriminationScore):
    term1 = discriminationScore.pCassumingP
    term2 = np.log(discriminationScore.pC)

    result = np.where(term1 == 0, -100000, discriminationScore.pPC * (np.log(term1) - term2))
    result = np.where(discriminationScore.pPC == 0, 0, result)

    return result
    
def GR(discriminationScore):
    numerator = discriminationScore.pPassumingC
    denominator = discriminationScore.pPassumingnotC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def InfGain(discriminationScore):
    term1 = -np.log(discriminationScore.pC)
    term2 = discriminationScore.pCassumingP

    # Use np.where to handle the special case
    result = np.where(term2 == 0, -100000, term1 * np.log(term2))
    return result

def Jacc(discriminationScore):
    numerator = discriminationScore.pPC
    denominator = discriminationScore.pP + discriminationScore.pC - discriminationScore.pPC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def Klos(discriminationScore):
    term1 = np.sqrt(discriminationScore.pPC)
    term2 = discriminationScore.pCassumingP - discriminationScore.pC

    # Use np.where to handle the special case
    result = np.where(term1 == 0, 0, term1 * term2)

    return result

def Lap(discriminationScore):
    return (discriminationScore.pPC + 1/discriminationScore.lenALL)/(discriminationScore.pP + 2/discriminationScore.lenALL)

def Lever(discriminationScore):
    return discriminationScore.pCassumingP-discriminationScore.pC*discriminationScore.pP

def Lift(discriminationScore):
    numerator = discriminationScore.pPC
    denominator = discriminationScore.pP * discriminationScore.pC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)
    return result

def MDisc(discriminationScore):
    print("MDisc")
    print(discriminationScore.pPC)
    print(discriminationScore.pnotPnotC)
    print(discriminationScore.pPnotC)
    print(discriminationScore.pnotPC)
    numerator = discriminationScore.pPC * discriminationScore.pnotPnotC 
    denominator = discriminationScore.pPnotC * discriminationScore.pnotPC 

    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    print("Result pour 238 MDISC",result[238])
    result = np.where(numerator == 0, -100000, np.log(result))

    return result

# Debut A Faire Lucas
def MutInf(discriminationScore):
    alpha = discriminationScore.pPC *np.log(discriminationScore.pPC/(discriminationScore.pP*discriminationScore.pC))
    beta = discriminationScore.pnotPC *np.log(discriminationScore.pnotPC/(discriminationScore.pnotP*discriminationScore.pC))
    gamma = discriminationScore.pPnotC *np.log(discriminationScore.pPnotC/(discriminationScore.pP*discriminationScore.pnotC))
    delta = discriminationScore.pnotPnotC *np.log(discriminationScore.pnotPnotC/(discriminationScore.pnotP*discriminationScore.pnotC))
    return alpha+beta+gamma+delta

def NetConf(discriminationScore):
    return (discriminationScore.pCassumingP-discriminationScore.pC)/(1-discriminationScore.pP)

def OddsR(discriminationScore):
    alpha = np.where(discriminationScore.pPC == 1 , float('inf'),discriminationScore.pPC/(1-discriminationScore.pPC))
    alpha = np.where(discriminationScore.pPC == 0 , 0 , alpha)
    beta = np.where(discriminationScore.pPnotC == 1 , float('inf'),discriminationScore.pPnotC/(1-discriminationScore.pPnotC))
    beta = np.where(discriminationScore.pPnotC == 0 , 0 , beta)

    return alpha/beta

def Pearson(discriminationScore):
    return (discriminationScore.pPC-discriminationScore.pP*discriminationScore.pC)/np.sqrt(discriminationScore.lenALL*discriminationScore.pP*discriminationScore.pC*discriminationScore.pnotP*discriminationScore.pnotC)

def RelRisk(discriminationScore): #GOOD
    numerator = discriminationScore.pCassumingP 
    denominator = discriminationScore.pCassumingnotP 

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def Sebag(discriminationScore): #GOOD
    numerator = discriminationScore.pPC
    denominator = discriminationScore.pPnotC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def Spec(discriminationScore): #GOOD
    return discriminationScore.pnotCassumingnotP

def Strenght(discriminationScore): #GOOD
    numerator = GR(discriminationScore)
    denominator = numerator + 1 

    # Use np.where to handle the special case
    result = np.where(numerator == np.inf, discriminationScore.pPC , (numerator / denominator)* discriminationScore.pPC)
    return result

def Supp(discriminationScore): #GOOD
    return discriminationScore.pPC

def SuppDif(discriminationScore): #GOOD
    return discriminationScore.pPassumingC - discriminationScore.pPassumingnotC

def SuppDifAbs(discriminationScore): #GOOD
    return np.abs(discriminationScore.pPassumingC - discriminationScore.pPassumingnotC)

def WRACC(discriminationScore): #GOOD
    return discriminationScore.pP * (discriminationScore.pCassumingP-discriminationScore.pC)

def chiTwo(discriminationScore): #GOOD
    return discriminationScore.lenALL*(discriminationScore.pPC*discriminationScore.pnotPnotC-discriminationScore.pPnotC*discriminationScore.pnotPC)**2/(discriminationScore.pP*discriminationScore.pC*discriminationScore.pnotP*discriminationScore.pnotC)

def Zhang(discriminationScore): #GOOD
    maxi = np.ones(len(discriminationScore.pP))
    for i in range(len(discriminationScore.pP)):
        maxi[i] = max(discriminationScore.pPC[i]*discriminationScore.pnotC,discriminationScore.pC*discriminationScore.pPnotC[i])
    return (discriminationScore.pPC - discriminationScore.pP*discriminationScore.pC)/(maxi)


#Fin A Faire Lucas
def TPR(discriminationScore):
    return discriminationScore.pCassumingP

def FPR(discriminationScore):
    result = np.where(discriminationScore.pnotCassumingP == 0, float('inf'), 1/discriminationScore.pnotCassumingP)
    return result


def CertaintyFactor(discriminationScore):
    return (discriminationScore.pCassumingP - discriminationScore.pC) / (1 - discriminationScore.pC)

# Les mesures à rajouter : 
def Gini(discriminationScore):
    gini_index = 1 - (discriminationScore.pCassumingP ** 2 + discriminationScore.pnotCassumingP ** 2)
    return 1/(gini_index+0.0000000001)

def Entropy(discriminationScore):
    epsilon = 1e-10  # Avoid log(0)
    p0 = discriminationScore.pnotCassumingP
    p1 = discriminationScore.pCassumingP
    
    entropy = - (p0 * np.log2(p0 + epsilon) + p1 * np.log2(p1 + epsilon))
    return 1 / (entropy + epsilon)

def Fisher(discriminationScore):
    epsilon = 1e-10  # Avoid division by zero
    mean_diff = (discriminationScore.pCassumingP - discriminationScore.pnotCassumingP) ** 2
    var_sum = discriminationScore.pCassumingP * (1 - discriminationScore.pCassumingP) + \
              discriminationScore.pnotCassumingP * (1 - discriminationScore.pnotCassumingP)
    
    return mean_diff / (var_sum + epsilon)

def creationDictionnaryScores():
    dico = {
        "Acc": Acc,
        "Brins": Brins,
        "CConf": CConf,
        "Cole": Cole,
        "ColStr": ColStr,
        "Conf": Conf,
        "Cos": Cos,
        "Cover": Cover,
        "Dep": Dep,
        "Entropy": Entropy,
        "Excex": Excex,
        "Fisher": Fisher,
        "Gain": Gain,
        "Gini": Gini,
        "GR": GR,
        "InfGain": InfGain,
        "Jacc": Jacc,
        "Klos": Klos,
        "Lap": Lap,
        "Lever": Lever,
        "Lift": Lift,
        "MDisc": MDisc,
        "MutInf": MutInf,
        "NetConf": NetConf,
        "OddsR": OddsR,
        "Pearson": Pearson,
        "RelRisk": RelRisk,
        "Sebag": Sebag,
        "Spec": Spec,
        "Strenght": Strenght,
        "Supp": Supp,
        "SuppDif": SuppDif,
        "SuppDifAbs": SuppDifAbs,
        "WRACC": WRACC,
        "chiTwo": chiTwo,
        "Zhang": Zhang,
        "FPR": FPR,
        "CertaintyFactor": CertaintyFactor}
    return {k: dico[k] for k in sorted(dico)}


def readLabels(fileLabel):
    """ this function reads the file containing the labels of the graphs
        and convert them into 2 classes : 0 and 1
        
    Input : fileLabel (string) : the name of the file containing the labels
    Output : labels (list of int) : the list of labels of the graphs"""
    
    file=open(fileLabel,"r")
    labels = []
    numero=0
    for line in file:
        line = str(line).split("\t")[0]
        if int(line)==-1:
            labels.append(0)
        elif int(line)>-1:
            labels.append(min(int(line),1))
        numero=numero+1
    return labels


def readLabels(fileLabel):
    """ this function reads the file containing the labels of the graphs
        and convert them into 2 classes : 0 and 1
        
    Input : fileLabel (string) : the name of the file containing the labels
    Output : labels (list of int) : the list of labels of the graphs"""
    
    file=open(fileLabel,"r")
    labels = []
    numero=0
    for line in file:
        line = str(line).split("\t")[0]
        if int(line)==-1:
            labels.append(0)
        elif int(line)>-1:
            labels.append(min(int(line),1))
        numero=numero+1
    return labels

def KVector(keep,K,diff,id_graphs,numberoccurences,LENGTHGRAPH,labels):
    """ this fuction creates the vectorial representation of the graphs
        Input : K (int) : the number of patterns to keep
        diff (list of int) : the list of discrimination scores of each pattern
        id_graphs (list of list of int) : the list of graphs containing of each pattern
        numberoccurences (list of list of int) : the list of number occurences of each pattern in each graph
        LENGTHGRAPH (int) : the number of graphs
        labels (list of int) : the list of labels of the graphs
        
        
        Output : X (list of list of int) : the vectorial representation of the graphs
        Y (list of int) : the list of labels of the graphs""" 
    keepPatterns = []
    for i in tqdm.tqdm(range(K)):
        if sum(diff)==0:
            break
        bestScore = np.max(diff)
        bestPattern = np.argmax(diff)
        keepPatterns.append(bestPattern)
        diff[bestPattern]=0
    vectorialRep = []
    newLabels = []
    c=0
    for j in tqdm.tqdm(range(LENGTHGRAPH)):#330
        if j in keep:
            vectorialRep.append([])
            for k in keepPatterns:
                if j in id_graphs[k]:
                    for t in range(len(id_graphs[k])):
                        if id_graphs[k][t]==j:
                            if numberoccurences==None:
                                occu=1
                            else:
                                occu = numberoccurences[k][t]
                    vectorialRep[c].append(occu)
                else:
                    vectorialRep[c].append(0)
            c=c+1
    X = vectorialRep
    return X,keepPatterns,labels

import sys, getopt

def graphKeep(Graphes,labels):
    """Equilibrate the number of graphs in each class"""
    ### Equilibre dataset
    if len(labels)-sum(labels)>sum(labels):
        minority=1
        NbMino=sum(labels)
    else:
        minority =0
        NbMino=len(labels)-sum(labels)
    keep = []
    count=0
    graphs=[]
    for i in range(len(labels)):
        if labels[i]==minority:
            keep.append(i)
    complete=NbMino
    for i in range(len(labels)):   
        if labels[i]!=minority:
            if count<complete:
                count=count+1
                keep.append(i)
    return keep



def cross_validation(X,Y,cv,classifier):
    #store for each fold the F1 score of each class
    F1_score0 = []
    F1_score1 = []
    # for each fold
    for train_index, test_index in cv.split(X,Y):
        #split the dataset into train and test
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        
    F1_score0_mean = np.mean(F1_score0)
    F1_score0_std = np.std(F1_score0)
    F1_score1_mean = np.mean(F1_score1)
    F1_score1_std = np.std(F1_score1)
    #return the mean and standard deviation of the F1 score of each class
    return F1_score0_mean,F1_score0_std,F1_score1_mean,F1_score1_std


def pangProcessing(Ks,keep,labels,id_graphs_mono,id_graphs_iso,occurences_mono,occurences_iso,LENGTHPATTERN,LENGTHGRAPHS):
    cv = StratifiedKFold(n_splits=10,shuffle=True)
    scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,LENGTHPATTERN)
    scoresGenOcc = computeScoreOccurences(keep,labels,id_graphs_mono,occurences_mono,LENGTHPATTERN)
    scoresIndBin = computeScoreMono(keep,labels,id_graphs_iso,LENGTHPATTERN)
    scoresIndOcc = computeScoreOccurences(keep,labels,id_graphs_iso,occurences_iso,LENGTHPATTERN)
    for K in Ks:
        X_GenBin = KVector(keep,K,scoresGenBin,id_graphs_mono,None,LENGTHGRAPHS,labels)
        X_GenOcc = KVector(keep,K,scoresGenOcc,id_graphs_mono,occurences_mono,LENGTHGRAPHS,labels)
        X_IndBin = KVector(keep,K,scoresIndBin,id_graphs_iso,None,LENGTHGRAPHS,labels)
        X_IndOcc = KVector(keep,K,scoresIndOcc,id_graphs_iso,occurences_iso,LENGTHGRAPHS,labels)
        results = np.zeros((4,2,2))
        representations = [[X_GenBin,labels],[X_GenOcc,labels],[X_IndBin,labels],[X_IndOcc,labels]]
        for i in range(len(representations)):
            results[i][0][0],results[i][0][1],results[i][1][0],results[i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(C=0.1))
        outputResults(K,results,"../data/"+str(K)+"results.txt")
    return 0

def outputResults(K,results, output_file):
    """ function to output the results of the cross validation
        Input : K : number of patterns
        results : array containing the results of the cross validation
        output_file : file where the results will be written"""
    f = open(output_file,"w")
    f.write("Results for K = "+str(K)+ " \n")
    for i in range(len(results)):
            f.write("Representation "+str(i)+"\n")
            f.write("F1 score for class 0: "+str(results[i][0][0])+" +/- "+str(results[i][0][1])+"\n")
            f.write("F1 score for class 1: "+str(results[i][1][0])+" +/- "+str(results[i][1][1])+"\n")
    f.close()

def printBestPattern(patterns,labels):
    for i in range(len(patterns)):
        if labels[i]==1:
            print(patterns[i])
def main(argv):
    opts, args = getopt.getopt(argv,"hd:o:k:",["ifile=","ofile="])
    for opt, arg in opts:
      if opt == '-h':
         print ('PANG.py -d <dataset> -k<values of K> -m<mode>')
         sys.exit()
      elif opt in ("-d"):
            folder="../data/"+str(arg)+"/"
            FILEGRAPHS=folder+str(arg)+"_graph.txt"
            FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
            FILEMONOSET=folder+str(arg)+"_mono.txt"
            FILEISOSET=folder+str(arg)+"_iso.txt"
            FILELABEL =folder+str(arg)+"_label.txt"
            TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
            TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)
      elif opt in ("-k"):
         Ks=[]
         temp=arg
         temp=temp.split(",")
         for j in range(len(temp)):
             Ks.append(int(temp[j]))
    keep= []
    dele =[]
    for i in range(TAILLEGRAPHE):
        if i not in dele:
            keep.append(i)
            
    """loading graphs"""
    print("Reading graphs")
    Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,TAILLEGRAPHE)
    """loading patterns"""
    print("Reading patterns")
    Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,TAILLEPATTERN)
    
    """loading processed patterns"""
    print("Reading processed patterns")
    xx,id_graphsMono,numberoccurencesMono = load_patterns(FILEMONOSET,TAILLEPATTERN)
    xx,id_graphsIso,numberoccurencesIso = load_patterns(FILEISOSET,TAILLEPATTERN)
    keep = graphKeep(PatternsRed,FILELABEL)
    print("Processing PANG")
    pangProcessing(Ks,keep,labels,id_graphsMono,id_graphsIso,numberoccurencesMono,numberoccurencesIso,TAILLEPATTERN,TAILLEGRAPHE)



import os

def plot_figures(arg, ress, originalSS, feature, scoressss, dataset_name):
    # create the directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists("results/" + arg):
        os.makedirs("results/" + arg)

    # plot the figures
    position = []
    position2 = []
    position3 = []
    position4 = []
    for i in range(len(ress[0])):
        #check the position of the feature in the list of features deleted  
        position.append(originalSS[0].index(ress[0][i]))
        position2.append(originalSS[1].index(ress[1][i]))
        position3.append(originalSS[2].index(ress[2][i]))
        position4.append(originalSS[3].index(ress[3][i]))
    plt.plot(feature,position,label="GenBin")
    plt.plot(feature,position2,label="GenOcc")
    plt.plot(feature,position3,label="IndBin")
    plt.plot(feature,position4,label="IndOcc") 
    #Name the x axis : features number
    plt.xlabel('Features number')
    plt.ylabel('Step of deletion')
    #Affiche la légende
    plt.legend()
    plt.title('Features importance')

    # save the plot in a specific directory
    name = "results/" + arg + "/" + dataset_name + str(len(ress)) + "NN.pdf"
    plt.savefig(name)

    # plot the second figure
    scoress0 = scoressss[0]
    scoress1 = scoressss[1]
    scoress2 = scoressss[2]
    scoress3 = scoressss[3]

    plt.figure()
    plt.xlabel('Number of features')
    plt.ylabel('FA-Score of the anomalous class')
    plt.plot([len(feature)+1]+feature,scoress0,label="GenBin")
    plt.plot([len(feature)+1]+feature,scoress1,label="GenOcc")
    plt.plot([len(feature)+1]+feature,scoress2,label="IndBin")
    plt.plot([len(feature)+1]+feature,scoress3,label="IndOcc")
    plt.legend()

    # save the second plot in a specific directory
    name = "results/" + arg + "/" + dataset_name + str(len(ress)) + "F1Score.pdf"
    plt.savefig(name)


def tableScore(K,patterns,file,h):
    f=open(file,"a")
    eachPatterns = []
    for i in range(len(patterns)):
         for j in range(len(patterns[i])):
             eachPatterns.append(patterns[i][j])
    uniquePatterns=len(set(eachPatterns))
    lenPatterns=len(eachPatterns)
    ratio = uniquePatterns/lenPatterns
    ratio = 1-ratio
    #ne garder que 3 chiffres apres la virgule
    ratio = round(ratio,3)
    #On ecrit la valeur de K
    f.write("K="+str(K)+"\n")
    #Pour chaque score, on ecrit le ratio UniquePatterns/lenPatterns

    for i in range(len(patterns)):
        f.write("K="+str(K)+" : Score "+str(h)+ ": "+str(ratio)+"\n")
    f.close()


import numpy as np

from sklearn.cluster import KMeans


def select_k(X,k_range):
    wss = np.empty(len(k_range))

    for k in k_range:
        kmeans = KMeans(k)
        kmeans.fit(X)
        wss[k-1] = ((X - kmeans.cluster_centers_[kmeans.labels_]) ** 2).sum()

    slope = (wss[0] - wss[-1]) / (k_range[0] - k_range[-1])
    intercept = wss[0] - slope * k_range[0]
    y = k_range * slope + intercept

    return k_range[(y - wss).argmax()]

def metricDotProduct(X):
    print("Avant")
    a = (len(X[0])-np.matmul(X,np.transpose(X)))/2
    print("Apres")
    return a


def keepOnlyPatterns(clusters):
    #Select 1 pattern per cluster
    #clusters = model.labels_
    nbClusters = max(clusters)+1
    patterns = []
    for i in range(nbClusters):
        for j in range(len(clusters)):
            if clusters[j]==i:
                patterns.append(j)
                break
    return patterns

def performClustering(pattern,distance):
    model = AgglomerativeClustering(distance_threshold=distance,metric="precomputed",n_clusters=None,linkage="complete")
    model = model.fit(pattern)
    return model

#import cdist
from scipy.spatial.distance import cdist
def selectCurrentClustering(pattern,distance,id_graphs,convertisseur,nbPointPerCluster,TypeMedoid,superMatrice):
    """ This function perform the full clustering for one specific value"""
    newID_graphs = []
    convertisseur = {}
    resUnique = []
    model = performClustering(pattern,distance)
    clusters = model.labels_
    n_clusters = max(clusters)+1
    #Create a dictionnary associating to each pattern the cluster it belongs to
    dicoClusterPattern = {}
    for i in range(len(clusters)):
        dicoClusterPattern[i]=clusters[i]
    res = []
    # Créer un dictionnaire qui associe à chaque cluster la liste des id des motifs qui lui appartiennent
    dicoCluster = {}
    # Calculer le centroïde de chaque cluster
    for cluster_id in range(n_clusters):
        cluster_points = []
        id_clusters_points = []
        for i in range(len(model.labels_)):
            if model.labels_[i]==cluster_id:
                cluster_points.append(superMatrice[i])
                id_clusters_points.append(i)
        for k in range(min(nbPointPerCluster,len(cluster_points))):
            cluster_centroid = np.mean(cluster_points, axis=0)
            # Calculer la distance de chaque point du cluster au centroïde
            distances = cdist(cluster_points, [cluster_centroid])
            
            # Trouver l'indice du point le plus proche du centroïde
            central_point_index = np.argmin(distances)

            # Trouver l'indice du point le plus loin du point le plus proche du centroïde
            long_point_index = np.argmax(distances)
            
            # Ajouter l'id du point le plus central au tableau
            # Supprimer le point le plus central du cluster
            if k==0:
                res.append(id_clusters_points[central_point_index])
                newID_graphs.append(id_graphs[id_clusters_points[central_point_index]])
                cluster_points.pop(central_point_index)
                id_clusters_points.pop(central_point_index)
            else:
                if TypeMedoid=="M":
                    res.append(id_clusters_points[central_point_index])
                    newID_graphs.append(id_graphs[id_clusters_points[central_point_index]])
                    cluster_points.pop(central_point_index)
                    id_clusters_points.pop(central_point_index)
                if TypeMedoid=="F":
                    res.append(id_clusters_points[long_point_index])
                    newID_graphs.append(id_graphs[id_clusters_points[long_point_index]])
                    cluster_points.pop(long_point_index)
                    id_clusters_points.pop(long_point_index)
    return model,res,convertisseur,newID_graphs

def ExtendDictionnary(dicoClusterPattern,dicoRepetition):
    newDictionnary = {}
    for key in dicoRepetition.keys():
        value=dicoRepetition[key]
        for patt in value:
            #si le pattern est dans le dictionnaire
            if patt in newDictionnary.keys():  
                print("??")
            else:
                newDictionnary[patt]=dicoClusterPattern[key] 
    #display the number of keys in the new dictionnary
    return newDictionnary

def ComputeRepresentation(keep,keepPatterns,id_graphs,labels,LENGTHGRAPH):
    numberoccurences=None
    vectorialRep = []
    newLabels = []
    c=0
    for j in tqdm.tqdm(range(LENGTHGRAPH)):#330
            if j in keep:
                newLabels.append(labels[j])
                vectorialRep.append([])
                for k in keepPatterns:
                    if j in id_graphs[k]:
                        for t in range(len(id_graphs[k])):
                            if id_graphs[k][t]==j:
                                if numberoccurences==None:
                                    occu=1
                                else:
                                    occu = numberoccurences[k][t]
                        vectorialRep[c].append(occu)
                    else:
                        vectorialRep[c].append(0)
                vectorialRep[c] = np.array(vectorialRep[c])
                c=c+1
    X = vectorialRep
    return X,newLabels

def AlterateMetric(metric,patternsAGarder):
    for i in range(len(metric)):
        if i not in patternsAGarder:
            metric[i]=-1000
    return metric


def replacePatternByInduced(induced):
    #Before : idgraphs represente les id des motifs generaux
    #After : idgraphs represente les id des motifs induits 
    #TAILLEPATTERN represente le nombre de motifs induits
    # quand on a un motif pas present dans les induits on le supprime
    motifs = []
    for i in range(len(induced)):
        if induced[i] != []:
            motifs.append(induced[i])
    return motifs,len(motifs)


def load_dataset(arg,mode):
    if mode == "c":
        motifs = "CLOSED"
    else:
        motifs = "GENERAUX"
    folder="../data/"+str(arg)+"/"
    FILEGRAPHS=folder+str(arg)+"_graph.txt"
    if motifs == "GENERAUX":
        FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
    if motifs == "CLOSED":
        FILESUBGRAPHS=folder+str(arg)+"_CGSPAN.txt"
    FILELABEL =folder+str(arg)+"_label.txt"
    FILEISOSET=folder+str(arg)+"_isoA.txt"
    TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
    TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)
    keep= []
    dele =[]
    for i in range(TAILLEGRAPHE):
        if i not in dele:
            keep.append(i)
    
    """loading graphs"""
    print("Reading graphs")
    Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,TAILLEGRAPHE)
    """loading patterns"""
    print("Reading patterns")
    Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,TAILLEPATTERN)

    """loading processed patterns"""
    if mode == "i":
        print("Reading processed patterns")
        xx,id_graphsIso,numberoccurencesIso = load_patterns(FILEISOSET,TAILLEPATTERN)

        id_graphs,LENGTHPATTERN = replacePatternByInduced(id_graphsIso)


    labelss = readLabels(FILELABEL)
    keep = graphKeep(PatternsRed,labelss)
    labels=[]
    for i in range(len(labelss)):
        if i in keep:
            labels.append(labelss[i])

    return id_graphs,labelss,keep,TAILLEGRAPHE



def KendallTauHistograms(argu,mode,id_graphsMono,labelss,keep,TAILLEGRAPHE):
    cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    #1) Clustering step
    #create a dictionnary, for each pattern , indicate the unique pattern it belongs to
    dicoRepetition = {}
    dicoUniqueToPattern = {}
    convertisseur = {}
    patternsUnique=[]
    dejaVu = []
    c=-1
    for i in tqdm.tqdm(range(len(id_graphsMono))):
        if id_graphsMono[i]==[]:
            dicoRepetition[i]=-1
            dicoUniqueToPattern[c] = []
        else:
            if id_graphsMono[i] not in dejaVu:
                patternsUnique.append(i)
                c = c+1
                dejaVu.append(id_graphsMono[i]) 
                dicoRepetition[i]=c
                dicoUniqueToPattern[c] = []
                convertisseur[c]=i
                dicoUniqueToPattern[c].append(i)
            else:
                dicoRepetition[i]=dejaVu.index(id_graphsMono[i])
                dicoUniqueToPattern[dejaVu.index(id_graphsMono[i])].append(i)

    dicoRepetition = {}
    patternsUnique=[]
    dejaVu = []
    for i in tqdm.tqdm(range(len(id_graphsMono))):
        if id_graphsMono[i] not in dejaVu:
            patternsUnique.append(i)
            dejaVu.append(id_graphsMono[i]) 
            dicoRepetition[i]=i
        else:
            dicoRepetition[i]=dejaVu.index(id_graphsMono[i])
    superMatrice = np.ones((len(dejaVu),TAILLEGRAPHE),dtype=np.int8)*-1
    for i in range(len(dejaVu)):
        for j in range(len(dejaVu[i])):
                superMatrice[i][dejaVu[i][j]]=1
    ### transform the matrix into a list of list


    ### Resume : superMatrice is a matrix where each row is a unique pattern
    # Each column is a graph
    # If the graph contains the pattern, the value is 1, else -1
    #Dejavu is the list of ids associated to each unique pattern
    #patternsUnique is the list of unique patterns
    superMatrice = superMatrice.tolist()
    dotProductMat = metricDotProduct(superMatrice)
    for i in range(len(superMatrice)):
        superMatrice[i]=np.array(superMatrice[i])

    vectRepresentation,vectLabels = ComputeRepresentation(keep,range(0,len(patternsUnique)),dejaVu,labelss,TAILLEGRAPHE)
    vectRepresentation = np.array(vectRepresentation)
    #RS = 7
    X_train, X_test, y_train, y_test = train_test_split(vectRepresentation, vectLabels, test_size=0.2, random_state=7, stratify=vectLabels)
    Range = []
    dicoFinal = {}
    #create dico of scores
    dicoScores = creationDictionnaryScores()
    #create dico of Rankings
    dicoRankings = {}
    for i in range(0,101):
        Range.append(int(i*TAILLEGRAPHE/100))
    for VALUECLUSTER in Range:
        model,patternsStepI,convertisseur,newIdGraphs = selectCurrentClustering(dotProductMat,VALUECLUSTER,dejaVu,convertisseur,1,"M",superMatrice)
        #2) Comparison step of discrimination score
        #Compute all patterns measures
        discriminationScores = patternMeasures(keep,labelss,dejaVu,len(dejaVu))
        #for each score
        compteur=0
        for score in dicoScores.keys():
            #compute the score
            scoresValues = dicoScores[score](copy.deepcopy(discriminationScores))
            scoresValues = AlterateMetric(scoresValues,patternsStepI)
            #Put -1 too if the pattern is not present in at least one graph
            # info is contained in the attribut toConsider of discriminationScores
            scoresValues = np.array(scoresValues)
            for i in range(len(scoresValues)):
                if discriminationScores.toConsider[i]==-2:
                    scoresValues[i]=-1000
            #count the number of -1 in the scores
            delete = np.count_nonzero(scoresValues == -1000)
            #Rank the patterns according to the score in decreasing order
            dicoRankings[compteur]=np.argsort(scoresValues)[::-1]
            #keep only the patterns with a score different from -1, i.e total number of patterns - number of -1, which is delete
            dicoRankings[compteur] = dicoRankings[compteur][0:len(dicoRankings[compteur])-delete]
            compteur=compteur+1
        dicoFinal[VALUECLUSTER]=copy.deepcopy(dicoRankings)
    import scipy
    
    NAMEBASE = "../results/"+argu+"/KendallTauHistograms/"
    #if the folder does not exist, create it
    if not os.path.exists(NAMEBASE):
        os.makedirs(NAMEBASE)
    plt.figure()
    res0 = []
    res20 = []
    res40 = []
    res60 = []
    res80 = []

    for i in range(0,len(dicoScores.keys())):
        for j in range(i+1,len(dicoScores.keys())):
            res = []
            for k in Range:
                res.append(scipy.stats.kendalltau(dicoFinal[k][i],dicoFinal[k][j])[0])
                #print(res)
            res0.append(res[0])
            res20.append(res[20])
            res40.append(res[40])
            res60.append(res[50])
            res80.append(res[80])
    bins = np.linspace(-1,1,100)
    histo0 = np.histogram(res0,bins=bins)
    histo20 = np.histogram(res20,bins=bins)
    histo40 = np.histogram(res40,bins=bins)
    histo60 = np.histogram(res60,bins=bins)
    histo80 = np.histogram(res80,bins=bins)

    histoVal0 = np.array(histo0[0])/sum(np.array(histo0[0]))
    histoVal20 = np.array(histo20[0])/sum(np.array(histo20[0]))
    histoVal40 = np.array(histo40[0])/sum(np.array(histo40[0]))
    histoVal60 = np.array(histo60[0])/sum(np.array(histo60[0]))

    plt.bar(histo60[1][0:len(histo60[1])-1],histoVal60,width=0.1,label="Clustering = 60%",alpha=1,color="blue")
    plt.bar(histo40[1][0:len(histo40[1])-1],histoVal40,width=0.1,label="Clustering = 40%",alpha=1,color="orange")
    plt.bar(histo0[1][0:len(histo0[1])-1],histoVal0,width=0.1,label="Clustering = 0%",alpha=1,color="green")
    plt.bar(histo20[1][0:len(histo20[1])-1],histoVal20,label="Clustering = 20%",width=0.1,alpha=1,color="red")

    order = [2,3,1,0] 
    handles, labels = plt.gca().get_legend_handles_labels()
    # pass handle & labels lists along with order as below 
    plt.legend([handles[i] for i in order], [labels[i] for i in order]) 

    plt.xlabel('Kendall Tau')
    plt.ylabel('Numbers of pairs (%)')
    #fixer les limites de l'axe des y entre 0 et 0.8
    plt.ylim(0,0.8)
    plt.savefig(NAMEBASE+"ComparisonMetricsKTALL.pdf")

    #Faire un graphique par val
    plt.figure()
    plt.bar(histo0[1][0:len(histo0[1])-1],histoVal0,width=0.1,label="Clustering = 0%",alpha=1,color="green")
    plt.xlabel("Kendall's Tau")
    plt.ylabel('Numbers of pairs (%)')
    #fixer les limites de l'axe des y entre 0 et 0.8
    plt.ylim(0,0.8)
    plt.legend()
    plt.savefig(NAMEBASE+"ComparisonMetricsKT0.pdf")

    plt.figure()
    plt.bar(histo20[1][0:len(histo20[1])-1],histoVal20,width=0.1,label="Clustering = 20%",alpha=1,color="red")
    plt.xlabel("Kendall's Tau")
    plt.ylabel('Numbers of pairs (%)')
    #fixer les limites de l'axe des y entre 0 et 0.8
    plt.ylim(0,0.8)
    plt.legend()
    plt.savefig(NAMEBASE+"ComparisonMetricsKT20.pdf")

    plt.figure()
    plt.bar(histo40[1][0:len(histo40[1])-1],histoVal40,width=0.1,label="Clustering = 40%",alpha=1,color="orange")
    plt.xlabel("Kendall's Tau")
    plt.ylabel('Numbers of pairs (%)')
    #fixer les limites de l'axe des y entre 0 et 0.8
    plt.ylim(0,0.8)
    plt.legend()
    plt.savefig(NAMEBASE+"ComparisonMetricsKT40.pdf")

    plt.figure()
    plt.bar(histo60[1][0:len(histo60[1])-1],histoVal60,width=0.1,label="Clustering = 60%",alpha=1,color="blue")
    plt.xlabel("Kendall's Tau")
    plt.ylabel('Numbers of pairs (%)')
    #fixer les limites de l'axe des y entre 0 et 0.8
    plt.ylim(0,0.8)
    plt.legend()
    plt.savefig(NAMEBASE+"ComparisonMetricsKT60.pdf")

    return 0


def main(argv):
    opts, args = getopt.getopt(argv,"d:m:",["ifile=","ofile="])
    mode = ""
    for opt, arg in opts:
        if opt == '-h':
          print ('PANG.py -d <dataset> -m<mode>')
          sys.exit()
        elif opt in ("-d"):
            dataset = arg
        elif opt in ("-m"):
            mode = arg
    
    #load the dataset
    id_graphs,labels,keep,TAILLEGRAPHE = load_dataset(dataset,mode)

    #Launch the experiment
    KendallTauHistograms(dataset,mode,id_graphs,labels,keep,TAILLEGRAPHE)


if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
