from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import ISMAGS
import copy
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
import rbo
import scipy as sp
from sklearn.cluster import AgglomerativeClustering
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV

import sys, getopt
from sklearn import metrics

def read_Sizegraph(fileName):
    """Read the number of graphs in a file.
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
    NbMino = 0
    count=0
    graphs=[]
    for i in range(len(labels)):
        if labels[i]==minority:
            NbMino=NbMino+1
            keep.append(i)
    complete=NbMino
    for i in range(len(labels)):   
        if labels[i]!=minority:
            if count<complete:
                count=count+1
                keep.append(i)

    return keep



def metricDotProduct(X):
    a = (len(X[0])-np.matmul(X,np.transpose(X)))/2
    return a

def ComputeRepresentation(keep,keepPatterns,id_graphs,labels,LENGTHGRAPH):
    numberoccurences=None
    vectorialRep = []
    newLabels = []
    c=0
    for j in range(LENGTHGRAPH):#330
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

def partialRepresentation(X,patterns):
    return X[:,np.array(patterns)]
def ExpeNumbersClusters(argu,mode,id_graphsMono,labels,keep,TAILLEGRAPHE):
    dicoC = {"MUTAG":0.1,"PTC":1000,"NCI1":1000,"FOPPA":1000,"DD":1000,"AIDS":1000,"FRANKENSTEIN":1000,"IMDB":1000}
    NBPATTERNBASE = len(id_graphsMono)
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
   
    superMatrice = superMatrice.tolist()
    dotProductMat = metricDotProduct(superMatrice)
    for i in range(len(superMatrice)):
        superMatrice[i]=np.array(superMatrice[i])
    
    vectTotal,vectLabelTotal = ComputeRepresentation(keep,range(0,len(id_graphsMono)),id_graphsMono,labels,TAILLEGRAPHE)
    vectRepresentation,vectLabels = ComputeRepresentation(keep,range(0,len(patternsUnique)),dejaVu,labels,TAILLEGRAPHE)
    vectRepresentation = np.array(vectRepresentation)
    vectLabels = np.array(vectLabels)

    #CV initialization
    cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    x_train_ind = []
    x_test_ind= []
    for train_index, test_index in cv.split(vectRepresentation,vectLabels):
        x_train_ind.append(train_index)
        x_test_ind.append(test_index)
    
    # for each fold of the cross validation
    F1_score0 = []
    F1_score1 = []
    #index of the fold are stored in x_train_ind and x_test_ind
    for i in range(len(x_train_ind)):
        x_train_np = []
        x_test_np = []
        for j in range(len(x_train_ind[i])):
            x_train_np.append(int(x_train_ind[i][j]))
        for j in range(len(x_test_ind[i])):
            x_test_np.append(int(x_test_ind[i][j]))

        x_test_np = np.array(x_test_np)
        x_train_np = np.array(x_train_np)
        vectTotal = np.array(vectTotal)
        vectLabelTotal = np.array(vectLabelTotal)
        X_train = vectTotal[x_train_np]
        X_test = vectTotal[x_test_np]
        y_train = vectLabelTotal[x_train_np]
        y_test = vectLabelTotal[x_test_np]
        #train the classifier
        clf = SVC(C=dicoC[argu])
        clf.fit(X_train,y_train)
        #predict the test set
        y_pred = clf.predict(X_test)
        #compute the F1 score of each class
        from sklearn.metrics import f1_score
        F1_score0.append(f1_score(y_test,y_pred,pos_label=0))
        F1_score1.append(f1_score(y_test,y_pred,pos_label=1))
    F1Debut = np.mean(F1_score1)

    Range = []
    nbRepres = []
    F1Score = []
    for i in range(0,100):
        Range.append(int(i*TAILLEGRAPHE/100))
    for distance in tqdm.tqdm(Range):
        # Clustering Step
        model,res,convertisseur,newIdGraphs = selectCurrentClustering(dotProductMat,distance,dejaVu,convertisseur,1,"M",superMatrice)
        nbRepres.append(len(res))
        #Reorganize the patterns
        patternsUnique = sorted(res)
        patternsUnique = np.array(patternsUnique)
        # Compute the vectorial representation with the representatives
        newVectRepresentation = partialRepresentation(copy.deepcopy(vectRepresentation),patternsUnique)
        # On calcule le F1 score
        # for each fold of the cross validation 
        F1_score0 = []
        F1_score1 = []
        #index of the fold are stored in x_train_ind and x_test_ind
        for i in range(len(x_train_ind)):
            x_train_np = []
            x_test_np = []
            for j in range(len(x_train_ind[i])):
                x_train_np.append(int(x_train_ind[i][j]))
            for j in range(len(x_test_ind[i])):
                x_test_np.append(int(x_test_ind[i][j]))

            x_test_np = np.array(x_test_np)
            x_train_np = np.array(x_train_np)

            X_train = newVectRepresentation[x_train_np]
            X_test = newVectRepresentation[x_test_np]
            y_train = vectLabels[x_train_np]
            y_test = vectLabels[x_test_np]
            #train the classifier
            SVC(C=dicoC[argu])
            clf.fit(X_train,y_train)
            #predict the test set
            y_pred = clf.predict(X_test)
            #compute the F1 score of each class
            from sklearn.metrics import f1_score
            F1_score0.append(f1_score(y_test,y_pred,pos_label=0))
            F1_score1.append(f1_score(y_test,y_pred,pos_label=1))
        #store the mean of F1 score of class 1 
        F1Score.append(np.mean(F1_score1))
    #plot the number of representatives and the F1 score, one per figure
    NAMEBASE = "../results/"+argu+"/ClusteringComparison/"
    #if the folder does not exist, create it
    if not os.path.exists(NAMEBASE):
        os.makedirs(NAMEBASE)

    plt.figure()
    plt.plot(np.linspace(0,100,100),nbRepres)
    plt.axhline(y=NBPATTERNBASE, color='r', linestyle='-')
    #put a logaritmic scale
    plt.yscale("log")
    plt.xlabel('Clustering threshold (%)')
    plt.ylabel('Number of representatives (log)')
    if mode == "c":
        NAMEMODE = "Closed"
    elif mode == "g":
        NAMEMODE = "General"
    else:
        NAMEMODE = "Induced"
    plt.savefig(NAMEBASE+"NumberRepresentatives"+NAMEMODE+".pdf")
    plt.close()
    plt.figure()
    #Find the best F1 score
    bestF1 = np.max(F1Score)
    #Find the position of the best F1 score
    bestPosition = np.argmax(F1Score)
    for i in range(len(F1Score)):
        if F1Score[i]>bestF1:
            bestPosition = i
    plt.plot(np.linspace(0,100,100),F1Score)
    plt.axhline(y=F1Debut, color='r', linestyle='-')
    plt.xlabel('Clustering threshold (%)')
    plt.ylabel('F1-score')
    if mode == "c":
        NAMEMODE = "Closed"
    elif mode == "g":
        NAMEMODE = "General"
    else:
        NAMEMODE = "Induced"
    plt.savefig(NAMEBASE+"F1Score"+NAMEMODE+".pdf")
    return 0

def main(argv):
    opts, args = getopt.getopt(argv,"d:m:",["ifile=","ofile="])
    print(opts)
    mode = ""
    for opt, arg in opts:
        if opt == '-h':
          print ('ClusteringComparison.py -d <dataset> -m<mode>')
          sys.exit()
        elif opt in ("-d"):
            dataset = arg
        elif opt in ("-m"):
            mode = arg

    #load the dataset
    id_graphs,labels,keep,TAILLEGRAPHE = load_dataset(dataset,mode)

    #Launch the experiment
    ExpeNumbersClusters(dataset,mode,id_graphs,labels,keep,TAILLEGRAPHE)

if __name__ == '__main__':
    argv = sys.argv[1:]
    print(argv)
    main(argv)
