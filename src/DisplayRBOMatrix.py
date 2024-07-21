import pandas
import numpy as np

# Read dataset
dataset = "MUTAG"
NAME = "results/"+dataset+"/PairwiseComparisons/PairwiseComparisonsKTGeneral.csv"
minDat = pandas.read_csv(NAME)

print(minDat)
#delete the two first columns
minDat.drop(minDat.columns[0], axis=1, inplace=True)
results = np.zeros((minDat.shape[0],minDat.shape[0]))
# Pour chaque colonne sauf la première
# Pour chaque ligne sauf la première
# Mettre la valeur

for i in range(0, minDat.shape[0]):
    for j in range(1, minDat.shape[1]):
        print("i: ", i, " j: ", j, " minDat[i,j]: ", minDat.iloc[i,j])
        results[i,j-1] = minDat.iloc[i,j]

columnsNames = list(minDat.columns)
print(columnsNames)
#delete the first column in columnsNames
columnsNames.pop(0)

AncienDictionnary = {}
for i in range(0, len(columnsNames)):
    AncienDictionnary[columnsNames[i]] = i


# Je veux faire des permutations
# Dico de permutation
PermutationDictionnary = {}
compteur = 0
Group1 = ["Conf","CertaintyFactor","GR",'Brins',"Cole","Lift","Sebag","Zhang","CConf","InfGain"]
Group2 = ["Acc","Lever","WRACC","SuppDif"]
Group3 = ["Cos","Strenght"]
Group4 = ["Cover","Supp"]
ComptSuivant = len(Group1)+len(Group2)+len(Group3)
for i in Group1:
    PermutationDictionnary[i] = compteur
    compteur += 1
for i in Group2:
    PermutationDictionnary[i] = compteur
    compteur += 1
for i in Group3:
    PermutationDictionnary[i] = compteur
    compteur += 1
for i in Group4:
    PermutationDictionnary[i] = compteur
    compteur += 1


for i in range(0, len(columnsNames)):
    if columnsNames[i] not in PermutationDictionnary:
        PermutationDictionnary[columnsNames[i]] = compteur
        compteur += 1
print(PermutationDictionnary)

newResults = np.zeros((minDat.shape[0],minDat.shape[0]))
newNames = []

for i in range(0, len(columnsNames)):
    # Append the name associated to the key i
    newNames.append([key for key, value in PermutationDictionnary.items() if value == i][0])
for i in range(0, len(columnsNames)): 
    for j in range(0, minDat.shape[0]):
        newResults[j,i] = results[AncienDictionnary[newNames[j]],AncienDictionnary[newNames[i]]]

# Noms des colonnes
dicoNomModifie = {
    "CertaintyFactor": "CFactor",
    "SuppDif": "SupDif",
    "Strenght": "Strength",
    "SuppDifAbs": "AbsSupDif",
}
#remplacer les noms dans newNames
for i in range(0, len(newNames)):
    if newNames[i] in dicoNomModifie:
        newNames[i] = dicoNomModifie[newNames[i]]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))

plt.imshow(newResults, interpolation='nearest',label='Correlation')
plt.colorbar()
#Add names
#aggrandir espace entre deux noms

plt.xticks(range(len(newNames)), newNames, rotation=90)
#mettre la lagende y en haut dans le meme ordre
plt.yticks(range(len(newNames)), newNames)

# Save the plt ing svg and pdf
#Rjouter de la marge en bas pour que les noms des colonnes soient visibles
plt.subplots_adjust(bottom=0.3)
