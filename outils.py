# Module de fonctions-outils pour l'analyse de données

import operator
import numpy as np # NUMerical PYthon
from numpy.random import randn, randint, rand
import pandas as pd # PANel DAtaS
from pandas import Series, DataFrame, Index
from pandas.tools.plotting import table
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import StandardScaler, maxabs_scale 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNetCV, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, cross_val_predict, train_test_split, \
StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from skll.metrics import kappa
from sklearn.metrics import precision_recall_fscore_support, roc_curve, roc_auc_score, \
confusion_matrix, classification_report, accuracy_score, average_precision_score, f1_score, \
cohen_kappa_score, make_scorer
from sklearn.svm import OneClassSVM
from sknn.mlp import Classifier, Layer
import warnings
from imblearn.over_sampling import SMOTE

def smote_multiclass(X, y):
    """
    Applique l'algorithme SMOTE pour équilibrer les nombres d'échantillons de chaque classe.
    X et y sont les données (matrice d'entrée et vecteur de sortie).
    Renvoie la DataFrame totale, X_smote et y_smote.
    """
    # on réassemble les données
    data = X.join(y)
    
    liste_classes = list(np.unique(y))
    
    # on trouve la classe la plus représentée
    liste_effectifs = data['quality'].value_counts().sort_values(inplace = False)
    classe_max = liste_effectifs[-1:].index[0]
    
    # on la supprime de la liste des classes à traiter
    liste_classes.remove(classe_max)
    
    # Création de la DataFrame qui va contenir toutes les données bruitées (et les autres)
    data_result = data[data['quality'] == classe_max]
    
    for classe in liste_classes:
        # extraction des données de la classe majoritaire et de classe
        data_classes = data[(data['quality'] == classe) | (data['quality'] == classe_max)]
        X_classes = data_classes[data_classes.columns[:-1]]
        y_classes = data_classes[data_classes.columns[-1:]]
        y_classes = np.ravel(y_classes)
        
        # Application de l'algorithme                   
        smote = SMOTE(ratio = 'auto', kind = 'regular',  n_neighbors = 3)                   
        X_classes_smote, y_classes_smote = smote.fit_sample(X_classes, y_classes)
        
        # On reforme une DataFrame à partir des deux tableaux
        data_classes_df = DataFrame(X_classes_smote, columns = X_classes.columns)
        data_classes_df['quality'] = y_classes_smote
        
        # Récupération des données bruitées
        data_classe = data_classes_df[data_classes_df['quality'] == classe]
        
        # Ajout des données bruitées au train set
        data_result = data_result.append(data_classe)
        
        #print("Fait pour la classe {}.".format(classe))
        
    return data_result, data_result[data_result.columns[:-1]], data_result[data_result.columns[-1:]]


def codage_ordinal(df, var, prefixe = ''):
    """
    Ajoute à la DataFrame df autant de colonnes qu'il y a de modalités de la variable var.
    Les modalités doivent se présenter sous la forme d'une liste d'entiers.
    Le codage est binaire ordinal, c'est-à-dire que pour 3 classes ordonnées 1, 2, 3 on va obtenir:
    - classe 1 : 100
    - classe 2 : 110
    - classe 3 : 111
    """
    # On détermine les modalités de var :
    tab = np.array(df[var])
    liste_mod = np.unique(tab)  
    nb_mod = len(liste_mod)
    
    for mod in liste_mod:      
        df[prefixe + '_' + str(mod)] =  (df[var] > mod - 1) * 1
    
    return df

def codage_disjonctif_adjacent(df, var, prefixe = ''):	
    """
 	Ajoute à la DataFrame df autant de colonnes qu'il y a de modalités de la variable var.
	Les modalités doivent se présenter sous la forme d'une liste d'entiers.
 	Le codage est binaire disjonctif, mais il tient compte des voisins immédiats :
    - classe 1 : 11000
 	- classe 2 : 11100
 	- classe 3 : 01100
 	- classe 4 : 00111
    - classe 5 : 00011
    """
    pass

def mapping(quality):
    """
    Création d'une nouvelle colonne 'Grade' à trois modalités :
    - 'Good' : si le vin appartient aux classes 7 ou 8
    - 'Average' : si le vin appartient aux classes 5 ou 6
    - 'Bad' : si le vin appartient aux classes 3 ou 4
    """
    if quality <= 4:
        return 'Bad'
    elif 5 <= quality <= 6:
        return 'Average'
    elif 7 <= quality <= 8:
        return 'Good'

def affichage_tchebychev():
    """
    Affiche les pourcentages du théorème de Tchebychev. On constate que pour obtenir un intervalle équivalent à
    3 sigma pour une loi normale, il faut tolérer un écart énorme pour une distribution non normale (K = 19).
    Un bon compromis pourrait être K = 10.
    """
    print('K pourcentage')
    for K in range(2, 20):
        print(K, '%.2f' %(100 * (1-(1/(K*K)))) )

def affiche_confmat(y_true, y_predict, liste_classes):
    """
    Affiche la matrice de confusion sous forme graphique.
    liste_classes est la liste des labels des classes.
    """
    confmat = confusion_matrix(y_true, y_predict)
    n, m = confmat.shape
    ax = sns.heatmap(confmat, annot = True, linewidth = 0.5, fmt = 'd', cbar = False)
    ax.set_xticklabels(liste_classes)
    ax.set_yticklabels(reversed(liste_classes), rotation = 0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    
    return ax
    
def resultats_multiclass(y_test, y_predict):
    """
    Affiche proprement les résultats de precision_recall_fscore_support.
    """
    liste_mod = np.unique(y_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
    df = DataFrame(index = ['precision', 'recall', 'fscore', 'support'])
    
    for i, mod in enumerate(liste_mod):
        df[mod] = [precision[i], recall[i], fscore[i], support[i]]
    
    return df

def proj_ACP(X, y):
    """
    Affiche les données dans le premier plan factoriel.
    X est la matrice d'entrée (var centrées réduites), y est le vecteur des classes de sortie.
    """
    # ACP 
    liste_classes = np.unique(y)
    nb_classes = len(liste_classes)
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X) # X est un tableau numpy
    # Attention, il ne faut pas perdre l'index !
    points = DataFrame(X_pca, columns = ['x', 'y'], index = X.index)
    points['classe'] = y # on ajoute l'info de classe
    
    # Liste des marqueurs possibles, 
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # couleurs
    colors = cm.rainbow(np.linspace(0, 1, nb_classes))
    
    # Affichage
    fig = plt.figure(figsize = (30, 20))
    ax = fig.add_subplot(111, axisbg = 'white')
    nb_points = len(X)
    for i, classe in enumerate(liste_classes):
        m = markers.pop(0)
        c = colors[i]
        ax.scatter(points[points['classe'] == classe].x, points[points['classe'] == classe].y, s = 40, marker = m, color = c, alpha = 0.6)
    ax.legend(labels = liste_classes)
    plt.show()

def proj_AD(X, y):
    """
    Affiche les données dans le premier plan discriminant.
    X est la matrice d'entrée (var centrées réduites), y est le vecteur des classes de sortie.
    """
    # AD 
    liste_classes = np.unique(y)
    nb_classes = len(liste_classes)
    lda = LinearDiscriminantAnalysis(n_components = 2)
    X_lda = lda.fit_transform(X, y) # X est un tableau numpy
    # Attention, il ne faut pas perdre l'index !
    points = DataFrame(X_lda, columns = ['x', 'y'], index = X.index)
    points['classe'] = y # on ajoute l'info de classe
    
    # Liste des marqueurs possibles, 
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # couleurs
    colors = cm.rainbow(np.linspace(0, 1, nb_classes))
    
    # Affichage
    fig = plt.figure(figsize = (30, 20))
    ax = fig.add_subplot(111, axisbg = 'white')
    nb_points = len(X)
    for i, classe in enumerate(liste_classes):
        m = markers.pop(0)
        c = colors[i]
        ax.scatter(points[points['classe'] == classe].x, points[points['classe'] == classe].y, s = 60, marker = m, color = c, alpha = 0.6)
    ax.legend(labels = liste_classes)
    
    return fig 


def proj(X, y):
    """
    Affiche les données à partir des coordonnées fournies dans X (DataFrame
    qui contient les coordonnées).
    y est le vecteur des classes de sortie.
    La fonction est généraliste par rapport à proj_pca par exemple.
    """
    liste_classes = np.unique(y)
    nb_classes = len(liste_classes)
    # Attention, il ne faut pas perdre l'index !
    points = DataFrame(X, columns = ['x', 'y'], index = X.index)
    points['classe'] = y # on ajoute l'info de classe
    
    # Liste des marqueurs possibles, 
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # couleurs
    colors = cm.rainbow(np.linspace(0, 1, nb_classes))
    
    # Affichage
    fig = plt.figure(figsize = (30, 20))
    ax = fig.add_subplot(111, axisbg = 'white')
    nb_points = len(X)
    for i, classe in enumerate(liste_classes):
        m = markers.pop(0)
        c = colors[i]
        ax.scatter(points[points['classe'] == classe].x, points[points['classe'] == classe].y, s = 40, marker = m, color = c, alpha = 0.6)
    ax.legend(labels = liste_classes)
    plt.show()

def affiche_importances_var(liste_noms_var, liste_importances):
    """
    Affiche sous forme d'histogramme les importances relatives des variables.
    """
    xy = tuple(zip(liste_importances, liste_noms_var))
    xy = sorted(xy, key = operator.itemgetter(0), reverse = True)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, axisbg = 'white')
    X = range(len(xy))
    ax.bar(X, [xy[i][0] for i in range(len(xy))], alpha = 0.7, align = 'center')
    ax.set_xticks(X)
    ax.set_xticklabels([xy[i][1] for i in range(len(xy))], rotation = 90)
    plt.show()
    return ax, fig

def resultats_gridsearch(clf):
    """
    A partir d'un clf issu d'un gridsearch, renvoie la DataFrame des scores et 
    des paramètres du gridsearch.
    """
    results = DataFrame(columns = ['Score', 'C', 'gamma'])
    for el in clf.grid_scores_:
        nouvelle_ligne = DataFrame({'Score' : [el.mean_validation_score], 'C' : [el.parameters['C']], 'gamma' : [el.parameters['gamma']]})
        results = results.append(nouvelle_ligne)
    return results

def mat_scores_gridsearch(resultats_gridsearch, nb_valeurs):
    """
    A partir de la DataFrame des résultats du gridsearch, renvoie
    le tableau 2D des scores pour servir de cote (z) aux graphes
    3D et aux affichages de matrices par code couleur.
    Rq : la DataFrame des résultats doit avoir une colonne 'Score'.
    """
    score = np.array(resultats_gridsearch['Score'])
    score_z = []
    for i, ligne in enumerate(score):
        if i%nb_valeurs == 0:
            score_z.append([])
        score_z[-1].append(ligne)
    return score_z

def reverse_ordinal(y_ordinal):
        """
        A partir d'un tableau numpy de codage ordinal, renvoie les classes
        originales sous forme de vecteur.
        """
        n_lignes = y_ordinal.shape[0]
        li = [] # la liste qu'on va ensuite transformer en vecteur
        for i in range(n_lignes):
            li.append(2 + sum(y_ordinal[i, :])) # on ajoute 2 car les classes commencent à 3
        y = np.array(li)
        return y
    
