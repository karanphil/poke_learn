# -*- coding: utf-8 -*-
from .classifieur import Classifieur
from sklearn.ensemble import RandomForestClassifier


class FAD(Classifieur):
    '''
    Implémentation du modèle de bayes naif. Cette classe possède
    seulement une initialisation.

    L'inititalisation prend en entrée les paramètres suivants :
    -le nombre d'arbres dans la forêt, un entier
    -le critère de séparation : gini, entropy
    -la profondeur maximale d'un arbre, un entier
    -la possibilité d'un 'bootstraping', un booléen

    Toutes les autres méthodes proviennent de la classe parent
    Classifieur, qui se charge des méthodes générales.
    '''
    def __init__(self, nb_arbres=10, critere='gini',
                    prof_max=None, bootstrap=True):
        self.modele = RandomForestClassifier(n_estimators=nb_arbres,
                        criterion=critere, max_depth=prof_max,
                        bootstrap=bootstrap)
