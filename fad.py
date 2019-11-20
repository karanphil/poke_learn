# -*- coding: utf-8 -*-
from classifieur import Classifieur
from sklearn.ensemble import RandomForestClassifier

class FAD(Classifieur):
    def __init__(self, nb_arbres = 10, critere = 'gini', prof_max = None, bootstrap = True):
        self.modele = RandomForestClassifier(n_estimators = nb_arbres, 
                        criterion = critere, max_depth = prof_max, bootstrap = bootstrap)