# -*- coding: utf-8 -*-
from classifieur import Classifieur
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost(Classifieur):
    def __init__(self, base_estimateur = None):
        self.modele = AdaBoostClassifier(base_estimator = base_estimateur)