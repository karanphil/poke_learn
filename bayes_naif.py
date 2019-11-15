# -*- coding: utf-8 -*-
from classifieur import Classifieur
from sklearn.naive_bayes import GaussianNB

class BayesNaif(Classifieur):
    def __init__(self):
        self.modele = GaussianNB(priors=None, var_smoothing=1e-09)
    
