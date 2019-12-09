# -*- coding: utf-8 -*-
from .classifieur import Classifieur
from sklearn.naive_bayes import GaussianNB


class BayesNaif(Classifieur):
    '''
    Implémentation du modèle de bayes naif. Cette classe possède
    seulement une initialisation, qui ne prend rien en entrée.
    Toutes les autres méthodes proviennent de la classe parent
    Classifieur, qui se charge des méthodes générales.
    '''
    def __init__(self):
        self.modele = GaussianNB(priors=None, var_smoothing=1e-09)
