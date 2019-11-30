# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix

class Analyse:
    def __init__(self, verite_terrain, resultats, probabilites):
        self.verite_terrain = verite_terrain
        self.resultats = resultats
        self.probabilites = probabilites
        self.vp = 0
        self.vn = 0
        self.fp = 0
        self.fn = 0

    def calculer_comptes(self, est_ech_poids = False, *args):
        # Cas avec poids variables
        if(est_ech_poids):
            matrice_confusion = confusion_matrix(self.verite_terrain, self.resultats, sample_weight = args[0])
        # Cas sans poids variables
        else:
            matrice_confusion = confusion_matrix(self.verite_terrain, self.resultats)
        self.vn, self.fp, self.fn, self.vp = matrice_confusion.ravel()
    
    def calculer_rappel(self):
        return(self.vp / (self.fn + self.vp))

    def calculer_justesse(self):
        return((self.vp + self.vn) / (self.vp + self.vn + self.fp + self.fn))

    def calculer_precision(self):
        return(self.vp / (self.vp + self.fp))

    def calculer_specificite(self):
        return(self.vn / (self.fp + self.vn))

    def calculer_mesure_f(self):
        rappel = self.calculer_rappel()
        precision = self.calculer_precision()
        return((2 * rappel * precision) / (rappel + precision))

    def calculer_courbe_roc(self):
        raise NotImplementedError