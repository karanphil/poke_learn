# -*- coding: utf-8 -*-
from classifieur import Classifieur

class Perceptron(Classifieur):
    def __init__(self):
        raise NotImplementedError

    def validation_croisee(self, x_entr, t_entr, est_ech_poids = False, *args):
        raise NotImplementedError
