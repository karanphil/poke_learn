# -*- coding: utf-8 -*-
from classifieur import Classifieur

class PerceptronMC(Classifieur):
    def __init__(self):
        raise NotImplementedError

    def validation_croisee(self, x_tab, t_tab, k = 10, est_ech_poids = False, *args):
        raise NotImplementedError