# -*- coding: utf-8 -*-

class Analyse:
    def __init__(self, verite_terrain, resultats, probabilites):
        self.verite_terrain = verite_terrain
        self.resultats = resultats
        self.probabilites = probabilites
        self.vp = 0
        self.vn = 0
        self.fp = 0
        self.fn = 0

    def calculer_comptes(self):
        raise NotImplementedError
    
    def calculer_rappel(self):
        raise NotImplementedError

    def calculer_justesse(self):
        raise NotImplementedError

    def calculer_precision(self):
        raise NotImplementedError

    def calculer_specificite(self):
        raise NotImplementedError

    def calculer_mesure_f(self):
        raise NotImplementedError

    def calculer_courbe_roc(self):
        raise NotImplementedError