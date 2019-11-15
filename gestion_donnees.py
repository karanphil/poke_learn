# -*- coding: utf-8 -*-

#####
# Gabrielle Grenier 15 065 448
# Philippe Karan 15 093 532
###

import numpy as np
import pandas as pd


class BaseDonnees:
    def __init__(self, fichier):
        """
        Classe effectuant le traitement de la base de données ainsi que 
        l'analyse des éléments utiles dans la base de données. 
        Prend en entrée le fichier .csv
        """
        self.fichier = fichier
        self.bd = pd.read_csv(self.fichier)

    def test_affichage(self):
        print("Les 5 premieres colonnes de la base de donnees" + self.bd[0:5])

        