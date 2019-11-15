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
        self.bd = pd.read_csv(fichier)


    def test_affichage(self):
        print(self.bd[0:5])


    def enlever_attributs(self, liste_att):
        """
        Enlève des colonnes de l'objet base de données de la classe.

        ``liste_att`` est une liste de string contenant le nom des
        colonnes de la base de données à retirer de l'objet base de données.

        Pour visualiser les noms des colonnes, appeler préalablement la
        fonction voir_att.
        """
        raise NotImplementedError


    def str_a_vec(self, liste_att):
        """
        Change les valeurs de string des colonnes de la base de données
        pour des vecteurs de style one-hot vector.

        ``liste_att`` est une liste de string contenant le nom des
        colonnes de la base de données pour lesquelles les valeurs doivent
        être changées en one hot vector.

        Pour visualiser les noms des colonnes, appeler préalablement la
        fonction voir_att.
        """
        raise NotImplementedError


    def normaliser_donnees(self):
        """
        Normalise et recentre les données en soustrayant avec la moyenne pour
        ensuite diviser par l'écart-type.

        Cette méthode suppose que toutes les données sont de types numériques
        Sinon, appelez la méthode str_a_vec avec le nom des attributs qui ne sont
        pas en valeurs numériques.
        """
        raise NotImplementedError


    def selectionner_att_corr(self, seuil_cc = 0.8):
        """
        À Revoir
        """
        raise NotImplementedError


    def selectionner_att_non_corr_leg(self, seuil_cc = 0.8):
        """
        À Revoir
        """
        raise NotImplementedError


    def definir_poids_att(self):
        """
        Retourne un vecteur de poids entre 0 et 1 de chacun des attributs
        selon le coefficient de corrélation entre l'attribut et is_legendary

        Cette méthode suppose que toutes les données sont de types numériques
        Sinon, appelez la méthode str_a_vec avec le nom des attributs qui ne sont
        pas en valeurs numériques.
        """
        raise NotImplementedError


    def faire_ens_entr_test(self, prop_entr = 0.7):
        """
        Retourne les ensembles de données et de cibles d'entrainement et de test
        en format ndarray de numpy

        ``prop_entr`` est un nombre entre 0 et 1 qui indique la proportion de
        données à mettre dans l'ensemble d'entrainement.

        Cette méthode suppose que toutes les données sont de types numériques
        Sinon, appelez la méthode str_a_vec avec le nom des attributs qui ne sont
        pas en valeurs numériques.
        """
        raise NotImplementedError


    def calculer_cc(self, att1, att2):
        """
        Retourne le coefficient de corrélation entre deux attributs.

        ***** À revoir

        Cette méthode suppose que toutes les données sont de types numériques
        Sinon, appelez la méthode str_a_vec avec le nom des attributs qui ne sont
        pas en valeurs numériques.
        """
        raise NotImplementedError


    def voir_att(self):
        """
        Affiche la liste des noms des colonnes, soit les attributs,
        et le type de données de chacun des attributs
        """
        raise NotImplementedError