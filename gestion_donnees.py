# -*- coding: utf-8 -*-

#####
# Gabrielle Grenier 15 065 448
# Philippe Karan 15 093 532
###

import numpy as np
import pandas as pd
from sklearn import preprocessing


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

        Pour visualiser les noms des colonnes, appeler préalableimportment la
        fonction voir_att.
        """
        self.bd.drop(liste_att, axis=1, inplace=True)


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
        enc = preprocessing.OneHotEncoder()
        for l in liste_att: 
            self.bd[l] = self.bd[l].fillna(value= 'vide')
            vec = enc.fit_transform(self.bd[l].to_numpy().reshape(-1,1)).toarray()
            categories = enc.categories_[0]
            for i in range(len(categories)):
                self.bd[categories[i]] = vec[:,i].astype(int)
        self.enlever_attributs(liste_att)
        print(self.bd.dtypes)


    def str_a_int(self, liste_att):
        """
        Change les valeurs de string des colonnes de la base de données
        pour des valeurs entières.
        
        ``liste_att`` est une liste de string contenant le nom des
        colonnes de la base de données pour lesquelles les valeurs doivent
        être changées en valeurs entières.

        Pour visualiser les noms des colonnes, appeler préalablement la
        fonction voir_att.
        """
        self.bd[liste_att] = self.bd[liste_att].astype(int)


    def normaliser_donnees(self):
        """
        Normalise et recentre les données en soustrayant avec la moyenne pour
        ensuite diviser par l'écart-type.

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
        nb_leg = 0
        nb_leg_requis = 20
        bd_entr = self.bd.drop('is_legendary', axis=1)
        bd_test = self.bd['is_legendary']

        while nb_leg < nb_leg_requis:
            masque = np.random.rand(len(self.bd)) < prop_entr
            x_entr = bd_entr[masque].to_numpy()
            t_entr = bd_test[masque].to_numpy()
            x_test = bd_entr[~masque].to_numpy()
            t_test = bd_test[~masque].to_numpy()
            nb_leg = np.sum(t_entr)
        
        return x_entr, t_entr, x_test, t_test

    def voir_att(self):
        """
        Affiche la liste des noms des colonnes, soit les attributs,
        et le type de données de chacun des attributs et
        retourne la liste du nom des colonnes.

        """
        # print(self.bd.dtypes)
        return self.bd.columns