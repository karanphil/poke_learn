# -*- coding: utf-8 -*-

#####
# Gabrielle Grenier 15 065 448
# Philippe Karan 15 093 532
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


class BaseDonnees:
    def __init__(self, fichier, attribut_interet):
        """
        Classe effectuant le traitement de la base de données ainsi que
        l'analyse des éléments utiles dans la base de données.
        Prend en entrée le fichier .csv
        """
        self.fichier = fichier
        self.bd = pd.read_csv(fichier)
        self.att_int = attribut_interet


    def enlever_attributs(self, liste_att):
        """
        Enlève des colonnes de l'objet base de données de la classe.

        ``liste_att`` est une liste de string contenant le nom des
        colonnes de la base de données à retirer de l'objet base de données.

        Pour visualiser les noms des colonnes, appeler préalableimportment la
        fonction voir_att.
        """
        self.bd.drop(liste_att, axis=1, inplace=True)
        print('Attributs ' + str(liste_att) + ' retirés de la base de données')


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
            self.bd[l] = self.bd[l].fillna(value= 'Vide')
            vec = enc.fit_transform(self.bd[l].to_numpy().reshape(-1,1)).toarray()
            categories = enc.categories_[0]
            if l == 'type2':
                self.bd[categories[1:]] += vec[:, 1:]
            else:
                for i in range(len(categories)):
                    self.bd[categories[i]] = vec[:,i].astype(int)
        self.enlever_attributs(liste_att)
        print('Attributs ' + str(liste_att) + ' changés en one hot vector')


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
        print('Attributs ' + str(liste_att) + ' changés en valeurs numériques entières')


    def normaliser_donnees(self):
        """
        Normalise et recentre les données en soustrayant avec la moyenne pour
        ensuite diviser par l'écart-type.

        Cette méthode suppose que toutes les données sont de types numériques
        Sinon, appelez la méthode str_a_vec avec le nom des attributs qui ne sont
        pas en valeurs numériques.
        """
        liste_norm = self.bd.loc[:, self.bd.std() > 1].columns
        self.bd[liste_norm] = (self.bd[liste_norm] - self.bd[liste_norm].mean()) / self.bd[liste_norm].std()
        print('Normalisation des données avec un écart-type supérieur à 1')


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
        bd_donnees = self.bd.drop(self.att_int, axis=1)
        bd_cible = self.bd[self.att_int]

        while nb_leg < nb_leg_requis:
            masque = np.random.rand(len(self.bd)) < prop_entr
            x_entr = bd_donnees[masque].to_numpy()
            t_entr = bd_cible[masque].to_numpy()
            x_test = bd_donnees[~masque].to_numpy()
            t_test = bd_cible[~masque].to_numpy()
            nb_leg = np.sum(t_entr)
        print('Ensembles dentrainement et de test générés')
        return x_entr, t_entr, x_test, t_test

    def voir_att(self):
        """
        Affiche la liste des noms des colonnes, soit les attributs,
        et le type de données de chacun des attributs et
        retourne la liste du nom des colonnes.

        """
        print(self.bd.dtypes)
        return self.bd.columns

    def enregistre_bd(self, nouvelle_bd, nom_fichier):
        nouvelle_bd.to_csv(nom_fichier, index= False)

    def comparer_attributs(self, att_1, att_2):
        x = self.bd[att_1].to_numpy()
        y = self.bd[att_2].to_numpy()
        colors = self.bd[self.att_int].to_numpy()

        plt.scatter(x, y, c= colors)
        plt.xlabel(att_1)
        plt.ylabel(att_2)
        plt.title('Comparaison ' + att_1 + ' et ' + att_2 + ' avec ' + self.att_int)
        plt.show()

