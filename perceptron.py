# -*- coding: utf-8 -*-
from classifieur import Classifieur
from sklearn.linear_model import Perceptron as skPerceptron
import numpy as np
from tqdm import tqdm

class Perceptron(Classifieur):
    def __init__(self):
        self.modele = skPerceptron(penalty = 'l2', shuffle = False)

    def validation_croisee(self, x_tab, t_tab, k = 10, est_ech_poids = False, *args):
        # Liste des lambda à explorer
        lamb_min = 0.000000001
        lamb_max = 2.
        liste_lamb = np.logspace(np.log(lamb_min), np.log(lamb_max), num = 25, base = np.e)

        nb_donnees = len(x_tab)
        # 20 % des donnees dans D_valid et 80% des donnees dans D_train
        nb_D_valid = int(np.floor(0.20 * nb_donnees))
        #nb_D_train = nb_donnees - nb_D_valid

        liste_erreur = np.zeros((len(liste_lamb)))
        for i in tqdm(range(len(liste_lamb))):
            self.modele = skPerceptron(penalty = 'l2', alpha = liste_lamb[i], shuffle = False)
            for j in range(k):
                # Masque de vrai ou de faux pour déterminer les ensembles D_valid et D_train
                liste_ind = np.ones(nb_donnees, dtype = bool)
                liste_ind[0:nb_D_valid] = 0
                np.random.shuffle(liste_ind)
                # D_valid correspond a faux 
                # Division de D en deux groupes formes aleatoirement : D_train et D_valid
                x_entr = x_tab[liste_ind]
                x_valid = x_tab[np.invert(liste_ind)]
                t_entr = t_tab[liste_ind]
                t_valid = t_tab[np.invert(liste_ind)]
                # Entrainement sur x_train et t_train
                self.entrainement(x_entr, t_entr, est_ech_poids, args[0])
                pred_valid = np.array([self.prediction(x) for x in x_valid])
                liste_erreur[i] += self.erreur(t_valid, pred_valid)
            # Moyenne des erreurs pour un lamb
            liste_erreur[i] /= k

        meilleur_lambda = liste_lamb[np.unravel_index(np.argmin(liste_erreur), liste_erreur.shape)[0]]
        self.modele = skPerceptron(penalty = 'l2', alpha = meilleur_lambda, shuffle = False)
        self.entrainement(x_tab, t_tab, est_ech_poids, args[0])
