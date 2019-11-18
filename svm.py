# -*- coding: utf-8 -*-

from classifieur import Classifieur
from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm

class SVM(Classifieur):
    def __init__(self, C = 1.0, noyau = 'linear', deg = 3, gamma = 'auto', coef0 = 0):
        self.C = C
        self.noyau = noyau
        self.deg = deg
        self.gamma = gamma
        self.coef0 = coef0
        self.modele = SVC(C = C, kernel = noyau, degree = deg, gamma = gamma, coef0 = coef0)

    def validation_croisee(self, x_tab, t_tab, k = 10, est_ech_poids = False, *args):
        # Liste des C à explorer
        C_min = 1
        C_max = 1000
        liste_C = int(np.linspace(C_min, C_max, 5))

        nb_donnees = len(x_tab)
        # 20 % des donnees dans D_valid et 80% des donnees dans D_entr
        nb_D_valid = int(np.floor(0.20 * nb_donnees))
        #nb_D_entr = nb_donnees - nb_D_valid

        # Noyau lineaire et rbf
        if(self.noyau == "linear" or self.noyau == "rbf"):
            liste_erreur = np.zeros((len(liste_C)))
            for i in tqdm(range(len(liste_C))):
                self.C = liste_C[i]
                self.modele = SVC(C = self.C, kernel = self.noyau,
                                degree = self.deg, gamma = self.gamma, coef0 = self.coef0)
                for j in range(k):
                    # Masque de vrai ou de faux pour déterminer les ensembles D_valid et D_entr
                    liste_ind = np.ones(nb_donnees, dtype = bool)
                    liste_ind[0:nb_D_valid] = 0
                    np.random.shuffle(liste_ind)
                    # D_valid correspond a faux 
                    # Division de D en deux groupes formes aleatoirement : D_entr et D_valid
                    x_entr = x_tab[liste_ind]
                    x_valid = x_tab[np.invert(liste_ind)]
                    t_entr = t_tab[liste_ind]
                    t_valid = t_tab[np.invert(liste_ind)]
                    # Entrainement sur x_entr et t_entr
                    self.entrainement(x_entr, t_entr, est_ech_poids, args[0])
                    pred_valid = self.prediction(x_valid)
                    liste_erreur[i] += self.erreur(t_valid, pred_valid)
                # Moyenne des erreurs pour un C
                liste_erreur[i] /= k

            self.C = liste_C[np.unravel_index(np.argmin(liste_erreur), liste_erreur.shape)[0]]
            self.modele = SVC(C = self.C, kernel = self.noyau,
                            degree = self.deg, gamma = self.gamma, coef0 = self.coef0)
            self.entrainement(x_tab, t_tab, est_ech_poids, args[0])

        # Noyau sigmoidal
        elif(self.noyau == "sigmoid"):
            # Liste des coef0 à explorer
            coef0_min = 0.00001
            coef0_max = 0.01
            liste_coef0 = np.logspace(np.log(coef0_min), np.log(coef0_max), num = 15, base = np.e)
            liste_erreur = np.zeros((len(liste_C), len(liste_coef0)))
            for i in tqdm(range(len(liste_C))):
                self.C = liste_C[i]
                for l in range(len(liste_coef0)):
                    self.coef0 = liste_coef0[l]
                    self.modele = SVC(C = self.C, kernel = self.noyau,
                                    degree = self.deg, gamma = self.gamma, coef0 = self.coef0)
                    for j in range(k):
                        # Masque de vrai ou de faux pour déterminer les ensembles D_valid et D_entr
                        liste_ind = np.ones(nb_donnees, dtype = bool)
                        liste_ind[0:nb_D_valid] = 0
                        np.random.shuffle(liste_ind)
                        # D_valid correspond a faux 
                        # Division de D en deux groupes formes aleatoirement : D_entr et D_valid
                        x_entr = x_tab[liste_ind]
                        x_valid = x_tab[np.invert(liste_ind)]
                        t_entr = t_tab[liste_ind]
                        t_valid = t_tab[np.invert(liste_ind)]
                        # Entrainement sur x_entr et t_entr
                        self.entrainement(x_entr, t_entr, est_ech_poids, args[0])
                        pred_valid = np.array([self.prediction(x) for x in x_valid])
                        liste_erreur[i, l] += self.erreur(t_valid, pred_valid)
                    # Moyenne des erreurs pour un C et un coef0
                    liste_erreur[i, l] /= k
            self.C = liste_C[np.unravel_index(np.argmin(liste_erreur), liste_erreur.shape)[0]]
            self.coef0 = liste_coef0[np.unravel_index(np.argmin(liste_erreur), liste_erreur.shape)[1]]
            self.modele = SVC(C = self.C, kernel = self.noyau,
                            degree = self.deg, gamma = self.gamma, coef0 = self.coef0)
            self.entrainement(x_tab, t_tab, est_ech_poids, args[0])

        # Noyau polynomial
        elif(self.noyau == "poly"):
            # Liste des coef0 à explorer
            coef0_min = 0
            coef0_max = 5
            liste_coef0 = np.arange(coef0_min, coef0_max, 1)
            # Liste des deg à explorer
            deg_min = 2
            deg_max = 6
            liste_deg = np.arange(deg_min, deg_max, 1)
            liste_erreur = np.zeros((len(liste_C), len(liste_coef0), len(liste_deg)))
            for i in tqdm(range(len(liste_C))):
                self.C = liste_C[i]
                for l in range(len(liste_coef0)):
                    self.coef0 = liste_coef0[l]
                    for m in range(len(liste_deg)):
                        self.deg = liste_deg[m]
                        self.modele = SVC(C = self.C, kernel = self.noyau,
                                        degree = self.deg, gamma = self.gamma, coef0 = self.coef0)
                        for j in range(k):
                            # Masque de vrai ou de faux pour déterminer les ensembles D_valid et D_entr
                            liste_ind = np.ones(nb_donnees, dtype = bool)
                            liste_ind[0:nb_D_valid] = 0
                            np.random.shuffle(liste_ind)
                            # D_valid correspond a faux 
                            # Division de D en deux groupes formes aleatoirement : D_entr et D_valid
                            x_entr = x_tab[liste_ind]
                            x_valid = x_tab[np.invert(liste_ind)]
                            t_entr = t_tab[liste_ind]
                            t_valid = t_tab[np.invert(liste_ind)]
                            # Entrainement sur x_entr et t_entr
                            self.entrainement(x_entr, t_entr, est_ech_poids, args[0])
                            pred_valid = np.array([self.prediction(x) for x in x_valid])
                            liste_erreur[i, l, m] += self.erreur(t_valid, pred_valid)
                        # Moyenne des erreurs pour un C, un coef0 et un deg
                        liste_erreur[i, l, m] /= k
            self.C = liste_C[np.unravel_index(np.argmin(liste_erreur), liste_erreur.shape)[0]]
            self.coef0 = liste_coef0[np.unravel_index(np.argmin(liste_erreur), liste_erreur.shape)[1]]
            self.deg = liste_deg[np.unravel_index(np.argmin(liste_erreur), liste_erreur.shape)[2]]
            self.modele = SVC(C = self.C, kernel = self.noyau,
                            degree = self.deg, gamma = self.gamma, coef0 = self.coef0)
            self.entrainement(x_tab, t_tab, est_ech_poids, args[0])