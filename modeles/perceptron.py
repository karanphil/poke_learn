# -*- coding: utf-8 -*-
from .classifieur import Classifieur
from sklearn.linear_model import Perceptron as skPerceptron
import numpy as np
from tqdm import tqdm


class Perceptron(Classifieur):
    '''
    Implémentation du modèle de perceptron. Cette classe possède
    une initialisation, ainsi qu'une validation croisée qui lui est
    propre.

    L'initialisation prend en entrée le nombre maximum d'itération
    et le critère d'arrêt, facultatifs.

    La validation croisée utilise le paramètre de régularisation
    comme hyperparamètre, en plus d'être de type "k-fold".

    Toutes les autres méthodes proviennent de la classe parent
    Classifieur, qui se charge des méthodes générales.
    '''
    def __init__(self, max_iter=1000, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.modele = skPerceptron(penalty='l2', max_iter=max_iter,
                                    tol=tol, shuffle=False)

    def validation_croisee(self, x_tab, t_tab, k=10,
                            est_ech_poids=False, *args):
        # Liste des lambda à explorer
        lamb_min = 0.000000001
        lamb_max = 2.
        liste_lamb = np.logspace(np.log(lamb_min), np.log(lamb_max),
                                    num=25, base=np.e)

        nb_donnees = len(x_tab)
        # 20 % des donnees dans D_valid et 80% des donnees dans D_ent
        nb_D_valid = int(np.floor(0.20 * nb_donnees))

        liste_erreur = np.zeros((len(liste_lamb)))
        for i in tqdm(range(len(liste_lamb))):
            self.modele = skPerceptron(penalty='l2', alpha=liste_lamb[i],
                            max_iter=self.max_iter, tol=self.tol,
                            shuffle=False)
            for j in range(k):
                # Masque de vrai ou de faux pour déterminer
                # les ensembles D_valid et D_ent
                liste_ind = np.ones(nb_donnees, dtype=bool)
                liste_ind[0:nb_D_valid] = 0
                np.random.shuffle(liste_ind)
                # D_valid correspond a faux
                # Division de D en deux groupes formes
                # aleatoirement : D_ent et D_valid
                x_entr = x_tab[liste_ind]
                x_valid = x_tab[np.invert(liste_ind)]
                t_entr = t_tab[liste_ind]
                t_valid = t_tab[np.invert(liste_ind)]
                # Entrainement sur x_ent et t_ent
                self.entrainement(x_entr, t_entr, est_ech_poids, args[0])
                pred_valid = self.prediction(x_valid)
                liste_erreur[i] += self.erreur(t_valid, pred_valid)
            # Moyenne des erreurs pour un lamb
            liste_erreur[i] /= k

        meilleur_lambda = liste_lamb[np.unravel_index(np.argmin(liste_erreur),
                                        liste_erreur.shape)[0]]
        self.modele = skPerceptron(penalty='l2', alpha=meilleur_lambda,
                        max_iter=self.max_iter, tol=self.tol, shuffle=False)
        self.entrainement(x_tab, t_tab, est_ech_poids, args[0])
