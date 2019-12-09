# -*- coding: utf-8 -*-
from .classifieur import Classifieur
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tqdm import tqdm


class AdaBoost(Classifieur):
    '''
    Implémentation du modèle de Adaboost. Cette
    classe possède une initialisation, ainsi qu'une validation
    croisée qui lui est propre. Prendre note que le modèle est
    initialisé avec un arbre décisionnel comme estimateur de base

    L'initialisation prend en entrée les paramètres suivants :
    -la profondeur maximale de l'arbre, un entier
    -le nombre d'estimateurs, un entier
    -le taux d'apprentissage,

    La validation croisée utilise le taux d'apprentissage et
    le nombre d'estimateurs comme hyperparamètre, en plus
    d'être de type "k-fold".

    Toutes les autres méthodes proviennent de la classe parent
    Classifieur, qui se charge des méthodes générales.
    '''
    def __init__(self, max_prof=1, nb_estimateur=50, lamb=1.):
        self.max_prof = max_prof
        self.nb_estimateur = nb_estimateur
        self.lamb = lamb
        self.modele = AdaBoostClassifier(
                        base_estimator=DecisionTreeClassifier(
                                        max_depth=max_prof),
                        n_estimators=nb_estimateur, learning_rate=lamb)

    def validation_croisee(self, x_tab, t_tab, k=10,
                            est_ech_poids=False, *args):
        # Liste des lambda à explorer
        lamb_min = 0.000000001
        lamb_max = 2.
        liste_lamb = np.logspace(np.log(lamb_min), np.log(lamb_max),
                                    num=10, base=np.e)
        # Liste des nb_estimateur à explorer
        estimateur_min = 20
        estimateur_max = 80
        liste_estimateur = np.linspace(estimateur_min,
                                        estimateur_max, 7).astype(int)

        nb_donnees = len(x_tab)
        # 20 % des donnees dans D_valid et 80% des donnees dans D_ent
        nb_D_valid = int(np.floor(0.20 * nb_donnees))

        liste_erreur = np.zeros((len(liste_lamb), len(liste_estimateur)))
        for i in tqdm(range(len(liste_lamb))):
            self.lamb = liste_lamb[i]
            for l in range(len(liste_estimateur)):
                self.nb_estimateur = liste_estimateur[l]
                self.modele = AdaBoostClassifier(
                                base_estimator=DecisionTreeClassifier(
                                                    max_depth=self.max_prof),
                                n_estimators=self.nb_estimateur,
                                learning_rate=self.lamb)
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
                    liste_erreur[i, l] += self.erreur(t_valid, pred_valid)
                # Moyenne des erreurs pour un lamb
                liste_erreur[i, l] /= k

        self.lamb = liste_lamb[np.unravel_index(np.argmin(liste_erreur),
                                liste_erreur.shape)[0]]
        self.nb_estimateur = liste_estimateur[np.unravel_index(
                                np.argmin(liste_erreur),
                                liste_erreur.shape)[1]]
        self.modele = self.modele = AdaBoostClassifier(
                                        base_estimator=DecisionTreeClassifier(
                                                    max_depth=self.max_prof),
                                        n_estimators=self.nb_estimateur,
                                        learning_rate=self.lamb)
        self.entrainement(x_tab, t_tab, est_ech_poids, args[0])
