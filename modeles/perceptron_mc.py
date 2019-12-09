# -*- coding: utf-8 -*-
from .classifieur import Classifieur
from sklearn.neural_network import MLPClassifier
import numpy as np
from tqdm import tqdm


class PerceptronMC(Classifieur):
    '''
    Implémentation du modèle de perceptron multi-couches. Cette
    classe possède une initialisation, ainsi qu'une validation
    croisée qui lui est propre.

    L'initialisation prend en entrée les paramètres suivants :
    -les couches cachées : nombre de neurones par couches
                            (ex : 5,10,15,5)
    -le type de fonction d'activation : identity, logistic, tanh, relu
    -le solutionneur : lbfgs, sgd, adam
    -le type de taux d'apprentissage : constant, invscaling, adaptive
                            (seulement avec sgd)
    -le maximum d'itérations

    La validation croisée utilise le paramètre de régularisation
    comme hyperparamètre, en plus d'être de type "k-fold".

    Toutes les autres méthodes proviennent de la classe parent
    Classifieur, qui se charge des méthodes générales.
    '''
    def __init__(self, couches_cachees, activation='relu',
                    solutionneur='sgd', apprentissage_type='constant',
                    max_iter=200):
        self.couches_cachees = couches_cachees
        self.activation = activation
        self.solutionneur = solutionneur
        self.apprentissage_type = apprentissage_type
        self.max_iter = max_iter
        self.modele = MLPClassifier(hidden_layer_sizes=couches_cachees,
                        activation=activation, solver=solutionneur,
                        learning_rate=apprentissage_type, max_iter=max_iter)

    def validation_croisee(self, x_tab, t_tab, k=10,
                            est_ech_poids=False, *args):
        # Liste des lambda à explorer
        lamb_min = 0.000000001
        lamb_max = 2.
        liste_lamb = np.logspace(np.log(lamb_min), np.log(lamb_max),
                                    num=10, base=np.e)

        nb_donnees = len(x_tab)
        # 20 % des donnees dans D_valid et 80% des donnees dans D_ent
        nb_D_valid = int(np.floor(0.20 * nb_donnees))

        liste_erreur = np.zeros((len(liste_lamb)))
        for i in tqdm(range(len(liste_lamb))):
            self.modele = MLPClassifier(hidden_layer_sizes=self.couches_cachees,
                                activation=self.activation,
                                solver=self.solutionneur,
                                alpha=liste_lamb[i],
                                learning_rate=self.apprentissage_type,
                                max_iter=self.max_iter)
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
        self.modele = MLPClassifier(hidden_layer_sizes=self.couches_cachees,
                                        activation=self.activation,
                                        solver=self.solutionneur,
                                        alpha=meilleur_lambda,
                                        learning_rate=self.apprentissage_type,
                                        max_iter=self.max_iter)
        self.entrainement(x_tab, t_tab, est_ech_poids, args[0])
