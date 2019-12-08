# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from matplotlib import pyplot as plt

class Analyse:
    def __init__(self, verite_terrain, resultats, probabilites):
        """
        Classe servant à faire l'analyse des résultats de 
        l'entraînement d'un modèle, basé sur des métriques
        telles que le rappel, la justesse, la précision, etc,
        ainsi que sur une courbe ROC.

        Prend en entrée la vérité terrain (donc les étiquettes
        de classe), les résultats (donc les prédictions du modèle)
        et les probabilités des résultats, tels que fournis 
        préalablement par la méthode confiance_test de la classe
        Classifieur.
        """ 
        self.verite_terrain = verite_terrain
        self.resultats = resultats
        self.probabilites = probabilites
        self.vp = 0
        self.vn = 0
        self.fp = 0
        self.fn = 0
        self.tvp = None
        self.tfp = None
        self.metriques = np.zeros(5)

    def calculer_comptes(self, est_ech_poids = False, *args):
        """
        Cette méthode compte le nombre de vrais positifs, de 
        faux positifs, de vrais négatifs et de faux négatifs.
        Elle doit être appelée avant de pouvoir utiliser les prochaines
        méthodes (autre que celles concernant la courbe ROC).
        """
        # Cas avec poids variables
        if(est_ech_poids):
            matrice_confusion = confusion_matrix(self.verite_terrain, 
                                self.resultats, sample_weight = args[0])
        # Cas sans poids variables
        else:
            matrice_confusion = confusion_matrix(self.verite_terrain, 
                                self.resultats)
        self.vn, self.fp, self.fn, self.vp = matrice_confusion.ravel()
    
    def afficher_comptes(self):
        """
        Cette méthode sert à afficher à l'écran les résultats
        de la méthode calculer_comptes.
        """
        print("Vrais positifs : ", self.vp)
        print("Faux positifs : ", self.fp)
        print("Vrais négatifs : ", self.vn)
        print("Faux négatifs : ", self.fn)

    def calculer_rappel(self):
        """
        Cette méthode calcule le rappel.
        """
        return(self.vp / (self.fn + self.vp))

    def calculer_justesse(self):
        """
        Cette méthode calcule la justesse.
        """
        return((self.vp + self.vn) / (self.vp + self.vn + self.fp + self.fn))

    def calculer_precision(self):
        """
        Cette méthode calcule la précision.
        """
        return(self.vp / (self.vp + self.fp))

    def calculer_specificite(self):
        """
        Cette méthode calcule la spécificité.
        """
        return(self.vn / (self.fp + self.vn))

    def calculer_mesure_f(self):
        """
        Cette méthode calcule la mesure-f.
        """
        rappel = self.calculer_rappel()
        precision = self.calculer_precision()
        return((2 * rappel * precision) / (rappel + precision))
    
    def calculer_metriques(self):
        """
        Cette méthode sert à calculer les  différentes
        métriques en utilisant les méthodes précédentes.
        """
        self.metriques[0] = self.calculer_rappel()
        self.metriques[1] = self.calculer_justesse()
        self.metriques[2] = self.calculer_precision()
        self.metriques[3] = self.calculer_specificite()
        self.metriques[4] = self.calculer_mesure_f()

    def afficher_metriques(self):
        """
        Cette méthode sert à afficher les résultats de
        la méthode calculer_metriques.
        """
        print("Rappel = ", self.metriques[0])
        print("Justesse = ", self.metriques[1])
        print("Précision = ", self.metriques[2])
        print("Spécificité = ", self.metriques[3])
        print("Mesure-f = ", self.metriques[4])

    def calculer_courbe_roc(self, est_ech_poids = False, *args):
        """
        Cette méthode calcule le nécessaire pour afficher
        une courbe ROC.
        """
        # Cas avec poids variables
        if(est_ech_poids):
            self.tfp, self.tvp, seuil = roc_curve(self.verite_terrain, 
                                                    self.probabilites, 
                                                    sample_weight = args[0], 
                                                    drop_intermediate = False) 
        # Cas sans poids variables
        else:
            self.tfp, self.tvp, seuil = roc_curve(self.verite_terrain, 
                                                    self.probabilites, 
                                                    drop_intermediate = False) 

    def afficher_courbe_roc(self):
        """
        Cette méthode sert à afficher le résultat de la méthode
        calculer_courbe_roc.
        """
        plt.figure()
        plt.title("Courbe ROC")
        plt.xlabel("TFP")
        plt.ylabel("TVP")
        plt.plot(self.tfp, self.tvp, "b-")
        plt.show()


class Analyse_multiple:
    def __init__(self, repetitions):
        self.repetitions = repetitions
        self.erreurs = np.ndarray((2, repetitions))
        self.metriques = np.ndarray((5, repetitions))

    def ajouter_erreurs(self, erreur_ent, erreur_test):
        return 0

    def calculer_moyennes(self):
        return 0