# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

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
        plot_init()
        plt.figure()
        #plt.title("Courbe ROC")
        plt.xlabel("TFP")
        plt.ylabel("TVP")
        plt.plot(self.tfp, self.tvp, "b-")
        plt.show()


class Analyse_multiple:
    def __init__(self, repetitions):
        self.repetitions = repetitions
        self.erreurs = np.ndarray((repetitions, 2))
        self.metriques = np.ndarray((repetitions, 5))
        self.erreurs_moy = np.array([0,0])
        self.metriques_moy = np.array([0,0,0,0,0])
        self.rep_courante = 0

    def ajouter_erreurs(self, erreur_ent, erreur_test):
        self.erreurs[self.rep_courante] = [erreur_ent, erreur_test]

    def ajouter_metriques(self, metriques):
        self.metriques[self.rep_courante] = metriques

    def calculer_moyennes(self):
        self.erreurs_moy = np.mean(self.erreurs, axis = 0)
        self.metriques_moy = np.mean(self.metriques, axis = 0)
    
    def augmenter_rep_courante(self):
        self.rep_courante += 1

    def afficher_moyennes(self):
        print("Erreur d'entrainement moyenne = ", self.erreurs_moy[0], '%')
        print("Erreur de test moyenne = ", self.erreurs_moy[1], '%')
        print("Rappel moyen = ", self.metriques_moy[0])
        print("Justesse moyenne = ", self.metriques_moy[1])
        print("Précision moyenne = ", self.metriques_moy[2])
        print("Spécificité moyenne = ", self.metriques_moy[3])
        print("Mesure-f moyenne = ", self.metriques_moy[4])

    def afficher_graphique(self):
        rep = np.arange(1, self.repetitions + 1, 1)
        fig = plt.figure(figsize=(15,15))
        plot_init()
        widths = [2]
        heights = [1,1]
        gs = gridspec.GridSpec(2, 1, figure = fig, width_ratios=widths,
                          height_ratios=heights)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(rep, 100 - self.erreurs[:, 0], "bs-", label = "Entrainement")
        ax.plot(rep, 100 - self.erreurs[:, 1], "gs-", label = "Test")
        ax.axhline(100 - self.erreurs_moy[0], color = "b", linestyle = "--", alpha = 0.3, linewidth = 1.5)
        ax.axhline(100 - self.erreurs_moy[1], color = "g", linestyle = "--", alpha = 0.3, linewidth = 1.5)
        plt.ylabel("Justesse en %")
        plt.legend(loc = 1)
        plt.xticks(rep)
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(rep, self.metriques[:, 0] * 100, "ro-", label = "Rappel")
        #plt.plot(rep, self.metriques[:, 1] * 100, "o-", label = "Justesse")
        ax.plot(rep, self.metriques[:, 2] * 100, "co-", label = "Précision")
        ax.plot(rep, self.metriques[:, 3] * 100, "yo-", label = "Spécificité")
        #plt.plot(rep, self.metriques[:, 4] * 100, "o-", label = "Mesure-f")
        plt.ylabel("Métriques en %")
        plt.xlabel("# de répétition")
        plt.xticks(rep)
        plt.legend(loc = 1)
        plt.show()

def plot_init():
    plt.style.use('seaborn-notebook')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = "darkgrey"
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = "-"
    plt.rcParams['grid.alpha'] = "0.5"
    plt.rcParams['figure.figsize'] = (13.0, 9.0)
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 0.9*plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 0.9*plt.rcParams['font.size']
    plt.rcParams['axes.linewidth'] =1
    plt.rcParams['lines.linewidth']=2
    plt.rcParams['lines.markersize']=8