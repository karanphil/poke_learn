# -*- coding: utf-8 -*-
import sys
from bayes_naif import BayesNaif
from perceptron import Perceptron
from perceptron_mc import PerceptronMC
from svm import SVM
from fad import FAD
from adaboost import AdaBoost

def main():
    vc = sys.argv[1]
    est_ech_poids = sys.argv[2]
    i = 3
    modele_choix = sys.argv[i]

    # Gestion des données

    # Gestion du modèle
    print("Création du mondèle...")
    if(modele_choix == "bayes_naif"):
        modele = BayesNaif()
    elif(modele_choix == "perceptron"):
        modele = Perceptron()
    elif(modele_choix == "perceptron_mc"):
        couches_cachees = tuple([int(x) for x in sys.argv[i+1].split(',')])
        activation = sys.argv[i+2]
        solutionneur = sys.argv[i+3]
        modele = PerceptronMC(couches_cachees = couches_cachees, 
                    activation = activation, solutionneur = solutionneur, max_iter = 200)
    elif(modele_choix == "svm"):
        noyau = sys.argv[i+1]
        modele = SVM(noyau = noyau, tol = 1e-3, max_iter = -1)
    elif(modele_choix == "fad"):
        nb_arbres = int(sys.argv[i+1])
        critere = sys.argv[i+2]
        prof_max = sys.argv[i+3]
        modele = FAD(nb_arbres = nb_arbres, critere = critere, prof_max = prof_max)
    elif(modele_choix == "adaboost"):
        max_prof = sys.argv[i+2]
        modele = AdaBoost(max_prof = max_prof)
    else:
        print("Oups, ce modèle n'existe pas!")
        return 0

    # Entrainement ou validation croisée
    if bool(vc) is False:
        print("Début de l'entrainement simple...")
        modele.entrainement(x_entr, t_entr, est_ech_poids, ech_poids) # Il faudra fournir ech_poids de Gestion des données!!!
    else:
        print("Début de l'entrainement par validation croisée...")
        modele.validation_croisee(x_entr, t_entr, 10, est_ech_poids, ech_poids) # Il faudra fournir ech_poids de Gestion des données!!!

    # Prédiction et erreur
    print("Calcul des erreurs...")
    predictions_entrainement = modele.prediction(x_entr)
    erreur_entrainement = modele.erreur(t_entr, predictions_entrainement) / len(t_entr) * 100

    predictions_test = modele.prediction(x_test)
    erreur_test = modele.erreur(t_test, predictions_test) / len(t_test) * 100

    # Analyse des résultats

if __name__ == "__main__":
    main()