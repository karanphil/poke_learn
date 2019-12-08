# -*- coding: utf-8 -*-
import sys
import argparse
from modeles.bayes_naif import BayesNaif
from modeles.perceptron import Perceptron
from modeles.perceptron_mc import PerceptronMC
from modeles.svm import SVM
from modeles.fad import FAD
from modeles.adaboost import AdaBoost
from gestion_donnees import BaseDonnees
from analyse import Analyse, Analyse_multiple

def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("fichier", type=str,
                    help="Nom du fichier de données.")
    p.add_argument("vc", type=int, choices=[0,1],
                    help="Choix de validation croisée ou pas.")
    p.add_argument("choix_modele", type=str, 
                    choices=["bayes_naif","perceptron",
                    "perceptron_mc","svm","fad","adaboost"],
                    help="Choix du modèle à utiliser.")
    p.add_argument("--repetitions", type=int, default=1,
                    help="Nombre de répétitions à faire pour moyenner.")
    p.add_argument("--est_ech_poids", type=int, default=0,
                    choices=[0,1],
                    help="Choix de poids d'échantillon ou pas.")
    p.add_argument("--tol_perceptron", type=float, default=1e-3,
                    help="Critère de tolérance (perceptron).")
    p.add_argument("--max_iter_perceptron", type=int, default=1000,
                    help="Maximum d'itérations (perceptron).")
    p.add_argument("--couches_cachees", type=str, default="2,5,2",
                    help="Nombre de neurones par couches (perceptron_mc).")
    p.add_argument("--activation", type=str, default="relu", 
                    choices=["identity","logistic","tanh","relu"],
                    help="Type de fonction d'activation (perceptron_mc).")
    p.add_argument("--solutionneur", type=str, default="sgd", 
                    choices=["lbfgs","sgd","adam"],
                    help="Type de solutionneur (perceptron_mc).")
    p.add_argument("--max_iter_perceptron_mc", type=int, default=2000,
                    help="Maximum d'itérations (perceptron_mc).")
    p.add_argument("--noyau", type=str, default="rbf",
                    choices=["linear","poly","rbf","sigmoid"],
                    help="Type de noyau (svm).")
    p.add_argument("--tol_svm", type=float, default=1e-3,
                    help="Critère de tolérance (svm).")
    p.add_argument("--max_iter_svm", type=int, default=-1,
                    help="Maximum d'itérations (svm).")
    p.add_argument("--nb_arbres", type=int, default=10,
                    help="Nombre d'arbres dans la forêt (fad).")
    p.add_argument("--critere", type=str, default="gini",
                    choices=["gini","entropy"],
                    help="Critère de séparation (fad).")
    p.add_argument("--prof_max_fad", type=int, default=None,
                    help="Profondeur maximale d'un arbre (fad).")
    p.add_argument("--prof_max_adaboost", type=int, default=1,
                    help="Profondeur maximale de l'arbre (adaboost).")

    return p

def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    #-------------------Gestion des données--------------------------
    bd = BaseDonnees(args.fichier, 'is_legendary')
    liste_colonne = bd.voir_att()
    bd.enlever_attributs(['abilities', 'japanese_name', 'name', 'generation', 'pokedex_number', 'classfication'])
    bd.str_a_int(['capture_rate'])
    bd.str_a_vec(['type1', 'type2'])
    bd.normaliser_donnees()
    bd.methode_filtrage()
    if(bool(args.est_ech_poids)):
        poids = bd.definir_poids_att()
    else:
        poids = []

    #-------------------Gestion du modèle----------------------------
    print("Création du modèle...")
    if(args.choix_modele == "bayes_naif"):
        modele = BayesNaif()
    elif(args.choix_modele == "perceptron"):
        modele = Perceptron(max_iter = args.max_iter_perceptron, tol = args.tol_perceptron)
    elif(args.choix_modele == "perceptron_mc"):
        modele = PerceptronMC(couches_cachees = tuple([int(x) for x in args.couches_cachees.split(',')]), 
                    activation = args.activation, solutionneur = args.solutionneur,
                    max_iter = args.max_iter_perceptron_mc)
    elif(args.choix_modele == "svm"):
        modele = SVM(noyau = args.noyau, tol = args.tol_svm, max_iter = args.max_iter_svm)
    elif(args.choix_modele == "fad"):
        modele = FAD(nb_arbres = args.nb_arbres, critere = args.critere, prof_max = args.prof_max_fad)
    elif(args.choix_modele == "adaboost"):
        modele = AdaBoost(max_prof = args.prof_max_adaboost)


    #-------------------Répétitions pour moyenner-----------
    analyse_mult = Analyse_multiple(args.repetitions)

    for i in range(args.repetitions):
        x_entr, t_entr, x_test, t_test= bd.faire_ens_entr_test()

        #-------------------Entrainement ou validation croisée-----------
        if bool(args.vc) is False:
            print("Début de l'entrainement simple...")
            modele.entrainement(x_entr, t_entr, args.est_ech_poids, poids) 
        else:
            print("Début de l'entrainement par validation croisée...")
            modele.validation_croisee(x_entr, t_entr, 10, args.est_ech_poids, poids)

        #-------------------Prédiction et erreur-------------------------
        print("Calcul des erreurs...")
        predictions_entrainement = modele.prediction(x_entr)
        erreur_entrainement = modele.erreur(t_entr, predictions_entrainement) / len(t_entr) * 100

        predictions_test = modele.prediction(x_test)
        erreur_test = modele.erreur(t_test, predictions_test) / len(t_test) * 100

        print('Erreur train = ', erreur_entrainement, '%')
        print('Erreur test = ', erreur_test, '%')

        #-------------------Analyse des résultats------------------------
        print("Analyse des résultats...")
        prob = modele.confiance_test(x_test)
        analyse = Analyse(t_test, predictions_test, prob)
        analyse.calculer_comptes()
        analyse.afficher_comptes()
        analyse.calculer_metriques()
        analyse.afficher_metriques()
        analyse.calculer_courbe_roc()
        analyse.afficher_courbe_roc()
    
if __name__ == "__main__":
    main()