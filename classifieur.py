# -*- coding: utf-8 -*-
import numpy as np

class Classifieur:
    def __init__(self):
        """
        Classe parent des classes modèles, qui contient les méthodes communes
        aux différents modèles. La class Classifieur ne devrait pas être
        instanciée directement, dans quel cas elle ne retourne rien.
        """
        self.modele = None
    
    def entrainement(self, x_entr, t_entr, est_ech_poids = False, *args):
        """
        Entraine une méthode d'apprentissage de type modèle, et prend en
        compte la possibilité d'un poids variables d'échantillons, qui donne
        une importance différente à divers attributs. 
        """
        # Cas avec poids variables
        if(est_ech_poids):
            self.modele.fit(x_entr, t_entr, args[0])
        # Cas sans poids variables
        else:
            self.modele.fit(x_entr, t_entr)

    def prediction(self, x_test):
        """
        Retourne la prédiction de classe pour une entrée representée par un 
        tableau 1D Numpy.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée.
        """
        return(self.modele.predict(x_test))

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``. 
        """   
        return np.sum((t-prediction)**2)

    def validation_croisee(self, x_entr, t_entr, k = 10, est_ech_poids = False, *args):
        """
        Cette méthode utilise simplement la méthode ``entrainement()``, qui est le cas
        le plus général pour certains modèles. Des véritables validations croisées 
        propres aux modèles sont appelées directement dans les modèles.
        """
        self.entrainement(x_entr, t_entr, est_ech_poids, args[0])

    def confiance_test(self, x):
        """
        Cette méthode renvoie le résultat de predict_proba ou decision_function,
        dépendant du modèle utilisé.
        """
        try:
            confiance = self.modele.predict_proba(x).T[1]
        except AttributeError:
            try:
                confiance = self.modele.decision_function(x)
            except AttributeError:
                print("Le modèle ne possède pas cette fonction.")
                confiance = None
        return confiance

    def affichage(self, x_tab, t_tab):
        raise NotImplementedError



