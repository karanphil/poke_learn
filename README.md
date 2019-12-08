# Poke_learn
**Poké_learn** est le nom du projet fait dans le cadre du projet de session du cours IFT712.

**Poké_learn** comprend la base de données pokemon prise sur [Kaggle] ainsi que le code traitant la base de données, testant six modèles différents d'entraînement et calculant des métriques de performance. 

### Dépendances et utilisation

On recommande d'exécuter le code sur un environnement virtuel correspondant aux requis du cours IFT712.

Voici comment obtenir la documentation sur le code: 

```
python poke_learn.py --h
```
et un exemple pour classifier avec le modèle du perceptron multi-couches:

```
python poke_learn.py pokemon.csv 1 perceptron_mc --couches_cachees 20,70,100,70,20 --max_iter_perceptron_mc 3000 --solutionneur lbfgs

```

[Kaggle]:https://www.kaggle.com/rounakbanik/pokemon
