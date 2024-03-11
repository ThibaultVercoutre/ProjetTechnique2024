import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import numpy as np

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
# from tensorflow.keras.applications import InceptionV3, ResNet50

from petitpoisson import *

# fonction de récupération des données

def get_dataframe():
    data, index = get_donnees()
    data = list(zip(*data))
    dic = {cle: value for cle, value in zip(index, data)}
    X = pd.DataFrame(dic)
    return X

# variables et séparation des données

df_modifie = get_dataframe()

# X = df_modifie[['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left']]
# Y = df_modifie['age']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# fonctions de modèles

def regression_logistique(tolerence, X_train, X_test, Y_train, Y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    # print(f"Précision du modèle Regressions Logistique : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def support_vector_machines(tolerence, X_train, X_test, Y_train, Y_test):
    model = SVC()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    # print(f"Précision du modèle Support Vector Machines : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def discriminant_analysis(tolerence, X_train, X_test, Y_train, Y_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    # print(f"Précision du modèle Discriminant Analysis : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def random_forests(tolerence, X_train, X_test, Y_train, Y_test):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    # print(f"Précision du modèle Random Forests : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def gradient_boosting_machines(tolerence, X_train, X_test, Y_train, Y_test, choix_n_estimators):
    model = GradientBoostingClassifier(n_estimators=choix_n_estimators, learning_rate=0.1, random_state=42)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    # print(f"Précision du modèle Gradient Boosting Machines : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def test_GBM():
    suite = ['nombre_raies', 'max_diametre', 'surface', 'diametre', 'relativeopacity', 'RC_top', 'RC_right', 'densite', 'RC_bottom', 'RC_left']

    Ygradient = []

    for i in range(100):
        nb_barres = '\033[91m' + '#' * int(i / (100) * 100) + ' ' * int(100 - i / (100) * 100) + '\033[0m'
        print(nb_barres, str(i / (100) * 100) + '%', end='\n')

        print("\033[92m" + f"N estimators : {i + 1}" + "\033[0m")
        X = df_modifie[suite]
        Y = df_modifie['age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        Ygradient.append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, i + 1))

        print("\033[94m" + f"Gradient Boosting Machines : {Ygradient}" + "\033[0m")

        os.system('cls')

    models = ['Régression logistique', 'Support Vector Machines', 'Discriminant Analysis', 'Random Forests', 'Gradient Boosting Machines']
    
    x = np.arange(1, 100)
    plt.plot(x, Ygradient, 'o-', label=models[4])
    plt.xlabel('Paramètres')
    plt.xticks(x, suite, rotation='horizontal')
    plt.ylabel('Précision')
    plt.title('Précision des modèles en fonction des paramètres')
    plt.legend()
    plt.show()

# utiliser tensorflow pour les modèles suivants
def cnn():
    pass


def inception():
    pass


def resnet():
    pass

# fonction principale

def meilleurs_données(colonnes: list, test: list):
    print("\033[92m" + f"Test : {test}" + "\033[0m")
    print("\033[94m" + f"Colonnes à test : {colonnes}" + "\033[0m")
    
    resultats = {
        'Régression logistique': [],
        'Support Vector Machines': [],
        'Discriminant Analysis': [],
        'Random Forests': [],
        'Gradient Boosting Machines': []
    }

    resultats_colonnes = {}

    for colonne in colonnes:
        colonnes_test = test.copy()
        colonnes_test.append(colonne)
        print("\033[91m" + f"Colonne : {colonne}" + "\033[0m")
        X = df_modifie[colonnes_test]
        Y = df_modifie['age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        resultats['Régression logistique'].append(regression_logistique(0, X_train, X_test, Y_train, Y_test))
        resultats['Support Vector Machines'].append(support_vector_machines(0, X_train, X_test, Y_train, Y_test))
        resultats['Discriminant Analysis'].append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test))
        resultats['Random Forests'].append(random_forests(0, X_train, X_test, Y_train, Y_test))
        resultats['Gradient Boosting Machines'].append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test))

    for key, value in resultats.items():
        # message = f"Meilleur colonne pour {key} : {colonnes[value.index(max(value))]}"
        resultats_colonnes[colonnes[value.index(max(value))]] = resultats_colonnes.get(colonnes[value.index(max(value))], 0) + 1
        # print(message)

    best_column = max(resultats_colonnes, key=resultats_colonnes.get)
    # print(f"Meilleure colonne pour {best_column} : {colonnes[resultats[best_column].index(max(resultats[best_column]))]}")
    test.append(best_column)
    colonnes.remove(best_column)
    if len(colonnes) != 0:
        meilleurs_données(colonnes, test)
    
    return test
    # print(f"Meilleure colonne globale : {best_column}")


def meilleure_combinaison():
    colonnes = ['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left', 'nombre_raies']
    
    # trouver toutes les combinaisons possibles de colonnes
    all_combinations = []
    for i in range(7, len(colonnes) + 1):
        combinations_i = list(itertools.combinations(colonnes, i))
        all_combinations.extend(combinations_i)

    # Afficher toutes les combinaisons
    for combination in all_combinations:
        print(combination)

    print("\033[93m" + f"Nombre de combinaisons possibles : {len(all_combinations)}" + "\033[0m")
    
    Ylogistique = []
    Ysvm = []
    Ydiscriminant = []
    Yrandom = []
    Ygradient = []

    for combination in all_combinations:
        
        nb_barres = '\033[91m' + '#' * int(all_combinations.index(combination) / (len(all_combinations)) * 100) + ' ' * int(100 - all_combinations.index(combination) / (len(all_combinations)) * 100) + '\033[0m'
        print(nb_barres, str(all_combinations.index(combination) / (len(all_combinations)) * 100) + '%', end='\n')

        print("\033[92m" + f"Combinaison : {combination}" + "\033[0m")
        X = df_modifie[list(combination)]
        Y = df_modifie['age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # print("\033[94m" + f"Régression logistique\033[0m")
        # Ylogistique.append(regression_logistique(0, X_train, X_test, Y_train, Y_test))
        # print("\033[94m" + f"Support Vector Machines\033[0m")
        # Ysvm.append(support_vector_machines(0, X_train, X_test, Y_train, Y_test))
        # print("\033[94m" + f"Discriminant Analysis\033[0m")
        # Ydiscriminant.append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test))
        # print("\033[94m" + f"Random Forests\033[0m")
        # Yrandom.append(random_forests(0, X_train, X_test, Y_train, Y_test))
        print("\033[94m" + f"Gradient Boosting Machines\033[0m")
        Ygradient.append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test))

        # print("\033[94m" + f"Régression logistique : {Ylogistique}" + "\033[0m")
        # print("\033[94m" + f"Support Vector Machines : {Ysvm}" + "\033[0m")
        # print("\033[94m" + f"Discriminant Analysis : {Ydiscriminant}" + "\033[0m")
        # print("\033[94m" + f"Random Forests : {Yrandom}" + "\033[0m")
        # print("\033[94m" + f"Gradient Boosting Machines : {Ygradient}" + "\033[0m")
        
        os.system('cls')

    # max_logistique_index = Ylogistique.index(max(Ylogistique))
    # max_svm_index = Ysvm.index(max(Ysvm))
    # max_discriminant_index = Ydiscriminant.index(max(Ydiscriminant))
    # max_random_index = Yrandom.index(max(Yrandom))
    max_gradient_index = Ygradient.index(max(Ygradient))

    # print(f"Meilleure combinaison pour Régression logistique : {all_combinations[max_logistique_index]} d'un taux de {round(Ylogistique[max_logistique_index], 2)}")
    # print(f"Meilleure combinaison pour Support Vector Machines : {all_combinations[max_svm_index]} d'un taux de {round(Ysvm[max_svm_index], 2)}")
    # print(f"Meilleure combinaison pour Discriminant Analysis : {all_combinations[max_discriminant_index]} d'un taux de {round(Ydiscriminant[max_discriminant_index], 2)}")
    # print(f"Meilleure combinaison pour Random Forests : {all_combinations[max_random_index]} d'un taux de {round(Yrandom[max_random_index], 2)}")
    print(f"Meilleure combinaison pour Gradient Boosting Machines : {all_combinations[max_gradient_index]} d'un taux de {round(Ygradient[max_gradient_index], 2)}")


def liste_para():
    colonnes = ['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left', 'nombre_raies']
    # suite = meilleurs_données(colonnes, [])
    # suite = ['max_diametre', 'densite', 'diametre', 'RC_top', 'nombre_raies', 'RC_right', 'surface', 'RC_bottom', 'relativeopacity', 'RC_left']
    suite = ['nombre_raies', 'max_diametre', 'surface', 'diametre', 'relativeopacity', 'RC_top', 'RC_right', 'densite', 'RC_bottom', 'RC_left']
    print(suite)

    params = []
    Ylogistique = []
    Ysvm = []
    Ydiscriminant = []
    Yrandom = []
    Ygradient = []

    for _, param in enumerate(suite):
        params.append(param)
        print("\033[92m" + f"Paramètres : {params}" + "\033[0m")
        X = df_modifie[params]
        Y = df_modifie['age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        Ylogistique.append(regression_logistique(0, X_train, X_test, Y_train, Y_test))
        Ysvm.append(support_vector_machines(0, X_train, X_test, Y_train, Y_test))
        Ydiscriminant.append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test))
        Yrandom.append(random_forests(0, X_train, X_test, Y_train, Y_test))
        Ygradient.append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test))

        print("\033[94m" + f"Régression logistique : {Ylogistique}" + "\033[0m")
        print("\033[94m" + f"Support Vector Machines : {Ysvm}" + "\033[0m")
        print("\033[94m" + f"Discriminant Analysis : {Ydiscriminant}" + "\033[0m")
        print("\033[94m" + f"Random Forests : {Yrandom}" + "\033[0m")
        print("\033[94m" + f"Gradient Boosting Machines : {Ygradient}" + "\033[0m")

    models = ['Régression logistique', 'Support Vector Machines', 'Discriminant Analysis', 'Random Forests', 'Gradient Boosting Machines']
    
    x = np.arange(0, len(suite))

    plt.plot(x, Ylogistique, 'o-', label=models[0])
    plt.plot(x, Ysvm, 'o-', label=models[1])
    plt.plot(x, Ydiscriminant, 'o-', label=models[2])
    plt.plot(x, Yrandom, 'o-', label=models[3])
    plt.plot(x, Ygradient, 'o-', label=models[4])
    plt.xlabel('Paramètres')
    plt.xticks(x, suite, rotation='horizontal')
    plt.ylabel('Précision')
    plt.title('Précision des modèles en fonction des paramètres')
    plt.legend()
    plt.show()


def main():
    colonnes = ['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left', 'nombre_raies']
    X = df_modifie[colonnes]
    Y = df_modifie['age']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Précision du modèle Discriminant Analysis : {regression_logistique(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Support Vector Machines : {support_vector_machines(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Discriminant Analysis : {discriminant_analysis(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Random Forests : {random_forests(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Gradient Boosting Machines : {gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test)}")


if __name__ == "__main__":
    # liste_para()
    # main()
    # meilleure_combinaison()
    test_GBM()