import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, LassoLarsIC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import time

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
# from tensorflow.keras.applications import InceptionV3, ResNet50

from recupparams import *

# fonction de récupération des données

def get_dataframe():
    data, index = get_donnees()
    data = list(zip(*data))
    dic = {cle: value for cle, value in zip(index, data)}
    X = pd.DataFrame(dic)
    return X.dropna()

# variables et séparation des données

df_modifie = get_dataframe()

def split_train_test_by_age(X, Y, test_size=0.2, random_state=42):
    # Créer un DataFrame avec X et Y
    data = pd.concat([X, Y], axis=1)

    # Séparer les données par âge
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for age in data['age'].unique():
        age_data: pd.DataFrame = data[data['age'] == age]
        X_age = age_data.drop(['age'], axis=1)
        Y_age = age_data['age']
        X_train_age, X_test_age, Y_train_age, Y_test_age = train_test_split(X_age, Y_age, test_size=test_size, random_state=random_state)
        train_data = pd.concat([train_data, pd.concat([X_train_age, Y_train_age], axis=1)])
        test_data = pd.concat([test_data, pd.concat([X_test_age, Y_test_age], axis=1)])

    # Séparer les données en X_train, X_test, Y_train, Y_test
    X_train = train_data.drop(['age'], axis=1)
    Y_train = train_data['age']
    X_test = test_data.drop(['age'], axis=1)
    Y_test = test_data['age']

    return X_train, X_test, Y_train, Y_test

# X = df_modifie[['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left']]
# Y = df_modifie['age']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# fonctions de modèles

def regression_logistique(tolerence, X_train, X_test, Y_train, Y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Overfitting': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def support_vector_machines(tolerence, X_train, X_test, Y_train, Y_test):
    model = SVC()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Overfitting': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def discriminant_analysis(tolerence, X_train, X_test, Y_train, Y_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Overfitting': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def random_forests(tolerence, X_train, X_test, Y_train, Y_test):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Overfitting': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def gradient_boosting_machines(tolerence, X_train, X_test, Y_train, Y_test):
    model = GradientBoostingClassifier() # n_estimators=choix_n_estimators, learning_rate=choix_learning_rate, random_state=42
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Overfitting': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def afficher_test_GBM(data):

    cursor = db.cursor()

    query = f'SELECT x, y FROM {data}'
    cursor.execute(query)
    rows = cursor.fetchall()

    x = [row[0] for row in rows]
    y = [row[1] for row in rows]

    plt.plot(x, y, 'o-', label=['Gradient Boosting Machines'])
    # for i, txt in enumerate(t):
    #     plt.annotate(txt, (x[i], y[i]))
    plt.xlabel(data)
    # plt.xticks(x, suite, rotation='horizontal')
    plt.ylabel('Précision')
    plt.title('Précision des modèles en fonction de ' + data)
    plt.legend()
    plt.show()

    cursor.close()

def afficher_test_GBM_3D():
    cursor = db.cursor()

    query = f'SELECT * FROM GBM_variations'
    cursor.execute(query)
    rows = cursor.fetchall()

    x = [row[0] for row in rows]
    y = [row[1] for row in rows]
    z = [row[2] for row in rows]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    nb_x_unique = len(np.unique(x))
    nb_y_unique = len(np.unique(y))

    print(f"Nombre de tests pour le learning rate : {nb_x_unique}")
    print(f"Nombre de tests pour le n_estimators : {nb_y_unique}")

    max_z = z.argmax()
    print(f"Learning Rate : {x[max_z]} N Enumerates : {y[max_z]} Score : {z[max_z]}")

    # Ygradient_array = z.reshape(nb_y_unique, nb_x_unique)

    X, Y = np.meshgrid(np.unique(x), np.unique(y))

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Ygradient_array)
    ax.scatter(x, y, z, c='r', marker='.')

    # dessiner le point maximum
    ax.scatter(x[max_z], y[max_z], z[max_z], color='b')

    # Set labels
    ax.set_xlabel('Learning Rate Test Number')
    ax.set_ylabel('N Enumerates Test Number')
    ax.set_zlabel('Gradient Boosting Machines Score')

    # Show the plot
    plt.show()

    cursor.close()


def test_GBM_learning_rate():
    suite = ['nombre_raies', 'max_diametre', 'surface', 'diametre', 'relativeopacity', 'RC_top', 'RC_right', 'densite', 'RC_bottom', 'RC_left']

    Ygradient = []

    nb_tests_learning_rate = 40
    pas_learning_rate = 1

    cursor = db.cursor()

    for i in range(0, nb_tests_learning_rate, pas_learning_rate):
        nb_tests_n_estimators = 40
        pas_n_estimators = 1

        YgradientBeta = []

        for j in range(0, nb_tests_n_estimators, pas_n_estimators):

            nb_barres = '\033[91m' + '#' * int(i / (nb_tests_learning_rate) * 100) + ' ' * int(100 - i / (nb_tests_learning_rate) * 100) + '\033[0m'
            print(nb_barres, str(i / (nb_tests_learning_rate) * 100) + '%', end='\n')
            
            print("\033[92m" + f"Learning rate : {(i + pas_learning_rate)/100}" + "\033[0m")
            print("\033[92m" + f"N estimators : {(j + 1)}" + "\033[0m")

            cursor.execute('SELECT * FROM GBM_variations WHERE learning_rate = ? and n_estimators = ?', 
                            ((i + pas_learning_rate)/100, j + 1))
            
            row = cursor.fetchone()

            if row is None:
                X = df_modifie[suite]
                Y = df_modifie['age']
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

                result = gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, (j + 1), (i + pas_learning_rate)/100)
                
                cursor.execute('INSERT INTO GBM_variations (learning_rate, n_estimators, y) VALUES (?, ?, ?)', 
                                ((i + pas_learning_rate)/100, j+1, result))
            
            # else:
            #     cursor.execute('UPDATE GBM_variations SET y = ? WHERE learning_rate = ? and n_estimators = ?', 
            #                     (result, (i + pas_learning_rate)/100, j+1))


            # print("\033[94m" + f"Gradient Boosting Machines : {Ygradient}" + "\033[0m")

            os.system('cls')
        
        # Ygradient.append(YgradientBeta)

    db.commit()
    cursor.close()

    afficher_test_GBM_3D()


def test_GBM_n_enumerate():
    suite = ['nombre_raies', 'max_diametre', 'surface', 'diametre', 'relativeopacity', 'RC_top', 'RC_right', 'densite', 'RC_bottom', 'RC_left']

    Ygradient = []
    Tgradient = []

    nb_tests = 100

    cursor = db.cursor()

    for i in range(nb_tests):
        start_time = time.time()

        nb_barres = '\033[91m' + '#' * int(i / (nb_tests) * 100) + ' ' * int(100 - i / (nb_tests) * 100) + '\033[0m'
        print(nb_barres, str(i / (nb_tests) * 100) + '%', end='\n')

        print("\033[92m" + f"N estimators : {i + 1}" + "\033[0m")
        X = df_modifie[suite]
        Y = df_modifie['age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        Ygradient.append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, i + 1))
        Tgradient.append(round(time.time() - start_time, 2))

        cursor.execute('SELECT x FROM GBM_n_estimators_variations WHERE x = ?', 
                        (i + 1,))
        
        row = cursor.fetchone()

        if row is None:
            cursor.execute('INSERT INTO GBM_n_estimators_variations (x, y, t) VALUES (?, ?, ?)', 
                            (i + 1, Ygradient[i], Tgradient[i]))
        else:
            cursor.execute('UPDATE GBM_n_estimators_variations SET y = ?, t = ? WHERE x = ?', 
                            (Ygradient[i], Tgradient[i], i + 1))


        print("\033[94m" + f"Gradient Boosting Machines : {Ygradient}" + "\033[0m")

        os.system('cls')

    db.commit()
    cursor.close()

    afficher_test_GBM("GBM_n_estimators_variations")


# utiliser tensorflow pour les modèles suivants
def cnn():
    pass


def inception():
    pass


def resnet():
    pass

# fonction principale

def meilleures_donnees(colonnes: list, test: list):
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

    model = 'Gradient Boosting Machines'

    for colonne in colonnes:
        colonnes_test = test.copy()
        colonnes_test.append(colonne)
        print("\033[91m" + f"Colonne : {colonne}" + "\033[0m")
        X = df_modifie[colonnes_test]
        Y = df_modifie['age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # resultats['Régression logistique'].append(regression_logistique(0, X_train, X_test, Y_train, Y_test))
        # resultats['Support Vector Machines'].append(support_vector_machines(0, X_train, X_test, Y_train, Y_test))
        # resultats['Discriminant Analysis'].append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test))
        # resultats['Random Forests'].append(random_forests(0, X_train, X_test, Y_train, Y_test))
        resultats['Gradient Boosting Machines'].append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test))

    # trouver la meilleure colonne pour la régression logistique
    resultats_colonnes[model] = [resultats[model][i]['Accuracy'] for i in range(len(resultats[model]))]

    best_column = colonnes[resultats_colonnes[model].index(max(resultats_colonnes[model]))]

    print(best_column)

    # print(f"Meilleure colonne pour {best_column} : {colonnes[resultats[best_column].index(max(resultats[best_column]))]}")
    test.append(best_column)
    colonnes.remove(best_column)
    if len(colonnes) != 0:
        meilleures_donnees(colonnes, test)
    
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


def liste_para(suite):
    # suite = meilleures_donnees(colonnes, [])
    # suite = ['max_diametre', 'densite', 'diametre', 'RC_top', 'nombre_raies', 'RC_right', 'surface', 'RC_bottom', 'relativeopacity', 'RC_left']
    # suite = ['nombre_raies', 'max_diametre', 'surface', 'diametre', 'relativeopacity', 'RC_top', 'RC_right', 'densite', 'RC_bottom', 'RC_left']
    suite = ['RC_left', 'densite', 'opacity', 'elongation', 'min_diametre', 'surface', 'max_diametre', 'growth', 'diametre', 'relative_opacity', 'nombre_raies', 'RC_right', 'RC_bottom', 'RC_top']

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
        # Y = df_modifie['seuil_age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        Ylogistique.append(regression_logistique(0, X_train, X_test, Y_train, Y_test)['Accuracy'])
        # Ysvm.append(support_vector_machines(0, X_train, X_test, Y_train, Y_test)['Accuracy'])
        # Ydiscriminant.append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test)['Accuracy'])
        # Yrandom.append(random_forests(0, X_train, X_test, Y_train, Y_test)['Accuracy'])
        # Ygradient.append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test)['Accuracy'])

        print("\033[94m" + f"Régression logistique : {Ylogistique}" + "\033[0m")
        # print("\033[94m" + f"Support Vector Machines : {Ysvm}" + "\033[0m")
        # print("\033[94m" + f"Discriminant Analysis : {Ydiscriminant}" + "\033[0m")
        # print("\033[94m" + f"Random Forests : {Yrandom}" + "\033[0m")
        # print("\033[94m" + f"Gradient Boosting Machines : {Ygradient}" + "\033[0m")

    models = ['Régression logistique', 'Support Vector Machines', 'Discriminant Analysis', 'Random Forests', 'Gradient Boosting Machines']
    
    x = np.arange(0, len(suite))

    plt.plot(x, Ylogistique, 'o-', label=models[0])
    # plt.plot(x, Ysvm, 'o-', label=models[1])
    # plt.plot(x, Ydiscriminant, 'o-', label=models[2])
    # plt.plot(x, Yrandom, 'o-', label=models[3])
    # plt.plot(x, Ygradient, 'o-', label=models[4])
    plt.xlabel('Paramètres')
    plt.xticks(x, suite, rotation='horizontal')
    plt.ylabel('Précision')
    plt.title('Précision des modèles en fonction des paramètres')
    plt.legend()
    plt.show()

def best_n_estimators():
    cursor = db.cursor()

    query = f'SELECT x, y FROM GBM_n_estimators_variations'
    cursor.execute(query)
    rows = cursor.fetchall()

    x = [row[0] for row in rows]
    y = [row[1] for row in rows]
    
    max_x_n_estimators = int(x[y.index(max(y))])

    query = f'SELECT x, y FROM GBM_learning_rate_variations'
    cursor.execute(query)
    rows = cursor.fetchall()

    x = [row[0] for row in rows]
    y = [row[1] for row in rows]

    max_x_learning_rate = x[y.index(max(y))]

    cursor.close()

    return max_x_n_estimators, max_x_learning_rate

def boxplot(col, ax, i):
    # Obtenir les âges uniques à partir des données
    ages_uniques = df_modifie['age'].unique()
    ages_uniques.sort()

    print(ages_uniques)

    # Générer les positions et les étiquettes en fonction des âges uniques
    positions = list(range(int(min(ages_uniques)), len(ages_uniques) + 1))
    etiquettes = list(range(int(min(ages_uniques)), len(ages_uniques)))

    donnees = df_modifie[[col, 'age']]

    donnees_par_age = donnees.groupby('age')[col].apply(list).tolist()

    fontsize = 7

    ax[i].boxplot(donnees_par_age)
    # ax[i].set_title(f'Box Plot de la caractéristique {col} en fonction de l\'Âge', fontsize=fontsize)
    ax[i].set_xlabel('Âge', fontsize=fontsize)
    ax[i].set_ylabel(col, fontsize=fontsize)
    ax[i].tick_params(axis='x', labelsize=fontsize) 
    ax[i].tick_params(axis='y', labelsize=fontsize) 
    ax[i].set_xticks(positions, positions)

def boxplot_all_columns(colonnes: list):
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))

    for i, col in enumerate(colonnes):
        boxplot(col, ax.flatten(), i)

    plt.show()

def meilleur_sous_ensemble(modele, X, Y, n_features):

    # Créer un objet RFE pour sélectionner les meilleures caractéristiques
    rfe = RFE(estimator=modele, n_features_to_select=n_features - 1)
    
    # Adapter le RFE au jeu de données
    rfe.fit(X, Y)
    
    # Renvoyer les indices des caractéristiques sélectionnées
    return X.columns[rfe.support_].tolist()

def meilleur_sous_ensemble_tout_classifieur(colonnes):
    X = df_modifie[colonnes]  # Vos données d'entraînement
    Y = df_modifie['age']

    modele_LR = LogisticRegression()
    modele_SVC = SVC()
    modele_LDA = LinearDiscriminantAnalysis()
    modele_RFC = RandomForestClassifier()
    modele_GBM = GradientBoostingClassifier()

    # Appeler la fonction pour obtenir le meilleur sous-ensemble
    caracteristiques_selectionnees_LR = meilleur_sous_ensemble(modele_LR, X, Y, n_features=len(colonnes))
    # meilleur_sous_ensemble_SVC = meilleur_sous_ensemble(modele_SVC, X, Y, n_features=5)
    caracteristiques_selectionnees_LDA = meilleur_sous_ensemble(modele_LDA, X, Y, n_features=len(colonnes))
    caracteristiques_selectionnees_RFC = meilleur_sous_ensemble(modele_RFC, X, Y, n_features=len(colonnes))
    # meilleur_sous_ensemble_GBM = meilleur_sous_ensemble(modele_GBM, X, Y, n_features=5)

    print("Les meilleures caractéristiques LR sélectionnées :")
    print(caracteristiques_selectionnees_LR)
    # print("Les meilleures caractéristiques SVC sélectionnées :")
    # print(caracteristiques_selectionnees_SVC)
    print("Les meilleures caractéristiques LDA sélectionnées :")
    print(caracteristiques_selectionnees_LDA)
    print("Les meilleures caractéristiques RFC sélectionnées :")
    print(caracteristiques_selectionnees_RFC)
    # print("Les meilleures caractéristiques GBM sélectionnées :")
    # print(caracteristiques_selectionnees_GBM)

def APRF1(colonnes: list):
    columns_to_normalize = df_modifie.columns.difference(['age', 'filepath'])

    # Créer un objet StandardScaler
    scaler = StandardScaler()

    # Normaliser les données
    df_modifie[columns_to_normalize] = scaler.fit_transform(df_modifie[columns_to_normalize])

    X = df_modifie[colonnes]
    Y = df_modifie['age']
    # Y = df_modifie['seuil_age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_train, X_test, Y_train, Y_test = split_train_test_by_age(X, Y, test_size=0.2, random_state=42)

    # max_x_n_estimators, max_x_learning_rate = best_n_estimators()
    # print(max_x_n_estimators, max_x_learning_rate)

    classifiers = {
        'Regression Logistique': regression_logistique(0, X_train, X_test, Y_train, Y_test),
        'Support Vector Machines': support_vector_machines(0, X_train, X_test, Y_train, Y_test),
        'Discriminant Analysis': discriminant_analysis(0, X_train, X_test, Y_train, Y_test),
        'Random Forests': random_forests(0, X_train, X_test, Y_train, Y_test),
        'Gradient Boosting Machines': gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test)
    }

    print(classifiers)

    # Affichage des scores de précision et de rappel
    classifiers_names = list(classifiers.keys())
    accuracy_scores = [classifiers[classifier]['Accuracy'] for classifier in classifiers_names]
    precision_scores = [classifiers[classifier]['Precision'] for classifier in classifiers_names]
    recall_scores = [classifiers[classifier]['Recall'] for classifier in classifiers_names]
    f1_scores = [classifiers[classifier]['F1_score'] for classifier in classifiers_names]

    barWidth = 0.2

    # Position des barres pour les différents scores
    r1 = np.arange(len(classifiers_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + 2*barWidth for x in r1]
    r4 = [x + 3*barWidth for x in r1]

    # Création du graphique avec les barres côte à côte pour chaque métrique
    plt.figure(figsize=(12, 6))
    plt.bar(r1, accuracy_scores, color='y', width=barWidth, edgecolor='grey', label='Accuracy')
    plt.bar(r2, precision_scores, color='b', width=barWidth, edgecolor='grey', label='Precision')
    plt.bar(r3, recall_scores, color='g', width=barWidth, edgecolor='grey', label='Recall')
    plt.bar(r4, f1_scores, color='r', width=barWidth, edgecolor='grey', label='F1 Score')

    # Ajout des étiquettes au-dessus des barres
    for i, value in enumerate(r1):
        plt.text(value, accuracy_scores[i]+0.02, str(round(accuracy_scores[i], 2)), ha='center', va='bottom')
    
    for i, value in enumerate(r2):
        plt.text(value, precision_scores[i]+0.02, str(round(precision_scores[i], 2)), ha='center', va='bottom')
        
    for i, value in enumerate(r3):
        plt.text(value, recall_scores[i]+0.02, str(round(recall_scores[i], 2)), ha='center', va='bottom')
        
    for i, value in enumerate(r4):
        plt.text(value, f1_scores[i]+0.02, str(round(f1_scores[i], 2)), ha='center', va='bottom')

    # Configuration du graphique
    plt.xlabel('Classifieurs')
    plt.ylabel('Scores')
    plt.title('Scores d Accuracy, Precision, Recall et F1 des Classifieurs')
    plt.xticks([r + 1.5*barWidth for r in range(len(classifiers_names))], classifiers_names)
    plt.legend()

    plt.ylim(0, 1)

    # Affichage du graphique
    plt.show()

def main(colonnes: list):
    # columns_to_normalize = df_modifie.columns.difference(['age', 'filepath'])

    # # Créer un objet StandardScaler
    # scaler = StandardScaler()

    # # Normaliser les données
    # df_modifie[columns_to_normalize] = scaler.fit_transform(df_modifie[columns_to_normalize])

    X = df_modifie[colonnes]
    Y = df_modifie['age']
    # Y = df_modifie['seuil_age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_train, X_test, Y_train, Y_test = split_train_test_by_age(X, Y, test_size=0.2, random_state=42)

    # max_x_n_estimators, max_x_learning_rate = best_n_estimators()
    # print(max_x_n_estimators, max_x_learning_rate)

    classifiers = {
        'Regression Logistique': regression_logistique(0, X_train, X_test, Y_train, Y_test),
        'Support Vector Machines': support_vector_machines(0, X_train, X_test, Y_train, Y_test),
        'Discriminant Analysis': discriminant_analysis(0, X_train, X_test, Y_train, Y_test),
        'Random Forests': random_forests(0, X_train, X_test, Y_train, Y_test),
        'Gradient Boosting Machines': gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test)
    }

    # tracer l'overfitting des modèles sur un graphique en barres
    classifiers_names = list(classifiers.keys())
    overfitting_scores = [classifiers[classifier]['Overfitting'] for classifier in classifiers_names]

    barWidth = 0.2

    # Position des barres pour les différents scores
    r1 = np.arange(len(classifiers_names))

    # Création du graphique avec les barres côte à côte pour chaque métrique
    plt.figure(figsize=(12, 6))

    # créer une ligne à 0.7 et 0.9 pour montrer les limites acceptables et remplir l'intervalle de vert clair
    # plt.axhline(y=0.7, color='g', linestyle='--')
    # plt.axhline(y=0.9, color='g', linestyle='--')
    # plt.fill_between(r1, 0.7, 0.9, color='g', alpha=0.1)

    plt.bar(r1, overfitting_scores, color='b', width=barWidth, edgecolor='grey', label='Overfitting')

    # Ajout des étiquettes au-dessus des barres
    for i, value in enumerate(r1):
        plt.text(value, overfitting_scores[i]+0.02, str(round(overfitting_scores[i], 2)), ha='center', va='bottom')

    # Configuration du graphique
    plt.xlabel('Classifieurs')
    plt.ylabel('Scores')
    plt.title('Scores d Overfitting des Classifieurs')
    
    plt.xticks([r for r in range(len(classifiers_names))], classifiers_names)

    

    plt.legend()
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    colonnes = ['growth', 'densite', 'relative_opacity', 'opacity', 'surface', 'diametre', 'min_diametre', 'max_diametre', 'elongation', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left', 'nombre_raies']

    # meilleur_sous_ensemble_tout_classifieur(colonnes)
    liste_para(colonnes)
    # main(colonnes)
    # meilleures_donnees(colonnes, [])
    # boxplot_all_columns(colonnes)
    # meilleure_combinaison()
    # test_GBM_learning_rate()
    # afficher_test_GBM_3D()
    # afficher_test_GBM("GBM_n_estimators_variations")