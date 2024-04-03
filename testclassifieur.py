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

total = 12174

class_weights = {
    -1: 21/total,
    0: 174/total,
    1: 1222/total,
    2: 2196/total,
    3: 2822/total,
    4: 2451/total,
    5: 1537/total,
    6: 879/total,
    7: 470/total,
    8: 226/total,
    9: 104/total,
    10: 39/total,
    11: 20/total,
    12: 7/total,
    13: 6/total
}

def regression_logistique(tolerence, X_train, X_test, Y_train, Y_test, version):
    if version:
        model = LogisticRegression(max_iter=10000, solver='lbfgs', multi_class='ovr', class_weight=class_weights)
    else:
        model = LogisticRegression()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy Test': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Accuracy Train': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def support_vector_machines(tolerence, X_train, X_test, Y_train, Y_test, version):
    if version:
        model = SVC(C=100, class_weight=class_weights)
    else:
        model = SVC()

    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy Test': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Accuracy Train': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def discriminant_analysis(tolerence, X_train, X_test, Y_train, Y_test, version):
    if version:
        total = sum([21, 174, 1222, 2196, 2822, 2451, 1537, 879, 470, 226, 104, 39, 20, 7, 6])
        priors = [x/total for x in [21, 174, 1222, 2196, 2822, 2451, 1537, 879, 470, 226, 104, 39, 20, 7, 6]]
        model = LinearDiscriminantAnalysis(priors=priors)
    else:
        model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy Test': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Accuracy Train': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def random_forests(tolerence, X_train, X_test, Y_train, Y_test, version):
    if version:
        model = RandomForestClassifier(random_state=42, n_estimators=20, max_depth=13, min_samples_split=6, min_samples_leaf=6, max_features=10)
    else:
        model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy Test': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Accuracy Train': accuracy_score(Y_train, prediction_train)
    }
    return metrics

def gradient_boosting_machines(tolerence, X_train, X_test, Y_train, Y_test, version, n):
    if version:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.4, random_state=42, max_depth=5)
    else:
        model = GradientBoostingClassifier(learning_rate=n, n_estimators=5)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_train = model.predict(X_train)
    metrics = {
        'Accuracy Test': accuracy_score(Y_test, prediction),
        'Precision': precision_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Recall': recall_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'F1_score': f1_score(Y_test, prediction, zero_division=np.nan, average='weighted'),
        'Accuracy Train': accuracy_score(Y_train, prediction_train)
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
        resultats['Régression logistique'].append(regression_logistique(0, X_train, X_test, Y_train, Y_test))
        # resultats['Support Vector Machines'].append(support_vector_machines(0, X_train, X_test, Y_train, Y_test))
        # resultats['Discriminant Analysis'].append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test))
        # resultats['Random Forests'].append(random_forests(0, X_train, X_test, Y_train, Y_test))
        # resultats['Gradient Boosting Machines'].append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 100, 0.32))

    for key, value in resultats.items():
        # message = f"Meilleur colonne pour {key} : {colonnes[value.index(max(value))]}"
        if value:  # check if the list is not empty
            resultats_colonnes[colonnes[value.index(max(value))]] = resultats_colonnes.get(colonnes[value.index(max(value))], 0) + 1
        # print(message)

    best_column = max(resultats_colonnes, key=resultats_colonnes.get)
    # print(f"Meilleure colonne pour {best_column} : {colonnes[resultats[best_column].index(max(resultats[best_column]))]}")
    test.append(best_column)
    colonnes.remove(best_column)
    if len(colonnes) != 0:
        meilleures_donnees(colonnes, test)
    
    return test
    # print(f"Meilleure colonne globale : {best_column}")


def meilleure_combinaison(colonnes):
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

        print("\033[94m" + f"Régression logistique\033[0m")
        Ylogistique.append(regression_logistique(0, X_train, X_test, Y_train, Y_test))
        print("\033[94m" + f"Support Vector Machines\033[0m")
        Ysvm.append(support_vector_machines(0, X_train, X_test, Y_train, Y_test))
        print("\033[94m" + f"Discriminant Analysis\033[0m")
        Ydiscriminant.append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test))
        print("\033[94m" + f"Random Forests\033[0m")
        Yrandom.append(random_forests(0, X_train, X_test, Y_train, Y_test))
        print("\033[94m" + f"Gradient Boosting Machines\033[0m")
        Ygradient.append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 100, 0.32))

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


def liste_para(colonnes):
    # suite = meilleurs_données(colonnes, [])
    # suite = ['RC_left', 'densite', 'elongation', 'min_diametre', 'surface', 'max_diametre', 'diametre']
    
    suite =  ['max_diametre', 'min_diametre', 'diametre', 'relative_opacity', 'densite', 'opacity', 'nombre_raies', 'elongation']
    
    # suite = ['RC_left', 'densite', 'elongation', 'surface', 'max_diametre', 'diametre', 'relative_opacity']
    # suite = ['max_diametre', 'densite', 'diametre', 'RC_top', 'nombre_raies', 'RC_right', 'surface', 'RC_bottom', 'relativeopacity', 'RC_left']
    # suite = ['nombre_raies', 'max_diametre', 'surface', 'diametre', 'relativeopacity', 'RC_top', 'RC_right', 'densite', 'RC_bottom', 'RC_left']
    print(suite)

    params = []
    Ylogistique = []
    Ysvm = []
    Ydiscriminant = []
    Yrandom = []
    Ygradient = []

    columns_to_normalize = df_modifie.columns.difference(['age', 'filepath'])
    # Créer un objet StandardScaler
    scaler = StandardScaler()

    # Normaliser les données
    df_modifie[columns_to_normalize] = scaler.fit_transform(df_modifie[columns_to_normalize])

    for _, param in enumerate(suite):
        params.append(param)
        print("\033[92m" + f"Paramètres : {params}" + "\033[0m")
        X = df_modifie[params]
        Y = df_modifie['age']
        # Y = df_modifie['seuil_age']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        Ylogistique.append(regression_logistique(0, X_train, X_test, Y_train, Y_test)['Accuracy Test'])
        # Ysvm.append(support_vector_machines(0, X_train, X_test, Y_train, Y_test)['Accuracy Test'])
        # Ydiscriminant.append(discriminant_analysis(0, X_train, X_test, Y_train, Y_test)['Accuracy Test'])
        # Yrandom.append(random_forests(0, X_train, X_test, Y_train, Y_test)['Accuracy Test'])
        # Ygradient.append(gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test)['Accuracy Test'])

        print("\033[94m" + f"Régression logistique : {Ylogistique}" + "\033[0m")
        # print("\033[94m" + f"Support Vector Machines : {Ysvm}" + "\033[0m")
        # print("\033[94m" + f"Discriminant Analysis : {Ydiscriminant}" + "\033[0m")
        # print("\033[94m" + f"Random Forests : {Yrandom}" + "\033[0m")
        # print("\033[94m" + f"Gradient Boosting Machines : {Ygradient}" + "\033[0m")

    models = ['Régression logistique', 'Support Vector Machines', 'Discriminant Analysis', 'Random Forests', 'Gradient Boosting Machines']
    
    x = np.arange(0, len(suite))

    # plt.plot(x, Ylogistique, 'o-', label=models[0])
    # plt.plot(x, Ysvm, 'o-', label=models[1])
    # plt.plot(x, Ydiscriminant, 'o-', label=models[2])
    # plt.plot(x, Yrandom, 'o-', label=models[3])
    plt.plot(x, Ygradient, 'o-', label=models[4])
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
    rfe = RFE(estimator=modele, n_features_to_select=n_features)
    
    # Adapter le RFE au jeu de données
    rfe.fit(X, Y)
    
    # Renvoyer les indices des caractéristiques sélectionnées
    return X.columns[rfe.support_].tolist()

def meilleur_sous_ensemble_tout_classifieur(colonnes):
    X = df_modifie[colonnes]  # Vos données d'entraînement
    Y = df_modifie['age']

    modele_LR = LogisticRegression(max_iter=10000)
    modele_SVC = SVC()
    modele_LDA = LinearDiscriminantAnalysis()
    modele_RFC = RandomForestClassifier()
    modele_GBM = GradientBoostingClassifier()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    meilleur_LR = [meilleur_sous_ensemble(modele_LR, X, Y, n_features=1), regression_logistique(0, Xtrain, Xtest, Ytrain, Ytest)]
    meilleur_SVC = [meilleur_sous_ensemble(modele_SVC, X, Y, n_features=1), support_vector_machines(0, Xtrain, Xtest, Ytrain, Ytest)]
    meilleur_LDA = [meilleur_sous_ensemble(modele_LDA, X, Y, n_features=1), discriminant_analysis(0, Xtrain, Xtest, Ytrain, Ytest)]
    meilleur_RFC = [meilleur_sous_ensemble(modele_RFC, X, Y, n_features=1), random_forests(0, Xtrain, Xtest, Ytrain, Ytest)]
    meilleur_GBM = [meilleur_sous_ensemble(modele_GBM, X, Y, n_features=1), gradient_boosting_machines(0, Xtrain, Xtest, Ytrain, Ytest, 100, 0.32)]

    for i in range(2, len(colonnes) + 1):

        print(f"Nombre de caractéristiques : {i}")

        # Appeler la fonction pour obtenir le meilleur sous-ensemble
        caracteristiques_selectionnees_LR = meilleur_sous_ensemble(modele_LR, X, Y, n_features=i)
        caracteristiques_selectionnees_SVC = meilleur_sous_ensemble(modele_SVC, X, Y, n_features=5)
        caracteristiques_selectionnees_LDA = meilleur_sous_ensemble(modele_LDA, X, Y, n_features=i)
        caracteristiques_selectionnees_RFC = meilleur_sous_ensemble(modele_RFC, X, Y, n_features=i)
        caracteristiques_selectionnees_GBM = meilleur_sous_ensemble(modele_GBM, X, Y, n_features=i)

        X = df_modifie[caracteristiques_selectionnees_LR]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
        scores_logistique = regression_logistique(0, Xtrain, Xtest, Ytrain, Ytest)

        X = df_modifie[caracteristiques_selectionnees_SVC]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
        scores_SVC = support_vector_machines(0, Xtrain, Xtest, Ytrain, Ytest)

        X = df_modifie[caracteristiques_selectionnees_LDA]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
        scores_LDA = discriminant_analysis(0, Xtrain, Xtest, Ytrain, Ytest)

        X = df_modifie[caracteristiques_selectionnees_RFC]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
        scores_RFC = random_forests(0, Xtrain, Xtest, Ytrain, Ytest)

        X = df_modifie[caracteristiques_selectionnees_GBM]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
        scores_GBM = gradient_boosting_machines(0, Xtrain, Xtest, Ytrain, Ytest, 100, 0.32)

        if scores_logistique > meilleur_LR[1]:
            meilleur_LR = [caracteristiques_selectionnees_LR, scores_logistique]
        if scores_SVC > meilleur_SVC[1]:
            meilleur_SVC = [caracteristiques_selectionnees_SVC, scores_SVC]
        if scores_LDA > meilleur_LDA[1]:
            meilleur_LDA = [caracteristiques_selectionnees_LDA, scores_LDA]
        if scores_RFC > meilleur_RFC[1]:
            meilleur_RFC = [caracteristiques_selectionnees_RFC, scores_RFC]
        if scores_GBM > meilleur_GBM[1]:
            meilleur_GBM = [caracteristiques_selectionnees_GBM, scores_GBM]

    print("Les meilleures caractéristiques LR sélectionnées :")
    print(meilleur_LR)
    print("Les meilleures caractéristiques SVC sélectionnées :")
    print(caracteristiques_selectionnees_SVC)
    print("Les meilleures caractéristiques LDA sélectionnées :")
    print(meilleur_LDA)
    print("Les meilleures caractéristiques RFC sélectionnées :")
    print(meilleur_RFC)
    print("Les meilleures caractéristiques GBM sélectionnées :")
    print(meilleur_GBM)

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
        'Regression Logistique': regression_logistique(0, X_train, X_test, Y_train, Y_test, 1),
        'Support Vector Machines': support_vector_machines(0, X_train, X_test, Y_train, Y_test, 1),
        'Discriminant Analysis': discriminant_analysis(0, X_train, X_test, Y_train, Y_test, 1),
        'Random Forests': random_forests(0, X_train, X_test, Y_train, Y_test, 1),
        'Gradient Boosting Machines': gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 1, 0)
    }

    print(classifiers)

    # Affichage des scores de précision et de rappel
    classifiers_names = list(classifiers.keys())
    accuracy_scores = [classifiers[classifier]['Accuracy Test'] for classifier in classifiers_names]
    precision_scores = [classifiers[classifier]['Precision'] for classifier in classifiers_names]
    recall_scores = [classifiers[classifier]['Recall'] for classifier in classifiers_names]
    f1_scores = [classifiers[classifier]['F1_score'] for classifier in classifiers_names]
    overfitting_scores = [classifiers[classifier]['Accuracy Train'] for classifier in classifiers_names]

    barWidth = 0.15

    # Position des barres pour les différents scores
    r1 = np.arange(len(classifiers_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + 2*barWidth for x in r1]
    r4 = [x + 3*barWidth for x in r1]
    r5 = [x + 4*barWidth for x in r1]

    # Création du graphique avec les barres côte à côte pour chaque métrique
    plt.figure(figsize=(12, 6))
    plt.bar(r1, accuracy_scores, color='y', width=barWidth, edgecolor='grey', label='Accuracy Test')
    plt.bar(r2, precision_scores, color='b', width=barWidth, edgecolor='grey', label='Precision')
    plt.bar(r3, recall_scores, color='g', width=barWidth, edgecolor='grey', label='Recall')
    plt.bar(r4, f1_scores, color='r', width=barWidth, edgecolor='grey', label='F1 Score')
    plt.bar(r5, overfitting_scores, color='c', width=barWidth, edgecolor='grey', label='Accuracy Train')

    # Ajout des étiquettes au-dessus des barres
    for i, value in enumerate(r1):
        plt.text(value, accuracy_scores[i]+0.02, str(round(accuracy_scores[i], 2)), ha='center', va='bottom')
    
    for i, value in enumerate(r2):
        plt.text(value, precision_scores[i]+0.02, str(round(precision_scores[i], 2)), ha='center', va='bottom')
        
    for i, value in enumerate(r3):
        plt.text(value, recall_scores[i]+0.02, str(round(recall_scores[i], 2)), ha='center', va='bottom')
        
    for i, value in enumerate(r4):
        plt.text(value, f1_scores[i]+0.02, str(round(f1_scores[i], 2)), ha='center', va='bottom')

    for i, value in enumerate(r5):
        plt.text(value, overfitting_scores[i]+0.02, str(round(overfitting_scores[i], 2)), ha='center', va='bottom')

    # Configuration du graphique
    plt.xlabel('Classifieurs')
    plt.ylabel('Scores')
    plt.title('Scores d Accuracy, Precision, Recall et F1 des Classifieurs')
    plt.xticks([r + 1.5*barWidth for r in range(len(classifiers_names))], classifiers_names)
    plt.legend()

    # plt.ylim(0, 1)

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
        'Regression Logistique': regression_logistique(0, X_train, X_test, Y_train, Y_test, 0),
        'Support Vector Machines': support_vector_machines(0, X_train, X_test, Y_train, Y_test, 0),
        'Discriminant Analysis': discriminant_analysis(0, X_train, X_test, Y_train, Y_test, 0),
        'Random Forests': random_forests(0, X_train, X_test, Y_train, Y_test, 0),
        'Gradient Boosting Machines': gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 0)
    }

    # tracer l'overfitting des modèles sur un graphique en barres
    classifiers_names = list(classifiers.keys())
    overfitting_scores = [classifiers[classifier]['Accuracy Train'] for classifier in classifiers_names]

    barWidth = 0.2

    # Position des barres pour les différents scores
    r1 = np.arange(len(classifiers_names))

    # Création du graphique avec les barres côte à côte pour chaque métrique
    plt.figure(figsize=(12, 6))

    # créer une ligne à 0.7 et 0.9 pour montrer les limites acceptables et remplir l'intervalle de vert clair
    plt.axhline(y=0.7, color='g', linestyle='--')
    plt.axhline(y=0.9, color='g', linestyle='--')
    plt.fill_between(r1, 0.7, 0.9, color='g', alpha=0.1)

    plt.bar(r1, overfitting_scores, color='b', width=barWidth, edgecolor='grey', label='Accuracy Train')

    # Ajout des étiquettes au-dessus des barres
    for i, value in enumerate(r1):
        plt.text(value, overfitting_scores[i]+0.02, str(round(overfitting_scores[i], 2)), ha='center', va='bottom')

    # Configuration du graphique
    plt.xlabel('Classifieurs')
    plt.ylabel('Scores')
    plt.title('Scores d Accuracy Train des Classifieurs')
    plt.xticks([r for r in range(len(classifiers_names))], classifiers_names)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

def regression_lineaire_var_C(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # C values 0.5 à 2.5:
    C_values = [0.001, 0.01, 0.1, 1, 10]#, 20, 30, 40, 50, 100, 150, 200, 250, 300]

    # Créer un objet LogisticRegression
    modele = LogisticRegression(max_iter=10000)

    # Créer une liste pour stocker les scores
    scores = []

    # Boucle sur les valeurs de C
    for C in C_values:
        print(f"Valeur de C : {C}")
        modele.C = C
        modele.fit(X_train, Y_train)
        scores.append(modele.score(X_test, Y_test))

    # Afficher les scores
    plt.plot(C_values, scores)
    plt.xlabel('Valeur de C')
    plt.ylabel('Score')
    plt.title('Score du SVM en fonction de C')
    plt.show()

def regression_lineaire_var_tol(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # valeur de tolérance de 0.00001 à 0.0001 avec x10:
    tol_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Créer un objet LogisticRegression
    modele = SVC()

    # Créer une liste pour stocker les scores
    scores = []

    # Boucle sur les valeurs de tolérance
    for tol in tol_values:
        print(f"Valeur de tolérance : {tol}")
        modele.tol = tol
        modele.fit(X_train, Y_train)
        scores.append(modele.score(X_test, Y_test))

    # Afficher les scores
    plt.plot(tol_values, scores)
    plt.xlabel('Valeur de tolérance')
    plt.ylabel('Score')
    plt.title('Score du SVM en fonction de la tolérance')
    plt.show()

def svm_var_kernel(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    kernel = ['linear', 'poly', 'rbf', 'sigmoid']

    # Créer un objet SVC
    modele = SVC()

    # Créer une liste pour stocker les scores
    scores = []

    # Boucle sur les valeurs de kernel
    for k in kernel:
        print(f"Valeur de kernel : {k}")
        modele.kernel = k
        modele.fit(X_train, Y_train)
        scores.append(modele.score(X_test, Y_test))

    # Afficher les scores en diagramme en barres
    plt.bar(kernel, scores)
    plt.xlabel('Valeur de kernel')
    plt.ylabel('Score')
    plt.title('Score du SVM en fonction de kernel')
    plt.show()

def DA_var_gamma(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    gamma = ['auto', 'scale']

    # Créer un objet Discriminant Analysis
    modele = LinearDiscriminantAnalysis()

    # Créer une liste pour stocker les scores
    scores = []

    # Boucle sur les valeurs de gamma
    for g in gamma:
        print(f"Valeur de gamma : {g}")
        modele.gamma = g
        modele.fit(X_train, Y_train)
        scores.append(modele.score(X_test, Y_test))

    # Afficher les scores en diagramme en barres
    plt.bar(gamma, scores)
    plt.xlabel('Valeur de gamma')
    plt.ylabel('Score')
    plt.title('Score du Discriminant Analysis en fonction de gamma')
    plt.show()

def DA_var_components(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Composants de 0 à 14:
    components = [1, 10, 50, 100, 500, 1000, 1500, 3000, 5000, 10000]

    # Créer un objet Discriminant Analysis
    modele = LinearDiscriminantAnalysis()

    # Créer une liste pour stocker les scores
    scores = []

    # Boucle sur les valeurs de composants
    for comp in components:
        print(f"Nombre de composants : {comp}")
        modele.n_components = comp
        modele.fit(X_train, Y_train)
        scores.append(modele.score(X_test, Y_test))

    # Afficher les scores
    plt.plot(components, scores)
    plt.xlabel('Nombre de composants')
    plt.ylabel('Score')
    plt.title('Score du Discriminant Analysis en fonction du nombre de composants')
    plt.show()

def RF_var_estimators(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Nombre d'estimateurs de 0 à 14:
    n_estimators = [i*10 for i in range(1, 50)]

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for n in n_estimators:
        print(f"Nombre d'estimateurs : {n}")
        resultat = random_forests(0, X_train, X_test, Y_train, Y_test, 0, n)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])


    # Afficher les scores
    plt.plot(n_estimators, scores1, label='Test')
    plt.plot(n_estimators, scores2, label='Train')
    plt.xlabel('Nombre d\'estimateurs')
    plt.ylabel('Score')
    plt.title('Score du Random Forest en fonction du nombre d\'estimateurs')
    plt.legend()
    plt.show()

def RF_var_depth(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Profondeur de 1 à 14:
    max_depth = [i for i in range(1, 15)]

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for depth in max_depth:
        print(f"Profondeur : {depth}")
        resultat = random_forests(0, X_train, X_test, Y_train, Y_test, 0, depth)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(max_depth, scores1, label='Test')
    plt.plot(max_depth, scores2, label='Train')
    plt.xlabel('Profondeur')
    plt.ylabel('Accuracy Train')
    plt.title('Overfitting du Random Forest en fonction de la profondeur')
    plt.legend()
    plt.xticks(np.arange(1, 15, 1))
    plt.show()

def RF_var_min_samples_split(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    # nombre d'éléments dans chaque classe Y
    repartition_element = Y.value_counts()
    trier = repartition_element.sort_values()
    repartition = trier.values
    # enlever toutes les valeurs supérieur à la moyenne des valeurs dans répartition
    repartition = [i for i in repartition if i < repartition.mean()]
    print(repartition)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Profondeur de 1 à 14:
    samples = [i for i in range(3, 50)]
    # samples = repartition

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for sample in samples:
        print(f"Nombre minimal d'échantillons pour split : {sample}")
        resultat = random_forests(0, X_train, X_test, Y_train, Y_test, 0, sample)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # trouver le meilleur score1 et donner la valeur correspondante
    max_score1 = max(scores1)
    max_score1_index = scores1.index(max_score1)
    print(f"Meilleur score1 : {max_score1} pour {samples[max_score1_index]}")

    # Afficher les scores
    plt.plot(samples, scores1, label='Test')
    plt.plot(samples, scores2, label='Train')
    plt.xlabel('Nombre minimal d\'échantillons pour split')
    plt.ylabel('Score')
    plt.title('Score du Random Forest en fonction du nombre minimal d\'échantillons pour split')
    plt.legend()
    plt.show()


def RF_var_min_samples_leaf(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Profondeur de 1 à 14:
    samples = [i for i in range(1, 14)]

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for sample in samples:
        print(f"Nombre minimal d'échantillons pour leaf : {sample}")
        resultat = random_forests(0, X_train, X_test, Y_train, Y_test, 0, sample)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(samples, scores1, label='Test')
    plt.plot(samples, scores2, label='Train')
    plt.xlabel('Nombre minimal d\'échantillons pour leaf')
    plt.ylabel('Score')
    plt.title('Score du Random Forest en fonction du nombre minimal d\'échantillons pour leaf')
    plt.legend()
    plt.show()

def RF_var_max_features(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # caractéristiques de 1 à 14:
    features = ['log2', 'auto', 'sqr', 'None']

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for feature in features:
        print(f"Nombre maximal de caractéristiques : {feature}")
        resultat = random_forests(0, X_train, X_test, Y_train, Y_test, 0, 'log2')
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(features, scores1, label='Test')
    plt.plot(features, scores2, label='Train')
    plt.xlabel('Nombre maximal de caractéristiques')
    plt.ylabel('Score')
    plt.title('Score du Random Forest en fonction du nombre maximal de caractéristiques')
    plt.legend()
    plt.show()

def RF_var_criterion(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    criterions = ['gini', 'entropy', 'log_loss']

    # Créer une liste pour stocker les scores
    scores = []

    # Boucle sur les valeurs de composants
    for criterion in criterions:
        print(f"Criterion : {criterion}")
        scores.append(random_forests(0, X_train, X_test, Y_train, Y_test, criterion)['Accuracy Test'])

    # Afficher les scores en diagramme en barres
    plt.bar(criterions, scores)
    plt.xlabel('Criterion')
    plt.ylabel('Score')
    plt.title('Score du Random Forest en fonction du criterion')
    plt.show()

def GBM_var_max_depth(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Profondeur de 1 à 14:
    max_depth = [i for i in range(1, 3)]

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for depth in max_depth:
        print(f"Profondeur : {depth}")
        resultat = gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 0, depth)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(max_depth, scores1, label='Test')
    plt.plot(max_depth, scores2, label='Train')
    plt.xlabel('Profondeur')
    plt.ylabel('Score')
    plt.title('Score du Gradient Boosting Machine en fonction de la profondeur')
    plt.legend()
    plt.show()

def GBM_var_subsample(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Profondeur de 1 à 14:
    subsample = [i/10 for i in range(1, 14)]

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for sample in subsample:
        print(f"Subsample : {sample}")
        resultat = gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 0, sample)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(subsample, scores1, label='Test')
    plt.plot(subsample, scores2, label='Train')
    plt.xlabel('Subsample')
    plt.ylabel('Score')
    plt.title('Score du Gradient Boosting Machine en fonction du subsample')
    plt.legend()
    plt.show()

def GBM_var_max_features(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # caractéristiques de 1 à 14:
    features = ['log2', 'sqr', 'None']

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for feature in features:
        print(f"Nombre maximal de caractéristiques : {feature}")
        resultat = gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 0, feature)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(features, scores1, label='Test')
    plt.plot(features, scores2, label='Train')
    plt.xlabel('Nombre maximal de caractéristiques')
    plt.ylabel('Score')
    plt.title('Score du Gradient Boosting Machine en fonction du nombre maximal de caractéristiques')
    plt.legend()
    plt.show()

def GBM_var_random_state(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test = split_train_test_by_age(X, Y, test_size=0.2, random_state=42)

    # random_state de 1 à 14:
    random_states = [i for i in range(1, 100)]

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for state in random_states:
        print(f"Random State : {state}")
        resultat = gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 0, state)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(random_states, scores1, label='Test')
    plt.plot(random_states, scores2, label='Train')
    plt.xlabel('Random State')
    plt.ylabel('Score')
    plt.title('Score du Gradient Boosting Machine en fonction du random state')
    plt.legend()
    plt.show()

def GBM_var_learning_rate(colonnes):
    X = df_modifie[colonnes]
    Y = df_modifie['age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # learning_rate de 0.01 à 0.1:
    learning_rates = [i/100 for i in range(1, 100)]

    # Créer une liste pour stocker les scores
    scores1 = []
    scores2 = []

    # Boucle sur les valeurs de composants
    for rate in learning_rates:
        print(f"Learning Rate : {rate}")
        resultat = gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, 0, rate)
        scores1.append(resultat['Accuracy Test'])
        scores2.append(resultat['Accuracy Train'])

    # Afficher les scores
    plt.plot(learning_rates, scores1, label='Test')
    plt.plot(learning_rates, scores2, label='Train')
    plt.xlabel('Learning Rate')
    plt.ylabel('Score')
    plt.title('Score du Gradient Boosting Machine en fonction du learning rate')
    plt.legend()
    plt.show()

def recup_donnees_class(X_train, X_test, Y_train, Y_test, version):
    classifiers = {
        'Regression Logistique': regression_logistique(0, X_train, X_test, Y_train, Y_test, version),
        'Support Vector Machines': support_vector_machines(0, X_train, X_test, Y_train, Y_test, version),
        'Discriminant Analysis': discriminant_analysis(0, X_train, X_test, Y_train, Y_test, version),
        'Random Forests': random_forests(0, X_train, X_test, Y_train, Y_test, version),
        'Gradient Boosting Machines': gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test, version)
    }

    return classifiers

def comparaison_avant_apres(colonnes):

    # columns_to_normalize = df_modifie.columns.difference(['age', 'filepath'])

    # # Créer un objet StandardScaler
    # scaler = StandardScaler()

    # # Normaliser les données
    # df_modifie[columns_to_normalize] = scaler.fit_transform(df_modifie[columns_to_normalize])

    X = df_modifie[colonnes]
    Y = df_modifie['age']
    # Y = df_modifie['seuil_age']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    resultats_avant = recup_donnees_class(X_train, X_test, Y_train, Y_test, 0)

    # normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    resultats_apres = recup_donnees_class(X_train, X_test, Y_train, Y_test, 1)


    # Affichage des scores de précision et de rappel
    classifiers_names = list(resultats_avant.keys())
    accuracy_scores_avant = [resultats_avant[classifier]['Accuracy Test'] for classifier in classifiers_names]
    precision_scores_avant = [resultats_avant[classifier]['Precision'] for classifier in classifiers_names]
    recall_scores_avant = [resultats_avant[classifier]['Recall'] for classifier in classifiers_names]
    f1_scores_avant = [resultats_avant[classifier]['F1_score'] for classifier in classifiers_names]
    overfitting_scores_avant = [resultats_avant[classifier]['Accuracy Train'] for classifier in classifiers_names]

    accuracy_scores_apres = [resultats_apres[classifier]['Accuracy Test'] for classifier in classifiers_names]
    precision_scores_apres = [resultats_apres[classifier]['Precision'] for classifier in classifiers_names]
    recall_scores_apres = [resultats_apres[classifier]['Recall'] for classifier in classifiers_names]
    f1_scores_apres = [resultats_apres[classifier]['F1_score'] for classifier in classifiers_names]
    overfitting_scores_apres = [resultats_apres[classifier]['Accuracy Train'] for classifier in classifiers_names]

    accuracy_scores = accuracy_scores_apres - accuracy_scores_avant
    precision_scores = precision_scores_apres - precision_scores_avant
    recall_scores = recall_scores_apres - recall_scores_avant
    f1_scores = f1_scores_apres - f1_scores_avant
    overfitting_scores = overfitting_scores_apres - overfitting_scores_avant

    barWidth = 0.15

    # Position des barres pour les différents scores
    r1 = np.arange(len(classifiers_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + 2*barWidth for x in r1]
    r4 = [x + 3*barWidth for x in r1]
    r5 = [x + 4*barWidth for x in r1]

    # Création du graphique avec les barres côte à côte pour chaque métrique
    plt.figure(figsize=(12, 6))
    plt.bar(r1, accuracy_scores, color='y', width=barWidth, edgecolor='grey', label='Accuracy Test')
    plt.bar(r2, precision_scores, color='b', width=barWidth, edgecolor='grey', label='Precision')
    plt.bar(r3, recall_scores, color='g', width=barWidth, edgecolor='grey', label='Recall')
    plt.bar(r4, f1_scores, color='r', width=barWidth, edgecolor='grey', label='F1 Score')
    plt.bar(r5, overfitting_scores, color='c', width=barWidth, edgecolor='grey', label='Accuracy Train')

    # Ajout des étiquettes au-dessus des barres
    for i, value in enumerate(r1):
        plt.text(value, accuracy_scores[i]+0.02, str(round(accuracy_scores[i], 2)), ha='center', va='bottom')
    
    for i, value in enumerate(r2):
        plt.text(value, precision_scores[i]+0.02, str(round(precision_scores[i], 2)), ha='center', va='bottom')
        
    for i, value in enumerate(r3):
        plt.text(value, recall_scores[i]+0.02, str(round(recall_scores[i], 2)), ha='center', va='bottom')
        
    for i, value in enumerate(r4):
        plt.text(value, f1_scores[i]+0.02, str(round(f1_scores[i], 2)), ha='center', va='bottom')

    for i, value in enumerate(r5):
        plt.text(value, overfitting_scores[i]+0.02, str(round(overfitting_scores[i], 2)), ha='center', va='bottom')

    # Configuration du graphique
    plt.xlabel('Classifieurs')
    plt.ylabel('Scores')
    plt.title('Scores d Accuracy, Precision, Recall et F1 des Classifieurs')
    plt.xticks([r + 1.5*barWidth for r in range(len(classifiers_names))], classifiers_names)
    plt.legend()

    # plt.ylim(0, 1)

    # Affichage du graphique
    plt.show()

if __name__ == "__main__":
    colonnes = ['growth', 'densite', 'relative_opacity', 'opacity', 'surface', 'diametre', 'min_diametre', 'max_diametre', 'elongation', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left', 'nombre_raies']

    # meilleur_sous_ensemble_tout_classifieur(colonnes)
    # liste_para(colonnes)
    # RF_var_max_features(colonnes)
    # RF_var_depth(colonnes)
    # GBM_var_learning_rate(colonnes)
    APRF1(colonnes)
    # regression_lineaire_var_tol(colonnes)
    # svm_var_kernel(colonnes)
    # main(colonnes)
    # meilleures_donnees(colonnes, [])
    # boxplot_all_columns(colonnes)
    # meilleure_combinaison(colonnes)
    # test_GBM_learning_rate()
    # afficher_test_GBM_3D()
    # afficher_test_GBM("GBM_n_estimators_variations")