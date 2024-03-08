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
    model = LogisticRegression(max_iter=1000)
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

def gradient_boosting_machines(tolerence, X_train, X_test, Y_train, Y_test):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    # print(f"Précision du modèle Gradient Boosting Machines : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

# utiliser tensorflow pour les modèles suivants
def cnn():
    # X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
    # X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1]))
    # X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1]))
    # model = Sequential()
    # model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # model.fit(X_train_reshaped, Y_train, epochs=10, batch_size=32)
    # prediction = model.predict(X_test_reshaped)
    # # accuracy = accuracy_score(Y_test, prediction)
    # r2 = r2_score(Y_test, prediction)
    # print(f"Précision du modèle CNN : {r2:.2%}")
    pass


def inception():
    # base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(height, width, 3))
    
    # # Ajouter vos propres couches supérieures
    # model = Sequential()
    # model.add(base_model)
    # model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1))
    
    # model.compile(optimizer='adam', loss='mean_squared_error')
    
    # # Entraîner le modèle
    # model.fit(X_train_reshaped, Y_train, epochs=10, batch_size=32)
    
    # # Faire des prédictions sur l'ensemble de test
    # prediction = model.predict(X_test_reshaped)
    
    # # Calculer R2 score
    # r2 = r2_score(Y_test, prediction)
    
    # print(f"R2 Score du modèle Inception : {r2:.2%}")
    pass


def resnet():
    pass
    # X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1]))
    # X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1]))
    # # Créer le modèle ResNet50 pré-entraîné sans les couches supérieures
    # base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
    
    # # Ajouter vos propres couches supérieures
    # model = Sequential()
    # model.add(base_model)
    # model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # model.fit(X_train_reshaped, Y_train, epochs=10, batch_size=32)
    # prediction = model.predict(X_test_reshaped)
    # # accuracy = accuracy_score(Y_test, prediction)
    # r2 = r2_score(Y_test, prediction)
    # print(f"Précision du modèle Resnet : {r2:.2%}")

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



def liste_para():
    colonnes = ['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left', 'age_estime']
    # suite = meilleurs_données(colonnes, [])
    # suite = ['max_diametre', 'densite', 'diametre', 'RC_top', 'age_estime', 'RC_right', 'surface', 'RC_bottom', 'relativeopacity', 'RC_left']
    suite = ['age_estime', 'max_diametre', 'surface', 'diametre', 'relativeopacity', 'RC_top', 'RC_right', 'densite', 'RC_bottom', 'RC_left']
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
    colonnes = ['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left', 'age_estime']
    X = df_modifie[colonnes]
    Y = df_modifie['age']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Précision du modèle Discriminant Analysis : {regression_logistique(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Support Vector Machines : {support_vector_machines(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Discriminant Analysis : {discriminant_analysis(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Random Forests : {random_forests(0, X_train, X_test, Y_train, Y_test)}")
    print(f"Précision du modèle Gradient Boosting Machines : {gradient_boosting_machines(0, X_train, X_test, Y_train, Y_test)}")


if __name__ == "__main__":
    liste_para()
    # main()
