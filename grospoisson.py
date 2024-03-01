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

X = df_modifie[['densite', 'relativeopacity', 'surface', 'diametre', 'max_diametre', 'RC_top', 'RC_bottom', 'RC_right', 'RC_left']]
Y = df_modifie['age']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# fonctions de modèles

def regression_logistique(tolerence):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    print(f"Précision du modèle Regressions Logistique : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def support_vector_machines(tolerence):
    model = SVC()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    print(f"Précision du modèle Support Vector Machines : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def discriminant_analysis(tolerence):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    print(f"Précision du modèle Discriminant Analysis : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def random_forests(tolerence):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    print(f"Précision du modèle Random Forests : {round(accuracy_with_tolerance, 2)}")
    return accuracy_with_tolerance

def gradient_boosting_machines(tolerence):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    accuracy_with_tolerance = sum(abs(Y_test - prediction) <= tolerence) / len(Y_test)
    print(f"Précision du modèle Gradient Boosting Machines : {round(accuracy_with_tolerance, 2)}")
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

def main():
    # tolerence = 0
    # regression_logistique(tolerence)
    # support_vector_machines(tolerence)
    # discriminant_analysis(tolerence)
    # random_forests(tolerence)
    # gradient_boosting_machines(tolerence)

    x = np.arange(0, 5)
    models = ['Régression logistique', 'Support Vector Machines', 'Discriminant Analysis', 'Random Forests', 'Gradient Boosting Machines']
    Ylogistique = [regression_logistique(tolerence) for tolerence in x]
    Ysvm = [support_vector_machines(tolerence) for tolerence in x]
    Ydiscriminant = [discriminant_analysis(tolerence) for tolerence in x]
    Yrandom = [random_forests(tolerence) for tolerence in x]
    Ygradient = [gradient_boosting_machines(tolerence) for tolerence in x]

    plt.plot(x, Ylogistique, 'o-', label=models[0])
    plt.plot(x, Ysvm, 'o-', label=models[1])
    plt.plot(x, Ydiscriminant, 'o-', label=models[2])
    plt.plot(x, Yrandom, 'o-', label=models[3])
    plt.plot(x, Ygradient, 'o-', label=models[4])
    plt.xlabel('Tolerence')
    plt.ylabel('Précision')
    plt.title('Précision des modèles en fonction de la tolerence')
    plt.legend()
    plt.show()


    # cnn()
    # inception()
    # print("1. Régression logistique")
    # print("2. Support Vector Machines")
    # print("3. Discriminant Analysis")
    # print("4. Random Forests")
    # print("5. Gradient Boosting Machines")
    # print("6. CNN")
    # print("7. Inception")
    # print("8. ResNet")
    # x = int(input("Entrez un nombre : "))
    # if x == 1:
    #     regression_logistique()
    # elif x == 2:
    #     support_vector_machines()
    # elif x == 3:
    #     discriminant_analysis()
    # elif x == 4:
    #     random_forests()
    # elif x == 5:
    #     gradient_boosting_machines()
    # elif x == 6:
    #     cnn()
    # elif x == 7:
    #     inception()
    # elif x == 8:
    #     resnet()
    # else:
    #     print("Nombre invalide.")


main()
