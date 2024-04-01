import sqlite3
import numpy as np
import pandas as pd
import os
import tifffile
import cv2
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math as m

db = sqlite3.connect('data.db')
plaice_dir_path = "F:\Téléchargements\96695\data_image_otolith_ple_mur\plaice"
df = pd.read_csv(plaice_dir_path + '\\metadata_plaice_2010-2019.csv', sep=';')

def get_echelle(name):
    if "ST_19_B68_O_C1" in name:
        return 125
    if "ST_19_B57_O" in name:
        return 111
    if "RE_18_PLEU-PLA_TRIM1_IBTS" in name:
        return 94
    if "RE_13_b1_C1_O" in name:
        return 94
    if "RE_12_b99_C1_O" in name:
        return 125
    if "RE_12_b98_C1_O" in name:
        return 94
    if "RE_12_b83_C1_O" in name:
        return 94
    if "RE_12_b75_C1_O" in name:
        return 94
    if "RE_12_b49_C1_O" in name:
        return 94
    if "RE_12_b48_C1_O" in name:
        return 94
    if "RE_12_b39_C1_O" in name:
        return 94
    if "RE_12_b38_C1_O" in name:
        return 94
    if "RE_12_b129_C1_O" in name:
        return 94
    if "RE_12_b118_C1_O" in name:
        return 94
    if "RE_12_b103_C1_O" in name:
        return 94
    if "MS_15_B95_C1_O" in name:
        return 112
    if "MS_15_B92_C1_O" in name:
        return 140
    if "MS_15_B91_C1_O" in name:
        return 140
    if "MS_15_B69_C1_O" in name:
        return 112
    if "MS_15_B52_C1_O" in name:
        return 140
    if "MS_15_B51_C1_O" in name:
        return 112
    if "KS_17_TRIM4_PL_CGFS" in name:
        return 111
    if "KS_17_TRIM3_PL_NOURSOMME" in name:
        return 140
    if "KS_17_TRIM4_7D_PL" in name:
        return 103
    if "KS_17_TRIM3_PL_COMOR" in name:
        return 103
    if "KS_17_TRIM1_IBTS_PL" in name:
        return 82
    if "KS_17_TRIM1_PL_LCN" in name:
        return 111
    if "KS_16_TRIM4_O_EVHOE_PL" in name:
        return 82
    if "KS_16_TRIM3_O_CGFS" in name:
        return 143
    if "KS_16_TRIM4_O_CGFS" in name:
        return 92
    if "KS_14_TRIM4_CGFS" in name:
        return 92
    if "KS_16_TRIM3_O_LCN" in name:
        return 82
    if "KS_16_TRIM2_XBL_O" in name:
        return 103
    if "KS_16_TRIM3_Noursomme_O" in name:
        return 112
    if "KS_14_TRIM3_LCN" in name:
        return 103
    if "KS_14_TRIM3_PENLY" in name:
        return 70
    if "KS_14_TRIM3_CAMANOC" in name:
        return 103
    if "KS_14_TRIM3_COMOR" in name:
        return 92
    if "KS_13_TRIM3_PENLY" in name:
        return 88
    if "KS_13_TRIM2_XBL" in name:
        return 70
    if "KS_10_TRIM3_XBL" in name:
        return 93
    if "KS_13_TRIM3_COMOR" in name:
        return 103
    if "KS_13_TRIM3_ESB" in name:
        return 93
    if "KS_13_TRIM3_LCN" in name:
        return 93
    if "KS_10_TRIM4_CGFS" in name:
        return 93
    if "KS_10_TRIM3_CGFS" in name:
        return 93
    if "KS_11_TRIM3_COMOR" in name:
        return 65
    if "KS_11_TRIM4_CGFS" in name:
        return 93
    if "GBD_19_B53_C1_O" in name:
        return 46
    if "GBD_19_B52_C1_O" in name:
        return 46
    if "GBD_19_B40_C1_O" in name:
        return 46
    if "GBD_19_B28_C1_O" in name:
        return 46
    if "GBD_19_B29_C1_O" in name:
        return 46
    if "GBD_19_B27_C1_O" in name:
        return 46
    if "GBD_19_B26_C1_O" in name:
        return 46
    if "GBD_19_B22_C1_O" in name:
        return 46
    if "GBD_19_B11_C1_O" in name:
        return 46
    if "GBD_18_B65_C1_O" in name:
        return 46
    if "GBD_18_B64_C1_O" in name:
        return 46
    if "GBD_18_B59_C1_O" in name:
        return 94
    if "GBD_18_B55_C1_O" in name:
        return 94
    if "GBD_18_B54_C1_O" in name:
        return 94
    if "GBD_18_B53_C1_O" in name:
        return 94
    if "GBD_18_B52_C1_O" in name:
        return 94
    if "GBD_18_B50_C1_O" in name:
        return 46
    if "GBD_18_B49_C1_O" in name:
        return 46
    if "GBD_18_B48_C1_O" in name:
        return 46
    if "GBD_18_B42_C1_O" in name:
        return 46
    if "CO_16_B76_C1_O" in name:
        return 114
    if "CO_16_B17_C1_O" in name:
        return 130
    if "CO_16_B75_C1_O" in name:
        return 92
    if "AD_19_B25_C1_O" in name:
        return 46
    if "AD_18_B81_C1_O" in name:
        return 46
    if "AD_18_B82_C1_O" in name:
        return 46
    if "AD_18_B76_C1_O" in name:
        return 46
    if "KS_10_TRIM3_COMOR" in name:
        return 84
    if "KS_10_TRIM3_LCN" in name:
        return 65
    if "KS_10_TRIM3_COLMATAGE" in name:
        return 84
    if "KS_13_TRIM3_MCE" in name:
        return 103
    if "KS_13_TRIM3_XBL" in name:
        return 70
    if "KS_13_TRIM4_CGFS" in name:
        return 84
    if "KS_13_TRIM4_LCN" in name:
        return 84
    if "KS_13_TRIM4_XFC" in name:
        return 84
    if "KS_16_TRIM1_LCN_O" in name:
        return 84
    if "KS_16_TRIM3_XLH_O" in name:
        return 103
    if "KS_16_TRIM4_O_XBL" in name:
        return 82
    if "KS_16_TRIM4_O_LCN_PL" in name:
        return 62
    if "KS_17_TRIM2_PL_LCN" in name:
        return 111
    if "KS_17_TRIM2_PL_XBL" in name:
        return 89
    if "KS_17_TRIM3_PL_XBL" in name:
        return 111
    if "ST_19_B87_C1_O" in name:
        return 115
    return 0
    

def get_relative_densite(image):
    if(image.shape[0] == 3):
        image = image.transpose(1, 2, 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_image != 0)

def get_densite(image):
    return np.mean(image)

d_ref = 69
def get_opacite_relative(density):
    return (density - d_ref) / d_ref


# ne fonctionne pas totalement
def get_surface(image: np.ndarray, name: str):
    if(image.shape[0] == 3):
        image = image.transpose(1, 2, 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    surface = np.sum(binary_image == 255)
    return surface / get_echelle(name)**2

def get_diametre(image, name: str):
    surface = get_surface(image, name)
    return 2 * np.sqrt(surface / np.pi)

def get_diametre_reel(image: np.ndarray, name: str):
    if(image.shape[0] == 3):
        image = image.transpose(1, 2, 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(max_contour)

    max_diametre = max(ellipse[1])
    min_diametre = min(ellipse[1])

    return max_diametre / get_echelle(name), min_diametre / get_echelle(name)

def remplissage():
    cursor = db.cursor()
    images = df[['Age', 'Reference_PC']]
    i = 0
    for index, row in images.iterrows():
        i += 1
        nb_barres = '#' * int(i / (len(images)) * 10) + ' ' * int(10 - i / (len(images)) * 10)
        print(nb_barres, str(i / (len(images)) * 100) + '%', end='\r')
        age = row['Age']
        image_path = f"{plaice_dir_path}/right/{row['Reference_PC']}.tif" if os.path.exists(f"{plaice_dir_path}/right/{row['Reference_PC']}.tif") else f"{plaice_dir_path}/left_and_right/{row['Reference_PC']}.tif"
        try:
            image = tifffile.imread(image_path)
        except: # l'image n'existe pas
            continue
        densite = get_densite(image)
        opacite = get_opacite_relative(densite)
        surface = get_surface(image_path)
        diametre = get_diametre(image)
        max_diametre = get_diametre_reel(image, image_path)
        surface = surface / 2 if '/left_and_right/' in image_path else surface
        cursor.execute('UPDATE donnees SET age = ?, densite = ?, relativeopacity = ?, surface = ?, diametre = ?, max_diametre = ? WHERE filepath = ?', (age, densite, opacite, surface, diametre, max_diametre, row['Reference_PC']))
    db.commit()

def get_donnees():
    cursor = db.cursor()
    cursor.execute('SELECT *, age > 7 as seuil_age FROM donnees')
    data = cursor.fetchall()
    index_names = [description[0] for description in cursor.description]
    cursor.close()
    return data, index_names

def destroy_data():
    cursor = db.cursor()
    cursor.execute('DELETE FROM donnees')
    db.commit()
    cursor.close()

def categorize_points(x_points, y_points):
    # Combiner les points x et y en un seul tableau pour faciliter le tri
    combined_points = np.array(list(zip(x_points, y_points)))
    
    # Trier les points en fonction des valeurs y (colonne 1)
    sorted_by_y = combined_points[combined_points[:, 1].argsort()]
    
    # Calculer les indices pour les seuils des 1/3 et 2/3 des points en fonction de y
    one_third_index = int(len(sorted_by_y) * (1/3))
    two_thirds_index = int(len(sorted_by_y) * (2/3))
    
    # Séparer les points en haut, milieu, et bas de l'image
    bottom_points = sorted_by_y[:one_third_index]
    middle_points = sorted_by_y[one_third_index:two_thirds_index]
    top_points = sorted_by_y[two_thirds_index:]
    
    # Trier les points du milieu en fonction des valeurs x (colonne 0)
    sorted_middle_by_x = middle_points[middle_points[:, 0].argsort()]
    
    # Calculer l'indice pour le seuil de la moitié des points du milieu
    half_middle_index = int(len(sorted_middle_by_x) / 2)
    
    # Séparer les points du milieu en gauche et droite
    middle_left_points = sorted_middle_by_x[:half_middle_index]
    middle_right_points = sorted_middle_by_x[half_middle_index:]
    
    # Retourner les quatre catégories
    return top_points, bottom_points, middle_left_points, middle_right_points

def calc_R(a_b, x, y):
    return (x - a_b[0])**2 + (y - a_b[1])**2

# Fonction de coût à minimiser
def cost_function(a_b, x, y):
    Ri = calc_R(a_b, x, y)
    return np.sum((Ri - np.mean(Ri))**2)

def plot_circle(ax, center, radius, color, linestyle):
    circle = plt.Circle(center, radius, color=color, fill=False, linestyle=linestyle)
    ax.add_artist(circle)

def centrer_image(image: np.ndarray):
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
        
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Supposons que le plus grand contour est notre région d'intérêt
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Calculer la boîte englobante du plus grand contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extraire la région d'intérêt
        regions_interet = gray_image[y:y+h, x:x+w]
        
        # Créer une nouvelle image de fond avec les mêmes dimensions que l'image originale
        new_image = np.zeros_like(gray_image)
        
        # Calculer les coordonnées pour centrer la ROI dans la nouvelle image
        start_y = (new_image.shape[0] - regions_interet.shape[0]) // 2
        start_x = (new_image.shape[1] - regions_interet.shape[1]) // 2
        
        # Placer la ROI au centre de la nouvelle image
        new_image[start_y:start_y+regions_interet.shape[0], start_x:start_x+regions_interet.shape[1]] = regions_interet
        
        return new_image

images_fails = []

def get_ring_curvature(image: np.ndarray, title):

    # if(len(image.shape) == 2):
    #     gray_image = image
    # else:
    gray_image = centrer_image(image)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # afficher gray_image avec plt et les contours
    # fig, ax = plt.subplots()
    # ax.imshow(gray_image, cmap='gray')
    
    # max_contour = max(contours, key=cv2.contourArea)
    # ax.plot(max_contour[:, 0, 0], max_contour[:, 0, 1], 'r')
    # plt.title(title)
    # plt.show()

    X = []
    Y = []
    # Parcourir tous les contours trouvés
    for contour in contours:
        # Parcourir chaque point du contour
        for point in contour:
            # Les points sont stockés sous la forme [[[x y]]], donc on extrait les coordonnées comme suit
            x, y = point[0]
            X.append(x)
            Y.append(y)
    
    # Assurez-vous que les points sont triés par x croissant
    sorted_indices = np.argsort(X)
    x_points = np.array(X)[sorted_indices]
    y_points = np.array(Y)[sorted_indices]

    part = 10
    m1 = ((image.shape[1] - image.shape[1]/part) - image.shape[1]/part) / image.shape[2]
    m2 = (image.shape[1]/part - (image.shape[1] - image.shape[1]/part)) / image.shape[2]

    x_points_np = np.array(x_points)
    y_points_np = np.array(y_points)

    f1 = x_points_np * m1 + image.shape[1] / part
    f2 = x_points_np * m2 + image.shape[1] - image.shape[1] / part

    # fig, ax = plt.subplots()
    # ax.imshow(gray_image, cmap='gray')
    
    # max_contour = max(contours, key=cv2.contourArea)
    # ax.plot(max_contour[:, 0, 0], max_contour[:, 0, 1], 'r')
    # # afficher f1 et f2
    # ax.plot(x_points_np, f1, 'm--', label='f1')
    # ax.plot(x_points_np, f2, 'c--', label='f2')
    # # afficher la legende pour f1, f2 et contour
    # plt.legend()
    # plt.title(title)
    # plt.show()

    bottom_mask = (f1 < y_points_np) & (f2 < y_points_np)
    top_mask = (f1 > y_points_np) & (f2 > y_points_np)
    middle_right_mask = (f1 > y_points_np) & (f2 < y_points_np)
    middle_left_mask = (f1 < y_points_np) & (f2 > y_points_np)

    print(bottom_mask, top_mask, middle_right_mask, middle_left_mask)

    bottom = np.column_stack((x_points_np[bottom_mask], y_points_np[bottom_mask]))
    top = np.column_stack((x_points_np[top_mask], y_points_np[top_mask]))
    middle_right = np.column_stack((x_points_np[middle_right_mask], y_points_np[middle_right_mask]))
    middle_left = np.column_stack((x_points_np[middle_left_mask], y_points_np[middle_left_mask]))

    top = np.array(top)
    bottom = np.array(bottom)
    middle_left = np.array(middle_left)
    middle_right = np.array(middle_right)

    if(top.shape[0] == 0 or bottom.shape[0] == 0 or middle_left.shape[0] == 0 or middle_right.shape[0] == 0):
        images_fails.append(title)
        # cv2.imwrite('gray_image.png', gray_image)
        return 0, 0, 0, 0

    # Initial guess for a and b (centre du cercle)
    # print(top.shape, bottom.shape, middle_left.shape, middle_right.shape)

    cx_top_cy_top_guess = [np.mean(top[:, 0]), np.mean(top[:, 1])]
    cx_bottom_cy_bottom_guess = [np.mean(bottom[:, 0]), np.mean(bottom[:, 1])]
    cx_middle_left_cy_middle_left_guess = [np.mean(middle_left[:, 0]), np.mean(middle_left[:, 1])]
    cx_middle_right_cy_middle_right_guess = [np.mean(middle_right[:, 0]), np.mean(middle_right[:, 1])]

    # Minimisation de la fonction de coût
    result_top = minimize(cost_function, cx_top_cy_top_guess, args=(top[:, 0],top[:, 1]))
    result_bottom = minimize(cost_function, cx_bottom_cy_bottom_guess, args=(bottom[:, 0],bottom[:, 1]))
    result_middle_left = minimize(cost_function, cx_middle_left_cy_middle_left_guess, args=(middle_left[:, 0],middle_left[:, 1]))
    result_middle_right = minimize(cost_function, cx_middle_right_cy_middle_right_guess, args=(middle_right[:, 0],middle_right[:, 1]))

    cx_top, cy_top = result_top.x  # Centre du cercle
    cx_bottom, cy_bottom = result_bottom.x  # Centre du cercle
    cx_middle_left, cy_middle_left = result_middle_left.x  # Centre du cercle
    cx_middle_right, cy_middle_right = result_middle_right.x  # Centre du cercle

    Ri_top = calc_R(result_top.x, top[:, 0], top[:, 1])
    Ri_bottom = calc_R(result_bottom.x, bottom[:, 0], bottom[:, 1])
    Ri_middle_left = calc_R(result_middle_left.x, middle_left[:, 0], middle_left[:, 1])
    Ri_middle_right = calc_R(result_middle_right.x, middle_right[:, 0], middle_right[:, 1])

    r_top = np.sqrt(np.mean(Ri_top))  # Rayon du cercle
    r_bottom = np.sqrt(np.mean(Ri_bottom))  # Rayon du cercle
    r_middle_left = np.sqrt(np.mean(Ri_middle_left))  # Rayon du cercle
    r_middle_right = np.sqrt(np.mean(Ri_middle_right))  # Rayon du cercle
    
    # print(f"Le rayon de courbure moyen Haut est: {r_top}")
    # print(f"Le rayon de courbure moyen Bas est: {r_bottom}")
    # print(f"Le rayon de courbure moyen Milieu Gauche est: {r_middle_left}")
    # print(f"Le rayon de courbure moyen Milieu Droit est: {r_middle_right}")

    fig, ax = plt.subplots()
    ax.imshow(gray_image, cmap='gray')
    # ax.axis('equal')

    # b = image.shape[1]/part
    # ax.plot([0, image.shape[2]], [b, m1*image.shape[2] + b], 'm--', label='m1')
    # b = image.shape[1] - image.shape[1]/part
    # ax.plot([0, image.shape[2]], [b, m2*image.shape[2] + b], 'c--', label='m2')

    plot_circle(ax, (cx_top, cy_top), r_top, 'r', '-')
    plot_circle(ax, (cx_bottom, cy_bottom), r_bottom, 'b', '-')
    plot_circle(ax, (cx_middle_left, cy_middle_left), r_middle_left, 'g', '-')
    plot_circle(ax, (cx_middle_right, cy_middle_right), r_middle_right, 'y', '-')

        # Tracer les points de contour et les lignes de division
    # ax.plot(max_contour[:, 0, 0], max_contour[:, 0, 1], 'r')
    ax.plot(top[:, 0], top[:, 1], 'r.', label='Top')
    ax.plot(bottom[:, 0], bottom[:, 1], 'b.', label='Bottom')
    ax.plot(middle_left[:, 0], middle_left[:, 1], 'g.', label='Middle Left')
    ax.plot(middle_right[:, 0], middle_right[:, 1], 'y.', label='Middle Right')

    plt.legend()
    plt.title(title)
    plt.show()

    return r_top, r_bottom, r_middle_left, r_middle_right

def filtrer_ellipses(ellipse, image: np.ndarray):
    ellipse_aera = m.pi * ellipse[1][0] * ellipse[1][1]
    return (ellipse_aera > 0.05 * image.shape[1] * image.shape[2] and
            (ellipse[0][0] > 0 
             and ellipse[0][1] > 0 
             and ellipse[0][0] < image.shape[2] 
             and ellipse[0][1] < image.shape[1]
             and ellipse[1][0] < image.shape[1] and ellipse[1][0] < image.shape[2]
             and ellipse[1][1] < image.shape[1] and ellipse[1][1] < image.shape[2])
            )

def get_image_flatten(image: np.ndarray):
    gray_image = centrer_image(image)
    return gray_image.flatten()


def get_age(image: np.ndarray, title, type):
    gray_image = centrer_image(image)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    gray_image[binary_image != 255] = 0

    # Égaliser l'histogramme des pixels de gray_image qui valent 1 dans binary_image
    equalized_image = cv2.equalizeHist(gray_image)

    equalized_image[binary_image != 255] = 255
    equalized_image = 255 - equalized_image

    # plt.imshow(equalized_image, cmap='gray')
    # plt.title(title)
    # plt.show()

    # Traiter l'image
    block_size = int(get_diametre_reel(image, title)[0]*get_echelle(title)) // 8 * 2 + 1
    # print("block size : ", block_size)
    
    binary_image_petit = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -25)
    binary_image = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 0)
    binary_image_grand = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 25)

    # afficher les images binaires avec plt sur une seule ligne
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # ax[0].imshow(binary_image_petit, cmap='gray')
    # ax[0].set_title('Constante = -25')
    # ax[1].imshow(binary_image, cmap='gray')
    # ax[1].set_title('Constante = 0')
    # ax[2].imshow(binary_image_grand, cmap='gray')
    # ax[2].set_title('Constante = 25')
    # plt.show()

    binary_image = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -25)
    
    binary_image_blur = cv2.GaussianBlur(binary_image, (15, 15), 0)

    contours, _ = cv2.findContours(binary_image_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = [contour for contour in contours if len(contour) >= 5]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # binary_image avec plt
    # plt.imshow(binary_eroded_image, cmap='gray')
    # # afficher les contours sur plt avec différentes couleurs
    # for i, new_contour in enumerate(contours):
    #     plt.plot(new_contour[:, 0, 0], new_contour[:, 0, 1], 'C' + str(i % 10))
    # plt.title(title)
    # plt.show()

    print("nombre de contours : ", len(contours))

    # trouver les ellipses ajustées aux contours
    ellipses = [cv2.fitEllipse(contour) for contour in contours]

    # enlever les ellipses par rapport à filtrer_ellipses
    ellipses = [ellipse for ellipse in ellipses if filtrer_ellipses(ellipse, image)]

    # prendre le centre de la plus petite ellipse
    if len(ellipses) == 0:
        return 0
    
    center = ellipses[0][0]

    # afficher le centre sur l'image avec plt
    # plt.imshow(gray_image, cmap='gray')
    # plt.plot(center[0], center[1], 'ro')
    # plt.title(title)
    # plt.show()

    equations_droites = []
    equations_pixels = []
    # Parcourir différentes valeurs de la pente
    pente = -10.0
    while pente <= 10.0:
        # Gérer le cas d'une pente verticale (infinie)
        if pente != 0:
            # Calculer l'ordonnée à l'origine
            c = center[1] - pente * center[0]
            # Former l'équation de la droite
            equation_droite = f"y = {pente}x + {c}"
            equations_droites.append([pente, c])

            pixels_droite = []
            for x in range(binary_image.shape[1]):
                y = int(pente * x + c)
                if 0 <= y < binary_image.shape[0]:
                    pixels_droite.append((x, y))
            
            equations_pixels.append((equation_droite, np.array(pixels_droite)))
        
        pente += 0.1        
    
    # Initialiser une liste pour stocker les résultats pour chaque droite
    resultats = []

    # Traitement pour chaque droite dans equations_pixels
    for equation, pixels in equations_pixels:
        x_values, y_values = zip(*pixels)
        
        # Calculer les valeurs de ligne et de colonne pour chaque droite
        droite = binary_image[y_values, x_values]
        
        # Calculer les différences et les sommes pour chaque droite
        diff_droite= np.diff(droite) / 255
        
        droite_0_1 = np.sum(diff_droite)
        
        # Ajouter les résultats à la liste
        resultats.append((equation, droite_0_1))

    # Calcul du vecteur final en combinant les résultats pour toutes les droites
    vecteur_final = np.array([resultat[1:] for resultat in resultats])
    moyenne_vecteur_final = np.mean(vecteur_final, axis=0)


    x = np.arange(binary_image.shape[1])

    for equation in equations_droites:
    # Définir les coordonnées du texte sur l'image (à ajuster selon vos besoins)
        pente= equation[0]
        c = equation[1]
        c = round(float(c))
        
        if pente == 0:
            y = np.full_like(x, center[1])
        else:
            y = pente * x + c
            y = np.clip(y, 0, binary_image.shape[0] - 1)  # Limiter les valeurs de y à la taille de l'image

        # Tracer la droite
    #     plt.plot(x, y, color='red')

    # plt.show()

    # afficher ligne, colonne et diagonales
    # Afficher la ligne en bleu
    # plt.imshow(binary_image, cmap='gray')
    # plt.plot([0, binary_image.shape[1]], [center[1], center[1]], color='blue')

    # # Afficher la colonne en bleu
    # plt.plot([center[0], center[0]], [0, binary_image.shape[0]], color='blue')

    # # Afficher la première diagonale en bleu
    # plt.plot([0, binary_image.shape[1]], [center[1] - center[0], center[1] - center[0] + binary_image.shape[1]], color='blue')

    # # Afficher la deuxième diagonale en bleu
    # plt.plot([0, binary_image.shape[1]], [center[1] + center[0], center[1] + center[0] - binary_image.shape[1]], color='blue')

    # plt.title(title)
    # plt.show()
    # print("moyenne_vecteur_final : ", moyenne_vecteur_final)
    age_estimation =  moyenne_vecteur_final[0]/2
    # print('age estimation : ', age_estimation)
    return age_estimation

def get_growth_ellipses(image: np.ndarray, title, type):
    # print(m.floor(1/diametre * echelle / 13))
    gray_image = centrer_image(image)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    gray_image[binary_image != 255] = 0

    # Égaliser l'histogramme des pixels de gray_image qui valent 1 dans binary_image
    equalized_image = cv2.equalizeHist(gray_image)

    equalized_image[binary_image != 255] = 255
    equalized_image = 255 - equalized_image

    # plt.imshow(equalized_image, cmap='gray')
    # plt.title(title)
    # plt.show()

    # Traiter l'image
    block_size = int(get_diametre_reel(image, title)[0]*get_echelle(title)) // 8 * 2 + 1
    # print("block size : ", block_size)
    
    binary_image_petit = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -25)
    binary_image = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 0)
    binary_image_grand = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 25)

    # afficher les images binaires avec plt sur une seule ligne
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # ax[0].imshow(binary_image_petit, cmap='gray')
    # ax[0].set_title('Constante = -25')
    # ax[1].imshow(binary_image, cmap='gray')
    # ax[1].set_title('Constante = 0')
    # ax[2].imshow(binary_image_grand, cmap='gray')
    # ax[2].set_title('Constante = 25')
    # plt.show()

    binary_image = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -25)
    
    binary_image = cv2.GaussianBlur(binary_image, (15, 15), 0)

    _, binary_eroded_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = [contour for contour in contours if len(contour) >= 5]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # binary_image avec plt
    # plt.imshow(binary_eroded_image, cmap='gray')
    # # afficher les contours sur plt avec différentes couleurs
    # for i, new_contour in enumerate(contours):
    #     plt.plot(new_contour[:, 0, 0], new_contour[:, 0, 1], 'C' + str(i % 10))
    # plt.title(title)
    # plt.show()

    print("nombre de contours : ", len(contours))

    # trouver les ellipses ajustées aux contours
    ellipses = [cv2.fitEllipse(contour) for contour in contours]

    # enlever les ellipses par rapport à filtrer_ellipses
    ellipses = [ellipse for ellipse in ellipses if filtrer_ellipses(ellipse, image)]

    # trouver l'air max des éllipses
    if len(ellipses) <= 1:
        return 0

    # trier les ellipses de la plus petite à la plus grande
    ellipses = sorted(ellipses, key=lambda x: x[1][0] * x[1][1])

    radius_diff = []

    # afficher avec print toutes les ellipses
    for ellipse in ellipses:
        print(ellipse)

    for i in range(len(ellipses)):
        if i == 0:
            radius_diff.append(ellipses[i][1][0]/2)
        else:
            radius_diff.append((ellipses[i][1][0] - ellipses[i - 1][1][0])/2)
    if len(radius_diff) == 0:
        return 0

    # trier valeur ordre decroissant
    radius_diff = sorted(radius_diff, reverse=True)

    min_radius_diff = min(radius_diff)

    # # enlever min_radius_diff de toute les valeurs
    radius_diff = [x - min_radius_diff for x in radius_diff]

    x = np.arange(len(radius_diff))

    a, b = np.polyfit(x, radius_diff, 1)

    # Find x for y = 0
    x_for_y_0 = - b / a
    x_for_y_0 = m.floor(x_for_y_0)
 
    # Plot the data and the line
    # plt.plot(x, radius_diff, 'o', label='Data')
    # plt.plot(x, a * x + b, label="Estimation de l'évolution de la différence de rayon")
    # plt.xlabel('Ellipse Number')
    # plt.ylabel('Radius Difference')
    # plt.legend()
    # titre = title, " : ", str(x_for_y_0)
    # plt.title(titre)
    # plt.show()

    # print(- b / x_for_y_0)

    return abs(a)

def get_flattened_image(image: np.ndarray, name: str):
    gray_image = centrer_image(image)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    gray_image[binary_image != 255] = 0

    # print(gray_image.flatten())

    return gray_image.flatten()

def split_image(image: np.ndarray):
    if image.shape[0] == 3:
        gray = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    gray[binary_image != 255] = 0

    equalized_image = cv2.equalizeHist(gray)

    equalized_image[binary_image != 255] = 255
    equalized_image = 255 - equalized_image

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Save the gray image
    # cv2.imwrite('gray_image.png', gray)

    # Trier les contours par aire (pour isoler les deux plus grands contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # fig, ax = plt.subplots()
    # ax.imshow(gray, cmap='gray')
    # for contour in contours:
    #     ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'r')
    # plt.show()

    # Créer un masque vide pour chaque contour
    mask1 = np.zeros_like(binary_image)
    mask2 = np.zeros_like(binary_image)

    # Dessiner chaque contour sur son masque respectif
    cv2.drawContours(mask1, [contours[0]], -1, (255), thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contours[1]], -1, (255), thickness=cv2.FILLED)

    # Trouver les points à l'intérieur de chaque contour
    points_inside_contour1 = np.transpose(np.nonzero(mask1))
    points_inside_contour2 = np.transpose(np.nonzero(mask2))

    # print(points_inside_contour1)
    X1, Y1 = points_inside_contour1[:, 1], points_inside_contour1[:, 0]
    X2, Y2 = points_inside_contour2[:, 1], points_inside_contour2[:, 0]

    # Assurez-vous que les points sont triés par x croissant
    sorted_indices_1 = np.argsort(X1)
    x1_points = np.array(X1)[sorted_indices_1]
    y1_points = np.array(Y1)[sorted_indices_1]

    sorted_indices_2 = np.argsort(X2)
    x2_points = np.array(X2)[sorted_indices_2]
    y2_points = np.array(Y2)[sorted_indices_2]

    # afficher l'image grisé et l'image binaire avec plt
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(gray, cmap='gray')
    # ax[0].set_title('Mask 1')
    # ax[0].plot(x1_points, y1_points, 'r.')
    # ax[1].imshow(gray, cmap='gray')
    # ax[1].plot(x2_points, y2_points, 'b.')
    # ax[1].set_title('Mask 2')
    # plt.show()

    # Utiliser np.column_stack pour combiner x_points et y_points en un seul tableau pour chaque ensemble de points
    points1 = np.column_stack((x1_points, y1_points))
    points2 = np.column_stack((x2_points, y2_points))
    
    points1 = np.array(points1)
    points2 = np.array(points2)

    diff = len(points2) / len(points1)

    #si la différence est trop grande, trouver le x moyen des points1
    if diff < 0.6:
        x_moyen = np.mean(points1[:, 0])
        points = points1
        points1 = points[points[:, 0] < x_moyen]
        points2 = points[points[:, 0] > x_moyen]

    #center les points1 et points2 dans les images1 et images2 de taille width/2 x height
    image1 = np.zeros((image.shape[0], image.shape[1], image.shape[2]//2))
    image2 = np.zeros((image.shape[0], image.shape[1], image.shape[2]//2))

    #calculer centre des points
    center1 = (np.mean(points1[:, 0]), np.mean(points1[:, 1]))
    center2 = (np.mean(points2[:, 0]), np.mean(points2[:, 1]))

    #centre des images1 et images2
    center_image1 = (image1.shape[2]//2, image1.shape[1]//2)
    center_image2 = (image2.shape[2]//2, image2.shape[1]//2)

    #décalage
    decalage1 = (center_image1[0] - center1[0], center_image1[1] - center1[1])
    decalage2 = (center_image2[0] - center2[0], center_image2[1] - center2[1])

    #décalage des points
    points1b = points1 + decalage1
    points2b = points2 + decalage2

    # mettre les points sur les images
    points1_array = np.array(points1, dtype=int)
    points1b_array = np.array(points1b, dtype=int)
    points2_array = np.array(points2, dtype=int)
    points2b_array = np.array(points2b, dtype=int)

    # Extraire les coordonnées x et y séparément
    y1, x1 = points1_array[:, 1], points1_array[:, 0]
    y1b, x1b = points1b_array[:, 1], points1b_array[:, 0]
    y2, x2 = points2_array[:, 1], points2_array[:, 0]
    y2b, x2b = points2b_array[:, 1], points2b_array[:, 0]

    # Ensure indices are within valid range
    y1 = np.clip(y1, 0, image.shape[1]-1)
    y1b = np.clip(y1b, 0, image1.shape[1]-1)
    x1 = np.clip(x1, 0, image.shape[2]-1)
    x1b = np.clip(x1b, 0, image1.shape[2]-1)
    y2 = np.clip(y2, 0, image.shape[1]-1)
    y2b = np.clip(y2b, 0, image2.shape[1]-1)
    x2 = np.clip(x2, 0, image.shape[2]-1)
    x2b = np.clip(x2b, 0, image2.shape[2]-1)

    # Verifier cette partie va avec la partie du dessus. Prendre en compte le décalage
    # Utiliser l'indexation avancée pour copier les valeurs pour tous les canaux à la fois

    image1[:, y1b, x1b] = image[:, y1, x1]
    image2[:, y2b, x2b] = image[:, y2, x2]

    image1 = cv2.normalize(image1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image2 = cv2.normalize(image2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Convertissez ensuite en 8 bits
    image1 = np.uint8(image1 * 255)
    image2 = np.uint8(image2 * 255)

    # enregistrer les images
    print(image1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(centrer_image(image1))
    ax[1].imshow(centrer_image(image2))
    plt.show()

    # # Tracer le centre
    # ax.plot(center1[0], center1[1], 'r*', label='Center')
    # ax.plot(center2[0], center2[1], 'r*', label='Center')

    # ax.plot(center_image1[0], center_image1[1], 'r*', label='Center')
    # ax.plot(center_image2[0], center_image2[1], 'r*', label='Center')

    # plt.legend()
    # plt.show()

    # Créer une image vide pour chaque objet
    object1 = np.zeros_like(image)
    object2 = np.zeros_like(image)

    # Enregistrer les deux objets comme des images séparées
    # cv2.imwrite('object1.jpg', object1)
    # cv2.imwrite('object2.jpg', object2)
    return image1, image2

def test_image():
    test = ["right/GBD_19_B40_C1_O_0090", "left_and_right/KS_10_TRIM3_LCN_073", "right/GBD_19_B40_C1_O_0092", "left_and_right/KS_13_TRIM3_XBL_0169", "left_and_right/KS_13_TRIM4_CGFS_0120"]
    image_n = 1
    image_path = f"{plaice_dir_path}/{test[image_n]}.tif"
    exist = False
    try:
        image = tifffile.imread(image_path)
        exist = True
    except: # l'image n'existe pas
        print("Image not found")
        pass
    
    print(test[image_n])

    if exist:
        if "left_and_right" in image_path:
            image_left, image_right = split_image(image)

            # r_top1, r_bottom1, r_middle_left1, r_middle_right1 = get_ring_curvature(image_left, test[image_n])
            # r_top2, r_bottom2, r_middle_left2, r_middle_right2 = get_ring_curvature(image_right, test[image_n])

            # r_top = (r_top1 + r_top2) / 2
            # r_bottom = (r_bottom1 + r_bottom2) / 2
            # r_middle_left = (r_middle_left1 + r_middle_left2) / 2
            # r_middle_right = (r_middle_right1 + r_middle_right2) / 2

            # age1 = get_growth_ellipses(image_left, test[image_n], 'left')
            # age2 = get_growth_ellipses(image_right, test[image_n], 'right')

            # age = (age1 + age2) / 2

            # density1 = get_relative_densite(image_left)
            # density2 = get_relative_densite(image_right)

            # density_retalive = (density1 + density2)/2
            # print(get_opacite_relative(density_retalive))

            # density1 = get_densite(image_left)
            # density2 = get_densite(image_right)

            # density = (density1 + density2)/2
            # print(get_opacite_relative(density))
            growth1 = get_growth_ellipses(image_left, test[image_n], 'left')
            growth2 = get_growth_ellipses(image_right, test[image_n], 'right')

            growth = (growth1 + growth2) / 2

            # get_growth_ellipses(image_left, test[image_n], 'left')
            # get_growth_ellipses(image_right, test[image_n], 'right')

        else:
            # r_top, r_bottom, r_middle_left, r_middle_right = get_ring_curvature(image, test[image_n])
            # age = get_growth_ellipses(image, test[image_n], '')

            # density_retalive = get_relative_densite(image)
            # print(get_opacite_relative(density_retalive))
            # density = get_densite(image)
            # print(get_opacite_relative(density))
            growth = get_growth_ellipses(image, test[image_n])
            # get_growth_ellipses(image, test[image_n], '')

        print(growth)

        # print('age estimation final : ', age)
    # print(r_top, r_bottom, r_middle_left, r_middle_right)

    # get_growth(age)
    # get_ring_curvature(image, "AD_18_B76_C1_O_0042.tif")
    # print(get_echelle(image))
    # print(image.shape)


# test_surfaces()

# destroy_data()
# remplissage()

def maj():
    cursor = db.cursor()
    images = df[['Age', 'Reference_PC']]    

    X = []
    Y = []

    for i, row in images.iterrows():
        nb_barres = '#' * int(i / (len(images)) * 100) + ' ' * int(100 - i / (len(images)) * 100)
        print(nb_barres, str(i / (len(images)) * 100) + '%', end='\n')
        # if(i / (len(images)) * 100 < 86):
        #     # os.system('cls')
        #     continue
        age = row['Age']
        image_path = f"{plaice_dir_path}/right/{row['Reference_PC']}.tif" if os.path.exists(f"{plaice_dir_path}/right/{row['Reference_PC']}.tif") else f"{plaice_dir_path}/left_and_right/{row['Reference_PC']}.tif"
        exist = False
        try:
            image = tifffile.imread(image_path)
            exist = True
        except: # l'image n'existe pas
            os.system('cls')  # Efface la console
            # print("Image not found")

        print(row['Reference_PC'])

        if exist:
            print(image.shape)
            if "left_and_right" in image_path:
                image_left, image_right = split_image(image)

                # r_top1, r_bottom1, r_middle_left1, r_middle_right1 = get_ring_curvature(image_left, row['Reference_PC'])
                # r_top2, r_bottom2, r_middle_left2, r_middle_right2 = get_ring_curvature(image_right, row['Reference_PC'])

                # r_top = (r_top1 + r_top2) / 2
                # r_bottom = (r_bottom1 + r_bottom2) / 2
                # r_middle_left = (r_middle_left1 + r_middle_left2) / 2
                # r_middle_right = (r_middle_right1 + r_middle_right2) / 2

                # age1 = get_growth(image_left, row['Reference_PC'], 'left')
                # age2 = get_growth(image_right, row['Reference_PC'], 'right')

                # if age1 == 0:
                #     age = age2
                # elif age2 == 0:
                #     age = age1
                # else:
                #     age = (age1 + age2) / 2

                # density1 = get_relative_densite(image_left)
                # density2 = get_relative_densite(image_right)

                # density_retalive = (density1 + density2)/2

                # density1 = get_densite(image_left)
                # density2 = get_densite(image_right)

                # density = (density1 + density2)/2

                # max_diametre_1, min_diametre_1 = get_diametre_reel(image_left, row['Reference_PC'])
                # max_diametre_2, min_diametre_2 = get_diametre_reel(image_right, row['Reference_PC'])

                # max_diametre = (max_diametre_1 + max_diametre_2) / 2

                # surface1 = get_surface(image_left, row['Reference_PC'])
                # surface2 = get_surface(image_right, row['Reference_PC'])

                # surface = (surface1 + surface2) / 2

                
                growth1 = get_age(image_left, row['Reference_PC'], 'left')
                growth2 = get_age(image_right, row['Reference_PC'], 'right')

                growth = (growth1 + growth2) / 2
                    
            else:
                # os.system('cls')  # Efface la console
                # continue
                # density_retalive = get_relative_densite(image)
                # density = get_densite(image)
                
                # max_diametre, min_diametre = get_diametre_reel(image, row['Reference_PC'])
                # surface = get_surface(image, row['Reference_PC'])

                growth = get_age(image, row['Reference_PC'], '')


            # print('age estimation final : ', growth)
                # r_top, r_bottom, r_middle_left, r_middle_right = get_ring_curvature(image, row['Reference_PC'])

            # print('age estimation final : ', age)

            # cursor.execute('SELECT age FROM donnees WHERE filepath = ?', 
            #                 (row['Reference_PC'],))
            
            # print(age, row['Reference_PC'])

            # result = cursor.fetchall()
            
            # if len(result) != 0:
            #     X.append(result[0])
            #     Y.append(age)
            #     cursor.execute('UPDATE donnees SET nombre_raies = ? WHERE filepath = ?', 
            #                     (age, row['Reference_PC']))
            
            # cursor.execute('UPDATE donnees SET max_diametre = ?, min_diametre = ?, elongation = ?, surface = ? WHERE filepath = ?', 
            #                      (max_diametre, min_diametre, max_diametre/min_diametre, surface, row['Reference_PC']))

            # cursor.execute('UPDATE donnees SET nombre_raies = ? WHERE filepath = ?', 
            #                      (growth, row['Reference_PC']))


            # plt.scatter(X, Y)
            # plt.xlabel('Age')
            # plt.ylabel('Coefficients Y')
            # plt.title('Coeff vs Age')
            # plt.show()

            # cursor.execute('UPDATE donnees SET RC_top = ?, RC_bottom = ?, RC_right = ?, RC_left = ? WHERE filepath = ?', 
            #                 (r_top, r_bottom, r_middle_right, r_middle_left, row['Reference_PC']))

            os.system('cls')  # Efface la console

    plt.scatter(X, Y)
    plt.xlabel('Age')
    plt.ylabel('Coefficients Y')
    plt.title('Coeff vs Age')
    plt.show()

    # print(len(images_fails))
    db.commit()

if __name__ == '__main__':
    # maj()
    test_image()