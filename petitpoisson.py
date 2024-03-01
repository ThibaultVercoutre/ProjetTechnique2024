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
plaice_dir_path = "C:\\Users\\thiba\\Downloads\\96695\\data_image_otolith_ple_mur\\plaice"
df = pd.read_csv(plaice_dir_path + '\\metadata_plaice_2010-2019.csv', sep=';')

def get_echelle(image: np.ndarray):
    return 125
    # motif = np.uint8(np.array([[[255]*3]*3]*21))
    # print(motif)
    # image = image.transpose(1, 2, 0)
    # print(image.shape, motif.shape)
    # # Vérifiez si l'image est déjà en niveaux de gris
    # # _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # _, binary_motif = cv2.threshold(motif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # result = cv2.matchTemplate(image, motif, cv2.TM_SQDIFF_NORMED)
    # seuil = 1
    # locations = np.where(result >= seuil)

    # print(result)
    # positions_haut_gauche = []

    # for pt in zip(*locations[::-1]):  # Inverser x et y dans locations
    #     positions_haut_gauche.append(pt)

    # print(len(positions_haut_gauche))
    #     # Dessiner un rectangle autour de la zone correspondante pour la visualisation
    #     cv2.rectangle(image, pt, (pt[0] + motif.shape[1], pt[1] + motif.shape[0]), (0,255,255), 2)

    # # Afficher l'image avec les rectangles dessinés
    # cv2.imshow('Correspondances', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def get_densite(image):
    return np.mean(image)

d_ref = 69
def get_opacite_relative(densite):
    return (densite - d_ref) / d_ref


# ne fonctionne pas totalement
def get_surface(image: np.ndarray):
    gray_image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('binary_image.png', binary_image)
    # binary_image = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    surface = 0
    for contour in contours:
        surface += cv2.contourArea(contour)
    return surface / get_echelle(image)**2

def get_diametre(image):
    surface = get_surface(image)
    return 2 * np.sqrt(surface / np.pi)

def get_max_diametre(image):
    gray_image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_diametre = 0
    for contour in contours:
        (x, y), rayon = cv2.minEnclosingCircle(contour)
        max_diametre = max(max_diametre, rayon*2)
    return max_diametre / get_echelle(image)

def get_min_diametre(image):
    gray_image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_diametre = 0
    for contour in contours:
        (x, y), rayon = cv2.minEnclosingCircle(contour)
        max_diametre = min(max_diametre, rayon*2)
    return max_diametre


# ne pas lancer cette fonction -> BDD déjà pleine
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
        surface = get_surface(image)
        diametre = get_diametre(image)
        max_diametre = get_max_diametre(image)
        surface = surface / 2 if '/left_and_right/' in image_path else surface
        cursor.execute('UPDATE donnees SET age = ?, densite = ?, relativeopacity = ?, surface = ?, diametre = ?, max_diametre = ? WHERE filepath = ?', (age, densite, opacite, surface, diametre, max_diametre, row['Reference_PC']))
    db.commit()

def get_donnees():
    cursor = db.cursor()
    cursor.execute('SELECT * FROM donnees')
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
    gray_image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Supposons que le plus grand contour est notre région d'intérêt
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Calculer la boîte englobante du plus grand contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # cv2.rectangle(gray_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # cv2.imwrite('blur.png', gray_image)
        
        # Extraire la région d'intérêt
        roi = gray_image[y:y+h, x:x+w]
        
        # Créer une nouvelle image de fond avec les mêmes dimensions que l'image originale
        new_image = np.zeros_like(gray_image)
        
        # Calculer les coordonnées pour centrer la ROI dans la nouvelle image
        start_y = (new_image.shape[0] - roi.shape[0]) // 2
        start_x = (new_image.shape[1] - roi.shape[1]) // 2
        
        # Placer la ROI au centre de la nouvelle image
        new_image[start_y:start_y+roi.shape[0], start_x:start_x+roi.shape[1]] = roi
        
        return new_image

images_fails = []

def get_ring_curvature(image: np.ndarray, title):

    if(len(image.shape) == 2):
        gray_image = image
    else:
        gray_image = centrer_image(image)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # cv2.imwrite('blur.png', gray_image)

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

    top = []
    bottom = []
    middle_left = []
    middle_right = []

    for i in range(len(x_points)):
        f1 = x_points[i]*m1 + image.shape[1]/part
        f2 = x_points[i]*m2 + image.shape[1] - image.shape[1]/part
        if f1 < y_points[i] and f2 < y_points[i]:
            bottom.append([x_points[i], y_points[i]])
        elif f1 > y_points[i] and f2 > y_points[i]:
            top.append([x_points[i], y_points[i]])
        elif f1 > y_points[i] and f2 < y_points[i]:
            middle_right.append([x_points[i], y_points[i]])
        elif f1 < y_points[i] and f2 > y_points[i]:
            middle_left.append([x_points[i], y_points[i]])

    # top, bottom, middle_left, middle_right = categorize_points(x_points, y_points)

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

    if(top.shape[0] == 0 or bottom.shape[0] == 0 or middle_left.shape[0] == 0 or middle_right.shape[0] == 0):
        fig, ax = plt.subplots()
        ax.axis('equal')

        b = image.shape[1]/part
        ax.plot([0, image.shape[2]], [b, m1*image.shape[2] + b], 'm--', label='m1')
        b = image.shape[1] - image.shape[1]/part
        ax.plot([0, image.shape[2]], [b, m2*image.shape[2] + b], 'c--', label='m2')

        plot_circle(ax, (cx_top, cy_top), r_top, 'r', '-')
        plot_circle(ax, (cx_bottom, cy_bottom), r_bottom, 'b', '-')
        plot_circle(ax, (cx_middle_left, cy_middle_left), r_middle_left, 'g', '-')
        plot_circle(ax, (cx_middle_right, cy_middle_right), r_middle_right, 'y', '-')

        # Tracer les points de contour et les lignes de division
        ax.plot(top[:, 0], top[:, 1], 'r*', label='Top')
        ax.plot(bottom[:, 0], bottom[:, 1], 'b*', label='Bottom')
        ax.plot(middle_left[:, 0], middle_left[:, 1], 'g*', label='Middle Left')
        ax.plot(middle_right[:, 0], middle_right[:, 1], 'y*', label='Middle Right')

        plt.legend()
        plt.title(title)
        plt.show()

        return 0, 0, 0, 0
    return r_top, r_bottom, r_middle_left, r_middle_right

def filtrer_ellipses(ellipse, image: np.ndarray):
    ellipse_aera = m.pi * ellipse[1][0] * ellipse[1][1]
    return (ellipse_aera > 0.05 * image.shape[1] * image.shape[2] and
            (ellipse[0][0] > 0 
             and ellipse[0][1] > 0 
             and ellipse[0][0] < image.shape[2] 
             and ellipse[0][1] < image.shape[1]) and
             (ellipse[1][0] / ellipse[1][1] <= 10
              and ellipse[1][1] / ellipse[1][0] <= 10))

def get_growth(image: np.ndarray, title):
    gray_image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    mean_value = 1*np.median(gray_image)
    gray_image = centrer_image(image)

    gray_image_init = gray_image

    mask = gray_image > mean_value
    # Égaliser l'histogramme sur les pixels non noirs
    gray_image[~mask] = 0
    # gray_image[mask] = 255
    non_zero_values = gray_image[mask]

    eq_values = cv2.equalizeHist(non_zero_values.reshape(-1,1)).flatten()

    # Créer une nouvelle image qui contiendra le résultat final
    eq_image = np.zeros_like(gray_image, dtype=np.uint8)

    # Remplacer les valeurs dans l'image égalisée là où le masque est vrai
    eq_image[mask] = eq_values

    binary_image = cv2.adaptiveThreshold(eq_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 2)
    
    binary_image = 255 - binary_image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    binary_image = opening

    dilated_image = cv2.dilate(binary_image, kernel, iterations=4)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=8)

    # save eroded image
    cv2.imwrite('eroded_image.png', eroded_image)

    binary_image = eroded_image

    binary_image = cv2.medianBlur(binary_image, 3)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Filtrer les contours pour ne conserver que ceux qui ont au moins 5 points
    filtered_contours = [contour for contour in contours if len(contour) >= 5]

    # Trouver les ellipses ajustées aux contours
    ellipses = [cv2.fitEllipse(contour) for contour in filtered_contours]

    # filtrer les ellipses
    ellipses = [ellipse for ellipse in ellipses if filtrer_ellipses(ellipse, image)]

    # trouver l'air max des éllipses
    max_ellipse = max(ellipses, key=lambda x: x[1][0] * x[1][1])

    # enlever toutes les éllipses dont le centre n'est pas contenu dans l'ellipse la plus grande
    ellipses = [ellipse for ellipse in ellipses if max_ellipse[0][0] - max_ellipse[1][0] < ellipse[0][0] < max_ellipse[0][0] + max_ellipse[1][0] and max_ellipse[0][1] - max_ellipse[1][1] < ellipse[0][1] < max_ellipse[0][1] + max_ellipse[1][1]]


    # trier les ellipses par taille
    ellipses = sorted(ellipses, key=lambda x: x[1][0] * x[1][1], reverse=False)

    # for i, ellipse in enumerate(ellipses):
    #    print(f"Ellipse {i+1} : Center = {ellipse[0]}, Axes = {ellipse[1]}, Angle = {ellipse[2]}")
    
    image_with_ellipses = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    filtered_ellipses = [ellipse for ellipse in ellipses if ellipse[1][0] > 0 and ellipse[1][1] > 0]

    for ellipse in filtered_ellipses:
        cv2.ellipse(image_with_ellipses, ellipse, (0, 255, 0), 2)


    cv2.imwrite('image_with_ellipses.png', image_with_ellipses)
    cv2.imwrite('binary_image.png', binary_image)
    cv2.imwrite('gray_image_init.png', gray_image_init)
    cv2.imwrite('eq_image.png', eq_image)

    radius_diff = []

    for i in range(len(ellipses) - 1):
        if i == 0:
            radius_diff.append(ellipses[i][1][0])
        radius_diff.append(np.linalg.norm(np.array(ellipses[i + 1][0]) - np.array(ellipses[i][0])))

    if len(radius_diff) == 0:
        return 0

    x = np.arange(len(radius_diff))

    slope, intercept = np.polyfit(x, radius_diff, 1)

    # Find x for y = 0
    x_for_y_0 = -intercept / slope
    x_for_y_0 = m.floor(x_for_y_0)

    # Plot the data and the line
    # plt.plot(x, radius_diff, 'o', label='Data')
    # plt.plot(x, slope * x + intercept, label='Linear Regression Line')
    # plt.xlabel('Ellipse Number')
    # plt.ylabel('Radius Difference')
    # plt.title('Average Radius Difference Between Consecutive Ellipses')
    # plt.legend()
    # titre = title, " : ", str(x_for_y_0)
    # plt.title(titre)
    # plt.show()

    return x_for_y_0

def split_image(image: np.ndarray):
    # Charger le modèle YOLO
    # gray = centrer_image(image)
    gray = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save the gray image
    # cv2.imwrite('gray_image.png', gray)

    # Trier les contours par aire (pour isoler les deux plus grands contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

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
    points1b_array = np.array(points1b, dtype=int)
    points2b_array = np.array(points2b, dtype=int)

    # Extraire les coordonnées x et y séparément
    y1, x1 = points1b_array[:, 1], points1b_array[:, 0]
    y2, x2 = points2b_array[:, 1], points2b_array[:, 0]

    # Ensure indices are within valid range
    y1 = np.clip(y1, 0, image1.shape[1]-1)
    x1 = np.clip(x1, 0, image1.shape[2]-1)
    y2 = np.clip(y2, 0, image2.shape[1]-1)
    x2 = np.clip(x2, 0, image2.shape[2]-1)

    # Utiliser l'indexation avancée pour copier les valeurs pour tous les canaux à la fois
    image1[:, y1, x1] = image[:, y1, x1]
    image2[:, y2, x2] = image[:, y2, x2]

    image1 = cv2.normalize(image1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image2 = cv2.normalize(image2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Convertissez ensuite en 8 bits
    image1 = np.uint8(image1 * 255)
    image2 = np.uint8(image2 * 255)

    # enregistrer les images
    # cv2.imwrite('image1.png', image1)
    # cv2.imwrite('image2.png', image2)

    # fig, ax = plt.subplots()
    # ax.axis(ymin=0, ymax=image.shape[1], xmin=0, xmax=image.shape[2])


    # # Tracer les points de contour et les lignes de division
    # ax.plot(points1[:, 0], points1[:, 1], 'r*', label='Top')
    # ax.plot(points2[:, 0], points2[:, 1], 'b*', label='Bottom')

    # # ax.plot(points1b[:, 0], points1b[:, 1], 'g*', label='Top')
    # # ax.plot(points2b[:, 0], points2b[:, 1], '*', label='Bottom')
    # # Tracer le centre
    # # ax.plot(center[0], center[1], 'g*', label='Center')

    # plt.legend()
    # plt.show()

    # # Créer une image vide pour chaque objet
    # object1 = np.zeros_like(image)
    # object2 = np.zeros_like(image)

    # # Enregistrer les deux objets comme des images séparées
    # cv2.imwrite('object1.jpg', object1)
    # cv2.imwrite('object2.jpg', object2)
    return image1, image2

def test_surfaces():
    test = ["left_and_right/KS_10_TRIM3_LCN_073", "right/GBD_19_B40_C1_O_0092", "left_and_right/KS_13_TRIM3_XBL_0169", "left_and_right/KS_13_TRIM4_CGFS_0120"]
    image_path = f"{plaice_dir_path}/{test[1]}.tif"
    try:
        image = tifffile.imread(image_path)
    except: # l'image n'existe pas
        print("Image not found")
        return "Image not found"
    
    if "left_and_right" in image_path:
        image_left, image_right = split_image(image)

        age1 = get_growth(image_left, test[1])   
        age2 = get_growth(image_right, test[1])   

        if age1 == 0:
            age = age2
        elif age2 == 0:
            age = age1
        else:
            age = m.floor((age1 + age2) / 2)
            
    else:
        age = get_growth(image, test[2])   
    
    print(age)
    # get_growth(age)
    # get_ring_curvature(image, "AD_18_B76_C1_O_0042.tif")
    # print(get_echelle(image))
    # print(image.shape)
    # return get_surface(image)


# test_surfaces()

# destroy_data()
# remplissage()

def maj():
    cursor = db.cursor()
    images = df[['Age', 'Reference_PC']]    

    for i, row in images.iterrows():
        nb_barres = '#' * int(i / (len(images)) * 100) + ' ' * int(100 - i / (len(images)) * 100)
        print(nb_barres, str(i / (len(images)) * 100) + '%', end='\n')
        age = row['Age']
        image_path = f"{plaice_dir_path}/right/{row['Reference_PC']}.tif" if os.path.exists(f"{plaice_dir_path}/right/{row['Reference_PC']}.tif") else f"{plaice_dir_path}/left_and_right/{row['Reference_PC']}.tif"
        exist = False
        try:
            image = tifffile.imread(image_path)
            exist = True
        except: # l'image n'existe pas
            pass

        print(row['Reference_PC'])

        if "left_and_right" in image_path and exist:
            
            image_left, image_right = split_image(image)

            age1 = get_growth(image_left, row['Reference_PC'])
            age2 = get_growth(image_right, row['Reference_PC'])

            if age1 == 0:
                age = age2
            elif age2 == 0:
                age = age1
            else:
                age = m.floor((age1 + age2) / 2)
            
        else:
            age = get_growth(image, row['Reference_PC'])  
        cursor.execute('UPDATE donnees SET age_estime = ? WHERE filepath = ?', 
                       (age, row['Reference_PC']))
        
        os.system('cls')  # Efface la console

    print(len(images_fails))
    db.commit()

if __name__ == '__main__':
    maj()
    # test_surfaces()