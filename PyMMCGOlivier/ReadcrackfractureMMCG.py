import os
import cv2
import numpy as np
import re
from PIL import Image

#exec(open('Database.py').read())
# Récupérer la liste des fichiers TIF dans le dossier Image_Selection
#path='D:\Recherche PRD\SPR_00_02\SPR_00_02'
#endS = os.path.join(os.getcwd(), path + '\Image_Selection')
maincwd='D:\Recherche PRD\EXP\MMCG_Olivier\Arcan30'
path = os.path.join(maincwd, Job)
endS = os.path.join(os.getcwd(), path)
os.chdir(endS)
os.chdir(endS)

fileNames = sorted([file for file in os.listdir() if file.endswith('.tiff')])
pattern = re.compile(r'\d+')
# Utiliser sorted() pour trier la liste en utilisant les nombres extraits des noms de fichier
#fileNames = sorted(fileNames, key=lambda x: int(pattern.findall(x)[0]))
fileNames = sorted(fileNames, key=lambda x: int(x.split('_')[1].split('.')[0]))

# Définir le nombre d'images
n_images = len(fileNames)

# Charger l'image
img = Image.open('D:\Recherche PRD\EXP\MMCG_Olivier\Arcan00\e0e1\e0e1_0.tiff')
#img = Image.open('D:\Recherche PRD\SPR_00_02\SPR_00_02\Image_Selection\Image1.tif')
# Obtenir la taille de l'image
largeur, hauteur = img.size
# Afficher la taille de l'image
print("La taille de l'image est de {} x {} pixels.".format(largeur, hauteur))


def select_points(image_path, n_points):
    """
    Sélectionne n_points points sur l'image spécifiée en cliquant sur l'emplacement souhaité avec la souris.
    Retourne les coordonnées des points sous forme d'une liste de tuples.
    """
    # Charge l'image
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', hauteur, largeur)
    image = cv2.imread(image_path)

    # Crée une fenêtre pour l'affichage de l'image
    cv2.namedWindow('image')

    # Fonction appelée lorsqu'un clic de souris est détecté
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            # Ajoute les coordonnées du clic à la liste des points
            #points.append((x, y))
            points.append((x))
            # Dessine un cercle à l'emplacement du clic sur l'image
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            # Rafraîchit l'affichage de l'image
            cv2.imshow('image', image)

    # Attache la fonction de rappel de souris à la fenêtre d'image
    cv2.setMouseCallback('image', mouse_callback)

    # Initialise la liste des points
    points = []

    # Boucle jusqu'à ce que le nombre de points souhaité soit atteint
    while len(points) < n_points:
        # Affiche l'image
        cv2.imshow('image', image)
        # Attend une touche de clavier
        key = cv2.waitKey(1) & 0xFF
        # Quitte la boucle si la touche 'q' est appuyée
        if key == ord('q'):
            break

    # Ferme la fenêtre d'image
    cv2.destroyAllWindows()

    # Retourne la liste des points
    return points

# Initialiser la matrice X
nbimages=2
#nbimages=10
X = np.zeros((nbimages, 2))#in pixel
Xmm=np.zeros((nbimages, 2))

nombre = input("La dernière image avant la rupture est : ")
nombre=int(nombre)
d=np.linspace(alpha_stages, nombre,nbimages)
#d=np.linspace(0, MatchID.stages,nbimages)
a=0
points_list = []
for k in range(nbimages):
    a=a+1
    points = select_points(fileNames[int(d[k])], 1)
    points_list.append(points)
    X[a-1,1]=k
    X[a-1,0]=points_list[a-1][0]
    Xmm[a-1,0]=X[a-1,0]*Test.mm2pixel
    Xmm[a-1,1]=int(d[k])
    
#cracklength=np.abs(Xmm[-1,0]-Xmm[0,0])+Test.a0
#print('The crack length with python is: ', cracklength)
a1=Xmm[0,0]
af=Xmm[-1,0]

########################################
#With Matchid

#cracklength=np.abs(a0.imgHuse-af.imgHuse)*Test.mm2pixel+Test.a0
#print('The crack length with Matchid is: ',cracklength)



# Sauvegarder la matrices X
#*np.savetxt('D:\Recherche PRD\SPR_00_02\SPR_00_02\X.txt', X)
