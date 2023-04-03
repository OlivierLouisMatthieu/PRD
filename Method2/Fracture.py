import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image

a1=7.219847
af=9.889156
print(af-a1)
if a1==0 or af==0:
    exec(open('Readcrackfracture.py').read())

# INITIALIZATION:
Valpixel = 391         # Number of pixels to perform calibration;
Valmm = 9.94           # Amount of mm to perform calibration;
Cal = Valmm/Valpixel   # Calibration factor from pixel to mm;

ina = 0.75
inb = 0.53
CTODimage = 281 # initial crack tip!!!!

path='D:\Recherche PRD\SPR_00_02\SPR_00_02'

# VIRTUAL STRAIN GAUGE:
endS = os.path.join(os.getcwd(), path + '\Image_Selection')
endC = os.path.join(os.getcwd(), path + '\Results_Correlation')

# READING THE FILES:
os.chdir(endS)

fileNames = sorted([file for file in os.listdir() if file.endswith('.tif')])
pattern = re.compile(r'\d+')
# Utiliser sorted() pour trier la liste en utilisant les nombres extraits des noms de fichier
fileNames = sorted(fileNames, key=lambda x: int(pattern.findall(x)[0]))
# Afficher la liste triée
nImagens = len(fileNames)


# Charger l'image
img = Image.open(path + '\Image_Selection\Image57.tif')
# Obtenir la taille de l'image
largeur, hauteur = img.size
# Afficher la taille de l'image
print("La taille de l'image est de {} x {} pixels.".format(largeur, hauteur))

I = np.zeros((hauteur, largeur, nImagens))

for k, fileName in enumerate(fileNames):
    I[:, :, k] = cv2.resize(cv2.imread(os.path.join(endS, fileName), cv2.IMREAD_GRAYSCALE), (largeur, hauteur))

os.chdir('..')

fileNamesX = sorted([file for file in os.listdir(endC) if file.startswith('xm')])
fileNamesY = sorted([file for file in os.listdir(endC) if file.startswith('ym')])
fileNamesUm = sorted([file for file in os.listdir(endC) if file.startswith('um')])
fileNamesVm = sorted([file for file in os.listdir(endC) if file.startswith('vm')])

nImagens = len(fileNamesX) - 1

#part 2
columns=201
Xm = np.zeros((2, columns, nImagens))
Ym = np.zeros((2, columns, nImagens))
Um = np.zeros((2, columns, nImagens))
Vm = np.zeros((2, columns, nImagens))
CODy = np.zeros((columns, nImagens))
CODx = np.zeros((columns, nImagens))

for k in range(nImagens):
    Xm[:, :, k] = np.loadtxt(endC + '/' + fileNamesX[k])
    Ym[:, :, k] = np.loadtxt(endC + '/' + fileNamesY[k])
    Um[:, :, k] = np.loadtxt(endC + '/' + fileNamesUm[k])
    Vm[:, :, k] = np.loadtxt(endC + '/' + fileNamesVm[k])
    # why 2*201*77
    CODy[:, k] = Ym[1, :, k] - Ym[0, :, k]
    CODx[:, k] = (Xm[1, :, k] + Xm[0, :, k]) / 2 
    #create a mean I think they have already chosen the pair of subset

dx = Xm[0, 1, 0] - Xm[0, 0, 0]
dy = CODy[:, 0]

CODy = CODy - CODy[:, [0]]#VD=0 for the 1st image to shift

STRAINy = np.zeros((columns, nImagens))

for k in range(nImagens):
    STRAINy[:, k] = CODy[:, k] / dy[0]

CODxx = np.zeros((1000, nImagens))
CODyy = np.zeros((1000, nImagens))
STRAINyy = np.zeros((1000, nImagens))
X = np.zeros((2, 1000, nImagens))
Y = np.zeros((2, 1000, nImagens))

MEANd=np.zeros(nImagens)
MEANs=np.zeros(nImagens)
aid=np.zeros(nImagens, dtype=int)
ad=np.zeros(nImagens)
ais=np.zeros(nImagens)
aas=np.zeros(nImagens)

for k in range(nImagens):
    CODxx[:, k] = np.linspace(CODx[0, k], CODx[-1, k], 1000)
    #same than CODx but 1000 values whereas 201
    CODyy[:, k] = np.interp(CODxx[:, k], CODx[:, k], CODy[:, k])
    #same
    STRAINyy[:, k] = np.interp(CODxx[:, k], CODx[:, k], STRAINy[:, k])

    X[0, :, k] = np.linspace(Xm[0, 0, k], Xm[0, -1, k], 1000)
    X[1, :, k] = np.linspace(Xm[1, 0, k], Xm[1, -1, k], 1000)
    Y[0, :, k] = np.interp(X[0, :, k], Xm[0, :, k], Ym[0, :, k])
    Y[1, :, k] = np.interp(X[1, :, k], Xm[1, :, k], Ym[1, :, k])
    #put all the variables with 1000 values

# trouver l'indice de la valeur la plus proche
indice_plus_prochea1 = int(np.abs(CODxx[0:1000, 1] - a1).argmin())
indice_plus_procheaf = int(np.abs(CODxx[0:1000, -1] - af).argmin())
#a1 et af good!

# Entrée des coefficients du système
a11 = np.mean(CODyy[:, -1])*(nImagens)#VDmeanf*if
a12 = np.mean(CODyy[:, -1])#VDmeanf
b1 = CODyy[indice_plus_procheaf, -1]#VDthf
a21 = np.mean(CODyy[:, 1])#VDmean1*i1
a22 = np.mean(CODyy[:, 1])#VDmean1
b2 = CODyy[indice_plus_prochea1, 1]#VDth1

# Application de la méthode d'élimination de Gauss
coeff = a21/a11
a22 = a22 - coeff*a12
b2 = b2 - coeff*b1
x2 = b2/a22
x1 = (b1 - a12*x2)/a11

# Affichage des résultats
print("La solution du système est :")
print("x1 = ", x1)
print("x2 = ", x2)

for k in range(nImagens):
    #aa1 = (ina - inb) / (1 - nImagens) #alpha and beta parameters!
    #bb1 = ina - aa1
    aa=x1
    bb=x2

    MEANd[k] = np.mean(CODyy[:, k]) * (aa * k + bb)#VDth
    MEANs[k] = np.mean(STRAINyy[:, k]) * (aa * k + bb)#VDth step
    #MEANs[k] = np.mean(CODyy[:, k]) * (aa1 * k + bb1)

    for i in range(1000):
        if CODyy[i, k] - MEANd[k] < 0.02: # if VD-VDth<0.02
            if CODxx[i, k] < CTODimage * Cal:#if 
                aid[k] = i
                ad[k] = CTODimage * Cal
            else:
                aid[k] = i
                ad[k] = CODxx[i, k]
            break
    
    for i in range(1000):
        if STRAINyy[i, k] - MEANs[k] < 0.02:
        #if CODyy[i, k] - MEANs[k] < 0.02:    
            if CODxx[i, k] < CTODimage * Cal:
                ais[k] = i
                aas[k] = CTODimage * Cal
            else:
                ais[k] = i
                aas[k] = CODxx[i, k]
            break

Md= np.zeros((nImagens)) 
Ms= np.zeros((nImagens))

Md = np.sort(np.unique(aid))
CTODid = int(Md[2])

Ms = np.sort(np.unique(ais))
CTODis = int(Ms[2])

CTOD=np.zeros((nImagens))
CTOA=np.zeros((nImagens))
tang1=np.zeros((41, nImagens))
tang2=np.zeros((41, nImagens))

for k in range(nImagens):
    CTOD[k] = CODyy[CTODid, k]
    dyy1 = (Y[0, CTODid+5, k] - Y[0, CTODid-5, k]) / (X[0, CTODid+5, k] - X[0, CTODid-5, k])
    dyy2 = (Y[1, CTODid+5, k] - Y[1, CTODid-5, k]) / (X[1, CTODid+5, k] - X[1, CTODid-5, k])
    tang1[:,k] = ((np.arange(-20,21)-X[0,CTODid,k])*dyy1) + Y[0,CTODid,k]
    tang2[:,k] = ((np.arange(-20,21)-X[1,CTODid,k])*dyy2) + Y[1,CTODid,k]
    CTOA[k] = np.arctan(dyy1) - np.arctan(dyy2)


for k in range(1, nImagens):
    plt.plot(CODxx[0:1000, k], CODyy[:, k], 'b-')
    plt.plot([0, 35], [MEANd[k], MEANd[k]], 'r-')
    plt.plot(ad[k], CODyy[int(aid[k]), k], 'gx')
    plt.xlabel('x11 [mm]', fontname='Times New Roman')
    plt.ylabel('COD [mm]', fontname='Times New Roman')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().xaxis.set_tick_params(width=0.5)
    plt.gca().yaxis.set_tick_params(width=0.5)
    plt.gca().set_xlim([4, 11])
    plt.gca().set_ylim([0, 8])
    plt.grid(False)
plt.show()

#part3
'''
for k in range(0, nImagens, 2):
    plt.imshow(I[:, :, k])
    
    # Ajouter les éléments supplémentaires
    plt.plot([ad[0]/Cal, ad[0]/Cal], [0, 1000], color=[0, 1, 0, 0.5], linewidth=2)
    plt.plot([ad[k]/Cal, ad[k]/Cal], [0, 1000], color='green', linewidth=2)
    plt.plot(np.arange(-20, 21)/Cal, tang1[:, k]/Cal, color='blue', linewidth=2)
    plt.plot(np.arange(-20, 21)/Cal, tang2[:, k]/Cal, color='blue', linewidth=2)
    plt.plot(X[0, range(0, 1000, 50), k]/Cal, Y[0, range(0, 1000, 50), k]/Cal, 'x', color='red', markersize=8, linewidth=2)
    plt.plot(X[1, range(0, 1000, 50), k]/Cal, Y[1, range(0, 1000, 50), k]/Cal, 'x', color='red', markersize=8, linewidth=2)
    plt.plot(X[0, CTODid, k]/Cal, Y[0, CTODid, k]/Cal, '.', color='blue', markersize=20, linewidth=2)
    plt.plot(X[1, CTODid, k]/Cal, Y[1, CTODid, k]/Cal, '.', color='blue', markersize=20, linewidth=2)
    plt.axis('equal')
    plt.axis([0, I.shape[1], 350, I.shape[0]-100])
    plt.axis('off')
    plt.grid(False)
    plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(k)+".png")
    plt.show()
    
path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
files = os.listdir(path)
files.sort()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(path+'\COD.avi', fourcc, 10.0, (640, 480))    
for j in range(0, nImagens, 2):
    img = cv2.imread(os.path.join(path, "Img"+str(j)+".png"))
    img = cv2.resize(img, (640, 480))
    output.write(img)
    os.remove(os.path.join(path, "Img"+str(j)+".png"))
output.release()
cv2.destroyAllWindows()
    
x = []
y = []
for k in range(0, nImagens, 2):
    plt.plot(CODxx[0:1000, 0], CODyy[:, 0], 'b-', label='VD')
    plt.plot([0, 35], [MEANd[0], MEANd[0]], 'r-',label='VDth')
    plt.plot(ad[0], CODyy[aid[0], 0], 'gx', label='Crack tip')
    plt.plot(CODxx[0:1000, 0:k], CODyy[:, 0:k], 'b-')
    plt.plot([0, 35], [MEANd[0:k], MEANd[0:k]], 'r-')
    
    x.append(ad[k])
    y.append(CODyy[aid[k], k])
    
    plt.plot(x, y, 'gx')
    # set the font and size for the axes and legend
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12)
    # set the axis limits and turn on the box
    plt.gca().set_xlim([4, 11])
    plt.gca().set_ylim([0, 8])
    # turn off the grid and set the background color of the plot
    plt.grid(False)
    plt.box(True)
    # display the plot
    plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(k)+".png")
    plt.show()

path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
files = os.listdir(path)
files.sort()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(path+'\VD.avi', fourcc, 10.0, (640, 480))    
for j in range(0, nImagens, 2):
    img = cv2.imread(os.path.join(path, "Img"+str(j)+".png"))
    img = cv2.resize(img, (640, 480))
    output.write(img)
    os.remove(os.path.join(path, "Img"+str(j)+".png"))
output.release()
cv2.destroyAllWindows()
'''
dad = ad - ad[0]
das = aas - aas[0]

#Plot crack tip vs Images
plt.plot(range(1,nImagens+1), dad, 'o')
plt.plot(range(1,nImagens+1), das, 'x')
plt.xlabel('Images')
plt.ylabel('Crack tip [mm]')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
plt.grid()
plt.box(True)

# Plot CTOD vs Images
fig, ax = plt.subplots()
ax.plot(range(1, nImagens+1), CTOD)
ax.set_xlabel('Images')
ax.set_ylabel('CTOD [mm]')
ax.tick_params(labelsize=16)
ax.set_facecolor('white')
ax.set_box_aspect(1)
ax.grid()
plt.show()

# Plot CTOA vs Images
fig, ax = plt.subplots()
ax.plot(range(1, nImagens+1), CTOA)
ax.set_xlabel('Images')
ax.set_ylabel('CTOA [º]')
ax.tick_params(labelsize=16)
ax.set_facecolor('white')
ax.set_box_aspect(1)
ax.grid()
plt.show()

# Plot CTOA vs CTOD
fig, ax = plt.subplots()
ax.plot(CTOD, CTOA)
ax.set_xlabel('CTOD [mm]')
ax.set_ylabel('CTOA [º]')
ax.tick_params(labelsize=16)
ax.set_facecolor('white')
ax.set_box_aspect(1)
ax.grid()
plt.show()


np.savetxt('das.txt', das)
np.savetxt('dad.txt', dad)
np.savetxt('CTOD.txt', CTOD)
np.savetxt('CTOA.txt', CTOA)


run=0
if run==1:
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video"
    cap1 = cv2.VideoCapture(path+'\COD.avi')
    cap2 = cv2.VideoCapture(path+'\VD.avi')
    
    # Récupérer les dimensions de la vidéo
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Créer un objet VideoWriter pour écrire la vidéo combinée
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    combined_video = cv2.VideoWriter(path+'\combined_video.mp4', fourcc, 25.0, (2*width, height))
    
    # Boucle pour lire les images de chaque vidéo et les combiner
    while True:
        # Lire les images des 2 vidéos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
    
        # Vérifier si toutes les vidéos ont été lues
        if not ret1 or not ret2:
            break
    
        # Redimensionner les images à la même taille
        frame1 = cv2.resize(frame1, (width, height))
        frame2 = cv2.resize(frame2, (width, height))
    
        # Combiner les 2 images en une seule
        combined_frame = cv2.hconcat([frame1, frame2])
    
        # Écrire la frame combinée dans la vidéo
        combined_video.write(combined_frame)
        
    # Fermer toutes les fenêtres et libérer les ressources
    cap1.release()
    cap2.release()
    combined_video.release()
    cv2.destroyAllWindows()

 
'''
#print(ad[-1]-ad[0])
print(ad[-1])
print(MEANd[-1])
print(np.mean(CODyy[:, -1]))
print(CODyy[int(aid[-1]), -1])   

print(ad[1])
print(MEANd[1])
print(np.mean(CODyy[:, 1]))
print(CODyy[int(aid[1]), 1]) 

print(np.mean(CODyy[:, -1])*(nImagens))
print(np.mean(CODyy[:, -1]))
print(CODyy[indice_plus_procheaf, -1])
print(np.mean(CODyy[:, 1]))
print(np.mean(CODyy[:, 1]))
print(CODyy[indice_plus_prochea1, 1])
'''
