import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import pandas as pd

plt.rcParams['axes.facecolor'] = (1, 1, 1)
plt.rcParams['figure.facecolor'] = (1, 1, 1)
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 18
plt.rcParams['image.cmap'] = 'pink' # 'afmhot', 'hot', 'gist_heat'
# plt.rcParams['text.usetex'] = True
params = {"ytick.color" : (0, 0, 0),
          "xtick.color" : (0, 0, 0),
          "grid.color" : (.8, .8, .8),
          "text.color" : (.7, .7, .7),
          "axes.labelcolor" : (0, 0, 0),
          "axes.edgecolor" : (0, 0, 0)}
plt.rcParams.update(params)

###  USER #####################################################################
Job = 'e2e2'

runuser = 'Olivier'
if runuser == 'Xavier':
    maincwd = "/home/slimbook/Documents/GitHub/OlivierLouisMatthieu/PRD/MMCGTests"
elif runuser == 'Olivier':
    maincwd = "D:\Recherche PRD\EXP\MMCGTests"

cwd = os.path.join(maincwd, Job)

###############################################################################


print('working directory: ', cwd)
# data structure, units: SI: Pa, N, m
print('starting structured variables..')
# anonymous class
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)

MatchID = Struct()
a0 = Struct()
COD = Struct()
Test = Struct()

# run database with data from the project/specimen
exec(open('Database.py').read())


if a1==0 or af==0 or nombre==0:
    exec(open('ReadcrackfractureMMCG.py').read())

#%% Read P-d curve data

print('reading : load and displacement from the test machine')

# read load file:

pathdados = os.path.join(cwd, Job + '_load.csv')
test = pd.read_csv(pathdados, delimiter=";", decimal=".", names=['Time', 'Load', 'Displ'])

Time = test.Time.values.astype(float)
Time = Time - Time[0]
incTime = int(1/Time[1])

Displ = test.Displ.values.astype(float)*Test.DisplConvFactor # unit: mm
Displ = Displ - Displ[0] + ddeplac
Load = test.Load.values.astype(float)*Test.LoadConvFactor # unit: N
if Load[0] < 0:  # unit: N
    Load = Load - Load[0] + Test.meanPreLoad

##########################################

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Displ, Load, 'k-', linewidth=3)
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.show()
#plot the shift data

#%% Read matchid DIC data

pathdados = os.path.join(cwd,'X[Pixels]',Job+'_0001_0.tiff_X[Pixels].csv')
MatchID.x_pic = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
MatchID.x_pic = MatchID.x_pic[:,0:-1]
MatchID.xCoord = MatchID.x_pic[0,:]
#take just the first line
pathdados = os.path.join(cwd,'Y[Pixels]',Job+'_0001_0.tiff_Y[Pixels].csv')
MatchID.y_pic = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
MatchID.y_pic = MatchID.y_pic[:,0:-1]
MatchID.yCoord = MatchID.y_pic[:,0]

# Area of Interest
MatchID.SubsetXi, MatchID.SubsetYi = int(MatchID.xCoord[0]), int(MatchID.yCoord[0])
MatchID.SubsetXf, MatchID.SubsetYf = int(MatchID.xCoord[-1]), int(MatchID.yCoord[-1])

# determining the number of stages by inspecting MatchID processing files
MatchID.stages = len(os.listdir(os.path.join(cwd,'U')))
MatchID.time = np.arange(1, MatchID.stages+1, 1)

auxD, auxL = [], []
for i in MatchID.time:
    aux = Time - MatchID.time[i-1]
    idx = np.argwhere(np.abs(aux) == np.min(np.abs(aux)))
    auxD.append(float(Displ[idx[0]]))
    auxL.append(float(Load[idx[0]]))

MatchID.displ = np.array(auxD)
MatchID.load = np.array(auxL)

print('Number of stages: ', str(MatchID.stages))

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Displ, Load, 'k-', linewidth=3)
plt.plot(MatchID.displ, MatchID.load, '+r',markersize=1)
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.show()

# Read results "....tif_#.csv" into 3D np.array

MatchID.SubsetsX = MatchID.x_pic.shape[1]
#.shape[1] for nb of column
MatchID.SubsetsY = MatchID.x_pic.shape[0]

# U displacement
UX = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
for i in np.arange(0, MatchID.stages, 1):
    readstr = Job+'_%04d_0.tiff_U.csv' % int(i+1)
    #print('reading : ',readstr)
    pathdados = os.path.join(cwd,'U',readstr)
    aux = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
    xend, yend = aux.shape[1], aux.shape[0]
    UX[0:yend, 0:xend-1, i] = aux[:, :-1]*Test.mm2pixel # unit: mm
# print(f'{toc():.1f} seg')

# V displacement
UY = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
# tic()
for i in np.arange(0, MatchID.stages, 1):
    readstr = Job+'_%04d_0.tiff_V.csv' % int(i+1)
    #print('reading : ',readstr)
    pathdados = os.path.join(cwd,'V',readstr)
    aux = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
    xend, yend = aux.shape[1], aux.shape[0]
    UY[0:yend, 0:xend-1, i] = aux[:, :-1]*Test.mm2pixel # unit: mm
# print(f'{toc():.1f} seg')

#%% Selecting a0
print('Selecting the subset closest to the initial crack tip..')

a0.X = int(np.argwhere(np.abs(MatchID.xCoord - a0.imgHuse) == 0))
a0.Y = int(np.argwhere(np.abs(MatchID.yCoord - a0.imgVuse) == 0))

pathdados = os.path.join(cwd,Job+'_0000_0.tiff')
img0 = cv.imread(pathdados, cv.IMREAD_GRAYSCALE) # cv.imread(pathdados, 0)
dpi = plt.rcParams['figure.dpi']
Height, Width = img0.shape

# What size does the figure need to be in inches to fit the image?
print('plotting: image + roi + subsets mesh..')

figsize = Width/float(dpi), Height/float(dpi)
fig = plt.figure(figsize=figsize)
cor = (255, 255, 255)
thickness = 1
start_point = (MatchID.SubsetXi,MatchID.SubsetYi)
end_point = (MatchID.SubsetXf,MatchID.SubsetYf)
img0 = cv.rectangle(img0, start_point, end_point, cor, thickness)
plt.imshow(img0, cmap='gray', vmin=0, vmax=255)
plt.plot(a0.imgHuse,a0.imgVuse, color='red', marker='+', markersize=50)
plt.show()

#%% Computing CTOD

# Uup, Udown, ||Uup-Udown||
CTODI  = np.zeros((ud_lim, 3, MatchID.stages))
# Vup, Vdown, ||Vup-Vdown||
CTODII = np.zeros((ud_lim, 3, MatchID.stages))

for J in np.arange(0, MatchID.stages, 1):
    # mode II:
    uXtemp = np.copy(UX[:, :, J])
    CTODII[:, 0, J] = np.flipud(uXtemp[a0.Y - ud_lim: a0.Y, a0.X])
    CTODII[:, 1, J] = uXtemp[a0.Y: a0.Y + ud_lim, a0.X]
    CTODII[:, 2, J] = np.abs(CTODII[:, 1, J] - CTODII[:, 0, J])
    # mode I:
    uYtemp = np.copy(UY[:, :, J])
    CTODI[:, 0, J] = np.flipud(uYtemp[a0.Y - ud_lim: a0.Y, a0.X])
    CTODI[:, 1, J] = uYtemp[a0.Y: a0.Y + ud_lim, a0.X]
    CTODI[:, 2, J] = np.abs(CTODI[:, 1, J] - CTODI[:, 0, J])

COD.wI = CTODI[COD.cod_pair, 2, :]
COD.wII = CTODII[COD.cod_pair, 2, :]

ud_limitup = COD.cod_pair + 1
ud_limitdown = COD.cod_pair - 1

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(MatchID.displ, CTODI[ud_lim-1, 2, :], 'g.', linewidth=1, label='COD pair : %d'%ud_lim)
plt.plot(MatchID.displ, COD.wI, 'b-', linewidth=4, label='Mode I with COD pair : %d' %COD.cod_pair) #chosen among 10 one by the user
plt.plot(MatchID.displ, CTODI[ud_limitup, 2, :], 'r--', linewidth=1, label='COD pair : %d' %ud_limitup)
plt.plot(MatchID.displ, CTODI[ud_limitdown, 2, :], 'k+', linewidth=2, label='COD pair : %d' %ud_limitdown)
plt.plot(MatchID.displ, COD.wII, 'k--', label='Mode II with COD pair : %d' %COD.cod_pair)
plt.xlabel('Displacement, mm')
plt.ylabel('CTOD, mm')
ax.set_xlim(xmin=0)
ax.set_ylim(bottom=0)
plt.grid()
plt.legend(loc=2, prop={'size': 8})
fig.tight_layout()
plt.show()
    
#%% Method2

#ina = 0.75
#inb = 0.53
CTODimage = MatchID.xCoord[a0.X]
print(CTODimage*Test.mm2pixel)

endS = os.path.join(os.getcwd(), cwd)

# READING THE FILES:
os.chdir(endS)

fileNames = sorted([file for file in os.listdir() if file.endswith('.tiff')])
pattern = re.compile(r'\d+')
# Utiliser sorted() pour trier la liste en utilisant les nombres extraits des noms de fichier
fileNames = sorted(fileNames, key=lambda x: int(pattern.findall(x)[0]))
# Afficher la liste triée
nImagens = len(fileNames)


# Charger l'image
cwd = os.path.join(cwd,Job+'_0001_0.tiff')
img = Image.open(cwd)
# Obtenir la taille de l'image
largeur, hauteur = img.size
# Afficher la taille de l'image
print("La taille de l'image est de {} x {} pixels.".format(largeur, hauteur))

'''
I = np.zeros((int(hauteur/8), int(largeur/8), nImagens))

for k, fileName in enumerate(fileNames):
    I[:, :, k] = cv.resize(cv.imread(os.path.join(endS, fileName), cv.IMREAD_GRAYSCALE), (int(largeur/8), int(hauteur/8)))

os.chdir('..')


for k in range(0, nImagens, 2):
    plt.imshow(I[:, :, k])
    plt.show()
'''

COD.cod_pair=  COD.cod_pair+20  #we are moving away from the fracture
Xm = np.zeros((2, a0.X+1, nombre))
Ym = np.zeros((2, a0.X+1, nombre))  
Xm[0,:,0]=MatchID.xCoord[0:a0.X+1]*Test.mm2pixel
Xm[1,:,0]=MatchID.xCoord[0:a0.X+1]*Test.mm2pixel  
Ym[0,:,0]=MatchID.yCoord[a0.Y-COD.cod_pair]*Test.mm2pixel        
Ym[1,:,0]=MatchID.yCoord[a0.Y+COD.cod_pair]*Test.mm2pixel  #Xm and Ym for stage 0  

for i in np.arange(0, nombre-1, 1):
    Xm[0,:,i+1]=UX[a0.Y-COD.cod_pair,0:a0.X+1,i]+MatchID.xCoord[0:a0.X+1]*Test.mm2pixel
    Xm[1,:,i+1]=UX[a0.Y+COD.cod_pair,0:a0.X+1,i]+MatchID.xCoord[0:a0.X+1]*Test.mm2pixel
    Ym[0,:,i+1]=-np.abs(UY[a0.Y-COD.cod_pair,0:a0.X+1,i]-UY[a0.Y+COD.cod_pair,0:a0.X+1,i])/2+MatchID.yCoord[a0.Y-COD.cod_pair]*Test.mm2pixel
    Ym[1,:,i+1]=np.abs(UY[a0.Y-COD.cod_pair,0:a0.X+1,i]-UY[a0.Y+COD.cod_pair,0:a0.X+1,i])/2+MatchID.yCoord[a0.Y+COD.cod_pair]*Test.mm2pixel

CODy = np.zeros((a0.X+1, nombre))
CODx = np.zeros((a0.X+1, nombre))

for k in range(nombre):
    CODy[:, k] = np.abs(Ym[1,:,k]-Ym[0,:,k]) #COD
    CODx[:, k] = (Xm[0,:,k]+Xm[1,:,k])/2
    #Coordinates of each subset in function of time
   
dx = Xm[0, 1, 0] - Xm[0, 0, 0]
dy = CODy[:, 0]

CODy = np.abs(CODy - CODy[:, [0]])


STRAINy = np.zeros((a0.X+1, nombre))

for k in range(nombre):
    STRAINy[:, k] = CODy[:, k] / dy[0]
#attention indice dans l'autre sens!!

CODxx = np.zeros((1000, nombre))
CODyy = np.zeros((1000, nombre))
STRAINyy = np.zeros((1000, nombre))
X = np.zeros((2, 1000, nombre))
Y = np.zeros((2, 1000, nombre))

MEANd=np.zeros(nombre)
MEANs=np.zeros(nombre)
aid=np.zeros(nombre, dtype=int)
ad=np.zeros(nombre)
ais=np.zeros(nombre)
aas=np.zeros(nombre)

for k in range(nombre):
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

# Entrée des coefficients du système
a11 = np.nanmean(CODyy[:, nombre-1])*(nombre-1)#VDmeanf*if
a12 = np.nanmean(CODyy[:, nombre-1])#VDmeanf
b1 = CODyy[indice_plus_procheaf, nombre-1]#VDthf
a21 = np.nanmean(CODyy[:, 1])#VDmean1*i1
a22 = np.nanmean(CODyy[:, 1])#VDmean1
b2 = CODyy[indice_plus_prochea1, 1]#VDth1

print(CODyy[indice_plus_prochea1, 1])
print(CODyy[indice_plus_procheaf, -1])

# Application de la méthode d'élimination de Gauss
coeff = a21/a11
a22 = a22 - coeff*a12
b2 = b2 - coeff*b1
x2 = b2/a22
x1 = (b1 - a12*x2)/a11

# Affichage des résultats
print("The system solution is :")
print("x1 = ", x1)
print("x2 = ", x2)



#aa = (ina - inb) / (1 - nImagens) #alpha and beta parameters!
#bb = ina - aa
aa=x1
bb=x2
MEANd = np.linspace(CODyy[indice_plus_prochea1, 1], CODyy[indice_plus_procheaf, nombre-1], nombre)
mean=np.zeros(nombre)

for k in range(nombre):

    MEANd[k] = np.nanmean(CODyy[:, k]) * (aa * k + bb)
    MEANs[k] = np.nanmean(STRAINyy[:, k]) * (aa * k + bb)
    mean[k]=np.nanmean(CODyy[:, k])

    
    a=int(np.abs(CODyy[0:1000, k] - MEANd[k]).argmin())
    aid[k] = a
    if CODxx[a, k] > CTODimage * Test.mm2pixel:
        ad[k] = CTODimage * Test.mm2pixel
    else:
        ad[k] = CODxx[a, k]
        
    
    b=int(np.abs(STRAINyy[0:1000, k] - MEANs[k]).argmin())
    ais[k] = b
    ais[0] = 999
    if CODxx[a, k] > CTODimage * Test.mm2pixel:
        aas[k] = CTODimage * Test.mm2pixel
    else:
        aas[k] = CODxx[b, k]
        
aid[0]=999 
ad[0]=CODxx[999, k]    
ais[0]=999 
aas[0]=CODxx[999, k]   

for k in range(1, nombre,4):
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
    plt.gca().set_xlim([0, 35])
    plt.gca().set_ylim([0, 1])
    plt.grid(False)
plt.show()  
 
x = []
y = []
for k in range(0, nombre, 4):
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
    plt.gca().set_xlim([0, 40])
    plt.gca().set_ylim([0, 1])
    # turn off the grid and set the background color of the plot
    plt.grid(False)
    plt.box(True)
    # display the plot
    plt.show()
 
ad.sort()
dad = ad - ad[0]+Test.a0 
aas.sort()  
das = aas - aas[0]+Test.a0
plt.plot(range(1,nombre+1), dad, 'o')
plt.plot(range(1,nombre+1), das, 'b')

#aid=sorted(aid, reverse=True)
#ais=sorted(ais, reverse=True)
'''
Md= np.zeros((nombre)) 
Ms= np.zeros((nombre))

Md = np.sort(np.unique(aid))
CTODid = int(Md[2])

Ms = np.sort(np.unique(ais))
CTODis = int(Ms[2])

CTOD=np.zeros((nombre))
CTOA=np.zeros((nombre))
tang1=np.zeros((41, nombre))
tang2=np.zeros((41, nombre))

for k in range(nombre):
    CTOD[k] = CODyy[CTODid, k]
    dyy1 = (Y[0, CTODid+5, k] - Y[0, CTODid-5, k]) / (X[0, CTODid+5, k] - X[0, CTODid-5, k])
    dyy2 = (Y[1, CTODid+5, k] - Y[1, CTODid-5, k]) / (X[1, CTODid+5, k] - X[1, CTODid-5, k])
    tang1[:,k] = ((np.arange(-20,21)-X[0,CTODid,k])*dyy1) + Y[0,CTODid,k]
    tang2[:,k] = ((np.arange(-20,21)-X[1,CTODid,k])*dyy2) + Y[1,CTODid,k]
    CTOA[k] = np.arctan(dyy1) - np.arctan(dyy2)


Cal=    Test.mm2pixel*8
for k in range(0, nombre-1, 1):
    plt.imshow(I[:, :, k])
    
    plt.plot([ad[-1]/Cal, ad[-1]/Cal], [0, 1000], color=[0, 1, 0, 0.5], linewidth=2)
    plt.plot([ad[-(1+k)]/Cal, ad[-(1+k)]/Cal], [0, 1000], color='green', linewidth=2)
    plt.plot(X[0, range(0, 1000, 50), k]/Cal, Y[0, range(0, 1000, 50), k]/Cal, 'x', color='red', markersize=8, linewidth=2)
    plt.plot(X[1, range(0, 1000, 50), k]/Cal, Y[1, range(0, 1000, 50), k]/Cal, 'x', color='red', markersize=8, linewidth=2)
    #plt.gca().set_xlim([0, 2200])
    plt.gca().set_ylim([0, int(hauteur/8)])
    plt.show()


#Plot crack tip vs Images
plt.plot(range(1,nombre+1), dad, 'o')
plt.plot(range(1,nombre+1), das, 'x')
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
ax.plot(range(1, nombre+1), CTOD)
ax.set_xlabel('Images')
ax.set_ylabel('CTOD [mm]')
ax.tick_params(labelsize=16)
ax.set_facecolor('white')
ax.set_box_aspect(1)
ax.grid()
plt.show()
'''
    
'''
print(a1+Test.a0)
print(af+Test.a0)
indicetestif = int(np.abs(CODyy[:, nombre-1] - CODyy[indice_plus_procheaf, nombre-1]).argmin())
print(indicetestif)
print(CODxx[indicetestif, nombre-1]+Test.a0)
indicetesti1 = int(np.abs(CODyy[:, 1] - CODyy[indice_plus_prochea1, 1]).argmin())
print(indicetesti1)
print(CODxx[indicetesti1,1]+Test.a0)

print(a1)
print(MatchID.xCoord[a0.X]*Test.mm2pixel)
print(MatchID.xCoord[-1]*Test.mm2pixel)
print(af)
a1=MatchID.xCoord[a0.X]*Test.mm2pixel-a1
af=MatchID.xCoord[a0.X]*Test.mm2pixel-a1

Xm = Xm[:,::-1,:]
Ym=Ym[:,::-1,:]
#CODy = CODy[::-1,:]

    for i in range(1000):
        if CODyy[i, k] - MEANd[k] < 0.02:
            if CODxx[i, k] < CTODimage * Test.mm2pixel:
                aid[k] = i
                ad[k] = CTODimage * Test.mm2pixel
            else:
                aid[k] = i
                ad[k] = CODxx[i, k]
            break

    for i in range(1000):
        if STRAINyy[i, k] - MEANs[k] < 0.02:
            if CODxx[i, k] < CTODimage * Test.mm2pixel:
                ais[k] = i
                aas[k] = CTODimage * Test.mm2pixel
            else:
                ais[k] = i
                aas[k] = CODxx[i, k]
            break
'''
