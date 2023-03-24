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
Job = 'e1o1'

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

Ubot  = np.zeros((a0.X, MatchID.stages))
Utop=  np.zeros((a0.X, MatchID.stages))
Vbot=  np.zeros((a0.X, MatchID.stages)) 
Vtop=  np.zeros((a0.X, MatchID.stages))                

for i in np.arange(0, MatchID.stages, 1):
    Ubot[:,i]=UX[a0.Y-COD.cod_pair,0:a0.X,i]
    Utop[:,i]=UX[a0.Y+COD.cod_pair,0:a0.X,i]
    Vbot[:,i]=UY[a0.Y-COD.cod_pair,0:a0.X,i]
    Vtop[:,i]=UY[a0.Y+COD.cod_pair,0:a0.X,i]

Xm = np.zeros((2, a0.X, MatchID.stages))
Ym = np.zeros((2, a0.X, MatchID.stages))

for i in np.arange(0, 2, 1):
    Xm[0,:,:]=Ubot[:,:]
    Xm[1,:,:]=Utop[:,:]
    
#############################################################################

# INITIALIZATION:
Valpixel = 391         # Number of pixels to perform calibration;
Valmm = 9.94           # Amount of mm to perform calibration;
Cal = Valmm/Valpixel   # Calibration factor from pixel to mm;

ina = 0.75
inb = 0.75
CTODimage = 281


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
img = Image.open(cwd + '\e1O1_0002_0.tiff')
# Obtenir la taille de l'image
largeur, hauteur = img.size
# Afficher la taille de l'image
print("La taille de l'image est de {} x {} pixels.".format(largeur, hauteur))

I = np.zeros((int(hauteur/4), int(largeur/4), nImagens))

for k, fileName in enumerate(fileNames):
    I[:, :, k] = cv.resize(cv.imread(os.path.join(endS, fileName), cv.IMREAD_GRAYSCALE), (int(largeur/4), int(hauteur/4)))

os.chdir('..')

#for k in range(0, nImagens, 2):
    #plt.imshow(I[:, :, k])
    #plt.show()

CODy = np.zeros((a0.X, MatchID.stages))
CODx = np.zeros((a0.X, MatchID.stages))

for k in range(MatchID.stages):
    CODy[:, k] = np.abs(Vtop[:,k]-Vbot[:,k])
    CODx[:, k] = (Ubot[:,i]+Utop[:,i])/2
    
dx = Xm[0, 1, 0] - Xm[0, 0, 0]
dy = CODy[:, 0]

CODy = np.abs(CODy - CODy[:, [0]])

STRAINy = np.zeros((a0.X, MatchID.stages))

for k in range(MatchID.stages):
    STRAINy[:, k] = CODy[:, k] / dy[0]
#attention indice dans l'autre sens!!

CODxx = np.zeros((1000, MatchID.stages))
CODyy = np.zeros((1000, MatchID.stages))
STRAINyy = np.zeros((1000, MatchID.stages))
X = np.zeros((2, 1000, MatchID.stages))
Y = np.zeros((2, 1000, MatchID.stages))

MEANd=np.zeros(MatchID.stages)
MEANs=np.zeros(MatchID.stages)
aid=np.zeros(MatchID.stages, dtype=int)
ad=np.zeros(MatchID.stages)
ais=np.zeros(MatchID.stages)
aas=np.zeros(MatchID.stages)

for k in range(MatchID.stages):
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

    aa = (ina - inb) / (1 - nImagens) #alpha and beta parameters!
    bb = ina - aa

    MEANd[k] = np.mean(CODyy[:, k]) * (aa * k + bb)
    MEANs[k] = np.mean(STRAINyy[:, k]) * (aa * k + bb)

    for i in range(1000):
        if CODyy[i, k] - MEANd[k] < 0.02:
            if CODxx[i, k] < CTODimage * Cal:
                aid[k] = i
                ad[k] = CTODimage * Cal
            else:
                aid[k] = i
                ad[k] = CODxx[i, k]
            break

    for i in range(1000):
        if STRAINyy[i, k] - MEANs[k] < 0.02:
            if CODxx[i, k] < CTODimage * Cal:
                ais[k] = i
                aas[k] = CTODimage * Cal
            else:
                ais[k] = i
                aas[k] = CODxx[i, k]
            break

Md= np.zeros((MatchID.stages)) 
Ms= np.zeros((MatchID.stages))

Md = np.sort(np.unique(aid))
CTODid = int(Md[2])

Ms = np.sort(np.unique(ais))
CTODis = int(Ms[2])

CTOD=np.zeros((MatchID.stages))
CTOA=np.zeros((MatchID.stages))
tang1=np.zeros((41, MatchID.stages))
tang2=np.zeros((41, MatchID.stages))

for k in range(MatchID.stages):
    CTOD[k] = CODyy[CTODid, k]
    dyy1 = (Y[0, CTODid+5, k] - Y[0, CTODid-5, k]) / (X[0, CTODid+5, k] - X[0, CTODid-5, k])
    dyy2 = (Y[1, CTODid+5, k] - Y[1, CTODid-5, k]) / (X[1, CTODid+5, k] - X[1, CTODid-5, k])
    tang1[:,k] = ((np.arange(-20,21)-X[0,CTODid,k])*dyy1) + Y[0,CTODid,k]
    tang2[:,k] = ((np.arange(-20,21)-X[1,CTODid,k])*dyy2) + Y[1,CTODid,k]
    CTOA[k] = np.arctan(dyy1) - np.arctan(dyy2)
