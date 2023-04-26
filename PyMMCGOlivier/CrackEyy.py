#%% Import modules + Database
print('import modules..')
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import csv
import re
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from PIL import Image
from scipy.spatial.distance import euclidean
# import tkinter as tk
# from PIL import Image, ImageTk
import glob
import time
# from ttictoc import tic,toc

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
# cwd = os.getcwd()
#Job = 'DCB_002'
Job = 'e1o1'

runuser = 'Olivier'
if runuser == 'Xavier':
    maincwd = "/home/slimbook/Documents/GitHub/OlivierLouisMatthieu/PRD/MMCGTests"
elif runuser == 'Olivier':
    maincwd = "D:\Recherche PRD\EXP\MMCGTests"

cwd = os.path.join(maincwd, Job)


###############################################################################

#%%
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

# Eyy strain
Eyy = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
for i in np.arange(0, MatchID.stages, 1):
    readstr = Job+'_%04d_0.tiff_Eyy.csv' % int(i+1)
    #print('reading : ',readstr)
    pathdados = os.path.join(cwd,'Eyy',readstr)
    aux = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
    xend, yend = aux.shape[1], aux.shape[0]
    Eyy[0:yend, 0:xend-1, i] = aux[:, :-1]

#%% Selecting a0
print('Selecting the subset closest to the initial crack tip..')

a0.X = int(np.argwhere(np.abs(MatchID.xCoord - a0.imgHuse) == 0))
a0.Y = int(np.argwhere(np.abs(MatchID.yCoord - a0.imgVuse) == 0))

'''
aux = MatchID.load - np.max(MatchID.load)
idx = np.argwhere(np.abs(aux) == np.min(np.abs(aux)))
fig = plt.figure()
plt.imshow(UY[:, :, int(idx[0])])
plt.plot(a0.X,a0.Y,'sr')
plt.colorbar()
plt.show()
'''

#Eyy=Eyy[a0.Y-30:a0.Y+30,0:a0.X,:]

fig = plt.figure()
plt.imshow(Eyy[:, :, 70])
#plt.plot(a0.X,a0.Y,'sr')
plt.colorbar()
plt.show()



# Find crack tip location for each time step
max_Eyy=np.zeros(MatchID.stages)
Eyy_threshold=np.zeros(MatchID.stages)
crack_tip = []
max_points = []

for t in range(MatchID.stages):
    # Step 1: Identify the highest Eyy values in the deformation data
    max_Eyy[t] = np.nanmax(Eyy[:, :, t])
    Eyy_threshold[t] = max_Eyy[t] * 0.1  # Set a threshold for the Eyy values to focus on
    #Eyy_threshold[t]=0.05
    
    # Step 2: Find the points where the Eyy values start to decrease or change direction
    points = np.argwhere(Eyy[:, :, t] >= Eyy_threshold[t])
    max_points.append(np.column_stack((points[:, 1], points[:, 0])))

    # Step 3: Sort the points by Eyy value in descending order
    max_points[t] = max_points[t][max_points[t][:, 1].argsort()[::-1]]

    # Step 4: Iterate through the list of maximum points and find the point where Eyy starts to decrease or change direction
    for i in range(len(max_points[t])):
        x, y = max_points[t][i]
        if t == 0:
            continue  # Ignore the first time step since it may not show the crack tip
        if Eyy[y, x, t] < Eyy[y, x, t-1]:
            crack_tip.append((x, y, t))
            break  # Stop searching once the crack tip is found   
           
'''
# Plot the crack tip location over time
fig, ax = plt.subplots()
im = ax.imshow(Eyy[:, :, 0], cmap='coolwarm')
for i in range(len(crack_tip)):
    x, y, t = crack_tip[i]
    ax.plot(x, y, 'rx', markersize=10)
plt.colorbar(im)
plt.show()
'''

x=np.zeros(len(crack_tip))
y=np.zeros(len(crack_tip))
t=np.zeros(len(crack_tip))
for i in range(len(crack_tip)):
    x[i], y[i], t[i] = crack_tip[i]
   
for i in range(len(crack_tip)-1):
    if x[i]< x[i+1]:
        x[i+1]=x[i]
     
fig = plt.figure()
for i in range(len(crack_tip)):
    plt.imshow(Eyy[:, :, i])
    plt.plot(x[i], y[i], 'rx', markersize=10)
    plt.plot(a0.X,a0.Y,'sr')
    plt.colorbar()
    plt.show()
    
plt.imshow(Eyy[:, :, -5])
plt.plot(x[0:-1], y[0:-1], 'rx', markersize=10)
plt.colorbar()
plt.show()

crackE=np.zeros(len(crack_tip))
for i in range(len(crack_tip)):
    crackE[i]=MatchID.xCoord[int(x[i])]*Test.mm2pixel
crackE=np.abs(crackE[0]-crackE)+Test.a0
#crackE.sort()
plt.plot(t,crackE)

# définir les seuils de gradient
threshold1 = 50
threshold2 = 150

for i in range(MatchID.stages):
    # appliquer un filtre gaussien pour réduire le bruit
    Eyy_blur = cv.GaussianBlur(Eyy[:, :, i], (5,5), 0)
    
    # normaliser les valeurs de Eyy pour qu'elles soient entre 0 et 255
    Eyy_norm = cv.normalize(Eyy_blur, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    
    # détecter les contours avec la méthode de Canny
    edges = cv.Canny(Eyy_norm, threshold1, threshold2)
    
    # afficher les contours détectés
    plt.imshow(edges, cmap='gray')
    plt.title('Contours de fissure détectés'+ str(i))
    plt.show()
