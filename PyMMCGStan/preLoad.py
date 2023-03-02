#%% Import modules + Database

print('import modules..')
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
# import tkinter as tk
# from PIL import Image, ImageTk
import glob
import time

from ttictoc import tic,toc

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
# Job = 'DCB_002'
Joblist = ['e1o1', 'e1o2', 'e1o3', 'e1p1', 'e1p2', 'e1p3', 'e2e2', 'e2o1',
           'e2o2', 'e2o3', 'e2p2', 'e2p3', 'e3o1', 'e3o2', 'e3o3', 'e3p1',
           'e3p2', 'e3p3', 'e4e1', 'e4i1', 'e4p1', 'e5i1', 'e5o1', 'e5p1']

DisplConvFactor = 1
LoadConvFactor = 1e3

runuser = 'Stanislas'
if runuser == 'Xavier':
    maincwd = "X:\\jxavier\\Orient\\Erasmus\\2021\\Polytech_Clermont-FD\\Stanislas\\EXP\\MMCGTests"
elif runuser == 'Stanislas':
    maincwd = "C:\\Users\\pc\\Desktop\\MMCGTests"

preload = []
for Job in Joblist:
    cwd = os.path.join(maincwd, Job)
    pathdados = os.path.join(cwd, Job + '_load.csv')
    test = pd.read_csv(pathdados, delimiter=";", decimal=".", names=['Time', 'Load', 'Displ'])
    Load = test.Load.values.astype(float)*LoadConvFactor
    print(Job,str(Load[0]))
    if Load[0] > 0:
        preload.append(Load[0])

preload = np.array(preload)
meanPreLoad = np.mean(preload)
print(f'mean Pre load = {meanPreLoad}')

DATA = { }
for Job in Joblist:
    cwd = os.path.join(maincwd, Job)
    pathdados = os.path.join(cwd, Job + '_load.csv')
    test = pd.read_csv(pathdados, delimiter=";", decimal=".", names=['Time', 'Load', 'Displ'])
    Time = test.Time.values.astype(float)
    Time = Time - Time[0]
    Displ = test.Displ.values.astype(float)*DisplConvFactor  # unit: mm
    Displ = Displ - Displ[0]
    Load  = test.Load.values.astype(float)*LoadConvFactor  # unit: N
    if Load[0] < 0:  # unit: N
        Load = Load - Load[0] + meanPreLoad
    RES = np.array([Time, Displ, Load])
    RES = np.transpose(RES)
    DATA.update({Job: RES})

#%% Ploting

xlmax = 5
ylmax = 900

deltadall = 0.1
fig = plt.figure(figsize=(8,6))
for Job in Joblist:
    xp = DATA[Job][:,1]
    yp = DATA[Job][:,2]
    plt.plot(xp+deltadall ,yp, label=Job)
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.grid("on")
plt.legend(loc=2, prop={'size': 8})
plt.tight_layout()
plt.show()

deltad = 0.1
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e1o1"][:,1], DATA["e1o1"][:,2]
plt.plot(xp+deltad+.2,yp, label='e1o1')
xp, yp = DATA["e1o2"][:,1], DATA["e1o2"][:,2]
plt.plot(xp+deltad+.05,yp, label='e1o2')
xp, yp = DATA["e1o3"][:,1], DATA["e1o3"][:,2]
plt.plot(xp+deltad,yp, label='e1o3')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.legend(loc=2, prop={'size': 8})
plt.grid("on")
plt.tight_layout()
plt.show()

deltad = 0.2
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e1p1"][:,1], DATA["e1p1"][:,2]
plt.plot(xp+deltad,yp, label='e1p1')
xp, yp = DATA["e1p2"][:,1], DATA["e1p2"][:,2]
plt.plot(xp+deltad,yp, label='e1p2')
xp, yp = DATA["e1p3"][:,1], DATA["e1p3"][:,2]
plt.plot(xp+deltad,yp, label='e1p3')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.legend(loc=2, prop={'size': 8})
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.grid("on")
plt.tight_layout()
plt.show()

deltad = 0.2
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e2o1"][:,1], DATA["e2o1"][:,2]
plt.plot(xp+deltad,yp, label='e2o1')
xp, yp = DATA["e2o2"][:,1], DATA["e2o2"][:,2]
plt.plot(xp+deltad,yp, label='e2o2')
xp, yp = DATA["e2o3"][:,1], DATA["e2o3"][:,2]
plt.plot(xp+deltad,yp, label='e2o3')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.legend(loc=2, prop={'size': 8})
plt.grid("on")
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.tight_layout()
plt.show()

deltad = 0.1
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e2e2"][:,1], DATA["e2e2"][:,2]
plt.plot(xp+deltad,yp, label='e2p1')
xp, yp = DATA["e2p2"][:,1], DATA["e2p2"][:,2]
plt.plot(xp+deltad,yp, label='e2p2')
xp, yp = DATA["e2p3"][:,1], DATA["e2p3"][:,2]
plt.plot(xp+deltad,yp, label='e2p3')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.grid("on")
plt.legend(loc=2, prop={'size': 8})
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.tight_layout()
plt.show()

deltad = 0.15
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e3o1"][:,1], DATA["e3o1"][:,2]
plt.plot(xp+deltad,yp, label='e3o1')
xp, yp = DATA["e3o2"][:,1], DATA["e3o2"][:,2]
plt.plot(xp+deltad,yp, label='e3o2')
xp, yp = DATA["e3o3"][:,1], DATA["e3o3"][:,2]
plt.plot(xp+deltad,yp, label='e3o3')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.grid("on")
plt.legend(loc=2, prop={'size': 8})
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.tight_layout()
plt.show()

deltad = 0.1
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e3p1"][:,1], DATA["e3p1"][:,2]
plt.plot(xp+deltad,yp, label='e3p1')
xp, yp = DATA["e3p2"][:,1], DATA["e3p2"][:,2]
plt.plot(xp+deltad,yp, label='e3p2')
xp, yp = DATA["e3p3"][:,1], DATA["e3p3"][:,2]
plt.plot(xp+deltad,yp, label='e3p3')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.grid("on")
plt.legend(loc=2, prop={'size': 8})
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.tight_layout()
plt.show()

deltad = 0.1
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e4e1"][:,1], DATA["e4e1"][:,2]
plt.plot(xp+deltad,yp, label='e4o1')
xp, yp = DATA["e4i1"][:,1], DATA["e4i1"][:,2]
plt.plot(xp+deltad,yp, label='e4i1')
xp, yp = DATA["e4p1"][:,1], DATA["e4p1"][:,2]
plt.plot(xp+deltad,yp, label='e4p1')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.grid("on")
plt.legend(loc=2, prop={'size': 8})
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.tight_layout()
plt.show()

deltad = 0.1
fig = plt.figure(figsize=(8,6))
xp, yp = DATA["e5i1"][:,1], DATA["e5i1"][:,2]
plt.plot(xp+deltad,yp, label='e5i1')
xp, yp = DATA["e5o1"][:,1], DATA["e5o1"][:,2]
plt.plot(xp+deltad,yp, label='e5o1')
xp, yp = DATA["e5p1"][:,1], DATA["e5p1"][:,2]
plt.plot(xp+deltad,yp, label='e5p1')
plt.xlabel('Displacement, mm')
plt.ylabel('Load, N')
plt.grid("on")
plt.legend(loc=2, prop={'size': 8})
plt.xlim(0, xlmax)
plt.ylim(0, ylmax)
plt.tight_layout()
plt.show()
