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
import glob
import time


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
Job = 'e30e2'

runuser = 'Olivier'
if runuser == 'Xavier':
    maincwd = "/home/slimbook/Documents/GitHub/OlivierLouisMatthieu/PRD/MMCGTests"
elif runuser == 'Olivier':
    #maincwd = "D:\Recherche PRD\EXP\MMCGTests"
    maincwd = "D:\Recherche PRD\EXP\MMCG_Olivier\Arcan30"

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
exec(open('Database_new.py').read())

#%% Read P-d curve data

print('reading : load and displacement from the test machine')

# read load file:

pathdados = os.path.join(cwd, Job + '_load.csv')
test = pd.read_csv(pathdados, delimiter=";", decimal=",", names=['Time', 'Load', 'Displ'])

Time = test.Time.values.astype(float)
Time = Time - Time[0]
incTime = int(1/Time[1])

Displ = test.Displ.values.astype(float)*Test.DisplConvFactor # unit: mm
#Displ = Displ - Displ[0]
Load = test.Load.values.astype(float)*Test.LoadConvFactor # unit: N
Load = Load - Load[0]
if Load[0]>Load[-1]:
    Load=Load+(Load[0]-np.min(Load))
    

X1 = Displ[0]
X2 = Displ[100]
Y1 = Load[0]
Y2 = Load[100]
a = (Y2 - Y1) / (X2 - X1)
b = Y1 - a * X1
x0=-b/a
print(a)
print(b)
Displ = Displ -x0

##########################################

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(test.Displ.values.astype(float), test.Load.values.astype(float), 'k-', linewidth=3)
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.show()
#plot the raw data

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Displ, Load, 'k-', linewidth=3)
#plt.plot(Displ, p(Displ), 'r-', linewidth=3)
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.show()

#plot the shift data

#%% Read matchid DIC data

pathdados = os.path.join(cwd,'x_pic',Job+'_1.tiff_x_pic.csv')
MatchID.x_pic = np.genfromtxt(pathdados, skip_header=0, delimiter=',')
MatchID.x_pic = MatchID.x_pic[:,0:-1]
MatchID.xCoord = MatchID.x_pic[0,:]
#take just the first line
pathdados = os.path.join(cwd,'y_pic',Job+'_1.tiff_y_pic.csv')
MatchID.y_pic = np.genfromtxt(pathdados, skip_header=0, delimiter=',')
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

curvesupx=[0,MatchID.displ[0]]
curvesupy=[0,MatchID.load[0]]

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(curvesupx, curvesupy, linestyle='dashed', color='red', linewidth=2)
plt.plot(MatchID.displ, MatchID.load, 'k-', linewidth=2)
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.title(Job)
plt.show()

# Read results "....tif_#.csv" into 3D np.array

MatchID.SubsetsX = MatchID.x_pic.shape[1]
#.shape[1] for nb of column
MatchID.SubsetsY = MatchID.x_pic.shape[0]

# U displacement
UX = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
# tic()
for i in np.arange(0, MatchID.stages, 1):
    readstr = Job+'_%01d.tiff_u.csv' % int(i+1)
    #print('reading : ',readstr)
    pathdados = os.path.join(cwd,'u',readstr)
    aux = np.genfromtxt(pathdados, skip_header=0, delimiter=',')
    xend, yend = aux.shape[1], aux.shape[0]
    UX[0:yend, 0:xend-1, i] = aux[:, :-1]*Test.mm2pixel # unit: mm
# print(f'{toc():.1f} seg')

# V displacement
UY = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
# tic()
for i in np.arange(0, MatchID.stages, 1):
    readstr = Job+'_%01d.tiff_v.csv' % int(i+1)
    #print('reading : ',readstr)
    pathdados = os.path.join(cwd,'v',readstr)
    aux = np.genfromtxt(pathdados, skip_header=0, delimiter=',')
    xend, yend = aux.shape[1], aux.shape[0]
    UY[0:yend, 0:xend-1, i] = aux[:, :-1]*Test.mm2pixel # unit: mm
# print(f'{toc():.1f} seg')

#%% Selecting a0
print('Selecting the subset closest to the initial crack tip..')

a0.X = int(np.argwhere(np.abs(MatchID.xCoord - a0.imgHuse) == 0))
a0.Y = int(np.argwhere(np.abs(MatchID.yCoord - a0.imgVuse) == 0))


#print('Selecting the subset closest to the end of the crack tip..')

#af.imgHuse, af.imgVuse = 299, 1013
#af.X = int(np.argwhere(np.abs(MatchID.xCoord - af.imgHuse) == 0))
#af.Y = int(np.argwhere(np.abs(MatchID.yCoord - af.imgVuse) == 0))

#cracklength=np.abs(a0.imgHuse-af.imgHuse)*Test.mm2pixel+Test.a0
#print(cracklength)

pathdados = os.path.join(cwd,Job+'_0.tiff')
img0 = cv.imread(pathdados, cv.IMREAD_GRAYSCALE) # cv.imread(pathdados, 0)
dpi = plt.rcParams['figure.dpi']
Height, Width = img0.shape

# What size does the figure need to be in inches to fit the image?
print('plotting: image + roi + subsets mesh..')
# tic()

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

'''
aux = MatchID.load - np.max(MatchID.load)
idx = np.argwhere(np.abs(aux) == np.min(np.abs(aux)))
fig = plt.figure()
plt.imshow(UY[:, :, int(idx[0])])
plt.plot(a0.X,a0.Y,'sr')
plt.colorbar()
plt.show()

fig = plt.figure()
plt.imshow(UY[:, :, 40])
plt.plot(a0.X,a0.Y,'sr')
plt.colorbar()
plt.show()
'''
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

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(COD.wI, MatchID.load, 'b-', linewidth=4, label='Mode I with COD pair : %d' %COD.cod_pair) #chosen among 10 one by the user
plt.plot(COD.wII, MatchID.load, 'k--', label='Mode II with COD pair : %d' %COD.cod_pair)
plt.ylabel('Load, N')
plt.xlabel('CTOD, mm')
ax.set_xlim(xmin=0)
ax.set_ylim(bottom=0)
plt.grid()
plt.legend(loc=2, prop={'size': 8})
fig.tight_layout()
plt.show()
    

#%% Computing aDIC
print('computing aDIC..')

roi = 'crop' # 'all'; 'crop'
i, incr = 1, 1
# incr : is used to step over stages if required (default = 1: all stages)
Y_i, Y_f = 0, UY.shape[0]
X_i, X_f = 0, a0.X
#donne nombre ligne UY.shape[0] et UY.shape[1] nb colonne
filtro = 'yes' # 'yes'; 'no'

####
#### 1st alpha evaluation :::::::::::::::::::::::::::::::::::::::::::::::::::::
### Selecting stage for investigating alpha from P-d curve
####

# least-squares linear regression
porder = 1
xx = MatchID.displ # displacement (mm)
yy = MatchID.load # load (N)
# Data point in the linear least-squares regression
limsup = int(0.75*np.argwhere(max(yy)==yy)[-1])
# number of maximum data points for LSR
liminf = int(np.round((1/3)*limsup))# number of minimum data points for LSR

xx, yy = xx[0:limsup], yy[0:limsup]
Rtot = np.zeros((limsup-liminf,1))
C_M = np.zeros((limsup-liminf,1))
for j in np.arange(0,limsup-liminf,1):
    limt_sup = liminf + j
    xfit, yfit = xx[0:limt_sup], yy[0:limt_sup]
    p  = np.polyfit(xfit, yfit, porder)
    C_M[j] = 1/p[0]
    dev = yfit - np.mean(yfit) # deviations - measure of spread
    SST = np.sum(dev**2) # total variation to be accounted for
    resid = yfit - np.polyval(p, xfit) # residuals - measure of mismatch
    SSE = np.sum(resid**2) # variation NOT accounted for
    Rtot[j] = 1 - SSE/SST #  variation NOT accounted for

# position for the best fitting point parameters
jmax = np.max(np.argwhere(np.max(Rtot)==Rtot))
J = int(liminf + jmax)

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(MatchID.displ, MatchID.load, 'k-', linewidth=3)
plt.plot(MatchID.displ[liminf:limsup], MatchID.load[liminf:limsup],'r--',linewidth=4)
plt.plot(MatchID.displ[J], MatchID.load[J],'bo', markersize=10)
x = np.linspace(0,MatchID.displ[liminf]*2.2)
y = np.linspace(0,MatchID.load[liminf]*2.2) #why?
plt.plot(x,y,'k--')
plt.xlabel('Displacment, mm')
plt.ylabel('Load, N')
fig.tight_layout()
ax.set_xlim(xmin=0)
ax.set_ylim(bottom=0)
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Rtot, 'k-', linewidth=3)
plt.plot(jmax,Rtot[jmax], 'sr', linewidth=4)
plt.ylabel('Coeff. Correlation R^2')
fig.tight_layout()
plt.grid()
plt.show()

####
#### 2nd alpha evaluation :::::::::::::::::::::::::::::::::::::::::::::::::::::
### 2 criterion for checking stage from displacament field processing
####
inb = 3
# standard deviation * inb (to be checked by user)
# inb = 1: 68.3 %
# inb = 2: 95.4 %
# inb = 3: 99.7 %
JJ = 1
while JJ == 1:
    ### crack location a = a(t) ------------------------------------------
    # transfering the displacements to a new variable
    displ_x = UX[:,:,J]
    displ_y = UY[:,:,J]
    # resize RoI containing the crack growth process
    if roi == 'crop':
            displ_x = displ_x[Y_i:Y_f,X_i:X_f]
            displ_y = displ_y[Y_i:Y_f,X_i:X_f]

    # find the dimensions of the matrix
    m = displ_x.shape[0] #  y - row
    n = displ_x.shape[1] # x - column
    # find the matrix with the displacements
    displ = (displ_x**2 + displ_y**2)**0.5
    # preallocation: zeros
    n_zeros = np.zeros((m, 1))
    m_zeros = np.zeros((1, n+1))
    # variables with displacements of the 4 corners of facets
    displ_A = np.vstack((np.hstack((displ,n_zeros)), m_zeros))/4
    # divided by 4 because sum: displ_A+displ_B+displ_C+displ_D
    displ_B = np.vstack((np.hstack((n_zeros,displ)), m_zeros))/4
    displ_C = np.vstack((m_zeros, (np.hstack((displ, n_zeros)))))/4
    displ_D = np.vstack((m_zeros, (np.hstack((n_zeros, displ)))))/4
    # auxiliar matrix 2 edges; 4 within the matrix 'matr_transf'
    matr_transf = np.ones((m+1, n+1))
    matr_transf[:, 0] = 2
    matr_transf[:, -1] = 2
    matr_transf[0, :] = 2
    matr_transf[-1, :] = 2
    matr_transf[0, 0] = 4
    matr_transf[0, -1] = 4
    matr_transf[-1, 0] = 4
    matr_transf[-1, -1] = 4
    grid_values = (displ_A + displ_B + displ_C + displ_D)*matr_transf
    # displacements of each corner on the facet
    displ_A = grid_values[0:-1, 0:-1]
    displ_B = grid_values[0:-1, 1:]
    displ_C = grid_values[1:, 0:-1]
    displ_D = grid_values[1:, 1:]
    # oblique distance between facet centroids
    displ_CA = np.abs(displ_C-displ_A)
    displ_DB = np.abs(displ_D-displ_B)
    # auxiliar function for the crack tip location criterion
    K = np.maximum(displ_CA, displ_DB)
    #prend le max de CA et de DB
    avgK = np.nanmean(K) #mean ignoring nan values.
    stdK = np.nanstd(K)
    maxK = np.nanmax(K)
    if maxK < avgK + inb*stdK:
        J = J + 1
    else:
        JJ = 0

# save selected stage - stage used to compute automatically the alpha thresh
# at this stage one assumes that there is no crack propagation a(t) = a0
alpha_stages = J

alphamin = 1
alphamax = int(maxK/avgK)
alphaint = np.arange(start=alphamin, stop=alphamax+10, step=.5)
#
# estimation of crack tip length
tipstep = np.zeros((alphaint.shape[0],1)) # unit: substep
tipmm = np.zeros((alphaint.shape[0],1)) # unit: mm

for kk in np.arange(0,alphaint.shape[0],1):
    alpha = alphaint[kk]
    # Criterion for crack tip location
    #  Kt = 0: region where the material is undamaged (default)
    Kt = np.zeros(K.shape)
    Kt[np.isnan(K)] = -1
    # Kt = -1: region where the material is completely damaged and no
    # information is available (NaN) using digital image correlation.
    # Kt[K<alpha*avgK] = 0
    # Kt = 1: are the region where a discontinuity is present but the material
    # is not completely damaged. This happens at the crack tip
    Kt[K>=alpha*avgK] = 1
    Ktemp = Kt
    # n1 deve ser relativo a 'a0'
    ind = np.argwhere(Ktemp==1)
    row, col = ind[:,0], ind[:,1]
    # TODO: origin of the Coordinate to be a0(X,Y)
    if len(col) == 0:
        tipstep[kk] = 0                  # X component
    else:
        tipstep[kk] = a0.X - np.min(col) # X component
    # pixel>mm: [macro-pixel*(pixel/macro-pixel)*(mm/pixel)
    tipmm[kk] = np.abs(tipstep[kk]*MatchID.mm2step - MatchID.mm2step)

# for selected  alpha parameters you compute crack length
# crack length however should be ZERO at the beginning (no crack propagation)
ind = np.where(np.abs(tipmm - np.min(tipmm)) == 0)
alpha_alphasel = alphaint[ind[0][0]]

###############################################################################
# analysis ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
###############################################################################

stagEval = np.arange(0, MatchID.stages, incr)
fract_K = np.zeros((Y_f-Y_i, X_f-X_i+1, stagEval.size))

for J in stagEval:

    ### crack location a = a(t) ------------------------------------------
    # transfering the displacements to a new variable
    displ_x, displ_y = np.copy(UX[:, :, J]), np.copy(UY[:, :, J])
    # resize RoI containing the crack growth process
    if roi == 'crop':
            displ_x = displ_x[Y_i:Y_f, X_i:X_f+1]
            displ_y = displ_y[Y_i:Y_f:, X_i:X_f+1]

    # find the dimensions of the matrix
    m = displ_x.shape[0] #  y - row
    n = displ_x.shape[1] # x - column
    # find the matrix with the displacements
    displ = (displ_x**2 + displ_y**2)**0.5
    # preallocation: zeros
    n_zeros = np.zeros((m, 1))
    m_zeros = np.zeros((1, n+1))
    # variables with displacements of the 4 corners of facets
    displ_A = np.vstack((np.hstack((displ,n_zeros)), m_zeros))/4
    # divided by 4 because sum: displ_A+displ_B+displ_C+displ_D
    displ_B = np.vstack((np.hstack((n_zeros,displ)), m_zeros))/4
    displ_C = np.vstack((m_zeros, (np.hstack((displ, n_zeros)))))/4
    displ_D = np.vstack((m_zeros, (np.hstack((n_zeros, displ)))))/4
    # auxiliar matrix 2 edges; 4 within the matrix 'matr_transf'
    matr_transf = np.ones((m+1, n+1))
    matr_transf[:, 0] = 2
    matr_transf[:, -1] = 2
    matr_transf[0, :] = 2
    matr_transf[-1, :] = 2
    matr_transf[0, 0] = 4
    matr_transf[0, -1] = 4
    matr_transf[-1, 0] = 4
    matr_transf[-1, -1] = 4
    grid_values = (displ_A + displ_B + displ_C + displ_D)*matr_transf
    # displacements of each corner on the facet
    displ_A = grid_values[0:-1, 0:-1]
    displ_B = grid_values[0:-1, 1:]
    displ_C = grid_values[1:, 0:-1]
    displ_D = grid_values[1:, 1:]
    # oblique distance between facet centroids
    displ_CA = np.abs(displ_C-displ_A)
    displ_DB = np.abs(displ_D-displ_B)
    # auxiliar function for the crack tip location criterion
    K = np.maximum(displ_CA, displ_DB)
    # TODO: The equivalent function of Matlab imfilter in Python
    # switch filtro
    # case 'yes'
    #     H = fspecial('gaussian', 5, .95);
    #     K = imfilter(K, H, 'replicate');
    # end
    fract_K[:,:,J] = K
    #where there is the greatest displacement there is the fracture?
    
'''
xplot = np.arange(X_i, X_f+1, 1)
yplot = np.arange(Y_i, Y_f, 1)
Xplt, Yplt = np.meshgrid(xplot, yplot)
fig = plt.figure()
ax = plt.axes(projection="3d")
Zplt = fract_K[:, :, j]
ax.plot_surface(Xplt, Yplt, Zplt)
ax.set_title('surface')
plt.show()
'''

# treshold range
# if int(alpha_alphasel/10) == 0:
#     if int(alpha_alphasel) == 1:
#         inc = 0
#     else:
#         inc = round(.5*alpha_alphasel)
# else:
#     if int(alpha_alphasel) == 10:
#         inc = 9
#     else:
#         inc = 10
# creating vector of alpha values
if re.match(r'^e15', Job):
    print("La racine de job commence par 'e15'")
    beta=15*np.pi/180
    alpha_alphaV = np.round(alpha_alphasel*np.arange(.1,0.8,.1),1) #for 15° except e15e1
if re.match(r'^e30', Job):
    print("La racine de job commence par 'e30'")
    beta=30*np.pi/180
    alpha_alphaV = np.round(alpha_alphasel*np.arange(.7,1.3,.1),1)
#print(alpha_alphasel)
#print(alpha_alphaV)
# differen values for alpha(treshold criterion)
d = { }
for i in np.arange(0,len(alpha_alphaV),1):
    alpha = alpha_alphaV[i]
    m3D =  np.zeros((Y_f-Y_i,X_f-X_i+1,stagEval.size))
    for J in stagEval:
        iAvg = 'NO'
        K = fract_K[:, :, J]
        avgK = np.nanmean(K)
        stdK = np.nanstd(K)
        # if iAvg == 'YES':
        #         # iteration process to get rig some outside values
        #         K1 = K
        #         K1(abs(K1) > avgK+stdK)  = NaN
        #         avgK = mean(K1(~isnan(K1(:))))
        #         stdK = std(K1(~isnan(K1(:))))
        # else:
        # Criterion for crack tip location
        #  Kt = 0: region where the material is undamaged (default)
        Kt = np.zeros(K.shape)
        Kt[np.isnan(K)] = -1
        # Kt = -1: region where the material is completely damaged and no
        # information is available (NaN) using digital image correlation.
        # Kt[K<alpha*avgK] = 0
        # Kt = 1: are the region where a discontinuity is present but the material
        # is not completely damaged. This happens at the crack tip
        Kt[K >= alpha * avgK] = 1
        m3D[:, :, J] = Kt

    d.update({'fract_Kt'+str(i): m3D})

# Zmap = d['fract_Kt4']
# j = 130
# fig = plt.figure()
# plt.imshow(Zmap[:,:,j])
# plt.colorbar()
# plt.show()

crackL_J_pixel_X = np.zeros((len(stagEval),len(alpha_alphaV)))
crackL_J_pixel_Y = np.zeros((len(stagEval),len(alpha_alphaV)))

for i in np.arange(0,len(alpha_alphaV),1):
    #pour tous les alphas
    for J in stagEval:
        #pour tte la FPZ
        istr = 'fract_Kt'+str(i)
        Ktemp = d[istr][:,:,J]
        # n1 deve ser relativo a 'a0'
        ind = np.argwhere(Ktemp == 1)
        row, col = ind[:, 0], ind[:, 1]
        if len(col) == 0:
            crackL_J_pixel_X[J, i] = 0  # X component
            crackL_J_pixel_Y[J, i] = 0  # Y component
        else:
            crackL_J_pixel_X[J, i] = a0.X - np.min(col)
            crackL_J_pixel_Y[J, i] = np.min(row)
        if J > 1:
            if crackL_J_pixel_X[J, i] < crackL_J_pixel_X[J - 1, i]:
                crackL_J_pixel_X[J, i] = crackL_J_pixel_X[J - 1, i]

# pixel > mm: [macro - pixel * (pixel / macro - pixel) * (mm / pixel)
crackL_J_mm = Test.a0 + crackL_J_pixel_X*MatchID.mm2step #crack length
indice = int(np.argmax(crackL_J_mm == np.max(crackL_J_mm))/crackL_J_mm.shape[1])
print('The last image to open for obtaining the crack length',indice)

if a1==0 or af==0 or nombre==0:
    exec(open('ReadcrackfractureMMCG.py').read())
#look at the crackJ in order to see which alpha is best in function of what you found for the crack length


j = 30
fig = plt.figure()
plt.imshow(UY[:, :, j])
plt.plot(UY.shape[1]-crackL_J_pixel_X[j, chos_alp],crackL_J_pixel_Y[j, chos_alp],'sr')
plt.colorbar()
plt.title(Job)
plt.show()

print('The crack length with alpha is:' ,np.max(crackL_J_mm[:,chos_alp]))

fig = plt.figure()
plt.plot(crackL_J_mm[:,0], 'k--', linewidth=1)
plt.plot(crackL_J_mm[:,1], 'c--', linewidth=1)
plt.plot(crackL_J_mm[:,2], 'g--', linewidth=1)
plt.plot(crackL_J_mm[:,3], 'b--', linewidth=1)
plt.plot(crackL_J_mm[:,4], 'r--', linewidth=1)
plt.plot(crackL_J_mm[:,5], 'y--', linewidth=1)
plt.plot(crackL_J_mm[:,6], 'm--', linewidth=1)
#plt.plot(MatchID.load,Abs, linewidth=4)
plt.ylabel('a(t)')
plt.xlabel('stages')
plt.title(Job)
plt.grid()
plt.show()


###############################################
#Least square
###############################################

Y_mean = np.mean(crackL_J_mm[:,chos_alp])
X_mean = np.mean(MatchID.displ)
XY_mean = np.mean (crackL_J_mm[:,chos_alp]*MatchID.displ)
X_car = np.mean(MatchID.displ**2)
slope_at = (XY_mean - (X_mean*Y_mean))/((X_car)-(X_mean**2))
Int_at = Y_mean-slope_at*X_mean
at_modif = slope_at*MatchID.displ+Int_at

fig = plt.figure()
plt.plot(at_modif, 'y--', linewidth=1)
plt.ylabel('a(t), mm')
plt.xlabel('displacement, mm')
plt.title(Job)
plt.grid()
plt.show()

C_modif = np.zeros(MatchID.stages)
i = 2
h = 1/(MatchID.stages+1)
while i < MatchID.stages :
    C_modif[i] = (at_modif[i-2]-4*at_modif[i-1]+3*at_modif[i])/(2*h)
    i=i+1
    

###############################################
CTODimage = MatchID.xCoord[a0.X]
#crack=(CTODimage-crack)*Test.mm2pixel+Test.a0

fig = plt.figure(figsize=(7,5))
plt.plot(MatchID.time,crackL_J_mm[:,chos_alp], '*r--', linewidth=3, label='Method1')
#plt.plot(indices, crack,'bo', markersize=5)
plt.xlabel('Images')
plt.ylabel('Crack length, a(t), mm')
plt.tick_params(axis='both', labelsize=14)
plt.legend(fontsize=12)
plt.title(Job)
fig.tight_layout()
plt.grid()
plt.show()   

#%% computing GI and GII (R-curve)

# Définir la fonction pour l'interpolation
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    # Calculer le R-squared
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    results['determination'] = ssreg / sstot
    return results


print('computing GI and GII (R-curve)..')
a_t = crackL_J_mm[:,chos_alp]
# Obtenir les indices des éléments uniques de x
indices_uniques1 = np.unique(a_t, return_index=True)[1]
CI = COD.wI/(MatchID.load*np.cos(beta))
CII = COD.wII/(MatchID.load*np.sin(beta))
a_t_uniques = a_t[Fc_indices]
CI_uniques = CI[Fc_indices]
CII_uniques = CII[Fc_indices]
ALPI = ((MatchID.load*np.cos(beta))**2)/(2*Test.thickness)
ALPII = ((MatchID.load*np.sin(beta))**2)/(2*Test.thickness)
ALPI_uniques = ALPI[Fc_indices]
ALPII_uniques = ALPII[Fc_indices]
P_uniques=MatchID.load[Fc_indices]
Px_uniques=P_uniques*np.cos(beta)
Py_uniques=P_uniques*np.sin(beta)

fig = plt.figure(figsize=(7,5))
plt.plot(MatchID.time,a_t, '*r--', linewidth=3, label='Method1')
plt.plot(Fc_indices, a_t_uniques, 'b', label='Method1 without duplicates')
plt.xlabel('Images')
plt.ylabel('Crack length, a(t), mm')
plt.tick_params(axis='both', labelsize=14)
plt.legend(fontsize=12)
plt.title(Job)
fig.tight_layout()
plt.grid()
plt.show()  

'''
G01 = ALPI*CI/a_t*10**3
G02 = ALPII*CII/a_t*10**3
Gtot=G01+G02
'''
G01 = ALPI_uniques*CI_uniques/a_t_uniques*10**3
G02 = ALPII_uniques*CII_uniques/a_t_uniques*10**3
Gtot=G01+G02


results = polyfit(a_t_uniques, CI_uniques, 3)
coeffs = results['polynomial']
r_squared = results['determination']
p = np.poly1d(coeffs)
dp = p.deriv()
GI = ALPI_uniques*dp(a_t_uniques)*10**3

results = polyfit(a_t_uniques, CII_uniques, 3)
coeffs = results['polynomial']
r_squared = results['determination']
p = np.poly1d(coeffs)
dp = p.deriv()
GII = ALPII_uniques*dp(a_t_uniques)*10**3

Gtot_interp=GI+GII

fig = plt.figure(figsize=(7,5))
plt.plot(a_t_uniques,CI_uniques, 'k-', linewidth=3)
plt.plot(a_t,CI)
plt.xlabel('Crack length, a(t), mm')
plt.ylabel('$C, {Pa}^{-1}$')
plt.grid()
plt.title(Job)
plt.show()

fig = plt.figure(figsize=(7,5))
plt.plot(a_t_uniques, GI, 'b:', linewidth=2, label='R-Curve-modeI alpha '+ str(chos_alp))
plt.plot(a_t_uniques, GII, 'r:', linewidth=2, label='R-Curve-modeII alpha '+ str(chos_alp))
plt.xlabel('Crack length, a(t), mm')
plt.ylabel('$G, J$')
plt.legend(loc=2, prop={'size': 8})
plt.grid()
plt.title(Job)
plt.show()

fig = plt.figure(figsize=(7,5))
plt.plot(a_t_uniques, G01, 'b:', linewidth=2, label='R-Curve-GI alpha '+ str(chos_alp))
plt.plot(a_t_uniques, G02, 'r:', linewidth=2, label='R-Curve-GII alpha '+ str(chos_alp))
plt.plot(a_t_uniques, Gtot, 'g:', linewidth=2, label='R-Curve-GI+GII alpha '+ str(chos_alp))
plt.ylabel('$G, J$')
plt.legend(loc=2, prop={'size': 8})
plt.grid()
plt.title(Job)
plt.show()


'''
Gc = np.max(G1)
Lc = np.max(Load)
COD_max = np.max(COD.wI)
'''
# write array results on a csv file:
RES = np.array([MatchID.displ[:], MatchID.load[:], CI[:],CII[:], COD.wI[:],COD.wII[:], a_t[:]])
RES = np.transpose(RES)
# pd.DataFrame(RES).to_csv("path/to/file.csv")
# converting it to a pandas dataframe
res_df = pd.DataFrame(RES)
#save as csv
savepath = os.path.join(cwd, Job + '_RES.csv')
tete = ["d, [mm]", "P [N]", "CI [mm/N]", "CII [mm/N]", "wI [mm]","wII [mm]", "a(t) [mm]"]
res_df.to_csv(savepath, header=tete, index=False)

out_file = open(maincwd+'\\Results.csv', "a",)
out_file.write(Job + '\n')
#out_file.write(str(COD_max) + '\n')
#out_file.write(str(Lc) + '\n')
#out_file.write(str(Gc) + '\n')
out_file.close()


RES1 = np.array([Px_uniques[:],Py_uniques[:],a_t_uniques[:],CI_uniques[:],CII_uniques[:],G01[:],G02[:]])
RES1 = np.transpose(RES1)
# pd.DataFrame(RES).to_csv("path/to/file.csv")
# converting it to a pandas dataframe
res_df = pd.DataFrame(RES1)
#save as csv
savepath = os.path.join(cwd, Job + '_RES1.csv')
tete = ["Py [N]","Px [N]","a(t) [mm]", "CI [mm/N]","CII [mm/N]", "G1 [J/m2]", "G2 [J/m2]"]
res_df.to_csv(savepath, header=tete, index=False)
#
# plt.legend(loc=2, prop={'size': 8})
# fig.tight_layout()
# ax.set_xlim(xmin=19)
# ax.set_ylim(bottom=0)
# plt.grid()
# plt.show()
    
#exec(open('Videomaker.py').read())