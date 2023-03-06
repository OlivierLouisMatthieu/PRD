#%% Import modules + Database

print('import modules..')
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
# import tkinter as tk
# from PIL import Image, ImageTk
import glob
import time
from ttictoc import tic,toc

plt.rcParams['axes.facecolor'] = (1, 1, 1)
plt.rcParams['figure.facecolor'] = (1, 1, 1) # parameters in order to plot
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
Job = 'DCB_002'

pwd_xavier = "X:\\jxavier\\Orient\\Erasmus\\2021\\Polytech_Clermont-FD" \
             "\\Wood_Fracture_Mechanics\\PyMMCG\\Data\\"
pwd_Stanislas = "C:\\Users\\pc\\Documents\\Documents\\Cours\\Polytech\\5A"\
                "\\PRD Lisbone\\Wood_Fracture_Mechanics\\PyMMCG"

cwd = os.path.join(pwd_Stanislas, Job) #join Job to pwd_Stanislas


#cwd = os.path.join('C:\\Users\\pc\\Desktop\\PRD essai\\DCB_002',Job) #why Job Here ?

cwd = os.path.join('C:\\Users\\pc\\Desktop\\Wood_Fracture_Mechanics_2\\PRD essai\\DCB_002')
###############################################################################

print('working directory: ', cwd)
# data structure, units: SI: Pa, N, m
print('starting structured variables..')
# anonymous class
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)

MatchID = Struct()
a0 = Struct()# create a struct ok!
COD = Struct()
Test = Struct()

# run database with data from the project/specimen
exec(open('Database.py').read())

#%% Read P-d curve data

print('reading : load and displacement from the test machine')

# read load file:
#pathdados = os.path.join(cwd,'DCB_002_load.dat')
pathdados = os.path.join(cwd,'DCB_002_load.dat')
#C:\Users\pc\Desktop\PRD essai\DCB_002\DCB_002_load.dat
load = np.genfromtxt(pathdados, skip_header=3, delimiter=';')
#filename,number of lines that we have to skip from the beginning of the file , string used to separate the values
Test.load = load[1:]*Test.LoadConvFactor

pathdados = os.path.join(cwd, 'DCB_002_displ.dat')
disp = np.genfromtxt(pathdados, skip_header=3, delimiter=';') 
#create a string
# conversion and shit P-d curve
Test.disp = disp[1:]*Test.DisplConvFactor + 0.587461272

del load, disp #Delete an user-defined object

#%% Read matchid DIC data

pathdados = os.path.join(cwd,'x_pic\\DCB_002_0001.tif_x_pic.csv')
MatchID.x_pic = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
MatchID.x_pic = MatchID.x_pic[:,0:-1] 
#takes all the lines and for the column it returns all elements [:] except the last one -1
MatchID.xCoord = MatchID.x_pic[0,:] #takes the whole first line of the 2D table
pathdados = os.path.join(cwd,'y_pic\\DCB_002_0001.tif_y_pic.csv')
MatchID.y_pic = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
MatchID.y_pic = MatchID.y_pic[:,0:-1]
MatchID.yCoord = MatchID.y_pic[:,0]

# determining the number of stages by inspecting MatchID processing files
stages = glob.glob(os.path.join(cwd, 'DCB_002_*.tif.dat'))
#the glob module is used to retrieve files/pathnames matching a specified pattern
MatchID.stages = stages.__len__()#number of elements (or length) in an object
del stages
print('Number of stages: ', str(MatchID.stages))
# str converts the specified value into a string

# Read results "....tif_#.csv" into 3D np.array

MatchID.SubsetsX = int((MatchID.Roi_PolyXf-MatchID.Roi_PolyXi)/MatchID.Step)-1
MatchID.SubsetsY = int((MatchID.Roi_PolyYf-MatchID.Roi_PolyYi)/MatchID.Step)

# U displacement
UX = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages)) #matrix of 0
tic() #easily timing sections of code
for i in np.arange(0, MatchID.stages, 1): #Return evenly spaced values within a given interval.
    readstr = 'DCB_002_%04d.tif_u.csv' % int(i+1)#modulo
    print('reading : ',readstr)
    pathdados = os.path.join(cwd,'u',readstr)
    #os.path.join() method in Python join one or more path components intelligently.
    aux = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
    UX[:, :, i] = aux[:, :-1]*Test.mm2pixel # unit: mm and UX 3D
print(f'{toc():.1f} seg')

# V displacement
UY = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
tic()
for i in np.arange(0, MatchID.stages, 1):
    readstr = 'DCB_002_%04d.tif_v.csv' % int(i+1)
    print('reading : ',readstr)
    pathdados = os.path.join(cwd,'v',readstr)
    aux = np.genfromtxt(pathdados, skip_header=0, delimiter=';') 
    #load the data from the text files
    UY[:, :, i] = aux[:, :-1]*Test.mm2pixel # unit: mm
print(f'{toc():.1f} seg')

#%% Selecting a0
print('Selecting the subset closest to the initial crack tip..')

a0.X = int(np.argwhere(np.abs(MatchID.xCoord - a0.imgH) == 0))
#argwhere function is used to find the indices of array elements that are non-zero, grouped by element.
#abs for absolute value
a0.Y = int(np.argwhere(np.abs(MatchID.yCoord - a0.imgV) == 0))

pathdados = os.path.join(cwd,'DCB_002_0000.tif')
img0 = cv.imread(pathdados, cv.IMREAD_GRAYSCALE) 
# cv.imread(pathdados, 0)
# imread loads an image from the specified file
dpi = plt.rcParams['figure.dpi']
Height, Width = img0.shape

# What size does the figure need to be in inches to fit the image?
print('plotting: image + roi + subsets mesh..')
tic()

figsize = Width/float(dpi), Height/float(dpi)
fig = plt.figure(figsize=figsize)
cor = (255, 255, 255)
thickness = 1
start_point = (MatchID.Roi_PolyXi,MatchID.Roi_PolyYi)
end_point = (MatchID.Roi_PolyXf,MatchID.Roi_PolyYf)
img0 = cv.rectangle(img0, start_point, end_point, cor, thickness)
#used to draw a rectangle on any image
plt.imshow(img0, cmap='gray', vmin=0, vmax=255)
#used to display data as an image, cmap colormap instance, vmin and vmax colorbar range
for i in np.arange(start=0, stop=MatchID.x_pic.shape[1], step=1):
    for j in np.arange(start=0, stop=MatchID.x_pic.shape[0], step=1):
        plt.plot(MatchID.x_pic[j,i], MatchID.y_pic[j,i], color='red',
                 marker='.', markerfacecolor='red', markersize=1)
plt.plot(a0.imgH,a0.imgV, color='blue', marker='.', markersize=1)
# x = plt.ginput(1)
# print(x)
plt.show()
print(f'{toc():.1f} seg')

####################################
#essai de graphique load en fonction du temps, ici les images
####################################
w=250
Abs =np.arange(w)
print(Abs)
fig, ax = plt.subplots(figsize=(7,5)) 
plt.plot(Test.load, 'k-', linewidth=3)
#plt.plot(Test.load,Abs, linewidth=4)
plt.ylabel('Load P [N]')
plt.xlabel('stages')
fig.tight_layout() #adjust the subplot
plt.grid()
plt.show()

# ######################################

j = 110
fig = plt.figure()
plt.imshow(UY[:, :, j])
plt.plot(a0.X,a0.Y,'sr')
plt.colorbar()
plt.show()

#%% Computing CTOD

pathdados = os.path.join(cwd, 'wI_aramis2D.csv')
wIaramis2D = np.genfromtxt(pathdados, skip_header=0, delimiter=';')

ud_lim = 10
# Uup, Udown, ||Uup-Udown||
CTODI  = np.zeros((ud_lim, 3, MatchID.stages)) #3D table of 0
# Vup, Vdown, ||Vup-Vdown||
CTODII = np.zeros((ud_lim, 3, MatchID.stages))

for J in np.arange(0, MatchID.stages, 1):
    #goes from 0 to matchid.stages with a step of 1
    # mode II:
    uXtemp = np.copy(UX[:, :, J])
    #Return an array copy of the given object
    CTODII[:, 0, J] = np.flipud(uXtemp[a0.Y - ud_lim: a0.Y, a0.X])
    #Reverse the order of elements along axis 0
    CTODII[:, 1, J] = uXtemp[a0.Y: a0.Y + ud_lim, a0.X]
    CTODII[:, 2, J] = np.abs(CTODII[:, 1, J] - CTODII[:, 0, J])
    # mode I:
    uYtemp = np.copy(UY[:, :, J])
    CTODI[:, 0, J] = np.flipud(uYtemp[a0.Y - ud_lim: a0.Y, a0.X])
    CTODI[:, 1, J] = uYtemp[a0.Y: a0.Y + ud_lim, a0.X]
    CTODI[:, 2, J] = np.abs(CTODI[:, 1, J] - CTODI[:, 0, J])

COD.wI = CTODI[COD.cod_pair, 2, :]
COD.wII = CTODII[COD.cod_pair, 2, :]

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(COD.wI , Test.load, 'k-', linewidth=3)
plt.plot(COD.wII , Test.load, 'k--')
plt.plot(wIaramis2D , Test.load, 'r:')
plt.xlabel('CTOD, mm')
plt.ylabel('Load, N')
fig.tight_layout()
ax.set_xlim(xmin=0)
ax.set_ylim(bottom=0)
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Test.disp, COD.wI, 'k-', linewidth=4)
plt.plot(Test.disp, COD.wII, 'k--')
plt.plot(Test.disp, wIaramis2D, 'r:')
plt.xlabel('Displacement, mm')
plt.ylabel('CTOD, mm')
fig.tight_layout()
ax.set_xlim(xmin=0)
ax.set_ylim(bottom=0)
plt.grid()
plt.show()

# pathdados = os.path.join(cwd,'wI_matchid.csv')
# np.savetxt(pathdados, cod.wI, delimiter=",")

#%% Computing aDIC
print('computing aDIC..')

roi = 'crop' # 'all'; 'crop'
i, incr = 1, 1
# incr : is used to step over stages if required (default = 1: all stages)
Y_i, Y_f = 0, UY.shape[0]
X_i, X_f = 0, a0.X
filtro = 'yes' # 'yes'; 'no'

####
#### alpha evaluation :::::::::::::::::::::::::::::::::::::::::::::::::::::
### Selecting stage for investigating alpha
####

# least-squares linear regression
porder = 1
xx = Test.disp # displacement (mm)
yy = Test.load # load (N)
# Data point in the linear least-squares regression
limsup = int(0.75*np.argwhere(max(yy)==yy)[-1])
# number of maximum data points for LSR
liminf = int(np.round((1/3)*limsup))#round to integer with 1/3*value
# number of minimum data points for LSR

xx, yy = xx[0:limsup], yy[0:limsup]
Rtot = np.zeros((limsup-liminf,1))
C_M = np.zeros((limsup-liminf,1))
for j in np.arange(0,limsup-liminf,1):
    limt_sup = liminf + j
    xfit, yfit = xx[0:limt_sup], yy[0:limt_sup]
    p  = np.polyfit(xfit, yfit, porder)
    #Least squares polynomial fit with porder as degree
    C_M[j] = 1/p[0]
    dev = yfit - np.mean(yfit) # deviations - measure of spread
    #mean of table
    SST = np.sum(dev**2) # total variation to be accounted for
    resid = yfit - np.polyval(p, xfit) # residuals - measure of mismatch
    SSE = np.sum(resid**2) # variation NOT accounted for
    Rtot[j] = 1 - SSE/SST #  variation NOT accounted for

# position for the best fitting point parameters
jmax = np.max(np.argwhere(np.max(Rtot)==Rtot))
J = int(liminf + jmax)

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Test.disp, Test.load, 'k-', linewidth=3)
plt.plot(Test.disp[liminf:limsup], Test.load[liminf:limsup],'r--',linewidth=4)
plt.plot(Test.disp[J], Test.load[J],'bo', markersize=10)
x = np.linspace(0,Test.disp[liminf]*2.2)
#Return evenly spaced numbers over a specified interval.
y = np.linspace(0,Test.load[liminf]*2.2)
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
#### alpha evaluation :::::::::::::::::::::::::::::::::::::::::::::::::::::
### 2 criterion for checking stage
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
    # vstack stacks arrays in sequence vertically and hstack horizontally
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
    avgK = np.nanmean(K)
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
alphaint = np.arange(start=alphamin,stop=alphamax+10,step=.5)
#gives indices from start to stop with a step
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

j = 20
fig = plt.figure()
plt.imshow(fract_K[:, :, j])
plt.plot(a0.X,a0.Y,'sr')
plt.colorbar()
plt.show()

xplot = np.arange(X_i, X_f+1, 1)
yplot = np.arange(Y_i, Y_f, 1)
Xplt, Yplt = np.meshgrid(xplot, yplot)
#Return coordinate matrices from coordinate vectors.
fig = plt.figure()
ax = plt.axes(projection="3d")
Zplt = fract_K[:, :, j]
ax.plot_surface(Xplt, Yplt, Zplt)
ax.set_title('surface')
plt.show()

# TODO: surface 3D plot K(mesh)
# from mpl_toolkits import mplot3d
# x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
# y = x.copy().T # transpose
# z = np.cos(x ** 2 + y ** 2)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
# ax.set_title('Surface plot')
# plt.show()

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
# creating vetor of alpha values
alpha_alphaV = np.round(alpha_alphasel*np.arange(.7,1.3,.1),1)

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

Zmap = d['fract_Kt4']
j = 130
fig = plt.figure()
plt.imshow(Zmap[:,:,j])
plt.colorbar()
plt.show()

crackL_J_pixel_X = np.zeros((len(stagEval),len(alpha_alphaV)))
crackL_J_pixel_Y = np.zeros((len(stagEval),len(alpha_alphaV)))

for i in np.arange(0,len(alpha_alphaV),1):
    for J in stagEval:
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
crackL_J_mm = Test.a0 + crackL_J_pixel_X*MatchID.mm2step

i = 1
fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Test.disp,crackL_J_mm[:,i], '*r--', linewidth=3)
plt.xlabel('Displacement, mm')
plt.ylabel('Crack length, a(t), mm')
fig.tight_layout()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(crackL_J_mm, 'k-', linewidth=3)
#plt.plot(Test.load,Abs, linewidth=4)
plt.ylabel('a(t)')
plt.xlabel('stages')
fig.tight_layout()
plt.grid()
plt.show()

#%% computing GI (R-curve)
print('computing GI (R-curve)..')

# TODO
#

# LOAD, DISP , B, CTOD, aDIC

#C = Test.disp/Test.load
#print(C)
# P**2/2/B* dC / da
ALP = (Test.load*Test.load)/(2*Test.thickness)
print(ALP)

C = Test.disp/Test.load
print(C)
fig, ax = plt.subplots(figsize=(7,5), dpi=80)
plt.plot(C, crackL_J_mm[:,i], 'k-.', linewidth=2, label='Compliance evolution')
plt.ylabel('$C, Pa^{-1}$')
plt.xlabel('Crack length, a(t), mm')
plt.legend(loc=2, prop={'size': 8})
fig.tight_layout()
plt.grid()
plt.show()

BET = C/crackL_J_mm[:,4]

G = ALP * BET

fig, ax = plt.subplots(figsize=(7,5), dpi=80)
plt.plot(crackL_J_mm[:,0], G, 'b-.', linewidth=2, label='R-Curve alpha 1')
plt.plot(crackL_J_mm[:,1], G, 'r--', linewidth=2, label='R-Curve alpha 2')
plt.plot(crackL_J_mm[:,2], G, 'g-', linewidth=2, label='R-Curve alpha 3')
plt.plot(crackL_J_mm[:,3], G, 'k:', linewidth=2, label='R-Curve alpha 4')
plt.plot(crackL_J_mm[:,4], G, 'c-.', linewidth=2, label='R-Curve alpha 5')
plt.plot(crackL_J_mm[:,5], G, 'm:', linewidth=2, label='R-Curve alpha 6')
plt.plot(crackL_J_mm[:,6], G, 'y--', linewidth=2, label='R-Curve alpha 7')
plt.xlabel('Crack length, a(t), mm')
plt.ylabel('$G_{Ic}, mm$')

plt.legend(loc=2, prop={'size': 8})
fig.tight_layout()
ax.set_xlim(xmin=19)
ax.set_ylim(bottom=0)
plt.grid()
plt.show()

# GI = f(CTOD) ?