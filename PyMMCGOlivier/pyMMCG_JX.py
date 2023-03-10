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
plt.rcParams['font.family'] = 'sans-serif'
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
Job = 'e1p3'

runuser = 'Xavier'
if runuser == 'Xavier':
    # maincwd = "X:\\jxavier\\Orient\\Erasmus\\2021\\Polytech_Clermont-FD\\Stanislas\\EXP\\MMCGTests"
    maincwd = "/home/slimbook/Documents/GitHub/OlivierLouisMatthieu/PRD/MMCGTests"
elif runuser == 'Olivier':
    maincwd = "C:\\Users\\pc\\Desktop\\MMCGTests"

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

##########################################

#%% Read matchid DIC data

pathdados = os.path.join(cwd,'X[Pixels]',Job+'_0001_0.tiff_X[Pixels].csv')
MatchID.x_pic = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
MatchID.x_pic = MatchID.x_pic[:,0:-1]
MatchID.xCoord = MatchID.x_pic[0,:]
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

# Creating a new axes with specified dimensions and some kwargs
plt.axes([.15, .15, .8, .8])
plt.plot(Displ, Load, 'k-', linewidth=3)
plt.plot(MatchID.displ, MatchID.load, 'r', markersize=8, marker='o', markerfacecolor = 'none')
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.xlim(0, int(np.max(Displ)+1))
plt.ylim(0, int(np.max(Load)*1.1))
plt.show()

# Read results "....tif_#.csv" into 3D np.array

MatchID.SubsetsX = MatchID.x_pic.shape[1]
MatchID.SubsetsY = MatchID.x_pic.shape[0]

# U displacement
UX = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
tic()
for i in np.arange(0, MatchID.stages, 1):
    readstr = Job+'_%04d_0.tiff_U.csv' % int(i+1)
    print('reading : ',readstr)
    pathdados = os.path.join(cwd,'U',readstr)
    # aux = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
    aux = pd.read_csv(pathdados, delimiter=';')
    xend, yend = aux.shape[1], aux.shape[0]
    UX[0:yend, 0:xend-1, i] = aux.values[:, :-1]*Test.mm2pixel # unit: mm
print(f'{toc():.1f} seg')

# V displacement
UY = np.zeros((MatchID.SubsetsY, MatchID.SubsetsX, MatchID.stages))
tic()
for i in np.arange(0, MatchID.stages, 1):
    readstr = Job+'_%04d_0.tiff_V.csv' % int(i+1)
    print('reading : ',readstr)
    pathdados = os.path.join(cwd,'V',readstr)
    # aux = np.genfromtxt(pathdados, skip_header=0, delimiter=';')
    aux = pd.read_csv(pathdados, delimiter=';')
    xend, yend = aux.shape[1], aux.shape[0]
    UY[0:yend, 0:xend-1, i] = aux.values[:, :-1]*Test.mm2pixel # unit: mm
print(f'{toc():.1f} seg')

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
# tic()

figsize = Width/float(dpi), Height/float(dpi)
fig = plt.figure(figsize=figsize)
cor = (255, 255, 255)
thickness = 1
start_point = (MatchID.SubsetXi,MatchID.SubsetYi)
end_point = (MatchID.SubsetXf,MatchID.SubsetYf)
img0 = cv.rectangle(img0, start_point, end_point, cor, thickness)
plt.imshow(img0, cmap='gray', vmin=0, vmax=255)
# for i in np.arange(start=0, stop=MatchID.x_pic.shape[1], step=1):
#     for j in np.arange(start=0, stop=MatchID.x_pic.shape[0], step=1):
#         plt.plot(MatchID.x_pic[j,i], MatchID.y_pic[j,i], color='red',
#                  marker='.', markerfacecolor='red', markersize=1)
plt.plot(a0.imgHuse,a0.imgVuse, color='red', marker='+', markersize=10)
plt.plot(a0.imgHuse,a0.imgVuse, color='red', marker='s', markersize=MatchID.Subset, alpha=0.5)
plt.show()
# print(f'{toc():.1f} seg')

#######################################
aux = MatchID.load - np.max(MatchID.load)
# idx = np.argwhere(np.abs(aux) == np.min(np.abs(aux)))
idx = np.array([0]) # select 1st image
plt.axes([.075, .0, .875, 1.2])
plt.imshow(UY[:, :, int(idx[0])])
plt.xticks(color='w')
plt.yticks(color='w')
plt.plot(a0.X,a0.Y,'sr')
plt.title('UY, mm', color='black')
plt.text(a0.X*.985, a0.Y*.95,'a0', color='black')
cax = plt.axes([0.075, 0.2, .875, 0.1])
plt.colorbar(cax=cax, orientation='horizontal')
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

#%% Computing aDIC
print('computing aDIC..')

exec(open('cal_aDIC_u.py').read())

#%% computing GI (R-curve)
print('computing GI (R-curve)..')

# compliance
delta = MatchID.displ[0:icrop]
P = MatchID.load[0:icrop]
C = delta/P

# Curve fitting
porder = 3
fitCa = np.polyfit(a_t, C, porder)

df = pd.DataFrame(columns=['C', 'a_t'])
df['C'] = C
df['a_t'] = a_t

weights = np.polyfit(a_t, C, porder)
# p[0] + p[1]*x + ... + p[N]*x**N
print(weights)
model = np.poly1d(weights)
print(model)
results = smf.ols(formula='C ~ model(a_t)', data=df).fit()

def compliancea(a,m,n):
    return m*a**3 + n

popa1, pcov1 = curve_fit(compliancea, a_t, C)
fit_a1 = compliancea(a_t, *popa1)

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(delta, C, 'k-.', linewidth=2, label='Compliance evolution')
xfit, yfit = a_t, fit_a1
plt.plot(xfit, yfit, 'r--', linewidth=2, label='Poly3')
plt.ylabel('$C$, mm/N')
plt.xlabel('$a(t)$, mm')
# plt.legend(loc=2, prop={'size': 8})
fig.tight_layout()
plt.grid()
plt.show()

# P**2/2/B* dC / da
ALP = (P**2)/(2*Test.thickness)
#
BET = C/a_t #changing the value of alpha from the crack length will change G values
#
G = ALP*BET
# G = np.dot(ALP,BET)

Gc = np.max(G)
Lc = np.max(Load)
COD_max = np.max(COD.wI)
fig = plt.figure(figsize=(7,5))
plt.plot(a_t, G, 'b:', linewidth=2, label='R-Curve alpha 6')
plt.xlabel('Crack length, a(t), mm')
plt.ylabel('$G_{Ic}, J$')
plt.grid()
plt.title(Job)
plt.show()

# write array results on a csv file:
RES = np.array([delta[0:icrop], P[0:icrop], C[0:icrop], COD.wI[0:icrop], a_t[0:icrop], G[0:icrop]])
RES = np.transpose(RES)
# pd.DataFrame(RES).to_csv("path/to/file.csv")
# converting it to a pandas dataframe
res_df = pd.DataFrame(RES)
#save as csv
savepath = os.path.join(cwd, Job + '_RES.csv')
tete = ["d, [mm]", "P [N]", "C [mm/N]", "wI [mm]", "a(t) [mm]", "GI [N/mm]"]
res_df.to_csv(savepath, header=tete, index=False)

#############################################################

# #Comparison of G values
#
# print('computing GI (R-curve)..')
# a_t_bis = Test.a0 + crackL_J_pixel_X[:,chos_alp]*MatchID.mm2step-Delta
#
# C_bis = delta/P
#
# ALP_bis = (MatchID.load**2)/(2*Test.thickness)
#
# BET_bis = C_bis/a_t_bis #changing the value of alpha from the crack length will change G values
#
# G_bis = ALP*BET
#
# fig = plt.figure(figsize=(7,5))
# plt.plot(a_t_bis, G_bis, 'r:', linewidth=2, label='R-Curve alpha %d'%chos_alp)
# plt.plot(a_t_bis,G, 'k:', linewidth=2, label='R-Curve alpha %d'%chos_alp)
# plt.xlabel('Crack length, a(t), mm')
# plt.ylabel('$G_{Ic}, J$')
# plt.grid()
# plt.legend(loc=2, prop={'size': 8})
# plt.title(Job)
# plt.show()
#
# # plt.legend(loc=2, prop={'size': 8})
# # fig.tight_layout()
# ax.set_xlim(xmin=19)
# ax.set_ylim(bottom=0)
# plt.grid()
# plt.show()

# GI = f(CTOD) ?

#############################################
#Interlaminar Fracture Thougness
#############################################


#a isn't the crack length chosen between some alphas values,
#it is obtained from linear least square regression fitting.
# IFT=Inter+Slop*(crackL_J_mm[:,4]*crackL_J_mm[:,4]*crackL_J_mm[:,4])

# aCUB=crackL_J_mm[:,i]*crackL_J_mm[:,i]*crackL_J_mm[:,i]
#a must be the CBBM one

#C is the same

# fig, ax = plt.subplots(figsize=(7,5), dpi=80)
# plt.plot(aCUB,C , 'k-.', linewidth=2, label='Compliance evolution')
# plt.ylabel('$C, Pa^{-1}$')
# plt.xlabel('Crack length, $a^{3}, mm^{3}$')
# plt.legend(loc=2, prop={'size': 8})
# fig.tight_layout()
# plt.grid()
# plt.show()

# Inter=C[0]
# X1=aCUB[0]
# Y1=C[0]
# X2=aCUB[230]
# Y2=aCUB[230]
#
# slope=(Y2-Y1)/(X2-X1)
#
# newC= slope*aCUB+Inter
# print('slope value is :',slope)
# print('Inter value is :',Inter)
# print('C with CBBM is now :',newC)
#Slop= will depend on the curve "cubic fit" which is an average on all the points

#############################################
#Mode II Interlaminar Fracture Thougness
#############################################
#
# GIIc=(3*slope*MatchID.load*MatchID.load*crackL_J_mm[:,i]*crackL_J_mm[:,i])/(2*Test.thickness)
# print('Mode II Interlaminar Fracture Thougness are :', GIIc)

# Issues with Units at the end =/ Joules because of the N²

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#MatLab process transcription
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#a_cbbm2 to create
#fitresult to create to

# Plot fit with data.
# fig1 = figure(Name=test.project.name)
# h0 = plt.plot(a_cbbm2,C,'--k',lineWidth=1,label='a_{eq}')
# h1 = plt.plot(xX,yY,'--r',MarkerSize=10,label='a_{DIC}')
# h2 = plt.plot(fitresult,'-k',lineWidth=2,label='cubic fit')
# plt.legend(loc=2, prop={'size': 8})
# plt.ylabel('$C, Pa^{-1}$')
# plt.xlabel('Crack length, $a^{3}, mm^{3}$')

# # Label axes
#
# fnome = [test.pathFiles,test.project.name,'_aDIC-a_C_',tipoanal];
# print(script.imgformat,script.resol,fnome)
# crop([fnome,script.filetype])
#
# || aDIC - aeq ||
# j1 = i1;
# stgmax = '1';
# switch stgmax
#     case '1' % step at maximum load
#         j2 = round(find(max(MatchID.load)==MatchID.load,1,'last')); % number of maximum data points for LSR
#     case '2' % last valid step
#         j2 = i2;
# end
# % -
# d3      = d(j1:j2);
# a_cbbm3 = a_cbbm(j1:j2);
# a_dic3  = a_dic(j1:j2);
#
# JJ1 = j1;
# JJ2 = j2;
# % Plot fit with data.
# close all
# fig1 = figure('Color',[1 1 1],'Name',test.project.name);
# axes('Position',[.12 .3 .35 .35],'FontName',script.nomeF,'FontSize',script.size_font1,'Parent',fig1);
# plot(MatchID.displ2,MatchID.load,'-k','LineWidth',4); hold on; box on;
# plot(MatchID.displ2(j1),MatchID.load(j1),'sr','MarkerFaceColor',[1 0 0]);
# plot(MatchID.displ2(j2),MatchID.load(j2),'sr','MarkerFaceColor',[1 0 0]);
# plot([0;MatchID.displ2],1./DCB.res.C.*[0;MatchID.displ2],'--k','MarkerFaceColor',[1 0 0]);
# xlim([0 max(MatchID.displ2)]); ylim([0 max(MatchID.load)])
# xlabel('{\it \delta}, mm','FontName',script.nomeF,'FontSize',script.size_font1)
# ylabel('{\it P}, N','FontName',script.nomeF,'FontSize',script.size_font1)
# axes('Position',[.6 .3 .35 .35],'FontName',script.nomeF,'FontSize',script.size_font1,'Parent',fig1);
# plot(d3,abs(a_dic3-a_cbbm3),'--k','LineWidth',2); hold on;
# % Label axes
# xlim([min(d3) max(d3)])
# ylim([min(abs(a_dic3-a_cbbm3)) max(abs(a_dic3-a_cbbm3))])
# xlabel('{\it \delta}, mm','FontName',script.nomeF,'FontSize',script.size_font1,'visible','on');
# ylabel('||{\it a}_{DIC} - {\it a}_{eq}||, mm','FontName',script.nomeF,'FontSize',script.size_font1,'visible','on')
# % -
# fnome = [test.pathFiles,test.project.name,'_aDIC-a_d_',tipoanal];
# print(script.imgformat,script.resol,fnome)
# crop([fnome,script.filetype])