import numpy as np
import matplotlib.pyplot as plt

# Test machine data. Delete the header in .txt and replace with ;
dados = np.loadtxt('D:\Recherche PRD\SPR_00_02\SPR_00_02\SPR_00_02.dat')

# Inform the speed of the test:
veloc = 4 # mm/min

# Images start;
ImgInit = 20

# Time start;
TimeInit = 2.5

# Acquisition time:
Time = 1

# Inform the number of images that have been captured:
nImagens = 300

# Calculation procedure:
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(dados[:,0], dados[:,1])
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.show()

tempoImagens = np.zeros(nImagens)

for k in range(1, nImagens):
    if k <= ImgInit:
        tempoImagens[k] = TimeInit + tempoImagens[k-1]
    else:
        tempoImagens[k] = Time + tempoImagens[k-1]

tempoMaq = (dados[:,0]/veloc)*60 #in mm/s

ind = np.unique(tempoMaq, return_index=True)[1]

ForcaMaq = dados[:,1]

DeslocMaq = dados[:,0]

ForcaImagens = np.interp(tempoImagens, tempoMaq[ind], ForcaMaq[ind])
DeslocImagens = np.interp(tempoImagens, tempoMaq[ind], DeslocMaq[ind])
#just 300 images the number of images

ForcaImagens = ForcaImagens[~np.isnan(ForcaImagens)]
DeslocImagens = DeslocImagens[~np.isnan(ForcaImagens)]

# Illustrative figure of each captured instant:
fig2, ax2 = plt.subplots()
ax2.plot(DeslocMaq, ForcaMaq)
ax2.plot(DeslocImagens, ForcaImagens, 'ro')

# Saves the correct .dat file considering the captured instants.
np.savetxt('forca.dat', ForcaImagens, fmt='%.18e')
np.savetxt('desloc.dat', DeslocImagens, fmt='%.18e')