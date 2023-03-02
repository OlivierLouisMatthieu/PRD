
print('import modules..')
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import csv
import pandas as pd
# import tkinter as tk
# from PIL import Image, ImageTk
import glob
import time

Job = 'e1o1'

pwd_Stanislas = "C:\\Users\\pc\\Desktop\\MMCGTests"

cwd = os.path.join(pwd_Stanislas, Job)

class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)

MatchID = Struct()
a0 = Struct()
COD = Struct()
Test = Struct()

# run database with data from the project/specimen
exec(open('Database.py').read())

print('reading : load and displacement from the test machine')

# read load file:

pathdados = os.path.join(cwd, Job + '_load.csv')
test = pd.read_csv(pathdados, delimiter=";", decimal=".", names=['Time', 'Load', 'Displ'])

# if Job == 'e3o1':
#     Lc = 2
#     Gc = 8.625
#     COD.wI = 2.15
# if Job == 'e3o2':
#     Lc = 5.6
#     Gc = 12
#     COD.wI = 3.56
# else :
#     Lc = 3.845
#     Gc = 9.56
#     COD.wI = 1.024
# # with open(pwd_Stanislas + 'Results.csv','a+',newline='', encoding= 'utf-8') as csvfile :
# #     writer = csv.writer(csvfile)
# #     writer.writerow = (Job, COD.wI, Lc, Gc)
# out_file = open(pwd_Stanislas+'\\Results.csv', "a",)
# out_file.write(Job + '\n')
# out_file.write(str(COD.wI) + '\n')
# out_file.write(str(Lc) + '\n')
# out_file.write(str(Gc) + '\n')
# out_file.close()



Time = test.Time.values.astype(float)
Time = Time - Time[0]
incTime = int(1/Time[1])

Displ = test.Displ.values.astype(float)*Test.DisplConvFactor # unit: mm
Displ = Displ - Displ[0]
Load = test.Load.values.astype(float)*Test.LoadConvFactor # unit: N
minL=np.min(Load)-1
Load = Load-minL
#
# X1 = Displ[0]
# X1bis = Displ[300]
# X2 = Displ[300]
# Y1 = Load[0]
# Y1bis = Load[300]
# Y2 = Load[300]
# slope = (Y2-Y1)/(X2-X1)
# pas = (Y1bis-Y1)/300
# pas_bis = (X1bis-X1)/300
# i = Y1
# j = X1
# k = 0
#
# while i > 0:
#     i = i-pas
#     j = j+pas_bis
#     k=k+1
#     print('k vaut : %d' %k)
#
# shift_right = j
#
# Displ = Displ+shift_right
#
# print(shift_right, 'en mm')

fig, ax = plt.subplots(figsize=(7,5))
plt.plot(Displ, Load, 'k-', linewidth=3)
plt.ylabel('Load [N]')
plt.xlabel('Displacement [mm]')
plt.grid()
plt.show()

i=j+k








# import urllib2
#
#
# def main():
#     # open a connection to a URL using urllib2
#     webUrl = urllib2.urlopen("https://ucafr-my.sharepoint.com/personal/stanislas_malfait_etu_uca_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fstanislas%5Fmalfait%5Fetu%5Fuca%5Ffr%2FDocuments%2FWood%20Fracture%20Mechanics%2FMMCGtests%2Fe1o1")
#
#     # get the result code and print it
#     print
#     "name: " + str(webUrl)
#
#     # read the data from the URL and print it
#     data = webUrl.read()
#     print(data)
#
#
# if __name__ == "__main__":
#     main()
#
# Python
# 3
# Example
#
# #
# # read the data from the URL and print it
# #
# import urllib.request
#
# # open a connection to a URL using urllib
# webUrl = urllib.request.urlopen('https://www.youtube.com/user/guru99com')
#
# # get the result code and print it
# print("result code: " + str(webUrl.getcode()))
#
# # read the data from the URL and print it
# data = webUrl.read()
# print(data)