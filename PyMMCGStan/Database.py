# Allied Manta G-505B
H, V  = 2448, 2050 # unit: pixels
# 'TC2336' telecentric lens
LX, LY = 34.98, 29.18 # unit: mm

# pixel to mm magnification factor
Test.mm2pixel = LX / H
# Load conversion factor - testing machine
Test.LoadConvFactor = 1000  # converning kN to N, unit: N
# Displacement conversion factor - testing machine
Test.DisplConvFactor = 1 # unit: mm
Test.meanPreLoad = 69.1 # unit: N

if Job == 'e1o1':
    ddeplac = 0.105
    ##########################################################################
    Test.thickness = 14 # unit  mm
    Test.a0 = 24.975 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID, unit: pixels
    a0.imgHuse, a0.imgVuse = 2314, 1013
    Delta = 1.525 # average okoume, unit: mm
    Deltapix = Delta*(1/Test.mm2pixel) # convert mm to pixel
    a0.imgH, a0.imgV = a0.imgHuse-Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 1
elif Job == 'e1o2':
    ddeplac = 0.057
    ##########################################################################
    Test.thickness = 14 # unit  mm
    Test.a0 = 24.25 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2151,1005
    Delta = 2.25 #average okoume
    Deltapix = Delta*(1/Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse-Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 1
elif Job == 'e1o3':
    ddeplac = 0.041
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 24.5  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2140, 1126
    Delta = 0.5
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 1
elif Job == 'e1p1':
    ddeplac = 0.3
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 23.25  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2120, 992
    Delta = 2.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 0
elif Job == 'e1p2':
    ddeplac = 0.044
    ##########################################################################
    Test.thickness = 13  # unit  mm
    Test.a0 = 23.4  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2197, 1028
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 1
elif Job == 'e1p3':
    ddeplac = 0.057
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 23.5 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2174, 1029
    Delta = 2
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 3
# elif Job == 'e1i1':
#     ##########################################################################
#     Test.thickness = 14.2  # unit  mm
#     Test.a0 = 24.975  # unit  mm
#     # Selecting image coordinates of subset directly from MatdhID
#     a0.imgHuse, a0.imgVuse = 0, 0
#     Delta = 1
#     Deltapix = Delta * (1 / Test.mm2pixel)
#     a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
#     ##########################################################################
#     # Selecting pair
#     COD.cod_pair = 2
elif Job == 'e2o1':
    ddeplac = 0.2
    ##########################################################################
    Test.thickness = 13  # unit  mm
    Test.a0 = 26.475  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 1940, 895
    Delta = 1.525  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 4
    # Selecting alpha
    chos_alp = 1
elif Job == 'e2o2':
    ddeplac = 0.2
    ##########################################################################
    Test.thickness = 14 # unit  mm
    Test.a0 = 27.975 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 1595, 868
    Delta = 1.525 #average okoume
    Deltapix = Delta*(1/Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse-Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 1
elif Job == 'e2o3':
    ddeplac = 0.2
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 29  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 1878, 982
    Delta = 1
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 1
elif Job == 'e2e2': #e2p1
    ddeplac = 0.17
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 27.4  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2161, 1002
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 0
elif Job == 'e2p2':
    ddeplac = 0.15
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 27.9  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2089,973
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 0
elif Job == 'e2p3':
    ddeplac = 0.28
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 27.5  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2096, 981
    Delta = 0.5
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 0
elif Job == 'e3o1':
    ddeplac = 0.15
    ##########################################################################
    Test.thickness = 14.2  # unit  mm
    Test.a0 = 21.475  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2274, 1024
    Delta = 1.525  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 2
elif Job == 'e3o2':
    ddeplac = 0.15
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 22.75 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2224, 1008
    Delta = 1.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 0
elif Job == 'e3o3':
    ddeplac = 0.15
    ##########################################################################
    Test.thickness = 13.8  # unit  mm
    Test.a0 = 21.975  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2162, 999
    Delta = 1.525
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 1
elif Job == 'e3p1':
    ddeplac = 0.1
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 21.9  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2379, 1018
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 1
    # Selecting alpha
    chos_alp = 1
elif Job == 'e3p2':
    ddeplac = 0.1
    ##########################################################################
    Test.thickness = 14.5 # unit  mm
    Test.a0 = 20.5 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2381, 958
    Delta = 1.6  # average padouck
    Deltapix = Delta*(1/Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse-Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 1
elif Job == 'e3p3':
    ddeplac = 0.1
    ##########################################################################
    Test.thickness = 15  # unit  mm
    Test.a0 = 24.6  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2274, 954
    Delta = 0.9
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 2
elif Job == 'e4e1':
    ddeplac = 0.1
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 23.975  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2251,944
    Delta = 1.525
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 2
elif Job == 'e4p1':
    ddeplac = 0.1
    ##########################################################################
    Test.thickness = 13.5  # unit  mm
    Test.a0 = 23.25  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2205, 1005
    Delta = 2.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 1
# elif Job == 'e4i1':
#     ddeplac = 0.1
#     ##########################################################################
#     Test.thickness = 14.2  # unit  mm
#     Test.a0 = 23.975  # unit  mm
#     # Selecting image coordinates of subset directly from MatdhID
#     a0.imgHuse, a0.imgVuse = 2028, 932
#     Delta = 1  # average okoume
#     Deltapix = Delta * (1 / Test.mm2pixel)
#     a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
#     ##########################################################################
#     # Selecting pair
#     COD.cod_pair = 2
elif Job == 'e5o1':
    ddeplac = 0.2
    ##########################################################################
    Test.thickness = 14.1  # unit  mm
    Test.a0 = 24 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2058, 1013
    Delta = 3  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 3
elif Job == 'e5p1':
    ddeplac = 0.2
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 22.75  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2307, 1029
    Delta = 1.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 3
# elif Job == 'e5i1':
#     ddeplac = 0.1
#     ##########################################################################
#     Test.thickness = 13.8  # unit  mm
#     Test.a0 = 23.975  # unit  mm
#     # Selecting image coordinates of subset directly from MatdhID
#     a0.imgHuse, a0.imgVuse = 2008, 1042
#     Delta = 1  # average okoume
#     Deltapix = Delta * (1 / Test.mm2pixel)
#     a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
#     ##########################################################################
# elif Job == 'pbis':
#     ##########################################################################
#     Test.thickness = 14.2  # unit  mm
#     Test.a0 = 23.9  # unit  mm
#     # Selecting image coordinates of subset directly from MatdhID
#     a0.imgHuse, a0.imgVuse = 0, 0
#     Delta = 1.6  # average padouck
#     Deltapix = Delta * (1 / Test.mm2pixel)
#     a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
#     ##########################################################################
elif Job == 'pter':
    ddeplac = 0.1
    ##########################################################################
    Test.thickness = 12.8  # unit  mm
    Test.a0 = 23.25  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2198, 1040
    Delta = 2.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0.imgHuse - Deltapix, a0.imgVuse
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 0
elif Job == 'others':
    pass

# Summary of DIC Settings
MatchID.CorrelationCoef = 'ZNSSD'
MatchID.InterpolationOrder = 'Bicubic spline'
MatchID.TransformationOrder = 'Affine'
MatchID.Subset, MatchID.Step = 21, 5
# Summary of Strain Settings
MatchID.StrainWindow = 7
MatchID.StrainConvention = 'GreenLagrange'
MatchID.StrainInterpolation = 'Q4'
##########################################################################
# selecting number of pairs to be evaluated
ud_lim = 10
##########################################################################
MatchID.mm2step = MatchID.Step*Test.mm2pixel