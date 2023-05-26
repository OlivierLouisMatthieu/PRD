# Allied Manta G-505B
H = 1339 # unit: pixels
# 'TC2336' telecentric lens
LX = 18 # unit: mm

# pixel to mm magnification factor
Test.mm2pixel = LX / H
# Load conversion factor - testing machine
Test.LoadConvFactor = 1000.  # converning kN to N, unit: N
# Displacement conversion factor - testing machine
Test.DisplConvFactor = 1. # unit: mm
##########################################################################
Test.thickness = 12.5 # unit  mm

if Job == 'e0e1':
    Test.a0 = 25.3 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID, unit: pixels
    a0.imgHuse, a0.imgVuse = 4124, 2285
    #af.imgHuse, af.imgVuse = 614, 1013
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 1
    inc_COD_line=31
    # Selecting alpha
    chos_alp = 0
    a1=55.53
    af=8.54
    nombre=49
    #indices = [0, 10, 20, 30, 40,50]
    #crack = [4124, 4120, 4104, 3457, 2636,87.9,1455,1229,1005,972,621]
elif Job == 'e0e2':
    Test.a0 = 24.85 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4188,2308
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    inc_COD_line=30
    # Selecting alpha
    chos_alp = 0
    a1=55.32
    af=3.02
    nombre=60
elif Job == 'e0e3':
    Test.a0 = 25.65  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4410, 2264
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    inc_COD_line=30
    # Selecting alpha
    chos_alp = 0
    a1=59
    af=3.07
    nombre=47
elif Job == 'e0e5':
    Test.a0 = 25.6  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4400, 2303
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 5
    inc_COD_line=31
    # Selecting alpha
    chos_alp = 4
    a1=59.44
    af=2.86
    nombre=59
elif Job == 'e0e6':
    Test.a0 = 26.3 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4403, 2289
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    inc_COD_line=30
    # Selecting alpha
    chos_alp = 1
    a1=59.34
    af=5.73
    nombre=59
elif Job == 'e15e1':
    Test.a0 = 25.85  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4146, 2652
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 4
    a1=55.92
    af=6.88
    nombre=71
elif Job == 'e15e2':
    Test.a0 = 25.6 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4166, 2580
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 1
    # Selecting alpha
    chos_alp = 0
    a1=55.48
    af=5.51
    nombre=39
elif Job == 'e15e3':
    Test.a0 = 25.55  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4189, 2469
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 2
    a1=56.53
    af=8.04
    nombre=41
elif Job == 'e15e4': #e2p1
    Test.a0 = 25.3  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4203, 2462
    #af.imgHuse, af.imgVuse = 1613, 1094
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 3
    # Selecting alpha
    chos_alp = 1
    a1=56.90
    af=0.50
    nombre=30#ou31
    indices = [0, 10, 20, 30, 40,50,60,70]
    crack = [2161, 2173, 1985, 1820, 1511,1503,1301,1203]
elif Job == 'e15e5':
    Test.a0 = 24.85  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4232,2484
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 1
    a1=57.29
    af=3.31
    nombre=51
elif Job == 'e30e1':
    Test.a0 = 25.55  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2096, 981
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 1
    # Selecting alpha
    chos_alp = 0
    a1=57.29
    af=3.31
    nombre=0
elif Job == 'e30e2':
    Test.a0 = 24.4  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4132, 2551
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 1
    # Selecting alpha
    chos_alp = 6
    a1=31.122
    af=18.077
    nombre=32
elif Job == 'e30e3':
    Test.a0 = 24.2 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4173, 2550
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 5
    a1=55.98
    af=2.37
    nombre=33
elif Job == 'e30e4':
    Test.a0 = 25.3  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2162, 999
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 1
elif Job == 'e30e5':
    Test.a0 = 24.4  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4109, 2501
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 0
    a1=31.122
    af=18.077
    nombre=32
    #COD damage
elif Job == 'e30e7':
    Test.a0 = 25.6  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 4030, 2572
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 0
    # Selecting alpha
    chos_alp = 2
    a1=53.93
    af=2.86
    nombre=40
elif Job == 'e45e1':
    Test.a0 = 25.8  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0.imgHuse, a0.imgVuse = 2251,944
    #af.imgHuse, af.imgVuse = 1886, 1064
    ##########################################################################
    # Selecting pair
    COD.cod_pair = 2
    # Selecting alpha
    chos_alp = 2
    a1=30.664534
    af=26.492304
    nombre=212
    #indices = [0, 20, 40, 60, 80,100,120,140,160,180,200,220,240]
    #crack = [2251, 2251, 2249, 2137,2099,1965,1878,1825,1715,1680,1603,1549,1443]
    indices = [0, 20, 40, 60, 80,100,120,140]
    crack = [2251, 2251, 2249, 2137,2099,1965,1878,1825]
elif Job == 'others':
    pass

# Summary of DIC Settings
MatchID.CorrelationCoef = 'ZNSSD'
MatchID.InterpolationOrder = 'Bicubic spline'
MatchID.TransformationOrder = 'Quadratic'
MatchID.Subset, MatchID.Step = 31, 10
# Summary of Strain Settings
MatchID.StrainWindow = 5
MatchID.StrainConvention = 'GreenLagrange'
MatchID.StrainInterpolation = 'Q4'
##########################################################################
# selecting number of pairs to be evaluated
ud_lim = 10
##########################################################################
MatchID.mm2step = MatchID.Step*Test.mm2pixel