# 'TC2336' telecentric lens
LX, H = 29.3, 1624  # mm / pixel

# pixel to mm magnification factor
Test.mm2pixel = LX / H
# Load conversion factor - testing machine
Test.LoadConvFactor = 1000  # N/V (gain = 500 N)
# Displacement conversion factor - testing machine
Test.DisplConvFactor = 1 # mm/V (gain = 20 mm)

if Job == 'e1o1':
    ##########################################################################
    Test.thickness = 14 # unit  mm
    Test.a0 = 24.975 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 2312, 1014
    Delta = 1.525 #average okoume
    Deltapix = Delta*(1/Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH-Deltapix, a0use.imgV
    ##########################################################################

elif Job == 'e1o2':
    ##########################################################################
    Test.thickness = 14 # unit  mm
    Test.a0 = 24.25 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0,0
    Delta = 2.25 #average okoume
    Deltapix = Delta*(1/Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH-Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e1o3':
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 24.5  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 0.5
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e1p1':
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 23.25  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 2.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e1p2':
    ##########################################################################
    Test.thickness = 13  # unit  mm
    Test.a0 = 23.4  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e1p3':
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 23.5 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 2
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e1i1':
    ##########################################################################
    Test.thickness = 14.2  # unit  mm
    Test.a0 = 24.975  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e2o1':
    ##########################################################################
    Test.thickness = 13  # unit  mm
    Test.a0 = 26.475  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.525  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e2o2':
    ##########################################################################
    Test.thickness = 14 # unit  mm
    Test.a0 = 27.975 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0,0
    Delta = 1.525 #average okoume
    Deltapix = Delta*(1/Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH-Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e2o3':
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 29  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e2p1':
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 27.4  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e2p2':
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 27.9  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e2p3':
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 27.5  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 0.5
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e3o1':
    ##########################################################################
    Test.thickness = 14.2  # unit  mm
    Test.a0 = 21.475  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.525  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e3o2':
    ##########################################################################
    Test.thickness = 14  # unit  mm
    Test.a0 = 22.75 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e3o3':
    ##########################################################################
    Test.thickness = 13.8  # unit  mm
    Test.a0 = 21.975  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.525
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e3p1':
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 21.9  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e3p2':
    ##########################################################################
    Test.thickness = 14.5 # unit  mm
    Test.a0 = 20.5 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0,0
    Delta = 1.6  # average padouck
    Deltapix = Delta*(1/Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH-Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e3p3':
    ##########################################################################
    Test.thickness = 15  # unit  mm
    Test.a0 = 24.6  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 0.9
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e4o1':
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 23.975  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.525
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e4p1':
    ##########################################################################
    Test.thickness = 13.5  # unit  mm
    Test.a0 = 23.25  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 2.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e4i1':
    ##########################################################################
    Test.thickness = 14.2  # unit  mm
    Test.a0 = 23.975  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e5o1':
    ##########################################################################
    Test.thickness = 14.1  # unit  mm
    Test.a0 = 24 # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 3  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e5p1':
    ##########################################################################
    Test.thickness = 14.5  # unit  mm
    Test.a0 = 22.75  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'e5i1':
    ##########################################################################
    Test.thickness = 13.8  # unit  mm
    Test.a0 = 23.975  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1  # average okoume
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'pbis':
    ##########################################################################
    Test.thickness = 14.2  # unit  mm
    Test.a0 = 23.9  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 1.6  # average padouck
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'pter':
    ##########################################################################
    Test.thickness = 12.8  # unit  mm
    Test.a0 = 23.25  # unit  mm
    # Selecting image coordinates of subset directly from MatdhID
    a0use.imgH, a0use.imgV = 0, 0
    Delta = 2.25
    Deltapix = Delta * (1 / Test.mm2pixel)
    a0.imgH, a0.imgV = a0use.imgH - Deltapix, a0use.imgV
    ##########################################################################
elif Job == 'others':
    pass

# Summary of DIC Settings
MatchID.CorrelationCoef = 'ZNSSD'
MatchID.InterpolationOrder = 'Bicubic spline'
MatchID.TransformationOrder = 'Affine'
MatchID.Subset, MatchID.Step = 21, 5
# Summary of Strain Settings
MatchID.StrainWindow = 77
MatchID.StrainConvention = 'GreenLagrange'
MatchID.StrainInterpolation = 'Q4'
# Area of Interest
MatchID.Roi_PolyXi, MatchID.Roi_PolyYi = 52, 259
MatchID.Roi_PolyXf, MatchID.Roi_PolyYf = 1587, 1119
##########################################################################
# Selecting pair
COD.cod_pair = 2
##########################################################################
MatchID.mm2step = MatchID.Step*Test.mm2pixel