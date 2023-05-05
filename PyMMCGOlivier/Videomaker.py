#Video Displ-Load
run=0
#run = int(input("Please enter 1 if you want the video: "))
if run == 1:
    for i in range(len(MatchID.displ)):
        fig, ax = plt.subplots(figsize=(7,5))
        plt.plot(MatchID.displ, MatchID.load, 'k-', linewidth=3)
        plt.plot(MatchID.displ[i], MatchID.load[i],'bo', markersize=10)
        plt.xlabel('Displacement, mm')
        plt.ylabel('Load, N')
        plt.title(Job)
        fig.tight_layout()
        plt.grid()
        plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(i)+".png")
        plt.show()
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
    files = os.listdir(path)
    files.sort()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(path+'\Disp-Load.avi', fourcc, 10.0, (640, 480))
    for j in range(len(MatchID.displ)): 
        img = cv.imread(os.path.join(path, "Img"+str(j)+".png"))
        img = cv.resize(img, (640, 480))
        output.write(img)
        os.remove(os.path.join(path, "Img"+str(j)+".png"))
    output.release()
    cv.destroyAllWindows()

#video specimen
run=0
#run = int(input("Please enter 1 if you want the video: "))
if run == 1:
    for i in MatchID.time:
        pathdados = os.path.join(cwd, Job + "_" + f"{i:04d}" + '_0.tiff')
        img0 = cv.imread(pathdados, cv.IMREAD_GRAYSCALE) # cv.imread(pathdados, 0)
        dpi = plt.rcParams['figure.dpi']
        Height, Width = img0.shape
        print(i)
        figsize = Width/float(dpi), Height/float(dpi)
        fig = plt.figure(figsize=figsize)
        cor = (255, 255, 255)
        thickness = 1
        start_point = (MatchID.SubsetXi,MatchID.SubsetYi)
        end_point = (MatchID.SubsetXf,MatchID.SubsetYf)
        img0 = cv.rectangle(img0, start_point, end_point, cor, thickness)
        plt.imshow(img0, cmap='gray', vmin=0, vmax=255)
        plt.plot(a0.imgHuse,a0.imgVuse, color='red', marker='+', markersize=50)
        plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(i)+".png")
        plt.show()
        
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
    files = os.listdir(path)
    files.sort()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(path+'\MMCG.avi', fourcc, 10.0, (640, 480))  
          
    for j in MatchID.time: 
        img = cv.imread(os.path.join(path, "Img"+str(j)+".png"))
        img = cv.resize(img, (640, 480))
        output.write(img)
        os.remove(os.path.join(path, "Img"+str(j)+".png"))
    output.release()
    cv.destroyAllWindows()

#Video CTOD    
run=0
#run = int(input("Please enter 1 if you want the video: "))
if run == 1:
    for i in range(len(COD.wI)):
        fig, ax = plt.subplots(figsize=(7,5))
        plt.plot(MatchID.displ[:i+1], COD.wI[:i+1], 'b-', linewidth=4, label='Mode I with COD pair : %d' %COD.cod_pair)
        plt.plot(MatchID.displ[:i+1], COD.wII[:i+1], 'k--', label='Mode II with COD pair : %d' %COD.cod_pair)
        plt.xlim(0, 1.4)
        plt.ylim(0, 0.35)
        plt.xlabel('CTOD, mm')
        plt.ylabel('Load, N')
        ax.set_xlim(xmin=0)
        ax.set_ylim(bottom=0)
        plt.grid()
        plt.legend(loc=2, prop={'size': 8})
        fig.tight_layout()
        plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(i)+".png")
        plt.show()
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
    files = os.listdir(path)
    files.sort()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(path+'\CTOD.avi', fourcc, 10.0, (640, 480))
    for j in range(len(COD.wI)): 
        img = cv.imread(os.path.join(path, "Img"+str(j)+".png"))
        img = cv.resize(img, (640, 480))
        output.write(img)
        os.remove(os.path.join(path, "Img"+str(j)+".png"))
    output.release()
    cv.destroyAllWindows()

#Video Crack
run=0
#run = int(input("Please enter 1 if you want the video: "))
if run == 1:
    for j in stagEval:
        fig = plt.figure()
        plt.imshow(fract_K[:, :, j])
        plt.plot(a0.X,a0.Y,'sr')
        plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(j)+".png")
        plt.colorbar()
        plt.show()
        
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
    files = os.listdir(path)
    files.sort()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(path+'\output.avi', fourcc, 10.0, (640, 480))  
          
    for j in stagEval: 
        img = cv.imread(os.path.join(path, "Img"+str(j)+".png"))
        img = cv.resize(img, (640, 480))
        output.write(img)
        os.remove(os.path.join(path, "Img"+str(j)+".png"))
    output.release()
    cv.destroyAllWindows()

#Video Crack2    
run=0
#run = int(input("Please enter 1 if you want the video: "))
if run == 1:
    for j in stagEval:
        fig = plt.figure()
        plt.imshow(UY[:, :, j])
        plt.plot(UY.shape[1]-crackL_J_pixel_X[j, chos_alp],crackL_J_pixel_Y[j, chos_alp],'sr')
        plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(j)+".png")
        plt.colorbar()
        plt.title(Job)
        plt.show()
        
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
    files = os.listdir(path)
    files.sort()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(path+'\Cracklength.avi', fourcc, 10.0, (640, 480))  
          
    for j in stagEval: 
        img = cv.imread(os.path.join(path, "Img"+str(j)+".png"))
        img = cv.resize(img, (640, 480))
        output.write(img)
        os.remove(os.path.join(path, "Img"+str(j)+".png"))
    output.release()
    cv.destroyAllWindows()

#Video Crack length    
run=1
#run = int(input("Please enter 1 if you want the video: "))
if run == 1:
    for i in range(len(MatchID.displ)):
        fig, ax = plt.subplots(figsize=(7,5))
        plt.plot(MatchID.time[:i],crackL_J_mm[:i,chos_alp], '*r--', linewidth=3, label='Method1')
        plt.plot(MatchID.time[:i], dad[:i], 'b', label='Method2')
        plt.xlabel('Images')
        plt.ylabel('Crack length, a(t), mm')
        plt.xlim(0, 175)
        plt.ylim(21, 55)
        plt.title(Job)
        fig.tight_layout()
        plt.grid()
        plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(i)+".png")
        plt.show()
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
    files = os.listdir(path)
    files.sort()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(path+'\Crack-Time.avi', fourcc, 10.0, (640, 480))
    for j in range(len(MatchID.displ)): 
        img = cv.imread(os.path.join(path, "Img"+str(j)+".png"))
        img = cv.resize(img, (640, 480))
        output.write(img)
        os.remove(os.path.join(path, "Img"+str(j)+".png"))
    output.release()
    cv.destroyAllWindows()

#Video G    
run=1
#run = int(input("Please enter 1 if you want the video: "))
if run == 1:
    for i in range(len(MatchID.displ)):
        fig, ax = plt.subplots(figsize=(7,5))
        #plt.plot(a_t[:i], G1[:i], 'r:', linewidth=2, label='R-Curve alpha '+ str(chos_alp))
        #plt.plot(dad[:i], G2[:i], 'b:', linewidth=2, label='Method2')
        plt.plot(a_interp[:i], G_interp1[:i], 'g:', linewidth=2, label='Method I interpolated alpha  '+ str(chos_alp))
        plt.xlabel('Crack length, a(t), mm')
        plt.ylabel('$G_{Ic}, J$')
        plt.legend(loc=2, prop={'size': 8})
        plt.xlim(23, 30)
        plt.ylim(0, 900)
        plt.title(Job)
        fig.tight_layout()
        plt.grid()
        plt.savefig("D:\Recherche PRD\EXP\MMCGTests\Video\Img"+str(i)+".png")
        plt.show()
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video" 
    files = os.listdir(path)
    files.sort()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter(path+'\Energy-Crack.avi', fourcc, 10.0, (640, 480))
    for j in range(len(MatchID.displ)): 
        img = cv.imread(os.path.join(path, "Img"+str(j)+".png"))
        img = cv.resize(img, (640, 480))
        output.write(img)
        os.remove(os.path.join(path, "Img"+str(j)+".png"))
    output.release()
    cv.destroyAllWindows()
    
#Combined videos
run=1
if run==1:
    path =  "D:\Recherche PRD\EXP\MMCGTests\Video"
    cap1 = cv.VideoCapture(path+'\Disp-Load.avi')
    cap2 = cv.VideoCapture(path+'\Crackspecimen.avi')
    cap3 = cv.VideoCapture(path+'\Crack-Time.avi')
    cap4 = cv.VideoCapture(path+'\Energy-Crack.avi')
    
    # Récupérer les dimensions de la vidéo
    width = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    # Créer un objet VideoWriter pour écrire la vidéo combinée
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    combined_video = cv.VideoWriter(path+'\combined_video.mp4', fourcc, 25.0, (2*width, 2*height))
    
    # Boucle pour lire les images de chaque vidéo et les combiner
    while True:
        # Lire les images des 4 vidéos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
    
        # Vérifier si toutes les vidéos ont été lues
        if not ret1 or not ret2 or not ret3 or not ret4:
            break
    
        # Redimensionner les images à la même taille
        frame1 = cv.resize(frame1, (width, height))
        frame2 = cv.resize(frame2, (width, height))
        frame3 = cv.resize(frame3, (width, height))
        frame4 = cv.resize(frame4, (width, height))
    
        # Combiner les 4 images en une seule
        combined_frame = cv.vconcat([cv.hconcat([frame1, frame2]), cv.hconcat([frame3, frame4])])
    
        # Écrire la frame combinée dans la vidéo
        combined_video.write(combined_frame)
        
    
    
    # Fermer toutes les fenêtres et libérer les ressources
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    combined_video.release()
    cv.destroyAllWindows()