import os

# USER ###################################
caminho = 'C:\\Users\\pc\\Desktop\\MMCGTests\\'
Job = 'pter'
old_Job = 'pterbis'
##########################################

# The listdir method lists out all the content of a given directory.
# list = os.listdir(cwd) # Src is the source to be listed out.

def upname(cwdi):
    for count, filename in enumerate(os.listdir(cwdi)):
        ini = os.path.join(cwdi, filename)
        out = os.path.join(cwdi, Job + filename[4:])
        # rename() function will rename all the files
        os.rename(ini, out)

upname(os.path.join(caminho, Job, 'X[Pixels]'))
upname(os.path.join(caminho, Job, 'Y[Pixels]'))
upname(os.path.join(caminho, Job, 'U'))
upname(os.path.join(caminho, Job, 'V'))
upname(os.path.join(caminho, Job, 'Exx'))
upname(os.path.join(caminho, Job, 'Eyy'))
upname(os.path.join(caminho, Job, 'Exy'))