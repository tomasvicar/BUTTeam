from driver import save_challenge_predictions
from read_file import read_data,LabelReader
import os

output_directory='../test'
try:
    os.mkdir(output_directory)
except:
    pass


labelReader=LabelReader()




filename=r"C:\Users\Tom\Desktop\tmp2_cinc2020\data\Training_WFDB\A0001.hea"

loaded=labelReader(filename)


path,filename=os.path.split(filename)

save_challenge_predictions(output_directory,filename,scores,labels,classes)


filename=r"C:\Users\Tom\Desktop\tmp2_cinc2020\data\Training_WFDB\A0001.hea"


path,filename=os.path.split(filename)


save_challenge_predictions(output_directory,filename,scores,labels,classes)












