from driver import save_challenge_predictions


output_directory='../test'
try:
    mkdir(output_directory)
except:
    pass







filename='test_file1'

save_challenge_predictions(output_directory,filename,scores,labels,classes)




filename='test_file2'

save_challenge_predictions(output_directory,filename,scores,labels,classes)












