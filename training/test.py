import driver_tom
import evaluate_12ECG_score

data_path='../../test_data'

res_path='../../res'


         
driver_tom.driver(data_path,res_path)


auroc,auprc,accuracy,f_measure,f_beta,g_beta = evaluate_12ECG_score.evaluate_12ECG_score(data_path,res_path)

output_string = 'AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure\n{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}'.format(auroc,auprc,accuracy,f_measure,f_beta,g_beta)


print(output_string)
    
    
    