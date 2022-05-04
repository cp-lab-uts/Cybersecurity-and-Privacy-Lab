


import numpy as np
import pandas as pd

from random import sample 
import random



class Fairness_metrics(object):
    def __init__(self,dataset, sensitive_attribute, y_test_prediction):
        """
        some popular fairness metrics
        dataset: dataset with true labels and predicted labels
        sensitive_attribute = 'sex' (for example)
        y_test_prediction = 'predicted label' in the test dataset
        """
        self.dataset = dataset
        self.sensitive_attribute = sensitive_attribute
        self.y_test_prediction = y_test_prediction
    
    def demographic_parity(self):
        """
        define demographic_parity
        """
        pro_positive = []
        pro_negative = []
        unpro_positive = []
        unpro_negative = []
        for i in range(0,len(self.dataset)):
            if self.dataset[self.sensitive_attribute][i] == 1 and self.dataset[self.y_test_prediction][i] == 1:
                pro_positive.append(i)
            elif self.dataset[self.sensitive_attribute][i] == 1 and self.dataset[self.y_test_prediction][i] == 0:
                pro_negative.append(i)
            elif self.dataset[self.sensitive_attribute][i] == 0 and self.dataset[self.y_test_prediction][i] == 1:
                unpro_positive.append(i)
            elif self.dataset[self.sensitive_attribute][i] == 0 and self.dataset[self.y_test_prediction][i] == 0:
                unpro_negative.append(i)
        num_pro = len(pro_positive) + len(pro_negative)
        num_unpro = len(unpro_positive) + len(unpro_negative)
        
        #n_p_P = len(p_P) # positive predictions in the protected group
        #n_p_N = len(p_N) # negative predictions in protected group
        #n_up_P = len(up_P) # positive predictions in the unprotected group
        #n_up_N = len(up_N) # negative predictions in the unprotected group
        #print(len(p_P),len(p_N),len(up_P),len(up_N))
        if num_pro == 0 or num_unpro == 0:
            pro_positive_rate = 100
            unpro_positive_rate = 100
            RD = 100  # risk difference
            
        elif len(pro_positive) == 0:
            pro_positive_rate = 100
            unpro_positive_rate = (len(unpro_positive) / num_unpro)
            RD = (len(unpro_positive) / num_unpro) - (len(pro_positive) / num_pro) 
            
        elif len(unpro_positive) == 0: 
            pro_positive_rate = (len(pro_positive) / num_pro) 
            unpro_positive_rate = 100
            RD = (len(unpro_positive) / num_unpro) - (len(pro_positive) / num_pro) 
             
        else:
            pro_positive_rate = (len(pro_positive) / num_pro) 
            unpro_positive_rate = (len(unpro_positive) / num_unpro)
            RD = (len(unpro_positive) / num_unpro) - (len(pro_positive) / num_pro)  # the lower, the better

        return RD, len(pro_positive), len(pro_negative), len(unpro_positive), len(unpro_negative), pro_positive_rate, unpro_positive_rate
    
    def equal_opportunity(self, true_label):
        """
        define fairness metrics: equal opporunity, equal odds, FRP and FNR
        """
        TP_up = []
        TN_up = []
        FP_up = []
        FN_up = []
        TP_p = []
        TN_p = []
        FP_p = []
        FN_p = []

        for i in range(0,len(self.dataset)):
            if self.dataset[true_label][i] == 1 and self.dataset[self.y_test_prediction][i] == 1 and self.dataset[self.sensitive_attribute][i] == 0:
                TP_up.append(i)
            elif self.dataset[true_label][i] == 1 and self.dataset[self.y_test_prediction][i] == 0 and self.dataset[self.sensitive_attribute][i] == 0:   
                FN_up.append(i)
            elif self.dataset[true_label][i] == 0 and self.dataset[self.y_test_prediction][i] == 1 and self.dataset[self.sensitive_attribute][i] == 0:
                FP_up.append(i) 
            elif self.dataset[true_label][i] == 0 and self.dataset[self.y_test_prediction][i] == 0 and self.dataset[self.sensitive_attribute][i] == 0:        
                TN_up.append(i)   

            elif self.dataset[true_label][i] == 1 and self.dataset[self.y_test_prediction][i] == 1 and self.dataset[self.sensitive_attribute][i] == 1:
                TP_p.append(i)
            elif self.dataset[true_label][i] == 1 and self.dataset[self.y_test_prediction][i] == 0 and self.dataset[self.sensitive_attribute][i] == 1: 
                FN_p.append(i)
            elif self.dataset[true_label][i] == 0 and self.dataset[self.y_test_prediction][i] == 1 and self.dataset[self.sensitive_attribute][i] == 1:
                FP_p.append(i)
            elif self.dataset[true_label][i] == 0 and self.dataset[self.y_test_prediction][i] == 0 and self.dataset[self.sensitive_attribute][i] == 1:
                TN_p.append(i)
                
        n_P_p = len(TP_p) + len(FN_p) # the number of postive labels in the protected group
        n_P_up = len(TP_up) + len(FN_up) # the number of positive labels in the unprotected group
        
        n_N_p = len(FP_p) + len(TN_p) # the number of negative labels in the protected group
        n_N_up = len(FP_up) + len(TN_up) #the number of negative labels in the unprotected group
        
        n = len(TP_up)+len(TN_up)+len(FP_up)+len(FN_up)+len(TP_p)+len(TN_p)+len(FP_p)+len(FN_p)
        if n_P_up == 0 or n_P_p == 0 or n_N_p == 0 or n_N_up==0:
            equal_oppo = 100
            equal_odds = 100
            TPR_p = 100
            TPR_up = 100
            FPR_p = 100
            FPR_up = 100
        else:
            TPR_p = len(TP_p) / (n_P_p)
            TPR_up = len(TP_up) / (n_P_up)
            FPR_p = len(FP_p) / (n_N_p)
            FPR_up = len(FP_up) / (n_N_up)
            FNR_p = len(FN_p) / n_P_p
            FNR_up = len(FN_up) / n_P_up
              
            equal_oppo = (len(TP_up) / n_P_up) - (len(TP_p) / n_P_p) # closer to 1, the better
            equal_odds = equal_oppo + abs((len(FP_up) / n_N_up) - (len(FP_p) / n_N_p))
            FPR = FPR_p - FPR_up
            FNR = FNR_p - FNR_up

            OMR = (FPR_p - FPR_up) + (FNR_p  - FNR_up)
        return equal_oppo, equal_odds, FPR, FNR, OMR, TPR_p, TPR_up, FPR_p, FPR_up

