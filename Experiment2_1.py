###############################################################################
# Experiment2_1 (correspoding to Table4)
###############################################################################
instrument = "GBP_JPY" 
start_train = '2017-04-01 00:00:00' 
end_train = '2018-04-01 00:00:00' 
start_test = '2018-04-01 00:00:00' 
end_test = '2019-04-01 00:00:00' 
model_list = ["LFGP","RF"]
d_list = [10,20,30]
n_min_list = [100,200,300] 
depth_list = [5,10,15] 
alpha = 0.40
beta = 0.48
iter_n = 5 
###############################################################################

from info import get_info
from BO import BinaryOption
import pickle

ID, token = get_info()
for d in d_list:
    BO = BinaryOption(ID, token, instrument, d=d)
    BO.get_data(start_train, end_train, start_test, end_test)    
    print('d =', d, 'n =', len(BO.X_train), 'n* =', len(BO.X_test))
    for model in model_list:
        if model == "LFGP":
            print("LFGP")
            for n_min in n_min_list:
                profit0 = 0
                for i in range(iter_n):
                    BO.LFGP_train(n_min)
                    profit = BO.LFGP_test(alpha)
                    if profit0 < profit[-1] or i == 0:
                        profit0 = profit[-1]
                        with open('LFGP_'+str(n_min)+'_'+str(d)+'.pickle', mode='wb') as fp:
                            pickle.dump(BO, fp)
                print('n0', n_min, 'profit', profit0)
        elif model == "RF":
            print("RF")
            for depth in depth_list:
                profit0 = 0
                for i in range(iter_n):
                    BO.RF_train(depth)
                    profit = BO.RF_test(beta)
                    if profit0 < profit[-1] or i == 0:
                        profit0 = profit[-1]
                        with open('RF_'+str(depth)+'_'+str(d)+'.pickle', mode='wb') as fp:
                            pickle.dump(BO, fp)
                print('depth', depth, 'profit', profit0)    