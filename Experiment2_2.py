###############################################################################
# Experiment2_2 (correspoding to Figure3)
###############################################################################
test = 'Backtesting' # 'Evaluation' 'Backtesting'
instrument = "GBP_JPY" 
model_list = ["LFGP","RF"]
d_LFGP = 10
d_RF = 20
n_min = 100 
depth = 10 
alpha_list = [0.5, 0.45, 0.4]
beta_list = [0.5, 0.49, 0.48]
c_list1 = ['red', 'blue', 'green']
c_list2 = ['pink', 'deepskyblue', 'lightgreen']
x_max = 140000
y_min = -1500
y_max = 2500 
start_train = '2017-04-01 00:00:00' 
end_train = '2018-04-01 00:00:00' 
if test == 'Evaluation' :
    start_test = '2018-04-01 00:00:00' 
    end_test = '2019-04-01 00:00:00' 
elif test == 'Backtesting' :
    start_test = '2019-04-01 00:00:00' 
    end_test = '2020-04-01 00:00:00'
###############################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

for model in model_list:    
    if model == "LFGP":
        with open('LFGP_'+str(n_min)+'_'+str(d_LFGP)+'.pickle', mode='rb') as fp:
                BO = pickle.load(fp)
        BO.get_data(start_train, end_train, start_test, end_test)
        print('n =', len(BO.X_train), 'n* =', len(BO.X_test))
        sns.set()
        for i, alpha in enumerate(alpha_list):
            profit = BO.LFGP_test(alpha)
            plt.plot(profit, c=c_list1[i], label='α = {:.2f}'.format(alpha))
            print('profit:', profit[-1])
        plt.legend(frameon=True, facecolor='white')
        plt.xlabel('Counts')
        plt.ylabel('Comulative Profit')    
        plt.xlim(0, x_max)
        plt.ylim(y_min, y_max)
        plt.title('LFGP')
        plt.savefig("./" + model + '_' + test + ".eps", bbox_inches='tight')
        plt.show()
    elif model == "RF":
        with open('RF_'+str(depth)+'_'+str(d_RF)+'.pickle', mode='rb') as fp:
                BO = pickle.load(fp)
        BO.get_data(start_train, end_train, start_test, end_test)
        print('n =', len(BO.X_train), 'n* =', len(BO.X_test))
        sns.set()
        for i, beta in enumerate(beta_list):
            profit = BO.RF_test(beta)
            plt.plot(profit, c=c_list2[i], label='β = {:.2f}'.format(beta))
            print('profit:', profit[-1])
        plt.legend(frameon=True, facecolor='white')
        plt.xlabel('Counts')
        plt.ylabel('Comulative Profit')    
        plt.xlim(0, x_max)
        plt.ylim(y_min, y_max)
        plt.title('RF')
        plt.savefig("./" + model + '_' + test + ".eps", bbox_inches='tight')
        plt.show()