###############################################################################
# Experiment2 (figure4, figure5)
###############################################################################
instrument = "GBP_JPY" 
start_train = '2018-09-01 00:00:00' 
end_train = '2019-09-01 00:00:00' 
start_test = '2019-09-01 00:00:00' 
end_test = '2020-09-01 00:00:00' 
test = 'back' # eval, back
save = '0'
d_main = 10
n_min_main = 100 
d_base = 30
n_min_base = 200 
x_max = 100000
y_min = -500
y_max = 2000
###############################################################################
from info import get_info
from BO_main import BinaryOption_main
from BO_base import BinaryOption_base
import matplotlib.pyplot as plt
import pickle

if save == '0' :
    ID, token = get_info()
    BO_main = BinaryOption_main(ID, token, instrument, d=d_main)
    BO_main.get_data(start_train, end_train, start_test, end_test)        
    BO_main.train(n_min_main)
    BO_base = BinaryOption_base(ID, token, instrument, d=d_base)
    BO_base.get_data(start_train, end_train, start_test, end_test)        
    BO_base.train(n_min_base)
if save != '0' :    
    with open('BO_main_'+str(n_min_main)+'_'+str(d_main)+'_'+test+'.pickle', mode='rb') as fp:
            BO_main = pickle.load(fp)
    with open('BO_base_'+str(n_min_base)+'_'+str(d_base)+'_'+test+'.pickle', mode='rb') as fp:
            BO_base = pickle.load(fp)
print('d:', d_main, 'n0:', n_min_main, 'n:', len(BO_main.X_train), 'n*:', len(BO_main.X_test))
print('d:', d_base, 'n0:', n_min_base, 'n:', len(BO_base.X_train), 'n*:', len(BO_base.X_test))
alpha_list = [0.5, 0.45, 0.4, 0.35]
for alpha in alpha_list:
    profit_main = BO_main.test(alpha)
    profit_base = BO_base.test(alpha)
    fig = plt.figure()
    plt.grid(which = "major", axis = "x", color = "black", linestyle = "--", linewidth = 0.1)
    plt.grid(which = "major", axis = "y", color = "black", linestyle = "--", linewidth = 0.1)            
    plt.plot(profit_main, label='Proposal', color="#FF1FEC")
    plt.plot(profit_base, label='Baseline', color="#20B9FF")
    plt.legend(frameon=True)
    plt.xlabel('Counts')
    plt.ylabel('Cumulative Profit')    
    plt.xlim(0, x_max)
    plt.ylim(y_min, y_max)
    plt.title('α = {:.2f}'.format(alpha), fontsize=30)
    plt.savefig("./alpha" + '{:.2f}'.format(alpha) + '_' + test + ".eps", bbox_inches='tight')
    plt.show()
if save == '0' :
    with open('BO_main_' + str(n_min_main) + '_' + str(d_main) + '_' + test + '.pickle', mode='wb') as fp:
        pickle.dump(BO_main, fp)
    with open('BO_base_' + str(n_min_base) + '_' + str(d_base) + '_' + test + '.pickle', mode='wb') as fp:
        pickle.dump(BO_base, fp)
