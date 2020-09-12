from scipy.stats import norm 
from HR import HistoricalRate 
from LFGP import LikelihoodFreeGaussianProcess

class BinaryOption_main:
    
    def __init__(self, ID, token, instrument, d=50):
        self.HR = HistoricalRate(ID, token, instrument, d)  
        self.d = d
        
    def get_data(self, start_train, end_train, start_test, end_test):
        self.X_train, self.y_train = self.HR.history_data(start_train, end_train)
        try:
            self.X_test, self.y_test = self.HR.history_data(start_test, end_test)
        except:
            self.X_test, self.y_test = self.HR.history_data(start_test, end_test)        
    
    def train(self, n_min):
        self.High = LikelihoodFreeGaussianProcess(map_X=None, map_y=95/195*100)        
        self.High.fit(self.X_train, self.y_train, n_min=n_min, display=False)
        self.Low = LikelihoodFreeGaussianProcess(map_X=None, map_y=100/195*100)        
        self.Low.fit(self.X_train, self.y_train, n_min=n_min, display=False)
    
    def test(self, alpha, payout=1.95):
        y_myu_High, y_sigma_High = self.High.predict(self.X_test)
        y_myu_Low, y_sigma_Low = self.Low.predict(self.X_test)
        conf = norm.ppf(1 - alpha)
        profit = [0]
        for i in range(len(self.y_test)):
            High = y_myu_High[i] - y_sigma_High[i] * conf
            Low = y_myu_Low[i] + y_sigma_Low[i] * conf
            if High >= 0.0005 and Low <= -0.0005:
                if High >= - Low:
                    if self.y_test[i] >= 0.0005:
                        profit += [profit[-1] + payout - 1]
                    else:
                        profit += [profit[-1] - 1]
                else:
                    if self.y_test[i] <= -0.0005:
                        profit += [profit[-1] + payout - 1]
                    else:
                        profit += [profit[-1] - 1]
            elif High >= 0.0005:
                if self.y_test[i] >= 0.0005:
                    profit += [profit[-1] + payout - 1]
                else:
                    profit += [profit[-1] - 1]
            elif Low <= -0.0005:
                if self.y_test[i] <= -0.0005:
                    profit += [profit[-1] + payout - 1]
                else:
                    profit += [profit[-1] - 1]
        return profit