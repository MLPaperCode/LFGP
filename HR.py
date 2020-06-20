import numpy as np
import pandas as pd
import datetime
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API

class HistoricalRate:
    
    def __init__(self, ID, token, instrument, d):
        self.ID = ID
        self.api = API(access_token=token, environment="practice")
        self.instrument = instrument
        self.d = d
            
    def get_data(self, count, location='Asia/Tokyo', start=None):
        def convert(x):
            try:
                x = pd.to_datetime(x).tz_convert(location)
            except:
                x = pd.to_datetime(x).tz_localize('utc').tz_convert(location)
            return str(x)[:-6]
        if start is None:
            params = {"count":count, "granularity":'S30'}
        else:
            start_ = start.timestamp()
            params = {"count":count, "granularity":'S30', "from":start_}            
        r = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
        self.api.request(r)
        candle = r.response['candles']
        history_ = [[convert(c['time']), float(c['mid']['c'])] for c in candle]
        return history_
    
    def W(self, t):
        if t.month == 1 and t.day <= 5:
            return False
        elif t.month == 12 and t.day >= 24:
            return False
        elif t.weekday() == 6:
            return False
        elif t.weekday() == 5:
            if t.hour >= 5:
                return False
            else:
                return True                    
        elif t.weekday() == 0:
            if t.hour < 8:
                return False
            else:
                return True
        else:
            if 8 > t.hour >= 5:
                return False
            else:
                return True    
    
    def history_data(self, start, end):
        if type(start) is str:
            start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        if type(end) is str:
            end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')        
        t = pd.date_range(start=start, end=end, freq='30S')
        t = pd.DataFrame(index=t, columns=[])
        history = []
        while end.timestamp() > start.timestamp():
            history += self.get_data(5000, start=start)
            last = history[-1][0]
            start = datetime.datetime.strptime(last, '%Y-%m-%d %H:%M:%S')
            start += datetime.timedelta(seconds=30)
        history = pd.DataFrame(history)
        history.columns = ['Time', 'Close']
        history = history.set_index('Time')
        C = t.join(history)
        C = C.fillna(method="ffill").fillna(method="bfill")
        t = C.index
        C = np.array(C)[:,0]
        X = np.array([[C[i - j] - C[i - j - 1] for j in range(self.d)] for i in range(self.d, len(C) - 1) 
            if self.W(t[i - self.d]) == True and self.W(t[i + 1]) == True and t[i].second == 30])
        y = np.array([C[i + 1] - C[i] for i in range(self.d, len(C) - 1) 
            if self.W(t[i - self.d]) == True and self.W(t[i + 1]) == True and t[i].second == 30])
        return X, y