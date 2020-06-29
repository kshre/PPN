# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
from pgportfolio.marketdata.coinlist import CoinList
import numpy as np
import pandas as pd
from pgportfolio.tools.data import panel_fillna
from pgportfolio.constants import *
import sqlite3
from datetime import datetime
import logging


class HistoryManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, coin_number, end, volume_average_days=1, volume_forward=0, online=True):
        self.initialize_db()
        self.__storage_period = DAY #FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        #if self._online:
        #    self._coin_list = CoinList(end, volume_average_days, volume_forward) 
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__coins = None

    @property
    def coins(self):
        return self.__coins

    def initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_panel(start, end, period, features).values


    def get_global_panel(self, start, end, period=300, features=('close',)):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        start = int(start - (start%period)) # Subtract the remainder
        end = int(end - (end%period))  # Subtract the remainder
        
        
        # read data
        df = pd.read_csv('./database/all_stocks_5yr.csv')
        #df.describe()
        df.head(5)
        
        # asset name list
        name_list = df['Name']
        coins = name_list.drop_duplicates(keep='first',inplace=False).tolist()
        self.__coins = coins 
        
        time = df['date']
        time_index=time.drop_duplicates(keep='first',inplace=False).tolist()
        #for coin in coins:
        #    self.update_data(start, end, coin)  # update data

        if len(coins)!=self._coin_number:
            raise ValueError("the length of selected coin (%d) is not equal to expected %d"
                             % (len(coins), self._coin_number))

        logging.info("feature type list is %s" % str(features))
        self.__checkperiod(period)  # check whether the period belongs to the required ranges
        features=['open','high','low','close']
        #time_index = pd.to_datetime(list(range(start, end+1, period)),unit='s')  
        panel = pd.Panel(items=features, major_axis=coins, minor_axis=time_index, dtype=np.float32)
        
        
        # obtain the global data
        if os.path.exists('./database/data_new.pkl'):
        #print('read_data')
            panel = pd.read_pickle("./database/data_new.pkl")
            
        else: 
        #print('process data')
            for sample_number, coin in enumerate(coins):
                for feature in features:
                  
                    temp_data = df.loc[df["Name"]==coin] 
                    temp_data = temp_data.set_index('date') 
                    temp_data =temp_data[feature]
                    if np.sum(np.isnan(temp_data))>0: 
                        temp_data =temp_data.fillna(method='backfill') 
                    temp=0
                    if np.shape(temp_data)[0]!= np.shape(time_index)[0]: 
                        temp_data1 = pd.DataFrame(np.random.randn(1259),index=time_index)
                        
                        for time_id in time_index: 
                            if time_id in temp_data.index.tolist(): 
                                temp = temp_data.loc[time_id] 
                                temp_data1.loc[time_id] = temp
                            else:
                                if time_id == '2013-02-08': 
                                    temp = 1
                                    temp_data1.loc[time_id] = temp
                                else:
                                    temp_data1.loc[time_id] = temp 
                        temp_data = temp_data1.squeeze() 
                    else:
                        temp_data.index = time_index 
                    panel.loc[feature, coin, time_index] = temp_data.squeeze() 
            f = open('./database/data_new.pkl','wb')
            panel.to_pickle(f)
            #print('save_data')
            f.close            
        return panel

 

    # add new history data into the database
    def update_data(self, start, end, coin):
        # update the data
        connection = sqlite3.connect(DATABASE_DIR)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]

            if min_date==None or max_date==None:
                self.__fill_data(start, end, coin, cursor)    
            else:
                if max_date+10*self.__storage_period<end:
                    if not self._online:
                        raise Exception("Have to be online")
                    self.__fill_data(max_date + self.__storage_period, end, coin, cursor)
                if min_date>start and self._online:
                    self.__fill_data(start, min_date - self.__storage_period-1, coin, cursor)
 
        finally:
            connection.commit()
            connection.close()
 
    def __checkperiod(self, period):
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')
  

    def __fill_data(self, start, end, coin, cursor):
        # fill the data into the databse
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.allActiveCoins.at[coin, 'pair'],
            start=start,
            end=end,
            period=self.__storage_period)
        logging.info("fill %s data from %s to %s"%(coin, datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                                            datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')))
        for c in chart: 
            if c["date"] > 0:
                if c['weightedAverage'] == 0:
                    weightedAverage = c['close']
                else:
                    weightedAverage = c['weightedAverage']
 
                if 'reversed_' in coin:
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                        (c['date'],coin,1.0/c['low'],1.0/c['high'],1.0/c['open'],
                        1.0/c['close'],c['quoteVolume'],c['volume'],
                        1.0/weightedAverage))
                else:
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                                   (c['date'],coin,c['high'],c['low'],c['open'],
                                    c['close'],c['volume'],c['quoteVolume'],
                                    weightedAverage))
