#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import tflearn 
import numpy as np

class NeuralNetWork:
    def __init__(self, feature_number, rows, columns, device):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth=True
        self.session = tf.Session(config=tf_config)
        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4

        self.input_num = tf.placeholder(tf.int32, shape=[])
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns])
        # rows indicate coin numbers; columns indicate features，feature_number indicates the feature number，None indicates the batch size
        self.previous_w = tf.placeholder(tf.float32, shape=[None, rows])
        self._rows = rows
        self._columns = columns

        self.layers_dict = {}
        self.layer_count = 0
        self.dropout = 0.2
        self.output = self._build_network()  
 
 
class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self, feature_number, rows, columns, device):
        # initial the network 
        NeuralNetWork.__init__(self, feature_number, rows, columns, device)

    def add_layer_to_dict(self, layer_type, tensor, weights=True):
        # define the dictornary name
        self.layers_dict[layer_type + '_' + str(self.layer_count) + '_activation'] = tensor
        self.layer_count += 1
 
    def _build_network(self):  
        """data reformulation""" 
        network = tf.transpose(self.input_tensor, [0, 2, 3, 1])      
        network = network / network[:, :, -1, 0, None, None]

        
        #######################       feature extraction      ###################################
        """feature extraction: temporal convolutional network (TCN) and LSTM """
        
        # TCN
        network1 = self._TemporalBlock(network, n_outputs=8, kernel_size=[1,3], 
                                     stride=1, dilation=(1,2 ** 0), padding=(3-1)*(2 ** 0), dropout=0.2,level=0) 
 
        network1 = self._TemporalBlock(network1, n_outputs=16, kernel_size=[1,3], 
                                     stride=1, dilation=(1,2 ** 1), padding=(3-1)*(2 ** 1), dropout=0.2,level=1) 
         
        network1 = self._TemporalBlock(network1, n_outputs=16, kernel_size=[1,3], 
                                     stride=1, dilation=(1,2 ** 2), padding=(3-1)*(2 ** 2), dropout=0.2,level=2) 
        
        width = int(network1.get_shape()[2])
        network1 = tflearn.layers.conv_2d(network1,16,[1,width],1,padding='valid',regularizer='L2',weight_decay=1e-09,activation='relu') 
        
        # LSTM
        coin_number = int(network.get_shape()[1]) 
        temporal_dimension = int(network.get_shape()[2]) 
        feature_dimension = int(network.get_shape()[3]) 
        network2 =  tf.reshape(network,[-1,temporal_dimension,feature_dimension])
        network2 = tflearn.layers.lstm(network2, n_units= 16, scope="lstm"+str(4)) 
        network2 = tf.reshape(network2, [-1, coin_number, 16])
        network2 = tf.reshape(network2,[-1,coin_number,1,16]) 
                
        """
        # old code
        network2 = tf.transpose(network, [0, 2, 3, 1])  
        resultlist = []
        reuse = False
        
        for i in range(coin_number):
            if i > 0:
                reuse = True    
            # LSTM    
            result = tflearn.layers.lstm(network2[:, :, :, i], n_units= 16, scope="lstm"+str(4), reuse=reuse)
            resultlist.append(result) 
            
            # GRU
            # result = tflearn.layers.recurrent.gru(network2[:, :, :, i], n_units= 16, scope="gru"+str(3), reuse=reuse) 
            #resultlist.append(result)
            
            # SRU
            # rnn_cell = tf.contrib.rnn.SRUCell(num_units=16,reuse=True) 
            # result, states = tf.nn.dynamic_rnn(rnn_cell, network2[:, :, :, i], dtype=tf.float32)
            # resultlist.append(result[:, -1, :]) 
            
            
        network2 = tf.stack(resultlist) 
        network2 = tf.transpose(network2, [1, 0, 2])   

        features = int(network2.get_shape()[2])
        network2 = tf.reshape(network2, [-1, coin_number, 1, features])  
        """
        

        
        
        ##########################       decision making        ########################################### 
        """decision making module""" 
        # FC
        w = tf.reshape(self.previous_w, [-1,self._rows,1,1]) # the portfolio vector from last period 
        network = tf.concat([network1, network2, w],axis=3)  # combine the features
        #network = tf.concat([network1, w],axis=3)  # combine the features
        # combine the btc bias
        btc_bias = tf.get_variable("btc_bias",[1,1,1,1],dtype=tf.float32, initializer=tf.zeros_initializer)
        features = int(network.get_shape()[3])
        btc_bias = tf.tile(btc_bias,[self.input_num,1,1,features])   
        network = tf.concat([btc_bias,network],1)     


        "three strategies to make decision, where the leverage operation is beyond the version of the paper."
        # (1) no leverage opperation
        # decision = tflearn.layers.fully_connected(network, coin_number+1, activation='softmax')
        
    
        # (2) with 2-head leverage opperation
        #  intial portfolio vector
        decision_i = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",regularizer='L2',weight_decay=1e-09)
        decision_i = decision_i[:,:,0,0]
        decision_i = tflearn.layers.core.activation(decision_i,activation='softmax')
        
        # short sale  vector
        decision_s = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",regularizer='L2',weight_decay=1e-09)
        decision_s = decision_s[:,:,0,0]
        decision_s = tflearn.layers.core.activation(decision_s,activation='softmax')

        # final decision
        decision =  2*decision_i - decision_s        
              
        
        
        # (3) with 3-head leverage opperation
        # intial portfolio vector
        #decision_i = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",regularizer='L2',weight_decay=1e-09)
        #decision_i = decision_i[:,:,0,0]
        #decision_i = tflearn.layers.core.activation(decision_i,activation='softmax')
        
        # short sale  vector
        #decision_s = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",regularizer='L2',weight_decay=1e-09)
        #decision_s = decision_s[:,:,0,0]
        #decision_s = tflearn.layers.core.activation(decision_s,activation='softmax')

        # reinvestment vector
        #decision_r = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",regularizer='L2',weight_decay=1e-09)
        #decision_r = decision_r[:,:,0,0]
        #decision_r = tflearn.layers.core.activation(decision_r,activation='softmax')
        
        # final decision
        #decision =  decision_i - decision_s + decision_r 
        
        return decision
 

    def _TemporalBlock(self, value, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,level=0): 
        
        n_inputs = int(value.get_shape()[3]) 
        padded_value1 = tf.pad(value, [[0,0], [0,0], [padding,0], [0,0]]) 
        self.conv1 = tf.layers.conv2d(padded_value1,
                                    filters=n_outputs,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='valid',
                                    dilation_rate=dilation,
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer'+str(level)+'_conv1')
        self.output1 = tf.nn.dropout(tf.nn.relu(self.conv1), keep_prob=1-dropout) 
        padded_value2 = tf.pad(self.output1, [[0,0], [0,0], [padding,0], [0,0]])
        
        
        self.conv2 = tf.layers.conv2d(inputs=padded_value2,
                                    filters=n_outputs,
                                    kernel_size=kernel_size,   
                                    strides=stride,
                                    padding='valid',
                                    dilation_rate=dilation,
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer'+str(level)+'_conv2')
        self.output2 = tf.nn.dropout(tf.nn.relu(self.conv2), keep_prob=1-dropout)

        asset_num = int(self.output2.get_shape()[1])
        self.conv3 = tf.layers.conv2d(inputs=self.output2,
                                    filters=n_outputs,
                                    kernel_size=[asset_num,1],
                                    strides=stride,
                                    padding='same',
                                    #dilation_rate=dilation,
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer'+str(level)+'_conv3')

        self.output3 = tf.nn.dropout(tf.nn.relu(self.conv3), keep_prob=1-dropout)
        #print("self.output2",self.output2.get_shape())   # [?,31,31,8]
        if n_inputs != n_outputs:
            res_x = tf.layers.conv2d(inputs=value,
                                    filters=n_outputs,
                                    kernel_size=[1,1],
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer'+str(level)+'_conv_res' )
            #print("res_x",res_x.get_shape())    
        else:
            res_x = value
        return tf.nn.relu(res_x + self.output3)
 
def allint(l):
    return [int(i) for i in l]

