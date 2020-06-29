# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import time
from datetime import datetime
import json
import os
rootpath = os.path.dirname(os.path.abspath(__file__)).\
    replace("\\pgportfolio\\tools", "").replace("/pgportfolio/tools","")

try:
    unicode        # Python 2
except NameError:
    unicode = str  # Python 3


def preprocess_config(config):
    fill_default(config)
    if sys.version_info[0] == 2:   
        return byteify(config)
    else:
        return config


def fill_default(config): 
    set_missing(config, "random_seed", 0)         #if missing random_seed, then 0
    set_missing(config, "agent_type", "NNAgent")    #if missing agent_type, then "NNAgent", i.e., portfolio policy network
    #fill_layers_default(config["layers"])
    fill_input_default(config["input"])
    fill_train_config(config["training"])


def fill_train_config(train_config):
    set_missing(train_config, "fast_train", True)
    set_missing(train_config, "decay_rate", 1.0)
    set_missing(train_config, "decay_steps", 50000)


def fill_input_default(input_config):
    set_missing(input_config, "save_memory_mode", False)
    set_missing(input_config, "portion_reversed", False)
    set_missing(input_config, "market", "poloniex")
    set_missing(input_config, "norm_method", "absolute")
    set_missing(input_config, "is_permed", False)
    set_missing(input_config, "fake_ratio", 1)


def fill_layers_default(layers):
    for layer in layers:
        if layer["type"] == "ConvLayer":
            set_missing(layer, "padding", "valid")
            set_missing(layer, "strides", [1, 1])
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_Dense":
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "DenseLayer":
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_LSTM" or layer["type"] == "EIIE_RNN":
            set_missing(layer, "dropouts", None)
        elif layer["type"] == "EIIE_Output" or\
                layer["type"] == "Output_WithW" or\
                layer["type"] == "EIIE_Output_WithW":
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "DropOut":
            pass
        else:
            raise ValueError("layer name {} not supported".format(layer["type"]))


def set_missing(config, name, value): 
    if name not in config:
        config[name] = value


def byteify(input):   
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return str(input)
    else:
        return input


def parse_time(time_string):
    """
    aim: to transform the time format: time -> localtime -> mktime
    example: 2017/07/01 -> datetime.datetime(2017, 7, 1, 0, 0) -> 
       # time.struct_time(tm_year=2017, tm_mon=7, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=5, tm_yday=182, tm_isdst=-1) ->1498838400.0s
    """ 
    return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())  

def load_config(index=None):
    """ 
    aim: to load the config
    @:param index: if None, load the default in pgportfolio;
     if a integer, load the config under train_package
    """
    if index:
        with open(rootpath+"/train_package/" + str(index) + "/net_config.json") as file:      
            config = json.load(file)
    else:
        with open(rootpath+"/pgportfolio/" + "net_config.json") as file:
            config = json.load(file)
    return preprocess_config(config)


def check_input_same(config1, config2):
    """aim: to judge whether two inputs are the same"""
    input1 = config1["input"]
    input2 = config2["input"]
    if input1["start_date"] != input2["start_date"]:
        return False
    elif input1["end_date"] != input2["end_date"]:
        return False
    elif input1["test_portion"] != input2["test_portion"]:
        return False
    else:
        return True

