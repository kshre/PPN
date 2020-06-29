# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import pandas as pd
import logging
import json
import numpy as np
import datetime
from pgportfolio.tools.indicator import max_drawdown, std, sharpe, positive_count, negative_count, moving_accumulate, calmar_ratio
from pgportfolio.tools.configprocess import parse_time, check_input_same
from pgportfolio.tools.shortcut import execute_backtest

# the dictionary of name of indicators mapping to the function of related indicators
# input is portfolio changes
INDICATORS = {"portfolio value": np.prod,
              "std":std,
              "sharpe ratio": sharpe,
              "max drawdown": max_drawdown,
              "calmar ratio":calmar_ratio,
              "positive periods": positive_count,
              "negative periods": negative_count,
              "postive day": lambda pcs: positive_count(moving_accumulate(pcs, 48)),
              "negative day": lambda pcs: negative_count(moving_accumulate(pcs, 48)),
              "postive week": lambda pcs: positive_count(moving_accumulate(pcs, 336)),
              "negative week": lambda pcs: negative_count(moving_accumulate(pcs, 336)),
              "average": np.mean}
# The time duration of a day is 48 rounds, and a time duration of a week is 7*48=336 

NAMES = {"best": "Best Stock (Benchmark)",
         "crp": "UCRP (Benchmark)",
         "ubah": "UBAH (Benchmark)",
         "anticor": "ANTICOR",
         "olmar": "OLMAR",
         "pamr": "PAMR",
         "cwmr": "CWMR",
         "rmr": "RMR",
         "ons": "ONS",
         "up": "UP",
         "eg": "EG",
         "bk": "BK",
         "corn": "CORN",
         "m0": "M0",
         "wmamr": "WMAMR"
         }

def plot_backtest(config, algos, labels=None):
    """
    @:param config: config dictionary
    @:param algos: list of strings representing the name of algorithms or index of pgportfolio result
    """
    results = []
    for i, algo in enumerate(algos):
        print('algo',algo)
        if algo.isdigit():
            results.append(np.cumprod(_load_from_summary(algo, config)[0]))
            # np.cumprod to obtain the final asset change verctor 
            logging.info("load index "+algo+" from csv file")
        else:
            logging.info("start executing "+algo)
            results.append(np.cumprod(execute_backtest(algo, config)[0]))
            logging.info("finish executing "+algo)

    start, end = _extract_test(config) 
    timestamps = np.linspace(start, end+10, len(results[0])//20+1) 
    dates = [datetime.datetime.fromtimestamp(int(ts)-int(ts)%config["input"]["global_period"])
             for ts in timestamps]

    weeks = mdates.WeekdayLocator()
    days = mdates.DayLocator()

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"],
                  "size": 8})

    """
    styles = [("-", None), ("--", None), ("", "+"), (":", None),
              ("", "o"), ("", "v"), ("", "*")]
    """
    
    #c = ["royalblue","r","coral","g"]  #for the parameter of gamma
    c=["black","brown","g","coral","royalblue","deepskyblue","darkviolet","r"]  #for different feature extractors
    #c=["black","black","black","black","black","black","black","black"]
    #style=["-+",":+",":x",":v","--*","--s","-D","-o"]
    #style=["-","-","-","-","-","-","-","-"]
    #style=["--","--","--","--","--","--","--","--"]
    style=["-^","-d","-s","->","-*","-<","-v","-o"]
    maker=["+","+","x","v","*","s","D","o"]
    fig, ax = plt.subplots() 
    fig.set_size_inches(12, 7) #(9, 6) 18,10
    for i, pvs in enumerate(results):
        if len(labels) > i:
            label = labels[i]
        else:
            label = NAMES[algos[i]]
        #print(np.array(pvs)[:2:])
        length = len(pvs)
        pvs = pvs[range(0,length,20)]
        #ax.semilogy(dates, pvs, linewidth=3, label=label) # 3 for gamma, 2 for feature extractors
        ax.semilogy(dates, pvs, style[i], color=c[i],linewidth=2, markersize=6,markevery = slice(4,length,7),label=label)
        #ax.semilogx(dates, pvs,style[i], color=c[i],linewidth=1.5, label=label)
        #ax.plot(dates, pvs,style[i], color=c[i],linewidth=1.5, label=label)
        #style[i],

    plt.ylabel("APV", fontsize=18)
    plt.xlabel("time", fontsize=18)
    xfmt = mdates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_minor_locator(days)
    datemin = dates[0]
    datemax = dates[-1]
    ax.set_xlim(datemin, datemax)

    ax.xaxis.set_major_formatter(xfmt)
    plt.grid(True)
    plt.tight_layout()  # to adjust exterior edges
    ax.legend(loc="upper left", prop={"size":16})   
    # Adaptively adjust the date of the figure  
    fig.autofmt_xdate()  
    plt.tick_params(labelsize=12)
    #==============================================================================
    #     plt.savefig("result.eps", bbox_inches='tight',
    #                 pad_inches=0)
    #==============================================================================
    
    # rename the file
    file_name = config["input"]["start_date"]+"_"+config["input"]["end_date"]+".eps"
    file_name = file_name.replace("/","")
    plt.savefig(file_name, bbox_inches='tight',
                pad_inches=0)
    

    plt.show()


def table_backtest(config, algos, labels=None, format="raw",
                   indicators=list(INDICATORS.keys())):
    """
    @:param config: config dictionary
    @:param algos: list of strings representing the name of algorithms
    or index of pgportfolio result
    @ output param format: "raw", "html", "latex" or "csv". If it is "csv",
    the result will be save in a csv file. otherwise only print it out
    @:return: a string of html or latex code
    """
    results = []
    labels = list(labels)
    for i, algo in enumerate(algos):
        print('algo,',algo)
        if algo.isdigit():
            portfolio_changes,turn_over = _load_from_summary(algo, config)
            logging.info("load index " + algo + " from csv file")
        else:
            logging.info("start executing " + algo)
            portfolio_changes,turn_over = execute_backtest(algo, config)
            logging.info("finish executing " + algo)

        indicator_result = {}
        for indicator in indicators:
            indicator_result[indicator] = INDICATORS[indicator](portfolio_changes)
        indicator_result["turn over"] = turn_over
        results.append(indicator_result)
        if len(labels)<=i:
            labels.append(NAMES[algo])

    dataframe = pd.DataFrame(results, index=labels)

    start, end = _extract_test(config)
    start = datetime.datetime.fromtimestamp(start - start%config["input"]["global_period"])
    end = datetime.datetime.fromtimestamp(end - end%config["input"]["global_period"])

    print("backtest start from "+ str(start) + " to " + str(end))
    if format == "html":
        print(dataframe.to_html())
    elif format == "latex":
        print(dataframe.to_latex())
    elif format == "raw":
        print(dataframe.to_string())
    elif format == "csv":
        dataframe.to_csv("./compare"+end.strftime("%Y-%m-%d")+".csv")
    else:
        raise ValueError("The format " + format + " is not supported")


def _extract_test(config): 
    #extract the start and the end of the test set 
    global_start = parse_time(config["input"]["start_date"])
    global_end = parse_time(config["input"]["end_date"])
    span = global_end - global_start
    start = global_end - config["input"]["test_portion"] * span
    end = global_end
    return start, end


def _load_from_summary(index, config):
    """ load the backtest result form train_package/train_summary
    @:param index: index of the training and backtest
    @:return: numpy array of the portfolio changes
    """
    dataframe = pd.DataFrame.from_csv("./train_package/train_summary.csv")
    history_string = dataframe.loc[int(index)]["backtest_test_history"]
    turn_over = dataframe.loc[int(index)]["backtest_turn_over"]
    
    if not check_input_same(config, json.loads(dataframe.loc[int(index)]["config"])):
        print(config)
        print("\n")
        print(json.loads(dataframe.loc[int(index)]["config"]))
        raise ValueError("the date of this index is not the same as the default config")
    return np.fromstring(history_string, sep=",")[:-1],turn_over

