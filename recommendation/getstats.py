"""
Get split statistics for models
split: train/test splits ratio
models: multimodal (flava) + 5 text models 

Average the user statistics to get an estimate of how well a split performs while training 
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from plotparams import init_plotting
from matplotlib import patheffects
from utils import plot_models_avgstats
init_plotting()

plt.rcParams['font.size'] = 16
LSIZE = 15

with open("splits_10_text_llama_flava.pkl", "rb") as f:
    splitdata = pkl.load(f)
    
    
# get average of liked_acc and disliked_acc for all users for each split for each model for each gender

def get_avg_usr_stats(splitdata):
    

    data = {}
    models = list(splitdata.keys())
    for model in models:
        data[model] = {}
        splitlist = splitdata[model].keys()
        for split in splitlist:
            data[model][split] = {}

    for model in models:

        
        for split in list(splitdata[model].keys()):
            for gender in ["Man", "Woman"]:
                # print(gender)
                data[model][split][gender] = {"liked_acc": [], "disliked_acc": [],
                                    "liked_var": [], "disliked_var": []}
                usr_liked_acc = []
                usr_disliked_acc = []
                usr_liked_var = []
                usr_disliked_var = []

                for user in list(splitdata[model][split][gender].keys()):
                    usrdata = splitdata[model][split][gender][user]
                    usr_liked_acc.append(usrdata["liked_acc"])
                    usr_disliked_acc.append(usrdata["disliked_acc"])
                    usr_liked_var.append(usrdata["liked_var"])
                    usr_disliked_var.append(usrdata["disliked_var"])

                data[model][split][gender]["liked_acc"] = np.mean(usr_liked_acc)
                data[model][split][gender]["disliked_acc"] = np.mean(usr_disliked_acc)
                data[model][split][gender]["liked_var"] = np.mean(usr_liked_var)
                data[model][split][gender]["disliked_var"] = np.mean(usr_disliked_var)

    return data

splitdata_avg = get_avg_usr_stats(splitdata)