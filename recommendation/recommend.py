import numpy as np 
import torch
import pickle as pkl 
import torch
from torch.nn import functional as F
from reco import RecoSystem
import matplotlib.pyplot as plt

from plotparams import init_plotting

init_plotting()


# store all embeddings in pkl file
# change path to file destination
# embeddings stored at this drive link: https://drive.google.com/file/d/1ouWQAvIting0n88X2ssEddytp5F0kU2w/view?usp=drive_link
with open('../embeddings/30embds_text_llama_flava.pkl', 'rb') as f:
    modelembeddings = pkl.load(f)
sources = list(modelembeddings.keys())
print(f"source models: {sources}")

# preference data of users
men_preferences = np.load('./men_profile_preferences.npy')
women_preferences = np.load('./women_profile_preferences.npy')


#train/test split ratio
splits = np.arange(1, 10)*0.1
# perform n_trials to randomize results
n_trials = 10

# data for each split
splitdata = {}
for s in sources:
    #initialize split data
    splitdata[s] = {}

for split in splits:
    expdata = {}
    for s in sources:
        man_embeddings = modelembeddings[s]['Man']
        woman_embeddings = modelembeddings[s]['Woman']
        r_man = RecoSystem(embeddings=man_embeddings, preferences=men_preferences)
        r_woman = RecoSystem(embeddings=woman_embeddings, preferences=women_preferences)
        man_data = r_man.train(split=split, epochs=n_trials)
        woman_data = r_woman.train(split=split, epochs=n_trials)
        expdata[s] = {"Man":man_data, "Woman":woman_data}
        splitdata[s][split] = expdata[s]


# results are written to this file
with open(f'exp_splits_{epochs}_text_llama_flava_train.pkl', 'wb') as f:
    pkl.dump(splitdata, f)
    
    
    
