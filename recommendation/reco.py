import numpy as np 
import pickle as pkl
import torch
from torch.nn import functional as F


class RecoSystem():
    """
    Recommendation system class
    Inputs: embeddings (all profile embeddings)
            preferences (all user feedback for each profile)
    """
    def __init__(self, embeddings, preferences):
        self.embeddings = embeddings
        self.preferences = preferences
   

    def get_groups(self):
        """
        Returns the embeddings of liked and disliked groups
        for every user in preference array
        """
        preferences = self.preferences

        #list of liked/disliked embeddings for each user preference 
        liked_all = []
        disliked_all = []

        #mean of all embeddings
        total_mean = np.mean(self.embeddings, axis=0)
        
        # iterate through each user preference
        for pnum, p in enumerate(preferences):
            
            liked_embeddings = []
            disliked_embeddings = []
            for i in range(len(p)):
                if p[i] == 1:
                    liked_embeddings.append(self.embeddings[i])
                else:
                    disliked_embeddings.append(self.embeddings[i])
            
            
            check_liked = (np.array(liked_embeddings).shape)
            check_disliked = (np.array(disliked_embeddings).shape)

            if(len(check_liked)==1 or len(check_disliked)==1):
                continue
            liked_embeddings = np.array(liked_embeddings) - total_mean
            disliked_embeddings = np.array(disliked_embeddings) - total_mean
            
            liked_all.append(liked_embeddings)
            disliked_all.append(disliked_embeddings)
        
        return liked_all, disliked_all
    
    def cosine_sim(self, le, de):
        # return cosine similarity between le and de
        return F.cosine_similarity(torch.tensor(le), torch.tensor(de), dim=1)

    def get_groups_similarity(self):
        """
        Returns the similarity between liked and disliked groups
        """
        liked_embeddings, disliked_embeddings = self.get_groups()
        for le, de in zip(liked_embeddings, disliked_embeddings):
            le_mean, de_mean = np.mean(le, axis=0), np.mean(de, axis=0)
            if(np.nan == le_mean.any() or np.nan == de_mean.any()):
                print("Error Seen!")
                return 'NANs'
            else:
                sim = F.cosine_similarity(torch.tensor(le_mean), torch.tensor(de_mean), dim=1)
                print(sim)
        
    def get_pref_embeddings(self, liked_disliked_group = None):
        """
        Returns the "preferred" embedding of all input users
        inputs: liked_disliked_group: (liked_embeddings, disliked_embeddings)
                if none, get groups of all user preferesces
        """

        if(liked_disliked_group is None):
            liked_embeddings, disliked_embeddings = self.get_groups()
        else:
            liked_embeddings, disliked_embeddings = liked_disliked_group
        
        pref_embds = []
        
        #compute preferred embedding (reference direction to compute similarity with other embeddings)

        for le, de in zip(liked_embeddings, disliked_embeddings):
            pref_embds.append(np.mean(le, axis=0) - np.mean(de, axis=0))

        
        return pref_embds
    
    def forward(self, split=0.8, verbose=False):
        """
        Returns liked/disliked accuracy for each user
        inputs: split (train/test split)
                verbose (enable debug prints)
        """
        liked, disliked = self.get_groups()
        
        # contain data for each user
        data = []

        # iterate over all users
        for i in range(len(liked)): 
            # get liked and disliked embeddings for user
            ul = liked[i]
            ud = disliked[i]

            # split data
            liked_len = len(ul)
            disliked_len = len(ud)

            train_liked_indices = np.random.choice(liked_len, int(liked_len * split), replace=False)
            test_liked_indices = np.setdiff1d(np.arange(liked_len), train_liked_indices)

            train_disliked_indices = np.random.choice(disliked_len, int(disliked_len * split), replace=False)
            test_disliked_indices = np.setdiff1d(np.arange(disliked_len), train_disliked_indices)

            
            # train
            ul_train = ul[train_liked_indices]
            ud_train = ud[train_disliked_indices]

            ul_test = ul[test_liked_indices]
            ud_test = ud[test_disliked_indices]

            
            ult = np.expand_dims(ul_train, axis=0)
            udt = np.expand_dims(ud_train, axis=0)

            u_pref = np.squeeze(self.get_pref_embeddings((ult, udt)), axis=0)

            # test performance of mode - compute classification accuracy

            ul_sims = []
            ud_sims = []
            
            for e in ul_test:
                sim = self.cosine_sim(u_pref, e)
                ul_sims.append(sim)
            for e in ud_test:
                sim = self.cosine_sim(u_pref, e)
                ud_sims.append(sim)
            
            ul_acc = len(np.where(np.array(ul_sims) > 0)[0]) / len(ul_sims)
            ud_acc = len(np.where(np.array(ud_sims) < 0)[0]) / len(ud_sims)


            userdata = {"liked_acc":ul_acc, "disliked_acc":ud_acc}
            data.append(userdata)

        return data
    
    def train(self, split=0.8, epochs=10, verbose=False):

        #compute results for each user
        usr_data = {}
        
        #number of trails in thie experiment
        for e in range(epochs):
            # output results of each user in the experiment
            iterdata = self.forward(split=split, verbose=verbose)
            
            usrlen = len(iterdata)
            
            if(e==0):
                for u in range(usrlen):
                    usr_data[u] = {"liked_acc":[], "disliked_acc":[]}
            
            for u in range(usrlen):
                usr_data[u]["liked_acc"].append(iterdata[u]["liked_acc"])
                usr_data[u]["disliked_acc"].append(iterdata[u]["disliked_acc"])
                
            
        usr_accdata = {}
        for u in range(usrlen):
            usr_accdata[u] = {"liked_acc":np.mean(usr_data[u]["liked_acc"]),
                             "disliked_acc":np.mean(usr_data[u]["disliked_acc"]),
                             "liked_var":np.var(usr_data[u]["liked_acc"]),
                             "disliked_var":np.var(usr_data[u]["disliked_acc"])}
        
        
        return usr_accdata
