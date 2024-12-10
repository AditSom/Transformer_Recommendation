# Recommendation

This folder contains scripts for out recommender and analysis obtained from the "train" results.

The recommender designed gives statistics (liked/disliked accuracy, variance) for each split, model, user, gender

To perform analysis the recommender output is saved as a pkl file and is processed as required.

The paper contains results computed using these scripts (example notebooks show sample output plots)

## Instructions to run code

The recommender system class is stored in `reco.py`. 

The recommendation train/evaluation is performed by `recommendation.py`

To run the script - 
1) Download pre-computed model embeddings from [link](https://drive.google.com/file/d/1ouWQAvIting0n88X2ssEddytp5F0kU2w/view?usp=drive_link)
2) run - `python recommendation.py`
3) The results summarize the prediction accuracy for both classes along with variance seen in accuracy for random sampling of the same train/test split ratio.
