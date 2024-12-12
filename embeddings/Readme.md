# Embeddings

This folder contains scripts to compute emebddings for different models

Three types of model embedding functions - 
1) embedModel - flava (Use Processor and Model api)
2) embedModelAutotokenizer - text models bert based (Use Autotokenizer and AutoModel api)
3) embedModelpipeline - llama3-8B (Use transformers.pipeline api)

## Instructions
To embed dataset using one of the embed classes, please follow example shown in .ipynb notebook.
Dataset can be loaded from link: [Profiles](https://drive.google.com/file/d/1EJyXe1Lsb_HWtzlnQN5crsReOaTFdQYR/view?usp=sharing)


>NOTE: For our analysis, we choose 30 profiles from the total profile pool generated initially for logistical reasons (user feedback). Profiles were sampled using random.seed(42) from index list np.arange(0,50) for both men and women profiles