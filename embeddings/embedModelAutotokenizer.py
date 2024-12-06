from transformers import pipeline
import os
import json
from transformers import AutoTokenizer, AutoModel
import torch

"""
# Choose a pre-trained model (e.g., BERT, RoBERTa, DistilBERT, etc.)
model_names = ["bert-base-uncased", "distilbert-base-uncased", "xlnet-base-cased", "xlm-roberta-base"]

# model_name = "bert-base-uncased"  # Replace with the model of your choice

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)


    # Input text
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt")

    # Forward pass
    outputs = model(**inputs)

    # Extract hidden states (list of tensors, one per layer)
    hidden_states = outputs.hidden_states

    # Use the last hidden state (or perform pooling across layers)
    last_hidden_state = hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

    # For a single vector representation, mean-pool or take the CLS token
    # feature_vector = last_hidden_state.mean(dim=1)  # Mean pooling
    feature_vector = last_hidden_state[:, 0, :]  # CLS token
    print(feature_vector.shape)  # Output: (batch_size, hidden_dim)
"""



class embedModel():
    def __init__(self, source="bert-base-uncased", Tokenizer=AutoTokenizer, Model=AutoModel):

        self.source = source
        self.tokenizer = Tokenizer.from_pretrained(source)
        self.model = Model.from_pretrained(source,  output_hidden_states=True)
        self.device = 'cpu'

        
    def embed(self, text_path, verbose=True):
        if(verbose):
            if(text_path is not None):
                print(f"processed profile : ",os.path.dirname(text_path))

        if(text_path is not None):
            # print(f"Input text is path = {os.path.isfile(text_path)}")
            if(os.path.isfile(text_path)):
                if(not os.path.exists(text_path)):
                    raise Exception(f"File {text_path} does not exist!")
                else:
                    with open(text_path, 'r') as f:
                        text = f.read()
            else:
                text = text_path
            
        
 
    
        if(text_path is not None):
            # print(text)

            inputs = self.tokenizer(text, return_tensors="pt")
        else:
            raise Exception('Required input to embed!')

            

   
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            hidden_states = outputs.hidden_states
            # print(hidden_states)
            last_hidden_state = hidden_states[-1]

            # For a single vector representation, mean-pool or take the CLS token
            # feature_vector = last_hidden_state.mean(dim=1)  # Mean pooli  
            feature_vector = last_hidden_state[:, 0, :]  # CLS token
            # print(feature_vector.shape)  # Output: (batch_size, hidden_dim)
                

        if(verbose):
            print(f"\tProcessed cls token shape = {feature_vector.shape}")
        return feature_vector



