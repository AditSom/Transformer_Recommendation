import transformers
import torch
import os

# set hugging face environment variable

model_id = "meta-llama/Meta-Llama-3-8B"


class embedModel():
    def __init__(self, model_id=model_id):
        self.model_id = model_id
        self.pipeline = transformers.pipeline("feature-extraction",
                                               model=model_id, 
                                               model_kwargs={"torch_dtype": torch.bfloat16})
        self.device = 'cpu' # tbd add gpu support

        
    def embed(self, text_path, verbose=True):
        if(verbose):
            if(text_path is not None):
                print(f"processed profile : ",os.path.dirname(text_path))

        if(text_path is not None):
            if(os.path.isfile(text_path)):
                if(not os.path.exists(text_path)):
                    raise Exception(f"File {text_path} does not exist!")
                else:
                    with open(text_path, 'r') as f:
                        text = f.read()
            else:
                text = text_path
            
        
 
    
        if(text_path is not None):
            features = self.pipeline(text)
            return features
        else:
            raise Exception('Required input to embed!')