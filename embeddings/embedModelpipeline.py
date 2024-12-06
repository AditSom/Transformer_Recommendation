import transformers
import torch
import os

os.environ["HF_TOKEN"] = "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"

model_id = "meta-llama/Meta-Llama-3-8B"

# pipeline = transformers.pipeline("feature-extraction", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16})
# pipeline("Hey how are you doing today?")


class embedModel():
    def __init__(self, model_id=model_id):
        self.model_id = model_id
        self.pipeline = transformers.pipeline("feature-extraction",
                                               model=model_id, 
                                               model_kwargs={"torch_dtype": torch.bfloat16})
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

            features = self.pipeline(text)
            return features
        else:
            raise Exception('Required input to embed!')