from transformers import FlavaProcessor, FlavaModel
from PIL import Image
import torch
import os

"""
# CODE sample for embedding extraction
# Load FLAVA model and processor
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model = FlavaModel.from_pretrained("facebook/flava-full")

# Prepare inputs
image = Image.open("dog.jpg")
text = "a description for the image"
inputs = processor(images=image, text=text, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    multimodal_embeddings = outputs.multimodal_embeddings
        

print("Multimodal Embeddings Shape:", multimodal_embeddings.shape)
"""



class embedModel():
    def __init__(self, source="facebook/flava-full", Processor=FlavaProcessor, Model=FlavaModel):
        self.source = source
        self.processor = Processor.from_pretrained(source)
        self.model = Model.from_pretrained(source)
        self.device = 'cpu'

        
    def embed(self, text_path, image_path=None, verbose=True):
        if(verbose):
            if(image_path is not None):
                print(f"processed profile : ",os.path.dirname(image_path))
            elif(text_path is not None):
                print(f"processed profile : ",os.path.dirname(text_path))
            
        if(image_path is not None):
            im = Image.open(image_path)
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
            
        
        if(image_path is not None):
            if(text_path is not None):
                inputs = self.processor(images=im, text=text, return_tensors="pt")
            else:
                inputs = self.processor(images=im, return_tensors="pt")
        else:
            if(text_path is not None):
                inputs = self.processor(images=None, text=text, return_tensors="pt")
            else:
                raise Exception('Required input to embed!')

            

        # if(self.device=='cuda'):
        #     inputs = inputs.to(self.device)
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            if(self.device=='cpu'):
                if(image_path==None):
                    embeddings = outputs.text_embeddings
                    cls_token = embeddings[:, 0, :]
                elif(text_path==None):
                    embeddings = outputs.image_embeddings
                    cls_token = embeddings[:, 0, :]
                else:
                    embeddings = outputs.multimodal_embeddings
                    cls_token = embeddings[:, 0, :]                    

            else:
                raise Exception("No gpu device!")
                

        if(verbose):
            print(f"\tProcessed cls token shape = {cls_token.shape}")
        return cls_token



