import os
os.environ['HF_HOME'] = '/Users/ekenem/Documents/GitProject/HugginFaceTest/HuggingFaceVenv/cache'
# os.environ['TRANSFORMERS_CACHE'] = '/Users/ekenem/Documents/GitProject/HugginFaceTest/HuggingFaceVenv/cache'

from transformers import pipeline
import torch 
import torch.nn.functional as F 

model_name = "bert-base-uncased"
unmasker = pipeline('fill-mask',model=model_name)
results = unmasker("You should have been there last night at the [MASK], we had a blast!")

for result in results:
    print(result)