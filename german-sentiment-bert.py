import os
os.environ['HF_HOME'] = '/Users/ekenem/Documents/GitProject/HugginFaceTest/HuggingFaceVenv/cache'

from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch 
import torch.nn.functional as F 
 
    
model_name = "oliverguhr/german-sentiment-bert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = ["Mit keinem quten Ergebnis","Das war unfair","Das ist gar nicht mal so gut",
         "nicht so schlecht wie erwartet","Das war gut!",
         "Sie fahrt ein grunes Auto."]



batch = tokenizer(texts,padding=True,truncation = True,max_length=512,return_tensors= "pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    label_ids = torch.argmax(outputs,dim=1)
    print(label_ids)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)
