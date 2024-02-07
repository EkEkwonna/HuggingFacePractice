#%%--------
# pip install: transformers, tensorflow,torch
import os

"Having issues so specifying cache folder prior to running code"
os.environ['TRANSFORMERS_CACHE'] = '/Users/ekenem/Documents/GitProject/HugginFaceTest/HuggingFaceVenv/cache'

from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification
# If Using Tensorflow use TFAutoModelForSequenceClassification


import torch 
import torch.nn.functional as F 

#%%--------

"""
Define a pipeline first with classifier/(TASK DEFINING THE PIPELINE)
Gives you a way to use model for inference and extracts details for you
You need to specify a model_name as well

"""

model_id= "distilbert-base-uncased-finetuned-sst-2-english"

".from_pretrained important hugginf function required"

model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

classifier = pipeline("sentiment-analysis",model=model, tokenizer = tokenizer)
results = classifier(["We are very happy to shou you the Transformers library.",
"We hope you don't hate it"])

for result in results:
    print(result)


#%%--------
model_id= "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)   

tokens = tokenizer.tokenize("We are very happy to show you the Transformenrs Library")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy to show you the Transformers Library")

print(f'    Tokens: {tokens}')
print(f' Token IDs: {token_ids}')
print(f' Input IDs: {input_ids}')

"""

input ID:101 - beginning of the string
input ID:102 - end of the string
"""

# %% 
"Training data"
X_train = ["We are very happy to show you the Transformers Library",
          "We hope you don't hate it"]


"""
pt = Pytorch Tensor
Will apply padding and truncation if necessary
Input id are unique ids that the model can understand
"""

batch = tokenizer(X_train,padding=True,truncation = True,max_length=512,return_tensors= "pt")
print(batch)
# %%

model_id= "distilbert-base-uncased-finetuned-sst-2-english"
".from_pretrained important hugginf function required"
model = AutoModelForSequenceClassification.from_pretrained(model_id)


"""
Disable the gradient tracking
For Tensor Flow no ** required but for PyTorch it is
"""

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits,dim=1)
    print(predictions)
    labels = torch.argmax(predictions,dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)

# OUTPUT:
"SEQUENCE CLASSIFIER OUTPUT, with logits arguements"
# SequenceClassifierOutput(loss=None, logits=tensor([[-4.1465,  4.3928],
#         [-0.8004,  0.7992]]), hidden_states=None, attentions=None)
"Acutal probabilities"
# tensor([[1.9560e-04, 9.9980e-01],
#         [1.6804e-01, 8.3196e-01]])
"Labels (argmax)"
# tensor([1, 1])
"Combined labels with actual class name "
# ['POSITIVE', 'POSITIVE']


# %%

"Training data"
X_train = ["We are very happy to show you the Transformers Library",
          "We hope you don't hate it"]

model_id= "distilbert-base-uncased-finetuned-sst-2-english"
".from_pretrained important hugginf function required"
model = AutoModelForSequenceClassification.from_pretrained(model_id)

batch = tokenizer(X_train,padding=True,truncation = True,max_length=512,return_tensors= "pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch, labels = torch.tensor([1,0]))
    print(outputs)
    predictions = F.softmax(outputs.logits,dim=1)
    print(predictions)
    labels = torch.argmax(predictions,dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)

# OUTPUT: 
# tensor([[1.9560e-04, 9.9980e-01],[1.6804e-01, 8.3196e-01]])
    
# SAME NUMBERS AS PIPELINE OUTPUT: 
# {'label': 'POSITIVE', 'score': 0.999729335308075}
# {'label': 'POSITIVE', 'score': 0.8319621086120605}

"""
model and tokeniser are required to fine tune your model
"""



# %%
save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)
# %%
