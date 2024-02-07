import os

os.environ['TRANSFORMERS_CACHE'] = '/Users/ekenem/Documents/GitProject/HugginFaceTest/HuggingFaceVenv/cache'

from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch 
import torch.nn.functional as F 

model_id= "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)   

classifier = pipeline("sentiment-analysis",model=model, tokenizer = tokenizer)
results = classifier(["We are very happy to shou you the Transformers library.",
"We hope you don't hate it"])

for result in results:
    print(result)

tokens = tokenizer.tokenize("We are very happy to show you the Transformenrs Library")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy to show you the Transformers Library")

print(f'    Tokens: {tokens}')
print(f' Token IDs: {token_ids}')
print(f' Input IDs: {input_ids}')

X_train = ["We are very happy to show you the Transformers Library",
          "We hope you don't hate it"]

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

save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)