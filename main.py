import pandas as pd
import time

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

class MainData(Dataset):
  def __init__(self, df, tokenizer) -> None:
    super().__init__()
    self.tokenizer = tokenizer
    self.df = df
  
  def __getitem__(self, index):
    text = self.df.iloc[index]['content']
    label = self.df.iloc[index]['reCheckedsubject']

    token = self.tokenizer(text, padding=True, truncation=True, max_length=512)

    input_ids = token['input_ids']
    token_ids = token['token_type_ids']
    attention_mask = token['attention_mask']
    label = torch.tensor(label)

    return (input_ids, token_ids, attention_mask, label)
  
  def __len__(self):
    return len(self.df)

def create_batch(datas):
    input_ids = [torch.Tensor(i[0]) for i in datas]
    token_type_ids = [torch.Tensor(i[1])for i in datas]
    attention_mask = [torch.Tensor(i[2]) for i in datas]
    
    if datas[0][3] is not None:
        labels = torch.stack([i[3] for i in datas])
    else:
        labels = None

    input_ids_tensors = pad_sequence(input_ids, batch_first=True)
    token_type_ids_tensors = pad_sequence(token_type_ids, batch_first=True)
    masks_tensors = pad_sequence(attention_mask, batch_first=True)

    input_ids_tensors      = input_ids_tensors.to(torch.long)
    token_type_ids_tensors = token_type_ids_tensors.to(torch.long)
    masks_tensors          = masks_tensors.to(torch.long)
    
    return input_ids_tensors, token_type_ids_tensors, masks_tensors, labels
    

class OurModel(nn.Module):
  def __init__(self, ori_model, num_labels):
    super(OurModel, self).__init__()
    self.num_labels = num_labels

    self.model = model = ori_model
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(768, num_labels)

  def forward(self, input_ids=None, attention_mask=None, labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    layers = self.dropout(outputs[0])
    logits = self.classifier(layers[:, 0, :].view(-1, 768))

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)

def eval(model, data_loader):
  model.eval()
  prediction = None
  true = None

  # label_pred = []
  # label_true = []
  with torch.no_grad():
    for index, data in enumerate(data_loader):
      input_id, mark, sgement_id, label = [t.to("cuda") for t in data]
      output = model(input_id, mark)

      logits = output[0]
      _, pred = torch.max(logits.data, 1)

      # label_pred.extend(pred.tolist())
      # label_true.extend(label.tolist())
      # print(type(label))
      pred = pred.cpu()
      label = label.cpu()
      

      if prediction is None:
        prediction = pred
        true = label
      else:
        prediction = torch.cat((prediction, pred))
        true = torch.cat((true, label))
      

  acc = accuracy_score(true, prediction)
  # f1 = f1_score(true, prediction)
  # prec = precision_score(true, prediction)
  # recall = recall_score(true, prediction)

  return acc

df = pd.read_csv('data_3000.csv')
val_df = pd.read_csv('testdata.csv')

NAME = "uer/roberta-base-finetuned-jd-full-chinese"
model = OurModel(AutoModel.from_pretrained(NAME), 32)
tokenizer = AutoTokenizer.from_pretrained(NAME)

train_set = MainData(df, tokenizer)
train_loader = DataLoader(train_set, shuffle=False, batch_size=16, collate_fn=create_batch)

val_set = MainData(val_df, tokenizer)
val_loader = DataLoader(val_set, shuffle=False, batch_size=8, collate_fn=create_batch)


device = 'cuda'
optimizer = Adam(model.parameters(), lr=5e-5)
loss = nn.BCELoss()
EPOCH = 10

model = model.to(device)
model.train()

loss_list = []
start = time.time()
for epoch in range(EPOCH):
    running_loss = 0.0
    bts = 0
    print('-'*50)
    for index, data in enumerate(train_loader):
        input_id, mark, sgement_id, label = [t.to(device) for t in data]
        optimizer.zero_grad()

        output = model(input_id, mark, label)
        l = output[0]
        l.backward()
        optimizer.step()
        running_loss+=l.item()
        bts+=len(data)
        print(index, running_loss)
    

    avg_loss = running_loss/bts
    print(epoch, avg_loss)
    loss_list.append(avg_loss)
    val_acc = eval(model, val_loader)
    print(f'val_acc: {val_acc}')
    print('-'*50)
    

end = time.time()
cost_time = end - start
print('training time', cost_time)


model.save_pretrained("suicide_predtion_model")
tokenizer.save_pretrained("suicide_predtion_token")