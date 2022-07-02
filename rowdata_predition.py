import pandas as pd
import torch
from torch import logit, nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel, AutoTokenizer


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

NAME = "uer/roberta-base-finetuned-jd-full-chinese"
model = OurModel(AutoModel.from_pretrained(NAME), 32)
checkpoint = torch.load('multi-class-model', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = AutoTokenizer.from_pretrained(NAME)

data = pd.read_csv('testdata.csv')
df = data.sample(frac=1).iloc[0]
x = df['content']
y = df['reCheckedsubject']

inputs = tokenizer(x, max_length=512, return_tensors='pt')
logits = model(inputs['input_ids'], inputs['attention_mask'])

prediotn_Y = torch.argmax(logits.logits[0].cpu()).tolist()

print(f"實際標籤: {y}")
print(f"預測標籤: {prediotn_Y}")


