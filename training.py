import pandas as pd
import time
import sys

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

# 使用torch的Dataset
class MainData(Dataset):
    def __init__(self, df, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df

    def __getitem__(self, index):
        text = self.df.iloc[index]['content']
        label = self.df.iloc[index]['reCheckedsubject']

        token = self.tokenizer(
            text, padding=True, truncation=True, max_length=512)

        input_ids = token['input_ids']
        token_ids = token['token_type_ids']
        attention_mask = token['attention_mask']
        label = torch.tensor(label)

        return (input_ids, token_ids, attention_mask, label)

    def __len__(self):
        return len(self.df)

# 方便DataLoader取batch資料
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

    input_ids_tensors = input_ids_tensors.to(torch.long)
    token_type_ids_tensors = token_type_ids_tensors.to(torch.long)
    masks_tensors = masks_tensors.to(torch.long)

    return input_ids_tensors, token_type_ids_tensors, masks_tensors, labels

# 更改原fine-turning模型
class OurModel(nn.Module):
    def __init__(self, ori_model, num_labels):
        super(OurModel, self).__init__()
        self.num_labels = num_labels

        self.model = model = ori_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        layers = self.dropout(outputs[0])
        logits = self.classifier(layers[:, 0, :].view(-1, 768))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# 評估模型
def eval(model, data_loader):
    loss = 0
    model.eval()
    prediction = None
    true = None

    with torch.no_grad():
        for index, data in enumerate(data_loader):
            input_id, mark, sgement_id, label = [t.to("cuda") for t in data]

            # 算Val_loss
            output_loss = model(input_id, mark, label)
            loss += output_loss[0].item()
            # ================================================

            output = model(input_id, mark)

            logits = output[0]
            _, pred = torch.max(logits.data, 1)

            pred = pred.cpu()
            label = label.cpu()

            if prediction is None:
                prediction = pred
                true = label
            else:
                prediction = torch.cat((prediction, pred))
                true = torch.cat((true, label))

    # =================
    loss = loss/len(data_loader)
    # =================

    acc = accuracy_score(true, prediction)

    return acc, loss

# 儲存模型
def save(model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'multi-class-model-2')


# 讀取訓練和驗證資料
df = pd.read_csv('data_3000.csv')
val_df = pd.read_csv('testdata.csv')

# 載入模型以及tokenizer
NAME = "uer/roberta-base-finetuned-jd-full-chinese"
model = OurModel(AutoModel.from_pretrained(NAME), 32)
tokenizer = AutoTokenizer.from_pretrained(NAME)

# 準備DataLoader
train_set = MainData(df, tokenizer)
train_loader = DataLoader(train_set, shuffle=False,
                          batch_size=16, collate_fn=create_batch)

val_set = MainData(val_df, tokenizer)
val_loader = DataLoader(val_set, shuffle=False,
                        batch_size=8, collate_fn=create_batch)

# 模型超參數設定
device = 'cuda'
optimizer = Adam(model.parameters(), lr=5e-5)
loss = nn.BCELoss()
EPOCH = 10

# 模型開始訓練-----
model = model.to(device)
model.train()

## 儲存Loss和Acc用
Train_Loss = []
Val_Loss = []
Train_Acc = []
Val_Acc = []
## 紀錄訓練時間
start = time.time()
for epoch in range(EPOCH):
    bts = 0
    for index, data in enumerate(train_loader):
        input_id, mark, sgement_id, label = [t.to(device) for t in data]
        optimizer.zero_grad()

        output = model(input_id, mark, label)
        l = output[0]
        l.backward()
        optimizer.step()
        bts += 1
        stats = 'Epoch [%d/%d], Step [%d/%d], Batch-Loss: %.4f' % (
            epoch+1, EPOCH, bts, len(train_loader), l.item())
        print('\r' + stats, end="")
        sys.stdout.flush()

    train_acc, train_loss = eval(model, train_loader)
    val_acc, val_loss = eval(model, val_loader)
    Train_Loss.append(train_loss)
    Val_Loss.append(val_loss)
    Train_Acc.append(train_acc)
    Val_Acc.append(val_acc)
    # print(f'val_acc: {val_acc}')
    stats = 'Epoch [%d/%d], Step [%d/%d], Train-Loss: %.4f, Val-Loss: %.4f, Train-Acc: %.4f, Val-Acc: %.4f' % (
        epoch+1, EPOCH, bts, len(train_loader), train_loss, val_loss, train_acc, val_acc)
    print('\r' + stats)

    # loss_list = np.array(loss_list)

end = time.time()
cost_time = end - start
print('training time', cost_time)
with open('log.txt', 'a+') as f:
    f.write(f"cost time: {cost_time}")
# 訓練完成-----

# 評估訓練、驗證準確度和loss值
training_acc = eval(model, train_loader)
val_acc = eval(model, val_loader)

# 紀錄log
with open('log.txt', 'a+') as f:
    f.write(f"training acc: {training_acc}\n")
    f.write(f"val acc: {val_acc}\n")

# 儲存模型
save(model, optimizer)
