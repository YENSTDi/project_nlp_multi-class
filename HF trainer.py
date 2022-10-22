import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

class MainData(Dataset):
    def __init__(self, df, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df

    def __getitem__(self, index):
        text = self.df.iloc[index]['content']
        label = self.df.iloc[index]['reCheckedsubject']

        token = self.tokenizer(
            text, padding=True, truncation=True, max_length=128)

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

    input_ids_tensors = input_ids_tensors.to(torch.long)
    token_type_ids_tensors = token_type_ids_tensors.to(torch.long)
    masks_tensors = masks_tensors.to(torch.long)


    res = {
        "input_ids": input_ids_tensors,
        "token_type_ids": token_type_ids_tensors,
        "attention_mask": masks_tensors,
        "labels": labels
    }

    return res


df = pd.read_csv('/content/drive/MyDrive/projects/Multi-class Classification/data/data_3000.csv')
test_df = pd.read_csv('/content/drive/MyDrive/projects/Multi-class Classification/data/testdata.csv')
len(df), len(test_df)

NAME = "uer/roberta-base-finetuned-jd-full-chinese"
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=32, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(NAME)

train_set = MainData(df, tokenizer)
val_set = MainData(test_df, tokenizer)

args = TrainingArguments(
    output_dir = '/content/drive/MyDrive/projects/Multi-class Classification/data/results/',
    overwrite_output_dir = True,
    do_train = True,
    do_eval = True,
    evaluation_strategy = 'epoch',
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 10,
    save_strategy = 'no',
    logging_strategy = 'epoch',
    logging_dir = '/content/drive/MyDrive/projects/Multi-class Classification/data/results/log/'
)

trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    args = args,
    train_dataset = train_set,
    eval_dataset = val_set,
    data_collator = create_batch
)

trainer.train()