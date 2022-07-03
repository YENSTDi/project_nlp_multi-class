# NLP多分類模型訓練

## 訓練平台

Colab

## 環境建置

`!pip install transformers`

## 預訓練模型

<https://huggingface.co/uer/roberta-base-finetuned-jd-full-chinese>

## 模型參數

| Name | Value |
| --- | --- |
| optimizer | Adam |
| loss function | CrossEntropyLoss |
| learning rate | 5e-5 |
| batch_size | 16 |
| epoch | 10 |

## 資料

| 類別 | 大小 |
| - | - |
| 訓練集 | 3000筆 |
| 驗證集 | 112筆 |

## 訓練結果

| 類別 | 準確度 |
| - | - |
| 訓練集 | 98.83% |
| 驗證集 | 98.33% |


> 準確度

![準確度](plot/acc.jpg)

>Loss

![loss](/plot/loss.jpg)