import logging
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, AutoModelForTokenClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import pandas as pd

FORMAT = '%(asctime)s [%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

with open('./data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 分割数据集
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42) 
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 创建 Huggingface 数据集
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# 初始化 tokenizer 和模型
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForTokenClassification.from_pretrained('ckiplab/bert-base-chinese-ws', num_labels=2)  

label_list = ["B", "I"]

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True, padding='max_length')
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_list.index(label[word_idx]))
            else:
                label_ids.append(label_list.index(label[word_idx]))
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets.map(lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_true_labels = [item for sublist in true_labels for item in sublist]
    all_true_predictions = [item for sublist in true_predictions for item in sublist]
    
    precision = precision_score(all_true_labels, all_true_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_true_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_true_predictions, average='weighted')
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

logging.info("Start training.")
trainer.train()

model.save_pretrained("./temp_model")
tokenizer.save_pretrained("./temp_model")
logging.info("Finish training.")

# 評估現有的最佳模型
existing_model = AutoModelForTokenClassification.from_pretrained("./best_model")
existing_tokenizer = BertTokenizerFast.from_pretrained("./best_model")

existing_tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_and_align_labels(examples, existing_tokenizer), batched=True)

trainer_existing = Trainer(
    model=existing_model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics
)

logging.info("Evaluating existing best model on test dataset.")
existing_results = trainer_existing.evaluate()

# 測試新訓練的模型
new_model = AutoModelForTokenClassification.from_pretrained("./temp_model")
new_tokenizer = BertTokenizerFast.from_pretrained("./temp_model")

new_tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_and_align_labels(examples, new_tokenizer), batched=True)

trainer_new = Trainer(
    model=new_model,
    args=training_args,
    eval_dataset=new_tokenized_test_dataset,
    compute_metrics=compute_metrics
)

logging.info("Evaluating new model on test dataset.")
new_results = trainer_new.evaluate()

logging.info(f"f1_score: current best_model= {existing_results['eval_f1']:.4f}, new model= {new_results['eval_f1']:.4f}")

# 比较 F1 分数
if new_results['eval_f1'] > existing_results['eval_f1']:
    model.save_pretrained("./best_model")
    tokenizer.save_pretrained("./best_model")
    logging.info("New best model saved.")
else:
    logging.info("The existing best model is better or equal to the new model.")
