import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random

def split_data(data_dir):
    datas = open(data_dir + "raw.txt", "r", encoding="utf-8").readlines()
    random.shuffle(datas)
    train_data = datas[:int(0.95 * len(datas))]
    test_data = datas[int(0.95 * len(datas)):]
    open(data_dir + "train.txt", "w", encoding="utf-8").writelines(train_data)
    open(data_dir + "test.txt", "w", encoding="utf-8").writelines(test_data)

def count_max_seq_len(data_dir, tokenizer):
    datas = open(data_dir + "train.txt", "r", encoding="utf-8").readlines()
    max_len = 0
    for index, data in enumerate(datas):
        try:
            en, zh, _ = data.strip().split("\t")
        except ValueError as e:
            print(f"Error at index {index}: {data} - {e}")
        max_len = max(max_len, len(tokenizer(en)["input_ids"]))
        max_len = max(max_len, len(tokenizer(zh)["input_ids"]))
    return max_len

class EnglishChineseDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_seq_len=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.datas = open(data_path, "r", encoding="utf-8").readlines()
        self.max_seq_len = max_seq_len
        self.data_cache = {}

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        if index in self.data_cache:
            return self.data_cache[index]
        try:
            en, zh, _ = self.datas[index].strip().split("\t")
        except ValueError as e:
            print(f"Error at index {index}: {self.datas[index]} - {e}")
            return None
        en_token = self.tokenizer(en + "[SEP]", padding="max_length", max_length=self.max_seq_len, truncation=True, return_tensors="pt", add_special_tokens=False)["input_ids"]
        zh_in_token = self.tokenizer("[CLS]" + zh + "[SEP]", padding="max_length", max_length=self.max_seq_len, truncation=True, return_tensors="pt", add_special_tokens=False)["input_ids"]
        zh_label_token = self.tokenizer(zh + "[SEP]", padding="max_length", max_length=self.max_seq_len, truncation=True, return_tensors="pt", add_special_tokens=False)["input_ids"]
        self.data_cache[index] = (torch.LongTensor(en_token)[0], torch.LongTensor(zh_in_token)[0], torch.LongTensor(zh_label_token)[0])
        return self.data_cache[index]

if __name__ == "__main__":
    data_dir = "./data/"
    # split_data(data_dir)
    tokenizer_path = "./tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir=tokenizer_path)
    x = [[101, 10560, 8171, 102], [101, 10005, 10117, 8118, 102]]
    results = [tokenizer.decode(code, skip_special_tokens=True).replace(' ', '') for code in x]
    print(results)
    pass

    
