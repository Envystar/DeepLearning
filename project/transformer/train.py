import torch
from torch import nn, optim
from model import Transformer
from dataset import *
import tqdm

class TranslationModel():
    def __init__(self, layer_num, heads_num, d_model, d_ff, dropout, max_seq_len, batch_size,
        padding_idx, tokenizer_type, tokenizer_path, train_path, test_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #数据
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, cache_dir=tokenizer_path)
        encoder_vocabulary_size = decoder_vocabulary_size = self.tokenizer.vocab_size + len(self.tokenizer.special_tokens_map)
        self.padding_idx = padding_idx
        train_datasets = EnglishChineseDataset(self.tokenizer, train_path, max_seq_len)
        test_datasets = EnglishChineseDataset(self.tokenizer, test_path, max_seq_len)
        self.train_loader = DataLoader(train_datasets, batch_size, shuffle=True)
        self.test_loader = DataLoader(test_datasets, batch_size, shuffle=False)
        #模型
        self.model = Transformer(encoder_vocabulary_size, decoder_vocabulary_size, layer_num, self.padding_idx, heads_num, d_model, d_ff, dropout, max_seq_len)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        self.max_seq_len = max_seq_len
        pass

    def train(self, epochs, pre_model):
        if pre_model != None:
            self.model.load_state_dict(torch.load(pre_model, weights_only=True))
        with tqdm.tqdm(total=epochs) as t:
            for epoch in range(epochs):
                self.model.train()
                for index, (encoder_in, decoder_in, decoder_label) in enumerate(self.train_loader):
                    encoder_in = encoder_in.to(self.device)
                    decoder_in = decoder_in.to(self.device)
                    decoder_label = decoder_label.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(encoder_in, decoder_in)

                    preds = torch.argmax(outputs,-1)
                    label_mask = (decoder_label != self.padding_idx)
                    correct = preds == decoder_label
                    acc = torch.sum(label_mask * correct)/torch.sum(label_mask)

                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    expect = decoder_label.reshape(-1)
                    train_loss = self.loss_function(outputs, expect)
                    
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()

                    if index % 100 == 0:
                        print(f"iter:{index} / {len(self.train_loader)} train loss = {train_loss.item()} acc = {acc}")
                        torch.save(self.model.state_dict(),f"model_saves/model_{epoch + 1}.pt")
                t.update(1)

                self.test()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for index, (encoder_in, _, decoder_label) in enumerate(self.test_loader):
                encoder_in = encoder_in.to(self.device)
                decoder_label = decoder_label.to(self.device)
                decoder_in = torch.full((encoder_in.size(0), 1), 101, dtype=torch.long).to(self.device)
                for _ in range(self.max_seq_len): #自回归方式生成
                    outputs = self.model(encoder_in, decoder_in)
                    next_token = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(1)
                    decoder_in = torch.cat([decoder_in, next_token], dim=-1)
                    if (next_token == 102).all():
                        break
                    
                preds = decoder_in[:, 1:]
                if preds.shape[1] < decoder_label.shape[1]:
                    pad_len = decoder_label.shape[1] - preds.shape[1]
                    preds = torch.nn.functional.pad(preds, (0, pad_len), value=0)
                elif preds.shape[1] > decoder_label.shape[1]:
                    preds = preds[:, :decoder_label.shape[1]]

                label_mask = (decoder_label != self.padding_idx)
                correct = (preds == decoder_label) * label_mask
                acc = torch.sum(correct) / torch.sum(label_mask)

                
                outputs = self.model(preds, encoder_in)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                expect = decoder_label.reshape(-1)
                test_loss = self.loss_function(outputs, expect)

                if index % 20 == 0:
                    print(f"iter:{index}/{len(self.test_loader)} test loss = {test_loss.item()} acc = {acc}")
                    print("Preds:", self.tokenizer.decode(preds[0].tolist(), skip_special_tokens=True).replace(' ', ''))
                    print("Label:", self.tokenizer.decode(decoder_label[0].tolist(), skip_special_tokens=True).replace(' ', ''))
    
    def inference(self, inputs, pre_model):
        if pre_model != None:
            self.model.load_state_dict(torch.load(pre_model, weights_only=True))
        inputs = ["[CLS]" + text for text in inputs]
        encoder_in = self.tokenizer(inputs, padding="max_length", max_length=self.max_seq_len, truncation=True, return_tensors="pt", add_special_tokens=False)["input_ids"]
        self.model.eval()
        with torch.no_grad():
            encoder_in = encoder_in.to(self.device)
            decoder_in = torch.full((encoder_in.size(0), 1), 101, dtype=torch.long).to(self.device)
            for _ in range(max_seq_len): #自回归方式生成
                outputs = self.model(encoder_in, decoder_in)
                next_token = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(1)
                decoder_in = torch.cat([decoder_in, next_token], dim=-1)
                if (next_token == 102).all():
                    break
            for i, code in enumerate(decoder_in):
                find_eos = False
                for j, token_id in enumerate(code):
                    if find_eos:
                        decoder_in[i][j] = 0
                    elif token_id == 102:
                        find_eos = True
            results = [self.tokenizer.decode(code, skip_special_tokens=True).replace(' ', '') for code in decoder_in]
            return results


if __name__ == "__main__":
    layer_num = 6
    heads_num = 8
    d_model = 512
    d_ff = 1024
    dropout = 0.2
    max_seq_len = 60
    batch_size = 16
    padding_idx = 0
    tokenizer_type = "bert-base-chinese"
    tokenizer_path = "./tokenizer/"
    train_path = "./data/train.txt"
    test_path = "./data/test.txt"

    translation_model = TranslationModel(layer_num, heads_num, d_model, d_ff, dropout, max_seq_len, batch_size, padding_idx, tokenizer_type, tokenizer_path, train_path, test_path)
    # translation_model.train(100, pre_model=None)
    x = [
        "do you need to work on sunday?",
        "a cooking course should be mandatory in schools.",
        "we love each other.",
        "i have just been to the post office.",
        "awesome!",
        "no one can help me.",
        "let's go.",
        "i might not see tom today.",
        "i like cat.",
        "we can't see it again.",
        "yet not nearly enough has been invested in this effort.",
        "i need your advice on what i should do next.",
        "i got him to repair my car.",
        "where did you learn to play tennis?",
        "she used margarine instead of butter."
    ]
    results = translation_model.inference(x, pre_model="./model_saves/model_100.pt")
    for result in results:
        print(result)    