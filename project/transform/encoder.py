import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#word embedding
#source sentence and target sentence
#build sentence

batch_size = 2
#单词表大小
max_num_src_words = 8
max_num_tar_words = 8
model_dim = 8   #模型特征大小

#最大序列长度
max_position_len = 5

src_len = torch.Tensor([2, 4]).to(torch.int32)
tar_len = torch.Tensor([4, 3]).to(torch.int32)

#单词索引的序列构成的句子
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (len, )), (0, max(src_len) - len)), 0) for len in src_len])
tar_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tar_words, (len, )), (0, max(tar_len) - len)), 0) for len in tar_len])

#构造word embedding
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tar_embedding_table = nn.Embedding(max_num_tar_words + 1, model_dim)

src_embedding = src_embedding_table(src_seq)
tar_embedding = tar_embedding_table(src_seq)

#构造positional embedding
pos_matrix = torch.arange(max_position_len).reshape(-1, 1)
i_matrix = torch.pow(10000, torch.arange(0, 8, 2).reshape(1, -1) / model_dim)
pe_embedding_table = torch.zeros(max_position_len, model_dim)

pe_embedding_table[:, 0::2] = torch.sin(pos_matrix / i_matrix)
pe_embedding_table[:, 1::2] = torch.cos(pos_matrix / i_matrix)

pe_embedding = nn.Embedding(max_position_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)

src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len])
tar_pos = torch.cat([torch.unsqueeze(torch.arange(max(tar_len)), 0) for _ in tar_len])

src_pe_embedding = pe_embedding(src_pos)
tar_pe_embedding = pe_embedding(tar_pos)

#构造encoder的self-attetion mask
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(len), (0, max(src_len) - len)), 0) for len in src_len]), 2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_encoder_pos_matrix = (1 - valid_encoder_pos_matrix)
mask_encoder_self_attetion = invalid_encoder_pos_matrix.to(torch.bool)

score = torch.randn(batch_size, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attetion, -1E9)
probability = F.softmax(masked_score, -1)
print(score.shape)
print(masked_score)
print(probability)