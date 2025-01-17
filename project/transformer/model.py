import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len = 512):
        super().__init__()
        position = torch.arange(0, max_seq_len).reshape(-1, 1)
        weight = (1 / 10000 ** (torch.arange(0, d_model, 2) / d_model)).reshape(1, -1)
        item = position * weight
        position_encoding = torch.zeros(max_seq_len, d_model)
        position_encoding[:, 0::2] = torch.sin(item) 
        position_encoding[:, 1::2] = torch.cos(item) 
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer("position_encoding", position_encoding, False)

    def forward(self, input_matrix):
        """
            input_shape [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = input_matrix.shape
        return input_matrix + self.position_encoding[:, :seq_len, :]
            
def attention(query, key, value, mask=None):
    d_model = key.shape[-1]
    result = torch.matmul(query, key.transpose(-1, -2)) / d_model ** 0.5
    if mask is not None:
        result = result.masked_fill_(mask, -1E9)
    attention_score = torch.softmax(result, -1)
    return torch.matmul(attention_score, value)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, heads_num, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads_num == 0
        self.query_linear = nn.Linear(d_model, d_model, bias=False) 
        self.key_linear = nn.Linear(d_model, d_model, bias=False) 
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.heads_num = heads_num
        self.k_num = d_model // heads_num
        self.d_model = d_model

    def forward(self, query, key, value, mask=None):
        """
            shape from [batch, seq_len, d_model] to [batch, heads_num, seq_len, k_num]
        """
        batch = query.shape[0]
        query = self.query_linear(query).reshape(batch, -1, self.heads_num, self.k_num).transpose(1, 2)                                                          
        key = self.key_linear(key).reshape(batch, -1, self.heads_num, self.k_num).transpose(1, 2)                                                                                                          
        value = self.value_linear(value).reshape(batch, -1, self.heads_num, self.k_num).transpose(1, 2)                                                                                                             
        result = attention(query, key, value, mask)
        result = result.transpose(1, 2).reshape(batch, -1, self.d_model)
        result = self.linear(result)
        result = self.dropout(result)
        return result

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feed_forward(x)

class EncoderLayer(nn.Module):
    def __init__(self, heads_num, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_multihead_attention = MultiHeadAttention(heads_num, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        multihead_attention_out = self.self_multihead_attention(x, x, x, src_mask)
        multihead_attention_out = self.add_norms[0](x + multihead_attention_out)
        feed_forward_out = self.feed_forward(multihead_attention_out)
        feed_forward_out = self.add_norms[1](multihead_attention_out + feed_forward_out)
        result = self.dropout(feed_forward_out)
        return result

class Encoder(nn.Module):
    def __init__(self, layer_num, vocabulary_size, padding_idx, heads_num, d_model, d_ff, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.word_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx)
        self.position_embedding = PositionEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(heads_num, d_model, d_ff, dropout) for _ in range(layer_num)])

    def forward(self, x, src_mask=None):
        x_embedding = self.word_embedding(x)   
        x_embedding = self.position_embedding(x_embedding)
        result = x_embedding
        for encoder_layer in self.encoder_layers:
            result = encoder_layer(result, src_mask)
        return result

class DecoderLayer(nn.Module):
    def __init__(self, heads_num, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.masked_multihead_attention = MultiHeadAttention(heads_num, d_model, dropout)
        self.multihead_attention = MultiHeadAttention(heads_num, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_kv, dst_mask=None, src_dst_mask=None):
        masked_multihead_attention_out = self.masked_multihead_attention(x, x, x, dst_mask)
        masked_multihead_attention_out = self.add_norms[0](x + masked_multihead_attention_out)
        multihead_attention_out = self.multihead_attention(masked_multihead_attention_out, encoder_kv, encoder_kv, src_dst_mask)
        multihead_attention_out = self.add_norms[1](masked_multihead_attention_out + multihead_attention_out)
        feed_forward_out = self.feed_forward(multihead_attention_out)
        feed_forward_out = self.add_norms[2](multihead_attention_out + feed_forward_out)
        result = self.dropout(feed_forward_out)
        return result

class Decoder(nn.Module):
    def __init__(self, layer_num, vocabulary_size, padding_idx, heads_num, d_model, d_ff, dropout, max_seq_len=512):
        super().__init__()
        self.word_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx)
        self.position_embedding = PositionEncoding(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(heads_num, d_model, d_ff, dropout) for _ in range(layer_num)])

    def forward(self, x, encoder_kv, dst_mask=None, src_dst_mask=None):
        x_embedding = self.word_embedding(x)
        x_embedding = self.position_embedding(x_embedding)
        result = x_embedding
        for decoder_layer in self.decoder_layers:
            result = decoder_layer(result, encoder_kv, dst_mask, src_dst_mask)
        return result

class Transformer(nn.Module):
    def __init__(self, encoder_vocabulary_size, decoder_vocabulary_size, layer_num,
        padding_idx, heads_num, d_model, d_ff, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.encoder = Encoder(layer_num, encoder_vocabulary_size, padding_idx, heads_num, d_model, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(layer_num, decoder_vocabulary_size, padding_idx, heads_num, d_model, d_ff, dropout, max_seq_len)
        self.linear = nn.Linear(d_model, decoder_vocabulary_size)
        self.padding_idx = padding_idx
    
    def generate_mask(self, query, key, is_triu_mask=False):
        device = query.device
        batch, seq_q = query.shape
        _, seq_k = key.shape
        mask = (key == self.padding_idx).unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch, 1, seq_q, seq_k).to(device)
        if is_triu_mask:
            dst_triu_mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)
            dst_triu_mask = dst_triu_mask.unsqueeze(0).unsqueeze(1)
            dst_triu_mask = dst_triu_mask.expand(batch, 1, seq_q, seq_k).to(device)
            return mask|dst_triu_mask
        return mask

    def forward(self, inputs, outputs):
        src_mask = self.generate_mask(inputs, inputs)
        dst_mask = self.generate_mask(outputs, outputs, is_triu_mask=True)
        src_dst_mask = self.generate_mask(outputs, inputs)
        encoder_out = self.encoder(inputs, src_mask)
        decoder_out = self.decoder(outputs, encoder_out, dst_mask, src_dst_mask)
        probability_out = self.linear(decoder_out)
        return probability_out 


if __name__ == "__main__":
    transformer = Transformer(10, 20, 1, 0, 4, 16, 8)
    inputs = torch.randint(0, 10, (2, 4))
    outputs = torch.randint(0, 20, (2, 5))
    result = transformer(inputs, outputs)
    print(result.shape)
    pass