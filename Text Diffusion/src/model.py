import torch
from torch import nn
from torch.nn import functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()

        self.query = nn.Linear(config.n_embd, config.proj_size)
        self.key = nn.Linear(config.n_embd, config.proj_size)
        self.value = nn.Linear(config.n_embd, config.proj_size)
        self.proj = nn.Linear(config.proj_size, config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_dropout_rate)
        self.proj_drop = nn.Dropout(config.proj_dropout_rate)

        self.num_heads = config.num_heads

    def forward(self, q, k, v, mask=None):
       
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        

        B, T, C = q.size()
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)


        y, att_weights = self.attention(q, k ,v, mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))

        return y, att_weights

    def attention(self, q, k, v, mask=None):
        att = (q @ k.transpose(-2, -1))
        att = att  * (1.0 / k.size(-1) ** 0.5)

        if mask is not None:
            # att = att + (mask * -1e9)
            att = att.masked_fill(mask == 1, float('-inf'))

        att_weights = F.softmax(att, dim=-1)
        att = self.attn_drop(att_weights)

        y = att @ v
        return y, att_weights

class EncoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super(EncoderBlock, self).__init__()

        self.mha = MultiheadAttention(config)
        self.dropout = nn.Dropout(config.resid_dropout_rate)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.n_embd),
            nn.Dropout(config.resid_dropout_rate),
        )

    def forward(self, x, padding_mask=None):
        attn_logits, attn_weights = self.mha(x, x, x, mask=padding_mask)
        attn_logits = self.dropout(attn_logits)
        x = self.ln1(x + attn_logits)
        x = x + self.ln2(x + self.mlp(x))

        return x, attn_weights

class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super(Encoder, self).__init__()


        self.token_embds = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_dropout_rate)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])

    def forward(self, idx, padding_mask=None):
        B, T = idx.size()

        token_embds = self.token_embds(idx)
        pos_embs = self.pos_emb[:, :T, :] # each position maps to a (learnable) vector
        x = self.drop(token_embds + pos_embs)

        attn_weights = {}

        for i, block in enumerate(self.blocks):
            x, weights = block(x, padding_mask=padding_mask)
            attn_weights[f'encoder_block_{i}'] = weights

        return x, attn_weights

class BertConfig():
    def __init__(self,
                vocab_size=512,
                n_embd=512,
                block_size=512,
                proj_size=512,
                d_model=512,
                num_heads=8,
                n_layers=12,
                attn_dropout_rate=0.1,
                proj_dropout_rate=0.1,
                resid_dropout_rate=0.1,
                embd_dropout_rate=0.1,
                **kwargs
                ) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.proj_size = proj_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.resid_dropout_rate = resid_dropout_rate
        self.embd_dropout_rate = embd_dropout_rate

class Bert(nn.Module):

    def __init__(self, config: BertConfig):
        super(Bert, self).__init__()

        self.encoder = Encoder(config)

        self.config = config

        self.logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    def forward(self, input_ids, timesteps=None, padding_mask=None, **kwargs):
        B_enc, T_enc = input_ids.size()

        if padding_mask is not None:
            padding_mask = padding_mask.view(B_enc, 1, 1, T_enc)

        enc_out, enc_attnetions = self.encoder(input_ids, padding_mask)
        
        # print(f'{enc_out.size()=}')
        logits = self.logits(enc_out)
        
        return logits #, enc_attnetions

    def __str__(self):
        return f"Bert_Layers_{self.config.n_layers}_Heads_{self.config.num_heads}_Emb_{self.config.n_embd}_Dmodel_{self.config.d_model}"

    @property
    def device(self):
        return next(self.parameters()).device