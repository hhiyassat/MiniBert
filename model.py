import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        position_emb = self.position_embedding(positions)
        embeddings = token_emb + position_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MiniBERT(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, num_heads=4, 
                 intermediate_dim=1024, max_len=128, dropout=0.1):
        super().__init__()
        
        self.embedding = BERTEmbedding(vocab_size, hidden_dim, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better training stability
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        encoded = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        logits = self.mlm_head(encoded)
        return logits
