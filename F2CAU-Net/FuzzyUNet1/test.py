import torch
import torch.nn as nn


class FuzzyMembershipFunction(nn.Module):
    def __init__(self, a, b, c, d):
        super(FuzzyMembershipFunction, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        return torch.max(torch.min((x - self.a) / (self.b - self.a), (self.d - x) / (self.d - self.c)), torch.tensor(0.))


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(SelfAttention, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fmf = FuzzyMembershipFunction(0., 0.3, 0.7, 1.)  # This is an example, you can modify it to fit your needs

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Query, Key, and Value
        q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key(x)    # (batch_size, seq_len, hidden_dim)
        v = self.value(x)  # (batch_size, seq_len, input_dim)

        # Attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        # Attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.fmf(attn_weights)

        # Attention result
        output = torch.bmm(attn_weights, v)  # (batch_size, seq_len, input_dim)

        return output, attn_weights


# 测试自注意力机制
input_dim = 512
seq_len = 512
batch_size = 3

x = torch.randn(batch_size, seq_len, input_dim)

attention = SelfAttention(input_dim)
output, attn_weights = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)
