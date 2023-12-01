import torch
import torch.nn as nn

class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len):
        super(TokenAndPositionalEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)
    
    def forward(self, tokens):
        token_embed = self.token_embedding(tokens)
        positions = torch.arange(0, tokens.size(1), device=tokens.device).unsqueeze(0)
        positional_embed = self.positional_embedding(positions)
        return token_embed + positional_embed

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.inv_freq = 1. / (10000 ** (torch.arange(0, embed_size, 2).float() / embed_size))

    def forward(self, positions):
        angles = positions.unsqueeze(-1) * self.inv_freq
        sinusoids = torch.sin(angles)
        cosinoids = torch.cos(angles)
        pos_embedding = torch.cat([sinusoids, cosinoids], dim=-1)
        return pos_embedding[:, :self.max_len, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        attention = torch.nn.functional.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_size)
        x = self.fc_out(x)
        
        return x

class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size, embed_size, num_heads):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Other required components such as query, key, value linear layers

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # Perform sliding window attention
        for i in range(0, seq_len, self.window_size):
            window_query = query[:, i:i+self.window_size, :]
            window_key = key[:, i:i+self.window_size, :]
            window_value = value[:, i:i+self.window_size, :]

            # Perform attention within the window
            # Implement the attention mechanism within the window here

        # Combine attention results from all windows

        return # Combine attention results and return

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups):
        super(GroupQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_size // num_heads
        
        # Initialize group query projection matrices
        self.query_group_proj = nn.Linear(embed_size, embed_size)
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)

        # Initialize attention computation layers for each group
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(self.embed_size, self.num_heads) for _ in range(self.num_groups)
        ])

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        assert seq_len % self.num_groups == 0, "Sequence length should be divisible by the number of groups."

        # Project queries into groups
        query = self.query_group_proj(query)
        query = query.view(batch_size, seq_len // self.num_groups, self.num_groups, self.head_dim)
        query = query.permute(0, 2, 1, 3).contiguous()

        # Project keys and values
        key = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        value = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        # Perform attention within each group separately
        group_outputs = []
        for i in range(self.num_groups):
            group_query = query[:, i, :, :]
            group_key = key[:, :, i*self.head_dim:(i+1)*self.head_dim]
            group_value = value[:, :, i*self.head_dim:(i+1)*self.head_dim]

            # Perform attention within each group using MultiheadAttention
            # Use attention computation layers initialized earlier for each group
            group_output, _ = self.attention_layers[i](group_query, group_key, group_value, attn_mask=mask)
            group_outputs.append(group_output)

        # Concatenate attention outputs from each group and reshape
        combined_output = torch.cat(group_outputs, dim=1)
        combined_output = combined_output.view(batch_size, self.num_groups, seq_len, self.embed_size)
        combined_output = combined_output.permute(0, 2, 1, 3).contiguous()
        combined_output = combined_output.view(batch_size, seq_len, -1)

        return combined_output

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, window_size, embed_size, num_heads, ff_hidden_size):
        super(TransformerBlock, self).__init__()
        self.attention = SlidingWindowAttention(window_size, embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + attention_output)
        
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm2(x + feed_forward_output)
        
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, window_size, embed_size, max_len, num_heads, ff_hidden_size, num_layers):
        super(GPT2, self).__init__()
        self.token_embedding = TokenAndPositionalEmbedding(vocab_size, embed_size, max_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(window_size, embed_size, num_heads, ff_hidden_size) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x, mask):
        x = self.token_embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        x = self.fc(x)
        return x

def load_weights(model, state_dict_path):
    # Load pre-trained weights into the model
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
# Example usage:
vocab_size = 50257  # Replace with actual vocabulary size
embed_size = 768   # Replace with desired embedding size
max_len = 512      # Maximum sequence length
num_heads = 12       # Number of attention heads
ff_hidden_size = 1024  # Hidden layer size of feed-forward network
num_layers = 12      # Number of transformer layers
window_size = 5

# GPT-2 model
model = GPT2(vocab_size, window_size, embed_size, max_len, num_heads, ff_hidden_size, num_layers)

# # Load pre-trained weights into the model
# pretrained_state_dict_path = 'models/124M/model.ckpt.data-00000-of-00001'  # Replace with the path to your pretrained weights
# load_weights(model, pretrained_state_dict_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate a sample input tensor and mask (replace with your actual data)
sample_input = torch.randint(0, vocab_size, (1, max_len))  # Example input
sample_mask = torch.ones(1, max_len).bool()  # Example mask

# Forward pass
output = model(sample_input, sample_mask)
print(output)  # This will print the shape of the output tensor
