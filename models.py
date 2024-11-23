import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper Classes
class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.transpose(0, 1)  # (B, T, D) -> (T, B, D)
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout(attn_output)
        return attn_output.transpose(0, 1)  # (T, B, D) -> (B, T, D)

class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size, 
            padding=kernel_size//2, 
            groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.GELU()
        self.pointwise_conv = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch_size, time, dim]
        residual = x
        x = self.layer_norm(x)
        
        # Rearrange for convolution
        x = x.transpose(1, 2)  # [batch_size, dim, time]
        
        # Pointwise conv
        x = self.conv1(x)
        x = self.glu(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv
        x = self.pointwise_conv(x)
        x = self.dropout(x)
        
        # Back to original shape
        x = x.transpose(1, 2)  # [batch_size, time, dim]
        
        return x + residual

# Main Building Blocks
class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor, conv_kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expansion_factor, dropout)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.conv = ConformerConvModule(dim, conv_kernel_size, dropout)
        self.ff2 = FeedForward(dim, ff_expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x shape: [batch_size, time, dim]
        x = self.norm1(x + 0.5 * self.ff1(x))
        x = self.norm2(x + self.self_attn(x))
        x = self.norm3(x + self.conv(x))
        x = self.norm4(x + 0.5 * self.ff2(x))
        return x

class MultiheadEmotionAttention(nn.Module):
    def __init__(self, model_dim, num_heads, emotion_embedding_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(model_dim, num_heads)
        self.emotion_proj = nn.Linear(emotion_embedding_dim, model_dim)
        
    def forward(self, x, emotion_embedding):
        emotion_key = self.emotion_proj(emotion_embedding)
        attended_features, _ = self.mha(x, emotion_key, emotion_key)
        return attended_features

class WaveformAdapter(nn.Module):
    def __init__(self, hidden_size, mask_time_prob, mask_time_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.mask_embedding = nn.Parameter(torch.FloatTensor(hidden_size).uniform_())
        
    def forward(self, x, apply_mask=True):
        if apply_mask:
            # Apply time masking
            mask = torch.bernoulli(torch.full(x.shape[:-1], self.mask_time_prob))
            mask = self._expand_mask(mask)
            x = torch.where(mask, self.mask_embedding, x)
        
        return self.proj(x)
    
    def _expand_mask(self, mask):
        expanded = mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        return expanded

# Main Model
class UrduClinicalEmotionTransformer(nn.Module):
    def __init__(self, num_emotions=4):
        super().__init__()
        
        # Input projection (mel_bins -> hidden_dim)
        self.input_projection = nn.Linear(80, 512)  # Changed from 301 to 80 mel bins
        
        # Conformer-based audio encoder
        self.audio_encoder = nn.ModuleList([
            ConformerBlock(
                dim=512,
                num_heads=8,
                ff_expansion_factor=4,
                conv_kernel_size=31,
                dropout=0.1
            ) for _ in range(12)
        ])
        
        # Clinical adapter
        self.clinical_adapter = WaveformAdapter(
            hidden_size=512,
            mask_time_prob=0.05,
            mask_time_length=10
        )
        
        # Final classification
        self.classifier = nn.Linear(512, num_emotions)
        
        # Layer norm for input
        self.input_norm = nn.LayerNorm(512)
    
    def forward(self, x):
        # x shape: [batch_size, channels, mel_bins, time]
        batch_size = x.size(0)
        time_steps = x.size(-1)
        
        # Reshape and transpose
        x = x.squeeze(1)  # [batch_size, mel_bins, time]
        x = x.transpose(1, 2)  # [batch_size, time, mel_bins]
        x = x.reshape(-1, x.size(-1))  # [batch_size * time, mel_bins]
        
        # Project to hidden dimension
        features = self.input_projection(x)  # [batch_size * time, hidden_dim]
        features = features.view(batch_size, time_steps, -1)  # [batch_size, time, hidden_dim]
        features = self.input_norm(features)
        
        # Apply Conformer layers
        for layer in self.audio_encoder:
            features = layer(features)
        
        # Apply clinical adapter
        features = self.clinical_adapter(features)
        
        # Global pooling
        features = torch.mean(features, dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        output = self.classifier(features)
        
        return output 