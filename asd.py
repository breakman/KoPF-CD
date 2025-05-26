import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader

class MultiScaleTemporalFeature(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.norm = nn.LayerNorm(hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim * 3, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = [conv(x) for conv in self.convs]
        out = torch.cat(features, dim=1)
        out = out.transpose(1, 2)
        out = self.norm(out)
        return self.proj(out)

class FractalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, norm in zip(self.attn_layers, self.norms):
            residual = x
            x, _ = attn(x, x, x)
            x = norm(x + residual)
        return x

class ScaleMixer(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 4.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, int(dim * expansion_factor))
        self.linear2 = nn.Linear(int(dim * expansion_factor), dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.norm(x + residual)

class MSTRBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mtf = MultiScaleTemporalFeature(dim, hidden_dim)
        self.attn = FractalSelfAttention(dim, num_heads, dropout)
        self.mixer = ScaleMixer(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mtf(x)
        x = self.attn(x)
        x = self.mixer(x)
        return x

class Wav2Vec2WithMSTR(nn.Module):
    def __init__(self, base_model_name="facebook/wav2vec2-base", num_layers=12, mstr_dim=768, mstr_hidden=256, num_heads=8):
        super().__init__()
        # HuggingFace 모델 로드
        self.wav2vec = Wav2Vec2Model.from_pretrained(base_model_name)
        
        # CNN feature encoder는 그대로 사용
        self.feature_extractor = self.wav2vec.feature_extractor
        
        # MSTR layer들
        self.mstr_layers = nn.ModuleList([
            MSTRBlock(dim=mstr_dim, hidden_dim=mstr_hidden, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # layer norm 및 projection
        self.encoder_layer_norm = self.wav2vec.encoder.layer_norm
        self.masked_spec_embed = None

    def forward(self, input_values, attention_mask=None):
        # 1. CNN Feature Extractor (B, T) → (B, Feature_T, D)
        extract_features = self.feature_extractor(input_values)
        
        # 2. MSTR layers 통과
        x = extract_features
        for layer in self.mstr_layers:
            x = layer(x)
        
        # 3. 최종 LayerNorm
        x = self.encoder_layer_norm(x)
        return x

class WavDataset(Dataset):
    def __init__(self, wav_dir, sample_rate=16000):
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_path = os.path.join(self.wav_dir, self.wav_files[idx])
        waveform, sr = torchaudio.load(wav_path)
        
        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 샘플링 레이트 변환
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
        return waveform.squeeze(), self.wav_files[idx]

# 테스트 코드
if __name__ == "__main__":
    # 모델 초기화
    model = Wav2Vec2WithMSTR()
    model.eval()  # 평가 모드로 설정
    
    # 데이터셋 및 데이터로더 설정
    dataset = WavDataset("./Sample/01.원천데이터/2.영어/건강,운동/8991")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 데이터 처리
    with torch.no_grad():
        for waveform, filename in dataloader:
            # 모델 입력
            output = model(waveform)
            
            print(f"처리 중인 파일: {filename}")
            print(f"입력 shape: {waveform.shape}")
            print(f"출력 shape: {output.shape}")
            print("-" * 50)