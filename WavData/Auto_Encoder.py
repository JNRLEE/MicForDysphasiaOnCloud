# Part1: Reconstruct audio from raw wav

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


device=torch.device('cpu')

config_path = "./WavTokenizer/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./WavTokenizer/avtokenizer_medium_music_audio_320_24k_v2.ckpt"
audio_outpath = "xxx"

'''
1. 原始數據結構 (Raw Data Structure)
/WavData/
├── NormalSubject/
│   ├── N002/
│   │   └── N002_2024-10-22_10-50-58/
│   │       ├── N002_info.json(或是N002_info)           # 包含實驗信息
│   │       └── Probe0_RX_IN_TDM4CH0.wav # 原始音頻文件
│   └── ... (其他正常受試者)
└── PatientSubject/
  ├── P001/
  │   └── P001-1/
  │       ├── P001_info.json(或是P001_info)
  │       └── DrySwallow_FB_left_240527.wav
  └── ... (其他病患受試者)
'''

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)


wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id) 
torchaudio.save(audio_outpath, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

# Part2: Generating discrete codecs

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device=torch.device('cpu')

config_path = "./WavTokenizer/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./WavTokenizer/avtokenizer_medium_music_audio_320_24k_v2.ckpt"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
_,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
print(discrete_code)

#Part3: Audio reconstruction through codecs

# audio_tokens [n_q,1,t]/[n_q,t]
features = wavtokenizer.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([0])  
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)