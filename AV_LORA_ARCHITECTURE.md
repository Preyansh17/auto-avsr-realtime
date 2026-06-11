# AV模型架构与LoRA微调详解

## 一、整体架构 (E2E Audiovisual Model)

### 1. 模型主要组件
位置: `espnet/nets/pytorch_backend/e2e_asr_conformer_av.py`

```
输入: video + audio
  ↓
[Video Encoder] ──┐
                  ├─→ [Fusion] → [CTC] + [Attention Decoder] → 输出
[Audio Encoder] ──┘
```

### 2. 三大核心模块

#### **2.1 Video Encoder (主编码器)**
```python
# 位置: line 27-43
self.encoder = Encoder(
    attention_dim=768,        # adim
    attention_heads=12,       # aheads
    linear_units=3072,        # eunits (FFN hidden)
    num_blocks=12,            # elayers (Conformer层数)
    input_layer='conv3d',     # 3D卷积前端
    ...
)
```

**内部结构:**
- **Frontend**: `Conv3dResNet` (3D卷积 + ResNet)
  - 处理原始视频帧 (B, T, C, H, W)
  - 提取时空特征 → 512维
  
- **Embedding**: `Linear(512, 768)` + `PositionalEncoding`
  
- **12层 Conformer Encoder Layers**, 每层包含:
  1. **Macaron-style FFN 1** (可选, scale=0.5)
  2. **Multi-Head Self-Attention** (12 heads, 768 dim)
  3. **Convolution Module** (kernel=31, 1D卷积)
  4. **FFN 2** (768 → 3072 → 768, scale=0.5 if macaron)
  
- **LayerNorm** 输出归一化

#### **2.2 Audio Encoder (辅助编码器)**
```python
# 位置: line 45-61
self.aux_encoder = Encoder(
    attention_dim=768,        # aux_adim
    attention_heads=12,       # aux_aheads
    linear_units=3072,        # aux_eunits
    num_blocks=12,            # aux_elayers
    input_layer='conv1d',     # 1D卷积前端
    ...
)
```

**内部结构:**
- **Frontend**: `Conv1dResNet` (1D卷积 + ResNet)
  - 处理音频波形
  - 提取声学特征 → 512维
  
- **Embedding**: `Linear(512, 768)` + `PositionalEncoding`
  
- **12层 Conformer Encoder Layers** (结构同Video Encoder)

#### **2.3 Fusion Module (融合模块)**
```python
# 位置: line 66-71
self.fusion = MLPHead(
    idim=768 + 768,          # video_dim + audio_dim = 1536
    hdim=8192,               # fusion hidden dim
    odim=768,                # 输出回768
    norm='batchnorm'
)
```

**结构:**
```python
# espnet/nets/pytorch_backend/nets_utils.py: line 505-526
x = Linear(1536 → 8192)(concat[video_feat, audio_feat])
x = BatchNorm1d(x)
x = ReLU(x)
x = Linear(8192 → 768)(x)
```

### 3. 解码部分

#### **3.1 CTC分支** (用于训练和beam search)
```python
# 位置: line 108-112
self.ctc = CTC(
    odim=vocab_size,
    eprojs=768,
    dropout_rate=0.1
)
```

#### **3.2 Attention Decoder** (自回归解码)
```python
# 位置: line 78-88
self.decoder = Decoder(
    odim=vocab_size,
    attention_dim=768,       # ddim
    attention_heads=12,      # dheads
    linear_units=3072,       # dunits
    num_blocks=6,            # dlayers
    ...
)
```

**结构:**
- **Embedding**: `Embedding(vocab_size, 768)` + `PositionalEncoding`
- **6层 Decoder Layers**, 每层包含:
  1. **Self-Attention** (masked, 12 heads)
  2. **Cross-Attention** (attend to encoder output, 12 heads)
  3. **FFN** (768 → 3072 → 768)
- **Output Layer**: `Linear(768 → vocab_size)`

### 4. 前向传播流程

```python
# espnet/nets/pytorch_backend/e2e_asr_conformer_av.py: line 114-142

def forward(video, audio, video_lengths, audio_lengths, label):
    # 1. Video编码
    video_feat, _ = encoder(video, video_mask)        # [B, T, 768]
    
    # 2. Audio编码
    audio_feat, _ = aux_encoder(audio, audio_mask)    # [B, T, 768]
    
    # 3. 融合
    x = fusion(concat(video_feat, audio_feat))        # [B, T, 768]
    
    # 4. CTC损失
    loss_ctc, _ = ctc(x, video_lengths, label)
    
    # 5. Attention Decoder损失
    pred, _ = decoder(label_in, label_mask, x, video_mask)
    loss_att = criterion(pred, label_out)
    
    # 6. 联合训练
    loss = mtlalpha * loss_ctc + (1 - mtlalpha) * loss_att
    
    return loss, loss_ctc, loss_att, acc
```

---

## 二、LoRA微调机制

### 1. LoRA原理

**核心思想:** 在预训练Linear层旁路添加低秩矩阵，只训练低秩参数

```
原始: y = W·x                    (W: [out, in])
LoRA: y = W·x + (α/r)·B·A·x     (A: [r, in], B: [out, r])
```

- **W**: 冻结的预训练权重
- **A, B**: 可训练的LoRA矩阵
- **r**: 秩 (通常8-64)
- **α**: 缩放因子 (通常=16)

### 2. LoRA实现

#### **2.1 LoRALinear模块**
```python
# modules/lora.py: line 9-41

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        # 冻结原始权重
        self.weight = nn.Parameter(base.weight.clone(), requires_grad=False)
        self.bias = nn.Parameter(base.bias.clone(), requires_grad=False) if base.bias else None
        
        # LoRA参数 (可训练)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))      # [8, in]
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))     # [out, 8]
        self.scaling = alpha / r  # 16/8 = 2.0
        
        # 初始化
        nn.init.kaiming_uniform_(lora_A)  # 高斯初始化
        nn.init.zeros_(lora_B)            # 零初始化 (开始时LoRA=0)
    
    def forward(self, x):
        # 原始输出
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA增量
        lora = self.lora_B @ (self.lora_A @ dropout(x).T)
        lora = lora.T
        
        return result + self.scaling * lora
```

#### **2.2 注入LoRA**
```python
# modules/lora.py: line 51-86

def inject_lora(model, target_scopes, r=8, alpha=16, dropout=0.0):
    """将指定scope内的Linear层替换为LoRALinear"""
    replaced = 0
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        # 检查是否在目标scope
        if not any(name.startswith(scope) for scope in target_scopes):
            continue
        
        # 替换为LoRALinear
        parent_name, child_name = name.rsplit('.', 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, LoRALinear(module, r, alpha, dropout))
        replaced += 1
    
    return replaced
```

### 3. 在AV模型中应用LoRA

#### **3.1 配置 (configs/train_config.yaml)**
```yaml
lora:
  enabled: true
  r: 8                  # 秩
  alpha: 16            # 缩放因子 (scaling = 16/8 = 2.0)
  dropout: 0.05        # LoRA层的dropout
  scopes: 
    - "encoder"        # Video Encoder所有Linear
    - "aux_encoder"    # Audio Encoder所有Linear
    - "decoder"        # Decoder所有Linear
  name_patterns: []    # 可选: 进一步用正则过滤层名
```

#### **3.2 注入和冻结 (lightning_av.py: line 48-62)**
```python
self.lora_enabled = cfg.lora.enabled

if self.lora_enabled:
    # 1. 注入LoRA
    replaced = inject_lora(
        self.model, 
        scopes=["encoder", "aux_encoder", "decoder"],
        r=8, alpha=16, dropout=0.05
    )
    
    # 2. 冻结所有参数, 只训练LoRA
    for name, param in self.model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True   # LoRA参数可训练
        else:
            param.requires_grad = False  # 其他参数冻结
    
    print(f"✓ LoRA enabled: replaced {replaced} Linear layers")
```

#### **3.3 优化器只训练LoRA参数**
```python
# lightning_av.py: line 79-84

def configure_optimizers(self):
    # 只收集requires_grad=True的参数 (即LoRA参数)
    params = [p for p in self.model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        [{"name": "lora", "params": params, "lr": cfg.optimizer.lr}],
        weight_decay=0.01, betas=(0.9, 0.98)
    )
    ...
```

### 4. 具体哪些层被替换

基于 `scopes=["encoder", "aux_encoder", "decoder"]`，所有以下Linear层会被替换:

#### **4.1 Video Encoder (encoder.)**
- `encoder.embed.0` - Embedding projection (512→768)
- `encoder.encoders.{0-11}.*` - 12层Conformer
  - `.self_attn.linear_q/k/v/out` - 每层4个attention Linear
  - `.feed_forward.w_1/w_2` - 每层2个FFN Linear
  - `.feed_forward_macaron.w_1/w_2` - 每层2个Macaron FFN (如果启用)
  - `.conv_module.*` - Conv模块中的Linear
  - `.concat_linear` - 如果concat_after=True
- `encoder.after_norm` - 无Linear

**Video Encoder总计: ~100-150个Linear层**

#### **4.2 Audio Encoder (aux_encoder.)**
结构同Video Encoder
**Audio Encoder总计: ~100-150个Linear层**

#### **4.3 Decoder (decoder.)**
- `decoder.embed.0` - Token embedding (vocab_size→768)
- `decoder.decoders.{0-5}.*` - 6层Decoder
  - `.self_attn.linear_q/k/v/out` - 每层4个self-attention Linear
  - `.src_attn.linear_q/k/v/out` - 每层4个cross-attention Linear
  - `.feed_forward.w_1/w_2` - 每层2个FFN Linear
- `decoder.output_layer` - 输出projection (768→vocab_size)

**Decoder总计: ~60-80个Linear层**

### 5. 参数量对比

#### **原始模型 (443M参数)**
- Video Encoder: ~150M
- Audio Encoder: ~150M  
- Decoder: ~100M
- Fusion + CTC: ~40M

#### **LoRA参数 (r=8, 约300个Linear层)**
假设平均每个Linear: in=768, out=768
- 原始参数: 768×768 = 589,824
- LoRA参数: 8×768 + 768×8 = 12,288 (**仅2%**)

**总LoRA参数: 300层 × 12,288 ≈ 3.7M** (约占原模型的0.8%)

### 6. LoRA微调优势

1. **参数高效**: 只训练<1%参数，显存需求低
2. **防止灾难性遗忘**: 预训练权重冻结，保留通用能力
3. **快速适配**: 90个样本即可微调到新领域 (ALS患者)
4. **易于部署**: 可以只存储LoRA参数 (3.7M vs 443M)

### 7. 训练流程

```python
# 前向传播 (所有权重参与)
loss = model(video, audio, ...)

# 反向传播 (只更新LoRA)
loss.backward()  # 只有lora_A, lora_B有梯度
optimizer.step() # 只更新这些参数

# 推理 (LoRA自动融合)
y = W·x + (α/r)·B·A·x  # LoRA增量在forward中自动加上
```

---

## 三、完整数据流示例

### Patient AV LoRA训练

**输入:**
- Video: [B, T, C, H, W] - 患者视频帧
- Audio: [B, T'] - 患者音频波形
- Label: [B, L] - 50个句子的token序列

**处理流程:**
```
1. Video → Conv3dResNet → [B,T,512] → Linear(冻结) + LoRA_A@LoRA_B → [B,T,768]
2. 12×Conformer(每层多个LoRA Linear) → video_feat [B,T,768]

3. Audio → Conv1dResNet → [B,T,512] → Linear(冻结) + LoRA → [B,T,768]
4. 12×Conformer(每层多个LoRA Linear) → audio_feat [B,T,768]

5. Concat → [B,T,1536] → Fusion(Linear冻结, 无LoRA) → [B,T,768]

6. CTC(冻结) + Decoder(6层LoRA Linear) → loss

7. 只更新encoder/aux_encoder/decoder中的lora_A, lora_B
```

**关键特点:**
- Fusion模块: **不在LoRA scope**, 使用冻结的预训练权重
- CTC: **冻结**, 保持预训练的声学建模能力
- 只有三个Transformer (encoder/aux_encoder/decoder) 被LoRA适配

---

## 代码位置总结

| 组件 | 文件路径 | 行号 |
|------|----------|------|
| AV模型定义 | `espnet/nets/pytorch_backend/e2e_asr_conformer_av.py` | 23-142 |
| Encoder | `espnet/nets/pytorch_backend/transformer/encoder.py` | 46-200 |
| EncoderLayer | `espnet/nets/pytorch_backend/transformer/encoder_layer.py` | 18-150 |
| Decoder | `espnet/nets/pytorch_backend/transformer/decoder.py` | 39-228 |
| Fusion | `espnet/nets/pytorch_backend/nets_utils.py` | 505-526 |
| LoRALinear | `modules/lora.py` | 9-41 |
| inject_lora | `modules/lora.py` | 51-86 |
| LoRA应用 | `lightning_av.py` | 48-62 |
| 配置 | `configs/train_config.yaml` | 36-42 |
| 模型超参 | `configs/model/audiovisual_backbone/resnet_conformer.yaml` | 1-49 |

