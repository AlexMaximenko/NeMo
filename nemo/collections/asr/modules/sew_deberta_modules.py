import math

from nemo.core.classes.module import NeuralModule

from nemo.collections.asr.modules.wav2vec_modules import SamePad
from nemo.collections.common.parts.transformer_utils import form_attention_mask, transformer_weights_init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.transformer import TransformerEncoder
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from omegaconf import DictConfig
from omegaconf.dictconfig import DictConfig

import random
from einops.layers.torch import Rearrange

try:
    from DeBERTa import deberta
except ImportError:
    print("Please install deberta")

class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


@torch.jit.script
def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    return torch.arange(0, lengths.max(), device=lengths.device).view(1, -1).expand(lengths.size(0), -1) >= lengths.view(-1, 1)


class SqueezeWav2VecDebertaTransformerEncoder(TransformerEncoder):
    """
        Encoder module following Transformer encoder paradigm 
		as described in Vaswani et al. (https://arxiv.org/abs/1706.03762). Used for SqueezedWav2Vec
		style encoding of context vectors as described by in Felix Wu et al (https://arxiv.org/pdf/2109.06870).
		Takes convolutional encodings of all time steps, adds to features and applies down-sampling by 
        squeezing.squeeze_factor before applying series of self-attention layers.
        After transformer layers applies up-sampling by the same squeeze_factor 
        Deberta means using Disentangled instead of usual Self-Attention
		
		Args:
			layer_drop: Floating point value specifying proportion of module for layer dropout (See Fan et al. https://arxiv.org/pdf/1909.11556.pdf).
				If non-zero, each layer will draw from uniform probability to determine if applied in current forward call.
				Occurs only during training step
			pos_embed: Config specifying parameters for contextual embedding convolutions. Module configures convolutional padding
				to maintain number of time steps
				Must contain following:
					embedding_dim: Depth/number of channels of each time step from feature encoding 
					conv_pos: Kernel size for convolution
					conv_pos_groups: Number of groups for convolution
			transformer: Config for transformer encoder. Uses self-attention layers found in: nemo.collections.nlp.modules.common.transformer
				Must contain followign:
					num_layers: Number of attention layers 
					hidden_size: Expected input depth (embedding size between model layers)
					inner_size: Depth of embeddings within feed-forward sections of encoder layers
					num_attention_heads: Number of attention heads
					attn_score_dropout: Probability of dropout applied to attention scores
					attn_layer_dropout: Probability of dropout applied to the output of the attention layers (prior to normalization)
					ffn_dropout: Probability of dropout applied to feed-forward modules
					hidden_act: Activation function for hidden layers
            squeezing: Config for squeezing
                Must contain followign:
                    squeeze_factor: downsample the sequece length by this factor in pos_conv and upsample after transformer
                    squeeze_method: method to squeeze the temporal dimension
    """

    def __init__(self, pos_embed: DictConfig, transformer: DictConfig, squeezing: DictConfig, layer_drop: float = 0.0):
        super().__init__(**transformer)

        # configuration deberta config
        ## common transformer config
        deberta_cfg = deberta.config.ModelConfig()

        deberta_cfg.num_hidden_layers = transformer.num_layers
        deberta_cfg.hidden_size = transformer.hidden_size
        deberta_cfg.intermediate_size = transformer.hidden_size * 4
        deberta_cfg.num_attention_heads = transformer.num_attention_heads
        deberta_cfg.attention_head_size = 64
        deberta_cfg.hidden_act = str(transformer.hidden_act)
        deberta_cfg.hidden_dropout_prob = transformer.get('activation_dropout', 0)
        deberta_cfg.conv_kernel_size = 0

        ## deberta only config
        deberta_cfg.layer_norm_eps = 1e-7
        deberta_cfg.norm_rel_ebd = transformer.get('norm_rel_ebd', "layer_norm")
        deberta_cfg.max_position_embeddings = transformer.get('max_position_embeddings', 512)
        deberta_cfg.max_relative_positions = transformer.get('max_relative_positions', -1)
        deberta_cfg.position_biased_input = transformer.get('position_biased_input', False)
        deberta_cfg.pos_att_type = transformer.get('pos_att_type', "p2c|c2p")
        deberta_cfg.position_buckets = transformer.get('position_buckets', 256)
        deberta_cfg.initializer_range = transformer.get('initializer_range', 0.02)
        deberta_cfg.relative_attention = transformer.get('relative_attention', True)
        deberta_cfg.share_att_key = transformer.get('share_att_key', True)

        self.encoder = deberta.bert.BertEncoder(deberta_cfg)
        self.encoder.layer = LayerDropModuleList(layer_drop, [l for l in self.encoder.layer])

        # convolution layer before transformer layers
        self.dropout = transformer.attn_layer_dropout
        self.pos_embed = pos_embed
        self.squeeze_factor = squeezing.squeeze_factor
        self.squeeze_method = squeezing.squeeze_method
        self.pos_conv = self._get_pos_conv()
        self.pool = self._get_pool()
        self.upsample = self._get_upsample()

        self.layer_drop = layer_drop
        self.layer_norm = nn.LayerNorm(pos_embed.embedding_dim)
        self.apply(lambda x: transformer_weights_init(x, xavier=False))

    @property
    def input_types(self):
        """Returns definitions of module output ports. 
        We treat features as SpectrogramType for Nemo compatibility
        audio_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        length:
            0: AxisType(BatchTag)
        """
        return {
            "audio_signal": NeuralType(('B', 'C', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports. 
        We're using SpectrogramType for now to keep things Nemo safe
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'C', 'T'), AcousticEncodedRepresentation()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def forward(self, audio_signal, length):
        
        # Padding mask needed for transformer
        padding_mask = self.create_padding_mask(length)

        T = audio_signal.shape[2] # Temporal dimension

        # Applying padding before convolution
        for idx, len in enumerate(length):
            audio_signal[idx, :, len:] = 0.0

        # Applying squeezing along temporal dimension
        if isinstance(self.pos_conv, nn.Sequential):
            signal_conv = self.pos_conv(audio_signal)
            signal_pool = self.pool(audio_signal)
            min_length = min(signal_conv.size(-1), signal_pool.size(-1))
            audio_signal = (signal_pool[...,:min_length] + signal_conv[...,:min_length])
        elif isinstance(self.pos_conv, nn.ModuleList):
            for conv in self.pos_conv:
                signal_conv = conv(audio_signal)
                signal_pool = self.pool(audio_signal)
                min_length = min(signal_conv.size(-1), signal_pool.size(-1))
                audio_signal = (signal_pool[...,:min_length] + signal_conv[...,:min_length])
        else:
            raise NotImplementedError

        audio_signal = audio_signal.transpose(1, 2) # B, C, T -> B, T, C
        audio_signal = self.layer_norm(audio_signal)

        # adjust the padding_mask
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = input_lengths // self.squeeze_factor
            output_lengths += audio_signal.size(1) - output_lengths.max().item()
            padding_mask = make_pad_mask(output_lengths).to(audio_signal.device) # 1 at padding
            attention_mask = padding_mask.eq(0).long()

        # Applying transformer
        context_emb = self.encoder(audio_signal, attention_mask=attention_mask)[-1]

        # Upsampling
        if self.upsample is not None:
            context_emb = self.layer_norm(context_emb)
            context_emb = self.upsample(context_emb)

        # Padding        
        if context_emb.size(1) < T:
            context_emb = F.pad(context_emb, (0, 0, 0, T - context_emb.size(1)))

        context_emb = context_emb.transpose(1, 2) # B T C -> B C T
        return context_emb, length  # Returning length for NeMo compatibility

    def _get_pool(self):
        if self.squeeze_factor == 1:
            return nn.Identity()
        if self.squeeze_method in {'default', 'default-v2'}:
            pool = nn.AvgPool1d(self.squeeze_factor, self.squeeze_factor)
        else:
            raise ValueError(f"squeeze_method={self.squeeze_method}")
        return pool

    def _get_pos_conv(self): 
        if self.squeeze_method in {'default', 'default-v2'}:
            pos_conv = nn.Conv1d(
                self.pos_embed.embedding_dim,
                self.pos_embed.embedding_dim,
                kernel_size=self.pos_embed.conv_pos,
                padding=self.pos_embed.conv_pos // 2,
                groups=self.pos_embed.conv_pos_groups,
                stride=self.squeeze_factor,
            )
            std = math.sqrt((4 * (1.0 - self.dropout)) / (self.pos_embed.conv_pos * self.pos_embed.embedding_dim))
            nn.init.normal_(pos_conv.weight, mean=0, std=std)
            nn.init.constant_(pos_conv.bias, 0)
            pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
            pos_conv = nn.Sequential(pos_conv, SamePad(self.pos_embed.conv_pos), nn.GELU())
        else:
            raise ValueError(f"squeeze_method={self.squeeze_method}")
        return pos_conv

    def _get_upsample(self):
        if self.squeeze_method == 'default':
            layers = [
                nn.Linear(self.pos_embed.embedding_dim, self.pos_embed.embedding_dim * self.squeeze_factor),
                nn.GELU(),
                Rearrange('b t (s c) -> b (t s) c', s=self.squeeze_factor, c=self.pos_embed.embedding_dim),
            ]
            upsample = nn.Sequential(*layers)
        elif self.squeeze_method == 'default-v2':
            layers = []
            for _ in range(int(np.log2(self.squeeze_factor))):
                layers += [
                    nn.Linear(self.pos_embed.embedding_dim, self.pos_embed.embedding_dim * 2),
                    nn.GELU(),
                    Rearrange('b t (s c) -> b (t s) c', s=2, c=self.pos_embed.embedding_dim),
                ]
            upsample = nn.Sequential(*layers)
        else:
            raise ValueError(f"squeeze_method={self.squeeze_method}")
        for m in upsample.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return upsample

    def create_padding_mask(self, length):
        # Broadcast to vectorize creating the padding mask
        max_len = int(max(length))
        padding_mask = torch.arange(max_len, device=length.device)

        # Switch to binary for transformer, 1 for valid tokens, 0 for padding
        padding_mask = (padding_mask.expand(len(length), max_len) < length.unsqueeze(1)).type(torch.uint8)

        return padding_mask
    