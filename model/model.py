import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import time
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import xformers.ops as xops

@dataclass
class ModelConfig:
    dim: int = 256  # Dimension of the model
    n_layers: int = 8  # Number of layers in the transformer
    n_heads: int = 16  # Number of attention heads
    n_kv_heads: int = 16
    vocab_size: int = 50000  # Vocabulary size
    norm_eps: float = 1e-5  # Epsilon value for normalization
    ffn_dim_multiplier: int = None
    multiple_of: int = 128
    max_batch_size: int = 150  # Maximum batch size for training
    max_seq_len: int = 512  # Maximum sequence length

    device: str = 'cuda'  # Device to run the model on (optional)

class RotaryPositionEmbedding(nn.Module):

    def __init__(self, head_dim: int, seq_len: int, device: str) -> None:
        super().__init__()
        self.dim = head_dim
        assert self.dim % 2 == 0, "head_dim must be divisible by 2"

        theta_numerator = torch.arange(0, self.dim, 2, dtype=torch.float32)
        theta = 1.0 / torch.pow(10000, theta_numerator / self.dim).to(device)

        m = torch.arange(seq_len, dtype=torch.float32).to(device)
        freqs = torch.outer(m, theta).float()

        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_complex", freqs_complex)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        _, seq_len, _, _ = x.shape

        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        freq_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)

        x_rotated = x_complex * freq_complex

        x_out = torch.view_as_real(x_rotated)

        x_out = x_out.reshape(*x.shape)

        return x_out.type_as(x).to(x.device)

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps  # Epsilon value for numerical stability
        self.gamma = nn.Parameter(torch.ones(dim))  # Learnable parameter for scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        normalized_x = (x / rms) * self.gamma

        return normalized_x

class SelfAttention(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.dim = args.dim

        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.head_dim = args.dim // args.n_heads

        self.Wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Initialize key and value caches with zeros
        '''self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim),
            device=args.device,
            requires_grad = False)
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim),
            device=args.device,
            requires_grad = False)'''

        # Rotary Position Embedding
        self.rope = RotaryPositionEmbedding(self.head_dim, args.max_seq_len, args.device)

    @staticmethod
    def repeat_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:

        # Repeat the heads of K and V to match the number of heads in Q

        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (x[:, :, :, None, :]
                    .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                    .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
                    )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape  # (B, 1, dim)
        assert dim == self.dim, "dim must be equal to self.dim"

        q = self.Wq(x)

        k = self.Wk(x)

        v = self.Wv(x)

        xq = q.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        xk = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        queries = self.rope(xq, start_pos)
        xk = self.rope(xk, start_pos)

        '''self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # Update key and value caches
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # Retrieve key and value caches
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
'''
        # Repeat the heads of K and V to match the number of heads in Q
        keys = self.repeat_heads(xk, self.n_rep)
        values = self.repeat_heads(xv, self.n_rep)

        output = xops.memory_efficient_attention(
            queries, keys, values,
            attn_bias=xops.LowerTriangularMask()
        )

        output = output.reshape(batch_size, seq_len, -1)

        return output

class AccurateGELUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))

class FeedForward(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.GELU = AccurateGELUActivation()
        self.fc1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.GELU(self.fc1(x))

        x = self.GELU(self.fc2(x))

        return x

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.norm1 = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        h = x + self.attention(self.norm1(x), start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelConfig) -> None:
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be specified"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, args.norm_eps)

        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:

        assert x.shape[1] == 1, "seq_len must be 1"

        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, start_pos)

        x = self.norm(x)

        x = self.output(x)

        return x

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, config: ModelConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, device: str):

        prev_time = time.time()

        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint from {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cuda")
            print("Loaded Checkpoint in {:.2f}s".format(time.time() - prev_time))
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "config.json", "r") as f:
            config = json.load(f)

        model_config = ModelConfig(device=device, **config)

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)

        model_config.vocab_size = tokenizer.vocab_size()

        model = Transformer(model_config).to(device)

        if load_model:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded Model in {:.2f}s".format(time.time() - prev_time))

        return LLaMA(model, tokenizer, model_config)

    @staticmethod
    def _sample_top_p(probs, top_p, eos_id, device):
        # Implement top-p sampling here
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        total_probs = torch.sum(probs, dim=-1, keepdim=True)
        probs = probs / total_probs  # re-normalize remaining probs
        next_token = torch.multinomial(probs, 1).squeeze(1)
        return next_token

    def generate(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9,
                max_tokens: Optional[int] = None) -> Tuple[List[List[int]], List[str]]:
        if max_tokens is None:
            max_tokens = self.config.max_seq_len - 1
  
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]

        batch_size = len(prompt_tokens)
        assert batch_size <= self.config.max_batch_size, f"Batch size {batch_size} exceeds max batch size {self.config.max_batch_size}"

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.config.max_seq_len, f"Prompt length {max_prompt_len} exceeds max sequence length {self.config.max_seq_len}"

        total_len = min(self.config.max_seq_len, max_prompt_len + max_tokens)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.config.device)
        

        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.config.device)

        eos_reached = torch.tensor([False] * batch_size, device=self.config.device)
        prompt_tokens_mask = tokens != pad_id

        for cur_pos in tqdm(range(1, total_len), desc="Generating Tokens.."):
            with torch.no_grad():
                cur_pos = torch.tensor(cur_pos, device = self.config.device)
                self.model = self.model.to(self.config.device)
                logits = self.model(tokens[:, cur_pos - 1:cur_pos], cur_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p, self.tokenizer.eos_id(), self.config.device)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

            if all(eos_reached):
                break

        out_tokens = []
        out_text = []

        for current_prompt_token in tokens.tolist():
            if self.tokenizer.eos_id() in current_prompt_token:
                eos_idx = current_prompt_token.index(self.tokenizer.eos_id())
                current_prompt_token = current_prompt_token[:eos_idx]
            out_tokens.append(current_prompt_token)
            out_text.append(self.tokenizer.decode(current_prompt_token))

        return out_tokens, out_text