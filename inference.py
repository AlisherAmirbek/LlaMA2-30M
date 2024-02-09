from model.model import ModelConfig, Transformer, LLaMA
from sentencepiece import SentencePieceProcessor
import json
import os
import argparse
import requests
from huggingface_hub import hf_hub_url
import torch

REPO_NAME = "AlisherAmirbek/LlaMA2-30M"
FILENAME = "llama2_30M.pth"

model_url = hf_hub_url(REPO_NAME, FILENAME)
state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True)

parser = argparse.ArgumentParser(description='Generate text based on prompts.')
parser.add_argument('prompts', type=str, nargs='+', help='Prompts for text generation')
parser.add_argument('--max_tokens', type=int, default=64, help='Maximum number of tokens to generate.')
args = parser.parse_args()

max_tokens = args.max_tokens if args.max_tokens else  64
prompts = args.prompts if args.prompts else ["Hello! Tell me about yourself."]   

tokenizer_path = "data/tokenizer/my_model50000.model"

ModelConfig = ModelConfig()
tranformer = Transformer(ModelConfig)
tokenizer = SentencePieceProcessor(tokenizer_path)
model = LLaMA(tranformer, tokenizer, ModelConfig)

model.build(tokenizer_path, device = 'cuda', state_dict = state_dict)

generated_tokens, generated_text = model.generate(prompts=prompts, max_tokens = max_tokens)
print(generated_text)