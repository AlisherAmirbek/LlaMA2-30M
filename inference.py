from model.model import ModelConfig, Transformer, LLaMA
from sentencepiece import SentencePieceProcessor
import json
import os
import argparse


parser = argparse.ArgumentParser(description='Generate text based on prompts.')
parser.add_argument('prompts', type=str, nargs='+', help='Prompts for text generation')
parser.add_argument('--max_tokens', type=int, default=64, help='Maximum number of tokens to generate.')
args = parser.parse_args()


max_tokens = args.max_tokens if args.max_tokens else 64
prompts = args.prompts if args.prompts else ["Hello! Tell me about yourself."] 


base_dir = os.path.expanduser('~/LlaMA2-30M')
config_path = os.path.join(base_dir, 'model', 'config.json')

with open(config_path, 'r') as file:
    config = json.load(file)

checkpoint_path = "model/"
tokenizer_path = "data/tokenizer/my_model50000.model"

ModelConfig = ModelConfig()
tranformer = Transformer(ModelConfig)
tokenizer = SentencePieceProcessor(tokenizer_path)
model = LLaMA(tranformer, tokenizer, ModelConfig)

model.build(checkpoint_path, tokenizer_path, load_model = True, device = 'cuda')

generated_tokens, generated_text = model.generate(prompts=prompts, max_tokens = max_tokens)
print(generated_text)
