from model.model import ModelConfig, Transformer, LLaMA
from data.textdataset import TextDataset
from sentencepiece import SentencePieceProcessor
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 

tokenizer_path = "data/tokenizer/my_model50000.model"
config = ModelConfig()
tranformer = Transformer(config)
tokenizer = SentencePieceProcessor(tokenizer_path)
optimizer = torch.optim.Adam(tranformer.parameters(), lr=0.0001)

loaded_dataset = torch.load("data/dataset/tokenized_dataset_3_6.pt")

def get_batches(dataset, batch_size):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for batch in dataloader:
        x = batch[:, :-1]
        y = batch[:, 1:]

        yield x, y

split = 'train'
batch_size = 150
len_batches = len(list(get_batches(loaded_dataset, batch_size)))


# Set the model to training mode
tranformer.train()
last_saved_model_path = None
train_loss = []
train_batches = get_batches(loaded_dataset, batch_size)
for step, (x, y) in tqdm(enumerate(train_batches), desc="Training", total=len_batches):
    # Assuming each batch is a tensor of token IDs with shape [batch_size, seq_length]
    input_seq, target_seq = x.to(config.device), y.to(config.device)

    # Zero gradients before forward pass
    # Initialize loss
    losses = 0
    optimizer.zero_grad()
    for cur_pos in range(input_seq.size(1) - 1):
        # Forward pass for the current position
        logits = tranformer(input_seq[:, cur_pos:cur_pos + 1], cur_pos + 1)
        # Compute loss
        # logits are expected to be of size [batch_size, 1, vocab_size]
        # target_seq[:, cur_pos] is of size [batch_size]
        loss = F.cross_entropy(logits[:, -1, :], target_seq[:, cur_pos])
        losses += loss

    (losses/batch_size).backward()
    optimizer.step()

    train_loss.append(losses.item())
    if step % 100 == 0:
        avg_train_loss = np.mean(train_loss) # Average loss per step in the current epoch
        print(avg_train_loss)
        model_filename = f'mini_llama2_step_{step}.pth'
        model_path = f'/model/trained/mini_llama/{model_filename}'

        if last_saved_model_path is not None and os.path.exists(last_saved_model_path):
            os.remove(last_saved_model_path)
            print(f"Deleted old model at {last_saved_model_path}")

        torch.save({
            'step': step,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'train_loss_history': train_loss,
        }, model_path)
        print(f"Model saved to {model_path}")
        last_saved_model_path = model_path

    print(f"   Step {step}, Loss: {losses.item()}")

print("Training complete.")
