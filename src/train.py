from tqdm import tqdm
import time
import torch
import math

def train_lm(self, model, dataloader, optimizer, grad_clip, log_inter, gumbel_softmax=False):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for idx, (src, trg, len, mask) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        loss = model.reconstruct_error(src, len, gumbel_softmax, mask)
        batch_loss = torch.mean(loss)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_loss += batch_loss.item()
        if idx % log_inter == 0 and idx > 0:
            print(f'| Batches: {idx}/{len(dataloader)} | Running Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
    elapsed = time.time() - start_time
    print(f'Epoch training time is: {elapsed}s.')
    return epoch_loss / len(dataloader)

def train_lm(self, model, dataloader, optimizer, grad_clip, log_inter, gumbel_softmax=False):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for idx, (src, trg, len, mask) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        loss = model.reconstruct_error(src, len, gumbel_softmax, mask)
        batch_loss = torch.mean(loss)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_loss += batch_loss.item()
        if idx % log_inter == 0 and idx > 0:
            print(f'| Batches: {idx}/{len(dataloader)} | Running Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
    elapsed = time.time() - start_time
    print(f'Epoch training time is: {elapsed}s.')
    return epoch_loss / len(dataloader)