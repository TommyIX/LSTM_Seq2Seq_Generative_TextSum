import os
import time
import math
import torch
from torch import optim
from pathlib import Path

from model_ebldl import *
from PreProcess import *

# 各项参数设置=======================================================
epoch_num = 40
clip_value = 1
batch_size = 64
# =================================================================


#function to train model
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        text, text_len = batch.text
        headline = batch.headline
        optimizer.zero_grad()
        output = model(text, text_len.cpu(), headline)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        headline = headline[1:].view(-1)
        loss = criterion(output, headline)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

#functio to retuen val loss
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            text, text_len = batch.text
            headline = batch.headline
            output = model(text, text_len.cpu(), headline, 0) #turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            headline = headline[1:].view(-1)
            loss = criterion(output, headline)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    articleField = Path("./DumpedField/article.Field")
    summaryField = Path("./DumpedField/summary.Field")
    if articleField.is_file() and summaryField.is_file():
        article, summary, train_loader, valid_loader, test_loader = buildloaders(store_dict = "./data", batch_size = batch_size, use_cache_fields = True)
    else:
        article, summary, train_loader, valid_loader, test_loader = buildloaders(store_dict = "./data", batch_size = batch_size, save = True, use_cache_fields = False)

    attention_layer = Attention(enc_hid_dim = 512, dec_hid_dim = 512)
    encode_layer = Encoder(vocab=len(article.vocab),embeding_dim=256, encoder_hidden_dim=512, decoder_hidden_dim=512, dropout=0.5)
    decode_layer = Decoder(output_dim=len(summary.vocab),emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5, attention=attention_layer)
    model = Seq2Seq(encode_layer,decode_layer, article.vocab.stoi[article.pad_token], device).to(device)
    
    print("=================== Model Structure ===================")
    print(model)
    print("=======================================================")
    print("Training Using Device: ", device)

    optimizer = optim.Adam(model.parameters())
    sum_pad_ids = summary.vocab.stoi[summary.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = sum_pad_ids)

    best_valid_loss = float('inf')
    for epoch in range(epoch_num):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip_value)
        valid_loss = evaluate(model, valid_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model_ebldl.pth')
        
        print(f'Epoch: {epoch+1:2} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


