import math
import spacy
import torch
from model_e2bgd2g import *
from train import evaluate
from PreProcess import *


def load_best_model (device, model_path = './model.pth', store_dict = "./data"):
    article = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True, tokenizer_language='en_core_web_trf' ,include_lengths=True)
    summary = Field(tokenize='spacy',init_token='<sos>',eos_token='<eos>',lower=True, tokenizer_language='en_core_web_trf')

    train_data, valid_data, test_data = TabularDataset.splits(path=store_dict, train='train.csv', validation='val.csv', test='test.csv', format='csv', fields=[("text",article),('headline',summary)])

    article.build_vocab(train_data, min_freq=2)
    summary.build_vocab(train_data, min_freq=2)

    _, _, test_loader = BucketIterator.splits((train_data, valid_data, test_data), batch_size=64, sort_within_batch=True, sort_key = lambda x:len(x.text), device=device)

    attention_layer = Attention(enc_hid_dim = 512, dec_hid_dim = 512)
    encode_layer = Encoder(vocab=len(article.vocab),embeding_dim=256, encoder_hidden_dim=512, decoder_hidden_dim=512, dropout=0.5)
    decode_layer = Decoder(output_dim=len(summary.vocab),emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5, attention=attention_layer)
    model = Seq2Seq(encode_layer,decode_layer, article.vocab.stoi[article.pad_token], device).to(device)

    sum_pad_ids = summary.vocab.stoi[summary.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = sum_pad_ids)

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss = evaluate(model, test_loader, criterion)
    print("Best Model Info:")
    print(f'Test Loss: {test_loss:.3f} / Test PPL: {math.exp(test_loss):7.3f}')

    return article,summary,model


def predict(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
    if sentence is str:
        nlp = spacy.load('en_core_web_trf')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len.cpu())
    mask = model.create_mask(src_tensor)        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        attentions[i] = attention            
        pred_token = output.argmax(1).item()        
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

# 各项参数设置=======================================================
model_path = './model_e2bgd2g.pth'
# =================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Evaluating using Device: ", device)

article,summary,model = load_best_model(device,model_path = model_path ,store_dict="./data")

parser = spacy.load('en_core_web_trf')

def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lower_ for tok in tokens]
    return tokens

import pandas as pd
import itertools
from tqdm import tqdm
toval = pd.read_csv("data/val.csv")['original'].to_list()
trueans = pd.read_csv("data/val.csv")[' sum'].to_list()

f = open("EvalResult/model_e2bgd2g.txt",'w',encoding='utf-8')
# ff = open("EvalResult/originalval.txt",'w',encoding='utf-8')
for num in tqdm(range(0,len(toval))):
    i = toval[num]
#     ff.write(trueans[num]+"\n")
    text_tokenized = spacy_tokenizer(i)
    prediction, _ = predict(text_tokenized, article, summary, model, device)
    curedprediction = [k for k, g in itertools.groupby(prediction)]
    towritestr = ""
    for i in curedprediction:
        if i!="<eos>": towritestr+=(i+" ")
    f.write(towritestr+"\n")
f.close()
# ff.close()