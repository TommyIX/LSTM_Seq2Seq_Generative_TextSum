import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab, embeding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab, embeding_dim)
        self.rnn = nn.GRU(embeding_dim, encoder_hidden_dim, bidirectional = True)
        self.fc = nn.Linear(encoder_hidden_dim*2, decoder_hidden_dim)
        self.fc2 = nn.Linear(encoder_hidden_dim*2, embeding_dim)
        self.dropout = nn.Dropout(p=dropout)
   

    def forward(self, text, text_len):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)

        packed_outputs, hidden = self.rnn(packed_embedded) 
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # second layer
        outputs = self.fc2(outputs)
        outputs, hidden = self.rnn(outputs)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden

class Attention(nn.Module):
    '''
    Attention machanism to take encoder hidden states and current decoder state and generate context vector
    '''
    def __init__(self, enc_hid_dim, dec_hid_dim ):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
    

    def forward(self, hidden, encoder_outputs, mask):      
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.fc2 = nn.Linear(512, 1280)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_, hidden, encoder_outputs, mask):
    
        input_ = input_.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_))

        atten = self.attention(hidden, encoder_outputs, mask)        
        atten = atten.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(atten, encoder_outputs)
        
        weighted = weighted.permute(1, 0, 2)        
        rnn_input_ = torch.cat((embedded, weighted), dim = 2)
            
        output, hidden = self.rnn(rnn_input_, hidden.unsqueeze(0))

        # second layer
        output = self.fc2(output)
        output, hidden = self.rnn(output)

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        return prediction, hidden.squeeze(0), atten.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, text_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.text_pad_idx = text_pad_idx
        self.device = device
        
    def create_mask(self, text):
        mask = (text != self.text_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, text, text_len, headline, teacher_forcing_ratio = 0.5):
        batch_size = text.shape[1]
        headline_len = headline.shape[0]
        headline_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(headline_len, batch_size, headline_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(text, text_len)
        input_ = headline[0,:]
        mask = self.create_mask(text)
        for t in range(1, headline_len):
            
            output, hidden, _ = self.decoder(input_, hidden, encoder_outputs, mask)           
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio            
            top1 = output.argmax(1)             
            input_ = headline[t] if teacher_force else top1
            
        return outputs