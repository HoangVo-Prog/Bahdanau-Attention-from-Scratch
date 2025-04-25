import torch
from torch import nn
import torch.nn.init as init
from config import *
from attention import BahdanauAttention
from Data.data import cache_or_process



torch.manual_seed(SEED)

train_data_loader, valid_data_loader, test_data_loader, en_tokenizer, vi_tokenizer = cache_or_process()

INPUT_DIM = en_tokenizer.get_vocab_size()
OUTPUT_DIM = vi_tokenizer.get_vocab_size()

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
       
        # x = x.to(DEVICE)
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        
        outputs = self.layer_norm(outputs)  
        
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        outputs = self.fc(outputs)
        hidden = self.fc_hidden(hidden)
        outputs = self.dropout(outputs)
        
        return outputs, hidden # (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM), (BATCH_SIZE, HIDDEN_DIM)

    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim*2, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        # GRU weight initialization
            for name, param in self.gru.named_parameters():
                if 'weight_ih' in name:  # input-to-hidden weights
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:  # hidden-to-hidden weights
                    init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif 'bias' in name:
                    init.constant_(param.data, 0)
            
            # Initialize Attention layers
            init.xavier_uniform_(self.attention.Wa.weight.data)
            init.xavier_uniform_(self.attention.Va.weight.data)
    
    def forward(self, input, encoder_outputs, hidden):
        # input.shape: (BATCH_SIZE, 1), encoder_outputs.shape: (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM), hidden.shape: (BATCH_SIZE, HIDDEN_DIM)
        # input = input.to(DEVICE)
        embedded = self.dropout(self.embedding(input))
        context, attn_weights = self.attention(hidden, encoder_outputs)
            
        rnn_input = torch.cat((embedded, context), dim=2)
                
        outputs, hidden = self.gru(rnn_input, (hidden.unsqueeze(0).repeat(self.gru.num_layers*(int(self.gru.bidirectional)+1), 1, 1)))
        
        outputs = self.layer_norm(outputs)
        
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
        hidden = self.fc_hidden(hidden).squeeze(0)  # (BATCH_SIZE, HIDDEN_DIM)
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)
        predictions = self.fc_out(outputs).squeeze(1)  # (BATCH_SIZE, OUTPUT_DIM)
        
        return predictions, hidden, attn_weights
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        # outputs = torch.zeros(batch_size, trg_len, OUTPUT_DIM).to(DEVICE)
        outputs = torch.zeros(batch_size, trg_len, OUTPUT_DIM)
        
        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]  # trg.shape: (BATCH_SIZE, MAX_LENGTH), input: (BATCH_SIZE, 1)
        
        for t in range(1, MAX_LENGTH):
            output, hidden, _ = self.decoder(
                input.unsqueeze(1), encoder_outputs, hidden
            )

            outputs[:, t, :] = output
            
            # teacher_force = torch.rand(1, device=DEVICE).item() < teacher_forcing_ratio
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs  
