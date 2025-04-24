import torch
import test_model
from model import *
from config import *
from Data.data import cache_or_process

for batch in train_data_loader:
    first_batch = batch
    break  # Just to check the first batch


def test_encoder_output_shapes():
    global first_batch
    
    encoder = Encoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_ENCODER
    )
    input_tensor = first_batch['src_ids']
    output, hidden = encoder(input_tensor)
    assert output.shape == (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)
    assert hidden.shape == (BATCH_SIZE, HIDDEN_DIM)


def test_attention_output_shapes():
    attention = BahdanauAttention(hidden_dim=HIDDEN_DIM)
    query = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    keys = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)
    context, weights = attention(query, keys)
    assert context.shape == (BATCH_SIZE, 1, HIDDEN_DIM)
    assert weights.shape == (BATCH_SIZE, MAX_LENGTH)


def test_decoder_output_shapes():
    global first_batch
    
    decoder = Decoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_DECODER
    )
    input_tensor = first_batch['trg_ids'][:, 0].unsqueeze(1)  
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)
    output, hidden_out, attn = decoder(input_tensor, encoder_outputs, hidden)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    assert hidden_out.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn.shape == (BATCH_SIZE, MAX_LENGTH)
   
    
def test_seq2seq_output_shapes():
    encoder = Encoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_ENCODER
    )
    decoder = Decoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_DECODER
    )
    seq2seq = test_model.Seq2Seq(encoder, decoder)
    input_tensor = first_batch['src_ids']
    target_tensor = first_batch['trg_ids']
    outputs = seq2seq(input_tensor, target_tensor, TEACHER_FORCING_RATIO)
    assert outputs.shape == (BATCH_SIZE, MAX_LENGTH, OUTPUT_DIM)
    
    
def test_data_loader_shapes():
    global train_data_loader
    for batch in train_data_loader:
        src_ids = batch['src_ids']
        trg_ids = batch['trg_ids']
        break
    assert src_ids.shape == (BATCH_SIZE, MAX_LENGTH)
    assert trg_ids.shape == (BATCH_SIZE, MAX_LENGTH)    
    