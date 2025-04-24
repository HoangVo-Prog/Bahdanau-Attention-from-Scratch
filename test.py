import torch
from model import *


# encoder = Encoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL)
# input = torch.randint(0, INPUT_DIM, (BATCH_SIZE, MAX_LENGTH))
# output, hidden = encoder(input)
# print("Encoder output shape:", output.shape)  # Should be (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)
# print("Encoder hidden shape:", hidden.shape)  # Should be (BATCH_SIZE, HIDDEN_DIM)

# attention = BahdanauAttention(hidden_dim=HIDDEN_DIM)
# query = torch.randn(BATCH_SIZE, HIDDEN_DIM) # Hidden
# keys = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM) # Encoder output
# context, weights = attention(query, keys)
# print("Attention context shape:", context.shape)  # Should be (BATCH_SIZE, 1, HIDDEN_DIM)
# print("Attention weights shape:", weights.shape)  # Should be (BATCH_SIZE, MAX_LENGTH)

decoder = Decoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL)
input = torch.randint(0, INPUT_DIM, (BATCH_SIZE, 1))
hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)
encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM) # Encoder output
output, hidden, attn = decoder(input, encoder_outputs, hidden)
print("Decoder output shape:", output.shape)  # Should be (BATCH_SIZE, OUTPUT_DIM)
print("Decoder hidden shape:", hidden.shape)  # Should be (BATCH_SIZE, HIDDEN_DIM)
print("Decoder attention shape:", attn.shape)  # Should be (BATCH_SIZE, MAX_LENGTH)

# seq2seq = Seq2Seq(encoder, decoder, device=DEVICE)
# source = torch.randint(0, INPUT_DIM, (BATCH_SIZE, MAX_LENGTH))
# target = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE, MAX_LENGTH))
# outputs = seq2seq(input, target, 0.5)
# print("Output shape:", outputs.shape)  # Should be (BATCH_SIZE, MAX_LENGTH, OUTPUT_DIM)

# for batch in train_data_loader:
#     src_ids = batch['src_ids']
#     trg_ids = batch['trg_ids']
#     break  # Just to check the first batch

# print(trg_ids.shape)  # Should be (BATCH_SIZE, MAX_LENGTH)
# print(src_ids.shape)  # Should be (BATCH_SIZE, MAX_LENGTH)
