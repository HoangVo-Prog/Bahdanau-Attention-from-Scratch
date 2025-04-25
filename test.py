# import torch
# from model import *


# for batch in train_data_loader:
#     first_batch = batch
#     break  # Just to check the first batch

# encoder = Encoder(
#     input_dim=INPUT_DIM,
#     hidden_dim=HIDDEN_DIM,
#     output_dim=OUTPUT_DIM,
#     num_layers=NUM_LAYERS,
#     dropout=DROPOUT,
#     bidirectional=BIDIRECTIONAL_ENCODER
# )
# decoder = Decoder(
#     input_dim=INPUT_DIM,
#     hidden_dim=HIDDEN_DIM,
#     output_dim=OUTPUT_DIM,
#     num_layers=NUM_LAYERS,
#     dropout=DROPOUT,
#     bidirectional=BIDIRECTIONAL_DECODER
# )
# seq2seq = Seq2Seq(encoder, decoder)
# input_tensor = first_batch['src_ids']
# target_tensor = first_batch['trg_ids']
# outputs = seq2seq(input_tensor, target_tensor, TEACHER_FORCING_RATIO)

