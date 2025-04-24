from training import train_and_evaluate
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from model import *
from Data.data import cache_or_process
from config import *

src_pad_index = en_tokenizer.token_to_id(pad_token)
trg_pad_index = vi_tokenizer.token_to_id(pad_token)

encoder = Encoder(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL
)

decoder = Decoder(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL
)

model = Seq2Seq(encoder, decoder, device=DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_index)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

checkpoint_path = ""

if checkpoint_path: 
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    train_losses = [checkpoint['train_loss']]
    val_losses = [checkpoint['val_loss']]
    bleu_scores = [checkpoint['bleu_score']]
    best_valid_loss = checkpoint['val_loss']  # assuming best = last val_loss
    
    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 0
    train_losses = []
    val_losses = []
    bleu_scores = []
    best_valid_loss = float("inf")



# Train the model
best_model = train_and_evaluate(
    model,
    train_data_loader,
    valid_data_loader,
    optimizer,
    criterion,
    scheduler,
    n_epochs=20,
    teacher_forcing_ratio=0.5,
    device='cuda',
    start_epoch=start_epoch,
    train_losses=train_losses,
    val_losses=val_losses,
    bleu_scores=bleu_scores,
    best_valid_loss=best_valid_loss
)