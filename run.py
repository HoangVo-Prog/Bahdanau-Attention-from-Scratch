from training import train_and_evaluate
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from model import *
from config import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model")
    parser.add_argument("--checkpoint", type=str, help="Path to training data", default="")
    parser.add_argument("--n_epochs", type=int, required=True, help="Path to number of epochs")
    
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    n_epochs = args.n_epochs
    
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
    print("Model initialized")
    
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    
    model = Seq2Seq(encoder, decoder, device=DEVICE)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_index)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

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

    print("Training model...")
    print("Start training on:", DEVICE)

    train_and_evaluate(
        model,
        train_data_loader,
        valid_data_loader,
        optimizer,
        criterion,
        scheduler,
        n_epochs=n_epochs,
        teacher_forcing_ratio=0.5,
        device=DEVICE,
        start_epoch=start_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        bleu_scores=bleu_scores,
        best_valid_loss=best_valid_loss
    )
    
    
if __name__ == "__main__":
    main()