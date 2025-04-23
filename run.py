from training import train_and_evaluate
from huggingface_hub import HfApi
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from model import Seq2Seq, Encoder, Decoder

# Login to Hugging Face (do this once)
from huggingface_hub import login
login(input("Enter your Hugging Face token: "))  

checkpoint_path = ""



model = Seq2Seq(encoder, decoder).to(DEVICE)  # or however you instantiated it

optimizer = optim.Adam(seq2seq.parameters(), lr=0.001)
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

# Save the model locally first
save_path = "model_weights"
os.makedirs(save_path, exist_ok=True)
torch.save(best_model.state_dict(), f"{save_path}/model.pt")

# Push to Hugging Face Hub
repo_name = "your-username/your-model-name"  # replace with your desired repository name
api = HfApi()
api.upload_file(
    path_or_fileobj=f"{save_path}/model.pt",
    path_in_repo="model.pt",
    repo_id=repo_name,
    repo_type="model"
)

# You can also save the configuration and tokenizer if needed
# model.config.save_pretrained(repo_name)
# tokenizer.save_pretrained(repo_name)
