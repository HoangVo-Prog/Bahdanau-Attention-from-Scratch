import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction




def plot_metrics(train_losses, val_losses, bleu_scores):
    epochs = len(train_losses)

    # Plot Train Loss vs Validation Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss', color='blue')
    plt.plot(range(epochs), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot BLEU score over epochs
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), bleu_scores, label='Validation BLEU Score', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.title('Validation BLEU Score over Epochs')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


def compute_bleu(predictions, targets):
    """
    Compute the BLEU score for the predictions against the targets.
    
    Args:
    predictions (list of list of int): Generated sequence of token IDs.
    targets (list of list of int): Reference target sequence of token IDs.
    
    Returns:
    float: BLEU score
    """
    # Define a smoothing function
    smoothing_function = SmoothingFunction().method4  # You can choose the method you prefer (e.g., method1, method2, etc.)

    # BLEU expects lists of n-grams. Each target is a list of token IDs,
    # and the predictions should be a list of token IDs too.
    return corpus_bleu([[target] for target in targets], predictions, smoothing_function=smoothing_function)


def train_fn(model, train_loader, optimizer, criterion, clip, teacher_forcing_ratio=0.5, device='cuda'):
    model.train()
    epoch_train_loss = 0
    total_batches = len(train_loader)
    valid_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        try:
            source = batch['src_ids'].to(device)
            target = batch['trg_ids'].to(device)

            optimizer.zero_grad()
            
            # Forward pass with gradient scaling
            with torch.cuda.amp.autocast(enabled=True):
                output = model(source, target, teacher_forcing_ratio)
                
                if torch.isnan(output).any():
                    print(f"NaN detected in model output at batch {batch_idx}")
                    continue
                
                # Add small epsilon to prevent log(0)
                output = torch.clamp(output, min=1e-7, max=1-1e-7)
                
                # Calculate loss
                loss = criterion(output[1:].view(-1, output.size(-1)), target[1:].view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected in loss computation at batch {batch_idx}")
                continue

            # Gradient clipping before backward pass
            loss.backward()
            
            if clip:
                # Clip gradients and check for NaN gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                
                # Check for NaN gradients
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN/Inf gradient detected in {name}")
                            valid_gradients = False
                            break
                
                if not valid_gradients:
                    optimizer.zero_grad()
                    continue

            optimizer.step()
            
            epoch_train_loss += loss.item()
            valid_batches += 1

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    # Calculate average training loss only for valid batches
    avg_train_loss = epoch_train_loss / max(valid_batches, 1)  # Prevent division by zero
    print(f"Training Loss: {avg_train_loss:.4f} (Valid batches: {valid_batches}/{total_batches})")
    return avg_train_loss

def evaluate_fn(model, val_loader, criterion, device='cuda'):
    model.eval()
    epoch_val_loss = 0
    val_predictions = []
    val_targets = []
    valid_batches = 0
    total_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                source = batch['src_ids'].to(device)
                target = batch['trg_ids'].to(device)

                # Forward pass with gradient scaling
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(source, target, teacher_forcing_ratio=0)
                    
                    if torch.isnan(output).any():
                        print(f"NaN detected in model output at validation batch {batch_idx}")
                        continue
                    
                    # Add small epsilon to prevent log(0)
                    output = torch.clamp(output, min=1e-7, max=1-1e-7)
                    
                    # Calculate loss
                    loss = criterion(output[1:].view(-1, output.size(-1)), target[1:].view(-1))

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected in validation loss at batch {batch_idx}")
                    continue

                epoch_val_loss += loss.item()
                valid_batches += 1

                # Store predictions and targets for BLEU score calculation
                pred = output.argmax(dim=-1)
                val_predictions.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

            except RuntimeError as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue

    # Calculate average validation loss only for valid batches
    avg_val_loss = epoch_val_loss / max(valid_batches, 1)
    print(f"Validation Loss: {avg_val_loss:.4f} (Valid batches: {valid_batches}/{total_batches})")

    # Calculate BLEU score only if we have predictions
    if val_predictions and val_targets:
        val_bleu_score = compute_bleu(val_predictions, val_targets)
        print(f"Validation BLEU Score: {val_bleu_score:.4f}")
    else:
        print("Warning: No valid predictions for BLEU score calculation")
        val_bleu_score = 0.0

    return avg_val_loss, val_bleu_score


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, scheduler,
                       n_epochs, initial_teacher_forcing_ratio, final_teacher_forcing_ratio, device='cuda',
                       start_epoch=0, train_losses=None, val_losses=None, bleu_scores=None,
                       best_valid_loss=float("inf")):

    train_losses = train_losses or []
    val_losses = val_losses or []
    bleu_scores = bleu_scores or []

    for epoch in tqdm(range(start_epoch, n_epochs), desc="Training Epochs"):
        print(f"Epoch {epoch+1}/{n_epochs}")

        teacher_forcing_ratio = initial_teacher_forcing_ratio - (epoch / n_epochs) * (initial_teacher_forcing_ratio - final_teacher_forcing_ratio)
        
        # Train the model
        train_loss = train_fn(model, train_loader, optimizer, criterion,
                              clip=1.0, teacher_forcing_ratio=teacher_forcing_ratio, device=device)
        train_losses.append(train_loss)

        # Evaluate the model
        val_loss, val_bleu_score = evaluate_fn(model, val_loader, criterion, device=device)
        val_losses.append(val_loss)
        bleu_scores.append(val_bleu_score)

        # Save best model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'bleu_score': val_bleu_score
                        }, 'checkpoint.pth')


        # Print stats
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}")
        print(f"Epoch {epoch+1} | Valid Loss: {val_loss:.3f} | Valid PPL: {np.exp(val_loss):.3f}")
        print(f"Epoch {epoch+1} | Valid BLEU Score: {val_bleu_score:.3f}")

        scheduler.step(val_loss)

    plot_metrics(train_losses, val_losses, bleu_scores)

