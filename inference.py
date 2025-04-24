from config import *
from model import *
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for a Seq2Seq model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--input_text", type=str, help="Input text to translate", default="")
    parser.add_argument("--index", type=int, help="index to generate sample input from train data", default=0)
    
    return parser.parse_args()


def translate_sentence(sentence, model, en_tokenizer, vi_tokenizer, vi_id_to_token, device, max_output_length=MAX_LENGTH):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        # Tokenize the input sentence (convert to token IDs)
        encoding = en_tokenizer.encode(sentence)
        
        # Add <SOS> and <EOS> tokens to the input
        src_ids = [en_tokenizer.token_to_id(sos_token)] + encoding.ids + [en_tokenizer.token_to_id(eos_token)]
        
        # Convert the source IDs to a tensor with shape [1, seq_len]
        tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)  # [1, seq_len]

        # Pass the tensor through the encoder
        encoder_outputs, hidden = model.encoder(tensor)  # hidden, cell from LSTM encoder

        # Initialize input for the decoder with <SOS> token
        input_token = torch.LongTensor([vi_tokenizer.token_to_id(sos_token)]).to(device)  # [1]
        
        outputs = [input_token.item()]  # Store the generated tokens
        for _ in range(max_output_length):
            
            # Pass the current input token along with hidden and cell states to the decoder
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)

            # Get the predicted token (index of the highest probability)
            predicted_token = output.argmax(-1).item()
            # print(f"Predicted Token ID: {predicted_token}")
            # Append the predicted token to the output sequence
            outputs.append(predicted_token)

            # Update the input token for the next step (teacher-forcing is not used here)
            input_token = torch.LongTensor([predicted_token]).to(device)

            # If the decoder outputs <EOS>, stop the generation
            if predicted_token == vi_tokenizer.token_to_id(eos_token):
                break
        
        # Convert the predicted token IDs back to words
        translated_tokens = [vi_id_to_token[idx] for idx in outputs]
        
    return translated_tokens


def en_vi(i):
    return train_data_loader.dataset[i]["en"], train_data_loader.dataset[i]["vi"]

def id_to_token(tokenizer):
    vocab = tokenizer.get_vocab()
    return {v: k for k, v in vocab.items()}

def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    input_text = args.input_text
    index = args.index
    
    # Load the model
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
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set the model to evaluation mode
    
    en_id_to_token = id_to_token(en_tokenizer)
    vi_id_to_token = id_to_token(vi_tokenizer)
    
    if input_text:
        sentence = input_text
        expected_translation = en_tokenizer.decode(en_tokenizer.encode(sentence).ids)
    else:
        # Use the index to get a sample from the training data
        src_ids = train_data_loader.dataset[index]['src_ids']
        sentence ,expected_translation = en_vi(src_ids)
        
    print("Source (English):", sentence)
    print("Expected Translation (Vietnamese):", expected_translation)

    # Run the translation function
    translation = translate_sentence(sentence, model, en_tokenizer, vi_tokenizer, vi_id_to_token, DEVICE)
    print("Model Translation:", translation)    
    
    
if __name__ == "__main__":
    main()