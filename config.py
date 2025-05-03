sos_token, eos_token, pad_token, unk_token, special_tokens = "<sos>", "<eos>", "<pad>", "<unk>", ["<sos>", "<eos>", "<pad>", "<unk>"]

MAX_LENGTH=50
BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=10
HIDDEN_DIM=1000
NUM_LAYERS=1
BIDIRECTIONAL_ENCODER=True
BIDIRECTIONAL_DECODER=False
DROPOUT=0.2
EMBEDDING_DIM=620
USE_CUDA=True
DEVICE='cuda' if USE_CUDA else 'cpu'
SEED=42
