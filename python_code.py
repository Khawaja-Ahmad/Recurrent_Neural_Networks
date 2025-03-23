import os
import numpy as np
import re
import shutil
import tensorflow as tf

DATA_DIR = "./data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")


def clean_logs():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    shutil.rmtree(LOG_DIR, ignore_errors=True)


# Modified for word-level vocabulary:
# Instead of extending the text character by character, we split the text into words.
def download_and_read(urls):
    words = []  # Will store words instead of individual characters
    for i, url in enumerate(urls):
        p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url, cache_dir=".")
        text = open(p, mode="r", encoding="utf-8").read()
        # Remove byte order mark and newlines; replace extra spaces with a single space
        text = text.replace("\ufeff", "")
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', " ", text)
        # Split text into words based on whitespace and add to the list
        word_list = text.split()
        words.extend(word_list)
    return words


def split_train_labels(sequence):
    input_seq = sequence[:-1]
    output_seq = sequence[1:]
    return input_seq, output_seq


# We keep the same model architecture.
# Now it processes sequences of words instead of characters.
class CharGenModel(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dim, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn_layer = tf.keras.layers.GRU(
            num_timesteps,
            recurrent_initializer="glorot_uniform",
            recurrent_activation="sigmoid",
            stateful=True,
            return_sequences=True
        )
        self.dense_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer(x)
        return x


def loss(labels, predictions):
    return tf.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=True)


# Modified generate_text function for word-level generation.
def generate_text(model, prefix_string, word2idx, idx2word, num_words_to_generate=100, temperature=1.0):
    # Split the prefix into words instead of iterating character by character.
    input_words = prefix_string.split()
    input_ids = [word2idx[s] for s in input_words]
    input_ids = tf.expand_dims(input_ids, 0)
    words_generated = []
    model.reset_states()
    for i in range(num_words_to_generate):
        preds = model(input_ids)
        preds = tf.squeeze(preds, 0) / temperature
        # Predict the next word's id using categorical sampling
        pred_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()
        words_generated.append(idx2word[pred_id])
        # Update the input to include the predicted word
        input_ids = tf.expand_dims([pred_id], 0)

    # Return the generated text, joining words with spaces.
    return prefix_string + " " + " ".join(words_generated)


# Download and read into a local data structure (list of words)
words = download_and_read([
    "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
    "https://www.gutenberg.org/files/12/12-0.txt"
])
clean_logs()

# Create the vocabulary from unique words
vocab = sorted(set(words))
print("vocab size: {:d}".format(len(vocab)))

# Create mapping from words to ints and vice versa.
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# Convert words into integers
words_as_ints = np.array([word2idx[w] for w in words])
data = tf.data.Dataset.from_tensor_slices(words_as_ints)

# Define sequence length (now the number of words per input sequence)
seq_length = 100
sequences = data.batch(seq_length + 1, drop_remainder=True)
sequences = sequences.map(split_train_labels)

# Print an example input and output sequence (words joined by spaces)
for input_seq, output_seq in sequences.take(1):
    print("input: [{}]".format(" ".join([idx2word[i] for i in input_seq.numpy()])))
    print("output: [{}]".format(" ".join([idx2word[i] for i in output_seq.numpy()])))

# Set up for training
batch_size = 64
steps_per_epoch = len(words) // seq_length // batch_size
dataset = sequences.shuffle(10000).batch(batch_size, drop_remainder=True)
print(dataset)

# Define network using word-level vocabulary size
vocab_size = len(vocab)
embedding_dim = 256

model = CharGenModel(vocab_size, seq_length, embedding_dim)
model.build(input_shape=(batch_size, seq_length))
model.summary()

# Validate model dimensions with one batch from the dataset
for input_batch, label_batch in dataset.take(1):
    pred_batch = model(input_batch)

print(pred_batch.shape)
assert(pred_batch.shape[0] == batch_size)
assert(pred_batch.shape[1] == seq_length)
assert(pred_batch.shape[2] == vocab_size)

model.compile(optimizer=tf.optimizers.Adam(), loss=loss)

# Train the model for 50 epochs and generate text every 10 epochs
num_epochs = 50
for i in range(num_epochs // 10):
    model.fit(
        dataset.repeat(),
        epochs=10,
        steps_per_epoch=steps_per_epoch
    )
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "model_epoch_{:d}".format(i+1))
    model.save_weights(checkpoint_file)

    # Create a generative model using the trained model so far
    gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
    gen_model.load_weights(checkpoint_file)
    gen_model.build(input_shape=(1, seq_length))

    print("After epoch: {:d}".format((i+1)*10))
    print(generate_text(gen_model, "Alice", word2idx, idx2word))
    print("---")
