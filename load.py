#%%
import pickle
import numpy as np
# %%
with open('./clean/audio.pickle', 'rb') as handle:
    x_train = pickle.load(handle)
# %%
with open('./clean/text.pickle', 'rb') as handle:
    y_train = pickle.load(handle)
# %%
x_train.shape
# %%
y_train.shape
# %%
from preprocessing.builder import DataBuilder
# %%
data_buider = DataBuilder()
# %%
dataset = data_buider.build_dataset(x_train, y_train, batch_size=64, buffer_size=64)
# %%
dataset
#%%
with open('./tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# %%
len(tokenizer.word_counts)
# %%
vocab_size = len(tokenizer.word_index) + 1
max_length = 40
# %%
from model.speech_transformer import SpeechTransformer
# %%
model = SpeechTransformer(vocab_size=vocab_size, max_length=max_length)
# %%
model.fit(dataset, epochs=10, saved_checkpoint_at=5)
# %%

# %%

# %%
