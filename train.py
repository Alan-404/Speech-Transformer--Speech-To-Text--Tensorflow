#%%
import pandas as pd
import numpy as np
import re
from preprocessing.text import TextProcessor
from preprocessing.audio import AudioProcessor
# %%
df = pd.read_csv('./wave-data/label/train.csv')
# %%
df.head(10)
# %%
y = np.array(df['label'].apply(lambda x: re.sub("_", " ", x)))
# %%
text_processor = TextProcessor("./tokenizer")
# %%
y_train = text_processor.process(y, max_length=40)
# %%
y_train.shape
# %%
audio_processor = AudioProcessor(sample_rate=22050, duration=10, mono=True, frame_size=705, hop_length=220)
# %%
x = np.array(df['file'])
# %%
x
#%%
x_train = []
# %%
for file in x:
    signal = audio_processor.process("./wave-data/dataset/" + file)
    x_train.append(signal)
# %%
import pickle
# %%
x_train = np.array(x_train)

# %%
x = np.expand_dims(x_train, axis=-1)
# %%
x.shape
# %%
with open('./clean/audio.pickle', 'wb') as handle:
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
y_train.shape
# %%
with open('./clean/text.pickle', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%

# %%
