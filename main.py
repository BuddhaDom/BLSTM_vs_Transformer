# %% [markdown] INTRO
# # BLSTM vs Transformers
# This script builds and runs two classification models. One mdoel is built with
# BLSTM layers while the other uses Transformer layers. The script also compares
# the two models in both performance and time. 

# %% LIBRARIES AND RESOURCES
from os import name
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import sys, re, time
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
import seaborn as sb
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_math_ops import Mod
from tensorflow.python.ops.math_ops import argmax

# NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print('Libraries Loaded')

# %% [markdown]
# # Utilities and Functions
# This long block are functions, utilities and variables that help
# throughout the implementation.

# %% 
# GPU MEMORY GROWTH
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", len(physical_devices))
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU memory growth activated.")
except:
    print("GPU device could not be found. Memory growth will not be available.")

# %% PREPROCESSING AND UTILITY FUNCTIONS

# Due to the low incidence of "love" and "surprise" on the EDNLP dataset, 
# they are being coupled together under the same category for this implementation.
emotionOther = ['love', 'surprise']

# Variables for cleaner() method
stemmer = PorterStemmer() # Stemmer
lem = WordNetLemmatizer()  # Lematizer
prelimStopwords = []
stopWordList = set(stopwords.words('english')) # List of Stopwords
minWordLength = 3  # Minimum word length
maxWordCount = 30  # Maximum words per post


# Simple Report of selected Dataframe
def dataFrameStatus(df: pd.DataFrame):
    print(f'Dataframe shape: {df.shape}')
    for name, _ in df.iteritems():
        print(f'Number of {name}: {df[name].nunique()}')
    print(df.sample(5),'\n')


# Remove invalid posts
def removeInvalidPosts(df: pd.DataFrame, invalidTextList: list) -> pd.DataFrame:
    for invalidText in invalidTextList:
        for name, _ in df.iteritems():
            df = df[df[name] != invalidText]
    df = df.reset_index(drop=True)
    return df

# Partial preprocessing of text.
def rawCleaner(post: str):
    post = re.sub(r'http\S+', '', post)  # Remove URLs
    post = re.sub(r'[^a-zA-Z\s]', '', post)  # Remove Non-letters
    post = post.lower()  # All to Lowercase
    post = word_tokenize(post)  # Tokenize text
    post = post[:maxWordCount]  # Limit word count (Post-Token Method)
    post = ' '.join(post)  # Rejoin token into a single string
    sys.stdout.write(f'\r{post[:30]}...')
    return post

# Preprocessing for NLP purposes
def nlpTreatment(entry: str, lemmatizer_over_stemmer: bool = True):
    entry = word_tokenize(entry)  # Tokenize text
    entry = [word for word in entry if word not in stopWordList] # All Stopwords
    if (lemmatizer_over_stemmer):
        entry = [lem.lemmatize(word=word, pos='v') for word in entry]  # Lematize
    else:
        entry = [stemmer.stem(entry) for word in entry] # Stem
    entry = [word for word in entry if len(word) >= minWordLength]  # No short words
    entry = ' '.join(entry)
    sys.stdout.write(f'\r{entry[:30]}...')
    return entry

# Run the RawCleaner on the selected Dataframe
def cleanDF(df: pd.DataFrame, rowInvalidationText: list) -> pd.DataFrame:
    tick = time.time()
    df = removeInvalidPosts(df, rowInvalidationText)
    for name, _ in df.iteritems():
        sys.stdout.write(f'Cleaning raw column "{name}"...\n')
        start = time.time()
        df[name] = df[name].apply(rawCleaner)
        end = time.time()
        sys.stdout.write(
            '\r\rColumn "{}" cleaned ({:.2f}s) ✓\n'.format(name, end - start))
        sys.stdout.flush()
    # removeInvalidPosts() is executed twice because it is possible for the clean()
    # function to return a now invalid value, which would invalidate the row.
    df = removeInvalidPosts(df, rowInvalidationText)
    tock = time.time()
    print('Finished preprocessing ({:.2f}s Total)\n'.format(tock-tick))
    return df

# Splitting function for EDNLP specifically.
def ednlpSplitter(df: pd.DataFrame, headers: list) -> set:
    global e_index
    sys.stdout.write('Cleaning for NLP...\n')
    tick=time.time()
    X = df[headers[0]].apply(nlpTreatment)
    tock = time.time()
    y = df[headers[1]]
    for drop in emotionOther:
        y = y.replace(drop, 'other')
    y, e_index = pd.factorize(y, sort=True)
    sys.stdout.write('\r\rDone ✓. ({:.2f}s)\n'.format(tock - tick))
    sys.stdout.flush()
    return X, y

# Counts the ocurrences of each word in an list of texts.
def counter_word(texts) -> int:
    count = Counter()
    for text in texts.values:
        for word in text.split():
            count[word] += 1
    return count

# Inverse translation from a Padded text sequence back to text (Which ahs been Lematized or Stemmed)
def decodeWordIndex(text, wordIndex):
    reverse_word_index = dict([(value, key) for (key, value) in wordIndex.items()])
    return ' '.join([reverse_word_index.get(i, "?") for i in text])

# Plotting of confussion matrixes or heatmaps.
def plot_confussion_matrix(data, labels, name='output', title='Confussion Matrix', annot=True, fmt='.2f', ylabel='True', xlabel='Predicted', vmin=None, vmax=None):
    sb.set(color_codes=True)
    plt.figure(1, figsize=(6, 5))
    plt.title = (f'{name} - {title}')
    
    ax = sb.heatmap(data, annot=annot, cmap='BuPu', linecolor='black', fmt=fmt, square=True, vmin=vmin, vmax=vmax)
    
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(f'{name} {title}')

    ax.set(ylabel=ylabel, xlabel=xlabel)

    plt.savefig(f'images/{name} {title}.png', dpi=300)
    plt.show()
    plt.close()

#Plotting of Countbar graphs.
def plot_countbars(data, name):
    sb.set(color_codes=True)
    plt.figure(figsize=(4,4))
    plt.title = (f'{name} Emotion Distribution'.capitalize())

    ax = sb.countplot(data=data, x=f'{name}_emotion', palette='YlGnBu')
    ax.set_xticklabels(e_index)
    ax.set_title(f'{name} Emotion Distribution'.capitalize())
    ax.set(ylabel='Count', xlabel='Emotion')
    
    plt.gcf().subplots_adjust(left=0.25)

    plt.savefig(f'images/{name}_emotion_distrubtion.png', dpi=300)
    plt.show()
    plt.close()

# Choose the highest values from a set of predicted features.
def predictAndChoose(model:Model, data):
    data = model.predict(data, verbose=1)
    data = data.argmax(axis=1)
    return data

# Return the sequence and padded sequence of a set of features.
def sequencerPadder(data, tokenizer):
    sequence = tokenizer.texts_to_sequences(data) # Features as Sequences
    paddedSequence = pad_sequences(sequence, maxlen=maxWordCount, padding='post', truncating='post') #Features as Padded Sequences
    return sequence, paddedSequence

def predictText(model:Model, text:str, tokenizer:Tokenizer):
    sequence = tokenizer.texts_to_sequences([text]) # Features as Sequences
    paddedSequence = pad_sequences(sequence, maxlen=maxWordCount, padding='post', truncating='post') #Features as Padded Sequences
    pred = model.predict(paddedSequence)
    pred = pred.argmax(axis=1)
    return f"Model: {model.name}\nText: {text}\nPrediction: {e_index[pred][0]}"
    

print("Utility and Preprocessing Functions loaded.")

# %% [markdown] 
# # EDNLP Dataset 
# Find and load the training, test and validation sets of the EDNLP Dataset. 
# Also store them in a dictionary for quote-unquote EASY referencing later.

# %% EDNLP DATASET BUILDING

# Loading EDNLP Data
ec_colnames = ['text', 'emotion']
try:
    ec_train = pd.read_csv('dataset/EDNLP/train.csv', names=ec_colnames, sep=";")
    ec_test = pd.read_csv('dataset/EDNLP/test.csv', names=ec_colnames, sep=";")
    ec_val = pd.read_csv('dataset/EDNLP/val.csv', names=ec_colnames, sep=";")
    print('EDNLP Datasets loaded.')


    ec_train_X, ec_train_y = ednlpSplitter(ec_train, ec_colnames)
    ec_test_X, ec_test_y = ednlpSplitter(ec_test, ec_colnames)
    ec_val_X, ec_val_y = ednlpSplitter(ec_val, ec_colnames)


    # Dictionary that will contain all the split sets of EDNLP
    ednlp = {
        'tr': {'X': ec_train_X, 'y': ec_train_y}, # Training Sets
        'te': {'X': ec_test_X, 'y': ec_test_y}, # Testing Sets
        'va': {'X': ec_val_X, 'y': ec_val_y} #Validation Sets
    }
    print()
    for key in ednlp:
        print('Shape of',key,'\b features:',ednlp[key]['X'].shape)
except:
    print('EDNLP Dataset(s) not found. Make sure test, val and train CSVs are available at at /dataset/EDNLP')

# %% WORD INDEXING

wordCounter = counter_word(ednlp['tr']['X'])
numWords = len(wordCounter)

tokenizer = Tokenizer(num_words=numWords)
tokenizer.fit_on_texts(ednlp['tr']['X'])
word_index = tokenizer.word_index

for key in ednlp:
    ednlp[key]['Xs'], ednlp[key]['Xp'] = sequencerPadder(ednlp[key]['X'], tokenizer)

print('Sequences and Padded Sequences generated.')

# %% [markdown] 
# # BLSTM Classification Model
# Attempt to load already-existing model.
# Otherwise, build and train the model. 

# %% EDNLP_BLSTM MODEL

# Attempt to load already-existing model.
# Otherwise, build and train the model.
try:
    BLSTM_model : Model = keras.models.load_model('models/EDNLP_BLSTM')
    print('EDNLP_BLSTM Model loaded.\n')
except:
    print('EDNLP_BLSTM Model not found at models/EDNLP_BLSTM. Building model.')
    # Model Structure
    BLSTM_model : Model = Sequential(name='EDNLP_BLSTM')

    scaleFactor = 8
    embedOutput = 32

    # Number of LSTM units necessary.
    lstmUnits = int(round(
        len(ednlp['tr']['Xp'])
        /
        (scaleFactor * (len(e_index) + embedOutput))
    ))

    print('Target Outputs for unidirectional LSTM Layer:',lstmUnits)

    BLSTM_model.add(Embedding(numWords, embedOutput, input_length=maxWordCount, name='embedding'))
    BLSTM_model.add(Bidirectional(LSTM(lstmUnits, dropout=0.1), name='blstm'))
    BLSTM_model.add(Dense(len(e_index), activation='sigmoid', name='dense'))

    BLSTM_model.get_layer('embedding')

    BLSTM_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = Adam(learning_rate=3e-4),
        metrics=['accuracy'])

    #Model Training
    history =  BLSTM_model.fit(
        ednlp['tr']['Xp'], ednlp['tr']['y'],
        epochs=21,
        use_multiprocessing=True,
        validation_data=(ednlp['va']['Xp'], ednlp['va']['y'])
    )
    # Save model
    print('Saving model...\n')
    BLSTM_model.save('models/EDNLP_BLSTM')

print(BLSTM_model.summary())
plot_model(BLSTM_model, to_file='images/BLSTM_model.png', show_shapes=True, show_layer_names=False)

# %% Transformer Classification Model

# Attempt to load already-existing model.
# Otherwise, build and train the model.
try:
    TRNS_model : Model = keras.models.load_model('models/EDNLP_TRNS')
    print('EDNLP_TRNS Model loaded.\n')
except:
    print('EDNLP_TRNS Model not found at models/EDNLP_TRNS. Building model.')
    
    # Transformer block layer class
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):
            super(TokenAndPositionEmbedding, self).__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

        def call(self, x):
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions

    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(maxWordCount,))
    embedding_layer = TokenAndPositionEmbedding(maxWordCount, numWords, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(len(e_index), activation="softmax")(x)

    TRNS_model : Model = keras.Model(inputs=inputs, outputs=outputs, name="ENDLP_TRNS")

    TRNS_model.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = Adam(learning_rate=3e-4),
        metrics=["accuracy"]
    )

    # Model Training
    history = TRNS_model.fit(
        ednlp['tr']['Xp'], ednlp['tr']['y'],
        batch_size=32,
        epochs=21,
        use_multiprocessing=True,
        validation_data=(ednlp['va']['Xp'], ednlp['va']['y'])
    )

    # Save model
    print('Saving model...\n')
    TRNS_model.save('models/EDNLP_TRNS')

print(TRNS_model.summary())
plot_model(TRNS_model, to_file='images/TRNS_model.png', show_shapes=True, show_layer_names=False)

# %% [markdown]
# # Testing the Models 
# Plots a Confussion Matrix and generates a Classification report for both models.

# %% TESTING BLSTM EDNLP

BLSTM_testPrediction = predictAndChoose(BLSTM_model, ednlp['te']['Xp'])

# Accuracy, Presicion, Recall, and F1-Score
blstm_report=classification_report(ednlp['te']['y'], BLSTM_testPrediction, target_names=e_index, output_dict=True)
blstm_report=pd.DataFrame(blstm_report).transpose()
blstm_report.to_csv('reports/blstm.csv', float_format='%.2f')
print(blstm_report)

# Confussion Matrix
blstm_cm=confusion_matrix(ednlp['te']['y'], BLSTM_testPrediction, normalize='pred')
plot_confussion_matrix(blstm_cm, e_index, name='EDNLP_BLSTM', fmt='.2f', vmin=0, vmax=1)

# %% TESTING TRNS EDNLP

TRNS_testPrediction = predictAndChoose(TRNS_model, ednlp['te']['Xp'])

# Accuracy, Presicion, Recall, and F1-Score
trns_report=classification_report(ednlp['te']['y'], TRNS_testPrediction, target_names=e_index, output_dict=True)
trns_report=pd.DataFrame(trns_report).transpose()
trns_report.to_csv('reports/trns.csv', float_format='%.2f')
print(trns_report)

# Confussion Matrix
trns_cm=confusion_matrix(ednlp['te']['y'], TRNS_testPrediction, normalize='pred')
plot_confussion_matrix(trns_cm, e_index, name='EDNLP_TRNS', fmt='.2f', vmin=0, vmax=1)

# %% [markdown]
# # Live Tests
# Test the models in real time.

# %% Live test: BLSTM
testText = input("Please input a string to predict:")
print(predictText(BLSTM_model, testText, tokenizer))

# %% Live test: TRANSFORMER
testText = input("Please input a string to predict:")
print(predictText(TRNS_model, testText, tokenizer))

# %% [markdown] 
# That was fun wasn't it?
# # :)