#@title Load the Universal Sentence Encoder's TF Hub module
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile

#Universal Sentence Encoder Comparer


print(os.path.join(tempfile.gettempdir(), "tfhub_modules"))

# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #"https://tfhub.dev/google/universal-sentence-encoder-lite/2" #"https://tfhub.dev/google/universal-sentence-encoder/4" #"https://tfhub.dev/google/universal-sentence-encoder-large/5"
# model = hub.load(module_url)
# print ("module %s loaded" % module_url)
model = tf.saved_model.load("data/universal-sentence-encoder.4")
def embed(input):
  return model(input)

#@title Compute a representation for each message, showing various lengths supported.
word = "limit"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
print(paragraph)
messages = [sentence, paragraph]

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

message_embeddings = embed(messages)

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  print("Message: {}".format(messages[i]))
  print("Embedding size: {}".format(len(message_embedding)))
  message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:10]))
  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")
  plt.show()

def run_and_plot(messages_):
  message_embeddings_ = embed(messages_)
  plot_similarity(messages_, message_embeddings_, 90)


messages = [
  # Smartphones
  "J'aime ma télécommande",
  "La télécommande n'est pas bonne.",
  "La télécommande ne marche pas.",

  # Weather
  "Will it snow tomorrow?",
  "Recently a lot of hurricanes have hit the US",
  "Global warming is real",

  # Food and health
  "An apple a day, keeps the doctors away",
  "Eating strawberries is healthy",
  "Is paleo better than keto?",

  # Asking about age
  "I like my baby",
  "what is your age?",
]

run_and_plot(messages)
message_embeddings = embed(messages)
message_embeddings = message_embeddings.numpy()
print(np.inner(message_embeddings[0],message_embeddings[0]))
print(np.inner(message_embeddings[0],message_embeddings[1]))
print(np.inner(message_embeddings[1],message_embeddings[0]))
corr = np.inner(message_embeddings, message_embeddings)
print(corr)

