import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

def write_embed_meta(LOG_DIR, word_dict, embeddings, writer):
  # write vocab to metadata.tsv
  file_path = os.path.join(LOG_DIR, 'metadata.tsv')
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'w') as f:
    for i in range(word_dict.get_vocab_size()):
      if word_dict.get_token(i) == '\n':
        f.write('\\n' + '\n')
      else:
        f.write(word_dict.get_token(i) + '\n')

  config = projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embeddings.name
  embedding_config.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

  projector.visualize_embeddings(writer, config)

