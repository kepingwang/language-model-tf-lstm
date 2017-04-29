import urllib.request
import re
import numpy as np

books = [
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/A+Child's+HIstory+of+England.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/A+Tale+of+Two+Cities.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/American+Notes+for+General+Circulation.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/Barnaby+Rudge.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/Bleak+House.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/David+Copperfield.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/Great+Expectations.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/Life+And+Adventures+Of+Martin+Chuzzlewit.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/Nicholas+Nickleby.txt",
  "https://s3.amazonaws.com/cmd-aws.kepingwang.bucket/public_data/novels/dickens/Oliver+Twist.txt"
]

def _get_book(data_url):
  with urllib.request.urlopen(data_url) as response:
    result = []
    for line in response:
      result.append(line)
    return result

def get_books_data():
  books_data = []
  for book_url in books:
    books_data = books_data + _get_book(book_url)
  print("total number of lines: {}".format(len(books_data)))
  return books_data

punctuation_set = [
  '`', ',', '.', '\"', '\'', '\n', 
  '[', ']', '(', ')', '{', '}', '<', '>',
  '-', '!', '?', ';', ':', '/', '|'
  ]


def clean_up_text(raw_text):
  # convert byte to str
  lines = [line.decode() for line in raw_text]
  # change line break style from \r\n to \n
  lines = [line.replace('\r\n', '\n') for line in lines]
  # remove line breaks inside sentences, and multiple newlines
  res = []
  prev_line = None
  for line in lines:
    if len(line) > 1:
      res.append(line.replace('\n', ' '))
    else:
      if prev_line != '\n':
        res.append(line)
    prev_line = line

  text = ''.join(res)
  # decapitalize all characters. (to reduce vocab size)
  text = text.lower()
  # replace all numbers with '#'
  text = re.sub('\d', '#', text)
  # use space as separator, so add spaces around punctuations
  for punctuation in punctuation_set:
    text = text.replace(punctuation, ' '+punctuation+' ')
  # clean up continuous spaces
  text = re.sub(' +', ' ', text)
  return text


class WordDict():
  """ A word dictionary mapping string tokens to integer ids.
  Id 0 is reserved for all unknown tokens, using key "_UNKNOWN".
  """

  def __init__(self, tokens):
    """ Takes an iterator/list of tokens and
    construct a dictionary of token key and integer values.
    """
    self.word_dict = {}
    self.word_list = []
    self.word_dict['_UNKNOWN'] = 0
    self.word_list.append('_UNKNOWN')
    for token in tokens:
      if token != '' and token not in self.word_dict:
        self.word_dict[token] = len(self.word_dict)
        self.word_list.append(token)

  def get_id(self, token):
    """ Get the integer id for the given token, return 0 if token not found. """
    if token in self.word_dict:
      return self.word_dict[token]
    else:
      return 0

  def get_token(self, id):
    """ Get the token for a given integer id """
    return self.word_list[id]

  def get_vocab_size(self):
    return len(self.word_dict)

  def ids_to_tokens(self, ids):
    return [self.get_token(id) for id in ids]

  def tokens_to_ids(self, tokens):
    return np.array([self.get_id(token) for token in tokens])