import numpy as np
import gensim.downloader as api

info = api.info()  # show info about available models/datasets
model = api.load("glove-wiki-gigaword-100")  # download the model and return as object ready for use
print(model.most_similar("cat"))

