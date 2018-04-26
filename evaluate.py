# -*- coding: utf-8 -*-
"""
Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

In this project we will be teaching a neural network to translate from
French to English.

::

    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .

... to varying degrees of success.

"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from seq2seq_translation_tutorial import *


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)



hidden_size = 256
print("initial model begin")
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = DecoderRNN(hidden_size, output_lang.n_words,1)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,1, dropout_p=0.1)
print("initial model finished")

print("use_cuda:"+str(use_cuda))
if use_cuda:
    print("encoder1 = encoder1.cuda()")
    print("encoder1 = encoder1.cuda()")
    encoder1 = encoder1.cuda()
    print("encoder1 move to cuda")
    attn_decoder1 = attn_decoder1.cuda()
    print("attn_decoder1 move to cuda")
#trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 100, print_every=100)
######################################################################
#
torch.save(encoder1, "./data/encode1_model")
torch.save(attn_decoder1, "./data/attn_decoder1_model")
#the_model = torch.load(PATH)


encoder1 = torch.load("./data/encode1_model")
attn_decoder1 = torch.load("./data/attn_decoder1_model")
if use_cuda:
    attn_decoder1 = attn_decoder1.cuda()
evaluateRandomly(input_lang, output_lang,pairs, encoder1, attn_decoder1)

encoder_outputs = encoderContext(input_lang, output_lang, encoder1, attn_decoder1,"je suis trop froid .")
#print(encoder_outputs)
######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

output_words = evaluate(
    input_lang, output_lang, encoder1, attn_decoder1, "je suis trop froid .")
#plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(input_lang,output_lang,
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


#evaluateAndShowAttention("elle a cinq ans de moins que moi .")

#evaluateAndShowAttention("elle est trop petit .")

#evaluateAndShowAttention("je ne crains pas de mourir .")

#evaluateAndShowAttention("c est un jeune directeur plein de talent .")


######################################################################
# Exercises
# =========
#
#
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#
