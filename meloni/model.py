import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import defaultdict


class Embedding(nn.Module):
    """Feature extraction:
    Looks up embeddings for a source character and the language,
    concatenates the two,
    then passes through a linear layer
    """
    def __init__(self, embedding_dim, langs, C2I):
        super(Embedding, self).__init__()
        self.C2I = defaultdict(lambda: '<unk>', C2I)
        self.langs = langs + ['sep']
        self.L2I = {l: idx for idx, l in enumerate(self.langs)}
        self.char_embedding = nn.Embedding(len(C2I), embedding_dim)
        self.lang_embeddings = nn.Embedding(len(self.langs), embedding_dim)
        # map concatenated source and language embedding to 1 embedding
        self.fc = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, char_idx, lang):
        # TODO: to device
        char_encoded = self.char_embedding(torch.tensor([char_idx]))
        lang_encoded = self.lang_embeddings(torch.tensor([self.L2I[lang]]))
        # concatenate the tensors to form one long embedding then map down to regular embedding size
        return self.fc(torch.cat((char_encoded, lang_encoded), dim=1))


class MLP(nn.Module):
    """
    Multi-layer perceptron to generate logits from the decoder state
    """
    def __init__(self, hidden_dim, feedforward_dim, output_size):
        # TODO: what are these magic numbers?
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 2 * feedforward_dim)
        self.fc2 = nn.Linear(2 * feedforward_dim, feedforward_dim)
        self.fc3 = nn.Linear(feedforward_dim, output_size, bias=False)

    # no need to perform softmax because CrossEntropyLoss does the softmax for you
    def forward(self, decoder_state):
        h = f.relu(self.fc1(decoder_state))
        scores = self.fc3(f.relu(self.fc2(h)))
        return scores


class Attention(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        # TODO: batch_first?
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c_s = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(self, query, keys, encoded_input):
        # query: decoder state. [1, 1, H]
        # keys: encoder states. [1, L, H]

        # TODO: why do we need this?
        query = self.W_query(query)
        # dot product attention to calculate similarity between the query and each key
        # scores: [1, L, 1]
        scores = torch.matmul(keys, query.transpose(1, 2))
        # TODO: do the softmax on the correct dimension
        # softmax to get a probability distribution over encoder states
        weights = f.softmax(scores, dim=-2)

        # TODO: do attention analysis and highlight the attention vector

        # weights: L x 1
        # encoded_input: L x E
        # keys: L x D
        # result: L x D - weighted version of the input
        # TODO: this is wrong!! result should be 1 x D
        weighted_states = weights * (self.W_c_s(encoded_input) + self.W_key(keys))

        return weighted_states


class Model(nn.Module):
    """
    Encoder-decoder architecture
    """
    def __init__(self, C2I, I2C,
                 num_layers,
                 dropout,
                 feedforward_dim,
                 embedding_dim,
                 model_size,
                 model_type,
                 langs):
        super(Model, self).__init__()
        # TODO: modularize so we can get dialects for Austronesian, Chinese, or Romance
        # TODO: can we modularize this better?
        self.I2C = I2C
        self.C2I = C2I

        # TODO: since there are really only two embedding matrices, can just create 2 separate ones
        encoder = Embedding(embedding_dim, langs, C2I)
        # share embedding across all languages, including the proto-language
        self.encoders = [encoder for _ in range(len(langs))]
        self.langs = langs
        self.protolang = langs[0]
        self.L2I = {l: idx for idx, l in enumerate(langs + ['sep'])}
        # TODO: maybe just explicitly share the encoders?
        self.l2e = {lang: self.encoders[i] for i, lang in enumerate(langs)}
        # separators have their own embedding
        self.l2e["sep"] = Embedding(embedding_dim, langs, C2I)

        if model_type == "gru":
            # TODO: beware of batching
            self.encoder_rnn = nn.GRU(input_size=embedding_dim,
                                      hidden_size=model_size,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      batch_first=True)
            self.decoder_rnn = nn.GRU(input_size=embedding_dim + model_size,
                                      hidden_size=model_size,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      batch_first=True)
        else:
            self.encoder_rnn = nn.LSTM(input_size=embedding_dim,
                                       hidden_size=model_size,
                                       num_layers=num_layers,
                                       dropout=dropout,
                                       batch_first=True)
            self.decoder_rnn = nn.LSTM(input_size=embedding_dim + model_size,
                                       hidden_size=model_size,
                                       num_layers=num_layers,
                                       dropout=dropout,
                                       batch_first=True)

        # TODO: shouldn't we share this with the encoder?
        self.lang_embedding = nn.Embedding(len(langs), embedding_dim)
        self.mlp = MLP(hidden_dim=model_size, feedforward_dim=feedforward_dim, output_size=len(C2I))
        self.attention = Attention(hidden_dim=model_size, embedding_dim=embedding_dim)

    def forward(self, daughter_forms, protoform):
        # encoder
        # TODO: is the encoder treating each input as separate?
        # encoder_states: 1 x L x H, memory: 1 x 1 x H, where L = len(daughter_forms)
        (encoder_states, memory), embedded_cognateset = self.encode(daughter_forms)

        # decoder
        # start of protoform sequence
        # TODO: is this really necessary? we already have < and > serving as BOS/EOS
        start_encoded = self.l2e["sep"](self.C2I["<s>"], "sep")
        # initalize weighted states to the final encoder state
        # TODO: there has to be a better way of doing this indexing - preserve batch dim
        attention_weighted_states = memory.squeeze(dim=0)
        # start_encoded: 1 x E, attention_weighted_states: 1 x H
        # concatenated into 1 x (H + E)
        decoder_input = torch.cat((start_encoded, attention_weighted_states), dim=1).unsqueeze(dim=0)
        # TODO: the decoder
        decoder_state, _ = self.decoder_rnn(decoder_input)
        scores = []  # TODO: it's faster to initialize the shape then fill it in

        # TODO: could we even do this batched or pass in the whole target?? but then we don't control the attention
        for lang, char in protoform:
            # lang will either be sep or the protolang
            # embedding layer
            true_char_embedded = self.l2e[self.protolang](char, lang)
            # MLP to get a probability distribution over the possible output phonemes
            char_scores = self.mlp(decoder_state + attention_weighted_states)
            scores.append(char_scores)
            # dot product attention over the encoder states
            attention_weighted_states = self.attention(decoder_state, encoder_states, embedded_cognateset)
            decoder_input = torch.cat((true_char_embedded, attention_weighted_states), dim=1).unsqueeze(dim=0)
            # TODO: make sure that we're really taking the decoder state
            decoder_state, _ = self.decoder_rnn(decoder_input)

        scores = torch.vstack(scores)
        return scores

    def encode(self, daughter_forms):
        # daughter_forms: list of lang and indices in the vocab
        embedded_cognateset = []
        for lang, idx in daughter_forms:
            # use the embedding corresponding to the language
            # shared for all languages (including the protolanguage), but separators have their own
            embedded_cognateset.append(self.l2e[lang](idx, lang))

        embedded_cognateset = torch.vstack(embedded_cognateset) # .to(DEVICE)
        # [1, C, E], batch size of 1, C = len(daughter_forms)
        embedded_cognateset = embedded_cognateset.unsqueeze(dim=0)

        # TODO: to(device)?
        # TODO: note that the LSTM returns something diff than the GRU in pytorch
        return self.encoder_rnn(embedded_cognateset), embedded_cognateset

    def decode(self, encoder_states, memory, embedded_cognateset, max_length):
        # greedy decoding - generate protoform by picking most likely sequence at each time step
        start_encoded = self.l2e["sep"](self.C2I["<s>"], "sep")
        # initalize weighted states to the final encoder state
        # TODO: there has to be a better way of doing this indexing - preserve batch dim
        attention_weighted_states = memory.squeeze(dim=0)
        # start_encoded: 1 x E, attention_weighted_states: 1 x H
        # concatenated into 1 x (H + E)
        decoder_input = torch.cat((start_encoded, attention_weighted_states), dim=1).unsqueeze(dim=0)
        decoder_state = self.decoder_rnn(decoder_input)
        reconstruction = []

        i = 0
        while i < max_length:
            # embedding layer
            # MLP to get a probability distribution over the possible output phonemes
            char_scores = self.mlp(decoder_state + attention_weighted_states)
            # TODO: make sure it's along the correct dimension
            predicted_char = torch.argmax(char_scores)
            predicted_char_encoded = self.l2e[self.protolang](predicted_char, self.protolang)

            # dot product attention over the encoder states
            attention_weighted_states = self.attention(decoder_state, encoder_states, embedded_cognateset)
            decoder_state = self.decoder_rnn(torch.cat(predicted_char_encoded, attention_weighted_states))

            reconstruction.append(predicted_char)

            i += 1
            # end of sequence generated
            # TODO: declare EOS as a global variable - same with BOS
            if self.I2C[predicted_char] == ">":
                break

        return reconstruction
