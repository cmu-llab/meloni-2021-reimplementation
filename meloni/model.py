import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Feature extraction:
    Looks up embeddings for a source character and the language,
    concatenates the two,
    then passes through a linear layer
    """
    def __init__(self, embedding_dim, langs, C2I, L2I):
        self.C2I = C2I
        self.langs = langs
        self.source_embedding = nn.Embedding(len(C2I), embedding_dim)
        self.lang_embedding = nn.Embedding(len(langs), embedding_dim)
        # map concatenated source and language embedding to 1 embedding
        self.fc = nn.Linear(2 * embedding_dim, embedding_dim)

    # TODO: use def forward instead?
    def encode(self, char, lang):
        char_idx = self.C2I[char] if char in self.C2I else self.C2I["<unk>"]
        char_encoded = self.source_embedding(char_idx)
        # TODO: stop doing a linear search - vocab lookup?
        lang_encoded = self.lang_embedding(self.langs.index(lang))
        # TODO: check the dimensions
        return torch.matmul(self.fc, torch.cat((char_encoded, lang_encode), dim=0))


class MLP(nn.Module):
    """
    Multi-layer perceptron to generate logits from the decoder state
    """
    def __init__(self, hidden_dim, feedforward_dim, output_size):
        # TODO: what are these magic numbers?
        self.fc1 = nn.Linear(hidden_dim, 2 * feedforward_dim)
        self.fc2 = nn.Linear(2 * feedforward_dim, feedforward_dim)
        self.fc3 = nn.Linear(feedforward_dim, output_size, bias=False)

    def forward(self, decoder_state):
        h = nn.ReLU(self.fc1(decoder_state))
        scores = self.fc3(nn.ReLU(self.fc2(h)))
        return scores


class Attention(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        # TODO: batch_first?
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_val = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_att = nn.Linear(embedding_dim, 1)
        self.W_c_s = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.W_direct = nn.Linear(len(self.C2I), hidden_dim)

    def forward(self, query, keys, encoded_input):
        # query: decoder state
        # keys: encoder states

        # TODO: why do we need this?
        query = self.W_query(query)
        # TODO: get dimensions right
        scores = torch.matmul(keys, query)
        # TODO: do the softmax on the correct dimension
        weights = nn.Softmax(torch.flatten(scores))

        # TODO: do attention analysis and highlight the attention vector

        # weights: L x 1
        # encoded_input: L x E
        # keys: L x D
        # result: L x D - weighted version of the input
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
        # TODO: modularize so we can get dialects for Austronesian, Chinese, or Romance
        # TODO: can we modularize this better?
        encoder = Encoder(C2I)
        # one encoder for the proto-language
        # share encoder across all languages
        self.encoders = [encoder for _ in range(len(langs) + 1)]
        # TODO: langs should include the protolang as the last language
        self.langs = langs
        self.l2e = { lang:self.encoders[i] for i, lang in enumerate(self.langs) }
        # separator language
        self.l2e["sep"] =self.encoders[-1]

        if model_type == "gru":
            # TODO: beware of batching
            self.encoder_rnn = nn.GRU(input_size=embedding_dim,
                                      hidden_size=model_size,
                                      num_layers=num_layers,
                                      dropout=dropout)
            self.decoder_rnn = nn.GRU(input_size=embedding_dim,
                                      hidden_size=model_size,
                                      num_layers=num_layers,
                                      dropout=dropout)
        else:
            self.encoder_rnn = nn.LSTM(input_size=embedding_dim,
                                       hidden_size=model_size,
                                       num_layers=num_layers,
                                       dropout=dropout)
            self.decoder_rnn = nn.LSTM(input_size=embedding_dim,
                                       hidden_size=model_size,
                                       num_layers=num_layers,
                                       dropout=dropout)

        # TODO: shouldn't we share this with the encoder?
        self.lang_embedding = nn.Embedding(len(langs), embedding_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, feedforward_dim=feedforward_dim, output_size=len(C2I))

        # TODO: the attention!
        self.W_query = self.model.add_parameters((lstm_size, lstm_size))
        self.W_key = self.model.add_parameters((lstm_size, lstm_size))
        self.W_val = self.model.add_parameters((lstm_size, lstm_size))
        self.W_att = self.model.add_parameters((1, EMBEDDING_SIZE))
        self.W_c_s = self.model.add_parameters((lstm_size, EMBEDDING_SIZE))
        self.W_direct = self.model.add_parameters((len(self.C2I), lstm_size))
        self.b_att = self.model.add_parameters((lstm_size, 1))
        self.b_direct = self.model.add_parameters((len(self.C2I), 1))

    def forward(self, cognate_set):
        # encoder
        # expects a cognate set mapping city to IPA - OUR. TODO: test on Romance data in our format
        # for all daughter languages
        for lang in self.langs[:-1]:
            # encodings for separator tokens - TODO: why do we need these?
            encoded_cognateset.append(self.encoders[0].encode("*", "sep"))
            encoded_cognateset.append(self.encoders[0].encode(lang, "sep")) # TODO: language embedding??
            encoded_cognateset.append(self.encoders[0].encode(":", "sep"))

            # encode each character of the input - TODO: can we do this in a batched way??
            for daughter_form in cognate_set[lang]:
                for char in daughter_form:
                    encoded_cognateset.append(self.encoders[0].encode(char, lang))

        # TODO: try seeing what happens if we format the data in a different way - like the transformer
        #  # all_vecs = torch.hstack(encoded_cognateset) = []

        all_vecs = [self.encoders[0].encode("<", "sep")] + vector_sequence + [self.encoders[0].encode(">", "sep")]
        # TODO: make sure we initialize
        s = self.encoder_rnn.initial_state()
        states = s.transduce(all_vecs)

        memory =

        # hstack instead of concatenating?

        # encoder-decoder
        # encoded_state = list of RNN output at each step. HIDDEN_SIZE
        # encoded_x = list of original embeddings. EMBEDDING_SIZE
        encoded_state, encoded_x = self.encode(x, y, train=True)

        # decoder


    def encode(self):
        # TODO torch.nograd?

