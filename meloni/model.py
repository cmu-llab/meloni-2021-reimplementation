import torch
import torch.nn as nn


class Embedding(nn.Module):
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
        # TODO: this should be done in the preprocessing
        char_idx = self.C2I[char] if char in self.C2I else self.C2I["<unk>"]
        # TODO: rename to char embedding
        char_encoded = self.source_embedding(char_idx)
        # TODO: stop doing a linear search - vocab lookup?
        lang_encoded = self.lang_embedding(self.langs.index(lang))
        # TODO: check the dimensions
        return torch.matmul(self.fc, torch.cat((char_encoded, lang_encoded), dim=0))


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
        encoder = Embedding(C2I)
        # one encoder for the proto-language
        # share encoder across all languages
        self.encoders = [encoder for _ in range(len(langs) + 1)]
        # TODO: langs should include the protolang as the last language
        self.langs = langs
        self.protolang = langs[-1]
        self.l2e = { lang:self.encoders[i] for i, lang in enumerate(self.langs) }
        # separator language
        self.l2e["sep"] = self.encoders[-1]

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

        # TODO: the attention!
        self.attention = Attention(hidden_dim=model_size, embedding_dim=embedding_dim)

    def forward(self, cognate_set, protoform):
        # encoder
        # expects a cognate set mapping city to IPA - OUR. TODO: test on Romance data in our format
        encoded_cognateset = []
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

        # TODO: rename to embedded
        encoded_cognateset = [self.encoders[0].encode("<", "sep")] + encoded_cognateset + [self.encoders[0].encode(">", "sep")]
        # TODO: make sure we initialize the state correctly??
        encoded_cognateset = torch.vstack(encoded_cognateset).to(DEVICE)

        # TODO: is the encoder treating each input as separate?
        (encoder_states, memory) = self.encoder_rnn(encoded_cognateset).to(DEVICE)
        # TODO: note that the LSTM returns something diff than the GRU

        # decoder

        # start of protoform sequence
        start_encoded = self.l2e["sep"].encode("<s>", "sep")
        attention_weighted_states = encoder_states[-1]
        decoder_state = self.decoder_rnn(torch.cat(start_encoded, attention_weighted_states))
        scores = []  # TODO: it's faster to initialize the shape then fill it in

        for char in protoform:
            # embedding layer
            true_char_encoded = self.l2e[self.protolang].encode(char, self.protolang)
            # MLP to get a probability distribution over the possible output phonemes
            char_scores = self.mlp(decoder_state + attention_weighted_states)
            scores.append(char_scores)
            # dot product attention over the encoder states
            attention_weighted_states = self.attention(decoder_state, encoder_states, encoded_cognateset)
            self.decoder_rnn(torch.cat(true_char_encoded, attention_weighted_states))

        return scores


    def encode(self):
        pass
        # TODO torch.nograd?

    def decode(self):
        pass
