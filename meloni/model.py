import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embedding_dim, langs, C2I, L2I):
        self.C2I = C2I
        self.langs = langs
        self.source_embedding = nn.Embedding(len(C2I), embedding_dim)
        self.lang_embedding = nn.Embedding(len(langs), embedding_dim)
        # map concatenated source and language embedding to 1 embedding
        self.fc = nn.Linear(2 * embedding_dim, embedding_dim)

    def encode(self, char, lang):
        char_idx = self.C2I[char] if char in self.C2I else self.C2I["<unk>"]
        char_encoded = self.source_embedding(char_idx)
        # TODO: stop doing a linear search - vocab lookup?
        lang_encoded = self.lang_embedding(self.langs.index(lang))
        # TODO: check the dimensions
        return torch.matmul(self.fc, torch.cat((char_encoded, lang_encode), dim=0))

