import os
import re
import sys
import random
import pickle
import argparse
from collections import defaultdict
import torch


random.seed(1234)


class DataHandler:
    """
    Data format:
    ex: { 'pi:num':
            {'protoform':
                {
                'Latin': ['p', 'i', 'n', 'ʊ', 'm']
                },
            'daughters':
                {'Romanian': ['p', 'i', 'n'],
                 'French': ['p', 'ɛ', '̃'],
                 'Italian': ['p', 'i', 'n', 'o'],
                 'Spanish': ['p', 'i', 'n', 'o'],
                 'Portuguese': ['p', 'i', 'ɲ', 'ʊ']
                }
            },
        ...
        }
    """
    def __init__(self, dataset_name):
        self._dataset_name = dataset_name

    def _read_tsv(self, fpath):
        """
        Assumes the first row contains the languages (daughter and proto-lang)
        Assumes the first column is the protoform (or characters in the case of Chinese)

        Returns a dict mapping a protoform to the daughter forms
        """
        with open(fpath) as fin:
            langs = fin.readline().strip().split('\t')
            if "chinese" in self._dataset_name:
                langs = langs[1:]  # first column is character
            d = {}
            for line in fin:
                tkns = line.strip().split('\t')
                d[tkns[0]] = tkns[1:]
        return langs, d

    def _clean_middle_chinese_string(self, clean_string):
        subtokens = clean_string.split('/')
        tone = None
        if len(subtokens) > 1:
            tone = subtokens[1]
        return subtokens[0], tone

    def _clean_sinitic_daughter_string(self, raw_string):
        # only keep first entry for multiple variants (polysemy, pronunciation variation, etc.)
        # selection is arbitrary -> can also be removed altogether
        clean_string = raw_string
        if '|' in raw_string:
            clean_string = raw_string.split('|')[0]
        if '/' in raw_string:
            clean_string = raw_string.split('/')[0]
        # remove chinese characters
        subtokens = re.findall('([^0-9]+)([0-9]+)([^0-9]*)', clean_string)
        tone = None
        if subtokens:
            subtokens = subtokens[0]
            clean_string = subtokens[0]
            tone =  subtokens[1]
        return clean_string, tone

    def sinitic_tokenize(self, clean_string, merge_diacritics=False):
        tkns = list(clean_string)

        ### TODO: middle chinese didn't connect affricates with '͡' --> will remove these tokens for now.
        # affricate - should always be merged
        # while '͡' in tkns:
        #     i = tkns.index('͡')
        #     tkns = tkns[:i-1] + [''.join(tkns[i-1: i+2])] + tkns[i+2:]

        tkns = [tkn for tkn in tkns if tkn != '͡']

        # diacritics - optionally merge
        if merge_diacritics:
            diacritics = {'ː', '̃', '̍', '̞', '̠', '̩'} # , 'ʰ', 'ʷ'}
            while diacritics & set(tkns):
                for i in range(len(tkns)):
                    if tkns[i] in diacritics:
                        tkns = tkns[:i-1] + [''.join(tkns[i-1: i+1])] + tkns[i+1:]
                        break
        return tkns

    def tokenize(self, string):
        return list(string)

    def generate_split_datasets(self):
        split_ratio = (70, 10, 20)  # train, dev, test
        langs, data = self._read_tsv(f'./data/{self._dataset_name}.tsv')
        protolang = langs[0]
        cognate_set = {}
        for cognate, tkn_list in data.items():
            entry = {}
            daughter_sequences = {}
            if "chinese" in self._dataset_name:
                mc_string, mc_tone = self._clean_middle_chinese_string(tkn_list[0])
                mc_tkns = self.sinitic_tokenize(mc_string, merge_diacritics=False)
                for dialect, tkn in zip(langs[1:], tkn_list[1:]):
                    if not tkn or tkn == '-':
                        continue
                    daughter_string, daughter_tone = self._clean_sinitic_daughter_string(tkn)
                    daughter_tkns = self.sinitic_tokenize(daughter_string, merge_diacritics=False)
                    daughter_sequences[dialect] = daughter_tkns
                entry['protoform'] = {
                    protolang: mc_tkns
                }
                entry['daughters'] = daughter_sequences
                cognate_set[cognate] = entry
            else:
                protolang_tkns = self.tokenize(cognate)
                for lang, tkn in zip(langs[1:], tkn_list):
                    if not tkn or tkn == '-':
                        continue
                    daughter_tkns = self.tokenize(tkn)
                    daughter_sequences[lang] = daughter_tkns

                entry['protoform'] = {
                    protolang: protolang_tkns
                }
                entry['daughters'] = daughter_sequences
                cognate_set[cognate] = entry

        dataset = {}
        proto_words = list(cognate_set.keys())
        random.shuffle(proto_words)
        dataset['train'] = proto_words[0: int(len(proto_words) * split_ratio[0]/sum(split_ratio))]
        dataset['dev'] = proto_words[len(dataset['train']): int(len(proto_words) * (split_ratio[0] + split_ratio[1])/sum(split_ratio))]
        dataset['test'] = proto_words[len(dataset['train']) + len(dataset['dev']): ]

        dataset_path = f'./data/{self._dataset_name}'
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        for data_type in dataset:
            subdata = {protoword: cognate_set[protoword] for protoword in dataset[data_type]}
            with open(f'./data/{self._dataset_name}/{data_type}.pickle', 'wb') as fout:
                pickle.dump((langs, subdata), fout)

    @classmethod
    def load_dataset(cls, fpath):
        vocab = set()  # set of possible phonemes in the daughters and the protoform
        with open(fpath, 'rb') as fin:
            (langs, data) = pickle.load(fin)
            # list format enables shuffling
            dataset = []
            for cognate, entry in data.items():
                for lang, target in entry['protoform'].items():
                    vocab.update(target)
                for lang, source in entry['daughters'].items():
                    vocab.update(source)
                dataset.append((cognate, entry))

        return dataset, vocab, langs

    @classmethod
    def get_cognateset_batch(cls, dataset, langs, C2I, I2C):
        """
        Convert both the daughter and protoform character lists to indices in the vocab
        """
        protolang = langs[0]
        for cognate, entry in dataset:
            # 1. convert the chars to indices
            # 2. then zip with the lang - List of (lang, index tensor)
            # 3. in the Embedding layer, add the BOS/EOS and the separator embeddings and do the language and char embeddings
            # the languages are supplied so the model knows what language embedding to apply
            target_batch = [('sep', C2I["<"])] + \
                [(protolang, C2I[char if char in C2I else "<unk>"]) for char in entry["protoform"][protolang]] + \
                [('sep', C2I[">"])]

            # example cognate set
            #      <*French:croître*Italian:crescere*Spanish:crecer*Portuguese:crecer*Romanian:crește*>
            # note that the languages will be treated as one token

            # start of sequence
            source_batch = [('sep', C2I["<"])]
            for lang in langs[1:]:
                source_index_sequence = [('sep', C2I['*']), ('sep', C2I[lang]), ('sep', C2I[':'])]
                # incomplete cognate set
                if lang not in entry['daughters']:
                    source_index_sequence.append((lang, C2I['-']))
                else:
                    raw_source_sequence = entry['daughters'][lang]
                    # note: C2I will recognize each language's name as a token, so it will not go to UNK
                    source_index_sequence += [(lang, C2I[char if char in C2I else "<unk>"])
                                              for char in raw_source_sequence]
                source_batch += source_index_sequence
            source_batch.append(('sep', C2I["*"]))
            # end of sequence
            source_batch.append(('sep', C2I[">"]))

            # used when calculating the loss
            target_tensor = torch.tensor([idx for _, idx in target_batch])
            yield source_batch, target_batch, target_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='chinese/romance_orthographic/romance_phonetic/austronesian')
    args = parser.parse_args()

    d = DataHandler(args.dataset)
    d.generate_split_datasets()
