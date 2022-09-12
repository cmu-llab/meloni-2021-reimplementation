# PyTorch re-implementation of Meloni et al 2021

[Meloni et al 2021](https://aclanthology.org/2021.naacl-main.353/) achieve state-of-the-art edit distances on the Latin protoform reconstruction task with an encoder-decoder 
model with cross-attention. The code is originally written in DyNet, so we translated it to PyTorch. We also tested the model
on a Chinese dataset, WikiHan (Chang et al 2022). 

The only difference between our PyTorch version and their code is that we do not implement variational dropout in the encoder (Gal and Ghahramani 2016), but DyNet comes with this flavor of dropout in its RNN modules. 
We do implement variational dropout for the decoder, though. Unfortunately, PyTorch's RNN, LSTM, and GRU modules do not come with variational dropout, though it is possible to overwrite the respective classes with a version that implements variational dropout.
We find that dropout makes a significant difference when trained on a Chinese dataset from Hou 2004. 


# Running the code
```
# install dependencies
pip install -r requirements.txt

# generate pickle files from the dataset
python preprocessing.py --dataset chinese_wikihan2022.tsv

# run the model
python main.py --batch_size=1 --beta1=0.9 --beta2=0.999 --dataset=chinese_wiktionary --dropout=0.2769919925175171 --embedding_size=489 --epochs=80 --eps=1e-08 --feedforward_dim=431 --lr=0.00012204821435543166 --model_size=28 --network=gru --num_layers=1
```


# Hyperparameters
These hyperparameters were obtained by doing a Wandb sweep. 

Hyperparameter | Explanation | Value
beta1 | Adam beta1 | 0.9
beta2 | Adam beta2 | 0.999
dataset | name of the TSV dataset | chinese_wikihan2022
dropout | dropout value; see note above on the importance of dropout | 0.2769919925175171
embedding_size | dimension of token embeddings | 489
epochs | number of full passes through the dataset | 80
eps | Adam epsilon | 1e-08
feedforward_dim | dimension of the final multi-layer perceptron | 431
lr | Adam learning rate | 0.00012204821435543166
model_size | RNN hidden layer size | 28
network | RNN flavor (only GRU is supported for now) | gru
num_layers | number of RNN layers | 1


# Data

### Current options
We currently offer 2 Chinese datasets:
* WikiHan (Chang et al 2022) https://github.com/cmu-llab/wikihan
* Hou 2004 - a dataset of 800+ cognate sets spanning 39 Sinitic varieties from Wiktionary (https://en.wiktionary.org/wiki/Module:zh/data/dial-pron/documentation) originally from Hou 2004

Hou 2004 as it appeared on Wiktionary originally contained 1,000 characters, but only around 800 had entries in the Qieyun. 
Meloni et al 2022's original code was tested on a Romance dataset from Ciobanu and Dinu 2014, but you will need to request the data from Ciobanu and Dinu. 

To run the model on additional datasets, add the dataset to the data/ folder and run preprocessing on the dataset using the preprocessing script as shown earlier. 
Our preprocessing expects that any Chinese dataset will be prefixed with "chinese" and will have the first column containing the character and the second column
containing the protoform, but for other language families, the dataset expects that the protoform be in the first column. 
In addition, the first row needs to be a header row denoting the languages for each column. 
Finally, we currently require the data to be in TSV form. 


### Adapting WikiHan for the protoform reconstruction task
In order for the model to learn correspondences between phonemes as linguists would, we tokenize by phonemes, for example /tʰ/ and /t͡ɕʰ/, instead of characters. 
These two example phonemes should each be treated as one consonant despite being represented with several Unicode characters. % affricates
We treat diphthongs and triphthongs as one token because they constitute one syllable, phonetically speaking. 
We also merge IPA diacritics for vowel length, nasalization, and syllabic nasals into one phoneme. 
Basically, IPA diacritics should be grouped along with the segment that they modify instead of being treated as a separate token (as is the case with Unicode denormalized form).
We also restrict ourselves to cognates with at least 4 entries including Middle Chinese to avoid being biased to varieties with more entries, such as Mandarin. 
Another decision we made is to arbitrarily take the first pronunciation variant for heteronyms. 
Let us clarify with an example. 車 has 2 cognate sets in the data, and its Hokkien entry for one of the sets has 3 variants ku˥/kɨ˥/ki˥. 
We keep both cognate sets but when we pick ku˥ when we assign a Min pronunciation for the cognate set containing ku˥/kɨ˥/ki˥. 


# References
Kalvin Chang, Chenxuan Cui, Youngmin Kim, and David R. Mortensen. 2022. WikiHan: A New Comparative Dataset for Sinitic Languages. In *Proceedings of the 29th International Conference on Computational Linguistics* (COLING 2022), Gyeongju, Korea.

Yarin Gal and Zoubin Ghahramani. 2016. A theoretically grounded application of dropout in recurrent neural networks. In *Advances in Neural Information Processing Systems*, volume 29. Curran Associates, Inc.

Carlo Meloni, Shauli Ravfogel, and Yoav Goldberg. 2021. Ab antiquo: Neural proto-language reconstruction. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4460–4473, Online. Association for Computational Linguistics.

Jīngyī Hóu (侯精一), editor. 2004. Xiàndài Hànyǔ fāngyán yīnkù 现代汉语方言音库 [Phonological database of Chinese dialects]. Shànghǎi Jiàoyù 上海教育, Shànghǎi 上海.
