lr = 0.00013
beta1 = 0.9
beta2 = 0.98
embedding_size = 128
dropout = 0.202
epochs = 200
dim_feedforward = 100

# TODO: fix
for dataset in ['chinese_baxter', 'romance_ipa', 'romance_orto']:
    python main.py --dataset dataset --lr lr --beta1 beta1, beta2, encoder_layers, decoder_layers, embedding_size, nhead, dim_feedforward, dropout, epochs)