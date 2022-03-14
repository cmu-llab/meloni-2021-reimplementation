import argparse
from model import *
from tqdm import tqdm
import torch
import time
import numpy as np
import random
from preprocessing import DataHandler
import os
from collections import defaultdict


def get_edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    # len(s1) <= len(s2)
    # TODO: understand
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]


def train_once(model, optimizer, loss_fn, train_data):
    model.train()  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch

    order = list(train_data.keys())
    random.shuffle(order)
    good, bad = 0, 0
    total_train_loss = 0
    for cognate in order:
        source_tokens, source_langs, target_tokens, target_langs = train_data[cognate]
        # TODO: check if i'm supposed to do this
        optimizer.zero_grad()

        # TODO: do the to(device) thingy here
        logits = model(source_tokens, source_langs, target_tokens, target_langs, DEVICE)
        # logits should be (1, T, |Y|)

        # reshape logits to (T, |Y|) - remove batch dim for now
        # protoform_tensor: (T,)
        loss = loss_fn(logits, target_tokens)
        loss.backward()
        total_train_loss += loss.item()

        optimizer.step()

        # TODO: check dimensions for everything!

        # compare indices instead of converting to string
        predicted = torch.argmax(logits, dim=1)
        if torch.equal(predicted, target_tokens):
            good += 1
        else:
            bad += 1

    return total_train_loss / len(train_data), good / (good + bad)


def train(epochs, model, optimizer, loss_fn, train_data, dev_data):
    mean_train_losses, mean_dev_losses = np.zeros(epochs), np.zeros(epochs)
    best_loss_epoch, best_ed_epoch = 0, 0
    best_dev_loss, best_dev_edit_distance = 0, 10e10

    # precompute the tensors once. reuse
    train_data = DataHandler.get_cognateset_batch(train_data, langs, C2I, L2I, DEVICE, I2C)
    dev_data = DataHandler.get_cognateset_batch(dev_data, langs, C2I, L2I, DEVICE, I2C)

    for epoch in tqdm(range(epochs)):
        t = time.time()

        train_loss, train_accuracy = train_once(model, optimizer, loss_fn, train_data)
        dev_loss, edit_distance, dev_accuracy, _ = evaluate(model, loss_fn, dev_data)
        print(f'< epoch {epoch} >  (elapsed: {time.time() - t:.2f}s)')
        print(f'  * [train]  loss: {train_loss:.6f}')
        dev_result_line = f'  * [ dev ]  loss: {dev_loss:.6f}'
        if edit_distance is not None:
            dev_result_line += f'  ||  edit distance: {edit_distance}  ||  accuracy: {dev_accuracy}'
        print(dev_result_line)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_loss_epoch = epoch
            save_model(model, optimizer, args, epoch, MODELPATH_LOSS)
        if edit_distance < best_dev_edit_distance:
            best_dev_edit_distance = edit_distance
            best_ed_epoch = epoch
            save_model(model, optimizer, args, epoch, MODELPATH_ED)

        mean_train_losses[epoch] = train_loss
        mean_dev_losses[epoch] = dev_loss

    # TODO: be more specific in the naming
    if not os.path.isdir('losses'):
        os.mkdir('losses')
    np.save("losses/train", mean_train_losses)
    np.save("losses/dev", mean_dev_losses)
    record(best_loss_epoch, best_dev_loss, best_ed_epoch, best_dev_edit_distance)


def record(best_loss_epoch, best_loss, best_ed_epoch, edit_distance):
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/' + DATASET):
        os.mkdir('results/' + DATASET)
    with open(f'./results/{DATASET}/params.txt', 'w') as fout:
        params = {'network': NETWORK,
                  'num_layers': NUM_LAYERS,
                  'model_size': HIDDEN_SIZE,
                  'lr': LEARNING_RATE,
                  'beta1': BETA_1,
                  'beta2': BETA_2,
                  'eps': EPS,
                  'embedding_size': EMBEDDING_SIZE,
                  'feedforward_dim': FEEDFORWARD_DIM,
                  'dropout': DROPOUT,
                  'epochs': NUM_EPOCHS,
                  'batch_size': 1}
        for k, v in params.items():
            fout.write(f'{k}: {v}\n')
    with open(f'./results/{DATASET}/metrics.txt', 'a') as fout:
        fout.write(f'{DATASET}. loss: {best_loss} ({best_loss_epoch})   ||   {edit_distance} ({best_ed_epoch})\n')


def evaluate(model, loss_fn, dataset):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        edit_distance = 0
        n_correct = 0
        predictions = []
        for _, (source_tokens, source_langs, target_tokens, target_langs) in dataset.items():
            # calculate loss
            logits = model(source_tokens, source_langs, target_tokens, target_langs, DEVICE)
            loss = loss_fn(logits, target_tokens)
            total_loss += loss.item()

            # calculate edit distance
            # necessary to have a separate encode and decode because we are doing greedy decoding here
            #   instead of comparing against the protoform
            (encoder_states, memory), embedded_x = model.encode(source_tokens, source_langs, DEVICE)
            prediction = model.decode(encoder_states, memory, embedded_x, MAX_LENGTH, DEVICE)
            # TODO: get the batching correct
            predict_str, protoform_str = \
                DataHandler.to_string(I2C, prediction), DataHandler.to_string(I2C, target_tokens)
            edit_distance += get_edit_distance(predict_str, protoform_str)
            if predict_str == protoform_str:
                n_correct += 1
            predictions.append((predict_str, protoform_str))

    accuracy = n_correct / len(dataset)
    mean_loss = total_loss / len(dataset)
    mean_edit_distance = edit_distance / len(dataset)

    return mean_loss, mean_edit_distance, accuracy, predictions


def save_model(model, optimizer, args, epoch, filepath):
    # TODO: store the vocabulary in the RNN Meloni format
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    torch.save(save_info, filepath)
    print(f'\t>> saved model to {filepath}')


def load_model(filepath):
    saved_info = torch.load(filepath)
    return saved_info


def write_preds(filepath, predictions):
    # predictions: predicted - original cognate
    # TODO: should we try adding the original cognate set
    with open(filepath, 'w') as f:
        f.write("prediction\tgold standard\n")
        for pred, gold_std in predictions:
            # remove BOS and EOS
            pred = pred[1:-1]
            gold_std = gold_std[1:-1]
            f.write(f"{pred}\t{gold_std}\n")


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='chinese/romance_orthographic/romance_phonetic/austronesian')
    parser.add_argument('--network', type=str, required=True, help='lstm/gru')
    parser.add_argument('--num_layers', type=int, required=True, help='number of RNN layers')
    parser.add_argument('--model_size', type=int, required=True, help='lstm hidden layer size')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--beta1', type=float, required=True, help='beta1')
    parser.add_argument('--beta2', type=float, required=True, help='beta2')
    parser.add_argument('--eps', type=float, required=True, help='eps')
    parser.add_argument('--embedding_size', type=int, required=True, help='embedding size')
    parser.add_argument('--feedforward_dim', type=int, required=True, help='dimension of the final MLP')
    parser.add_argument('--dropout', type=float, required=True, help='dropout value')
    parser.add_argument('--epochs', type=int, required=True)
    # TODO: batching
    parser.add_argument('--batch_size', type=int, required=True, help='batch_size')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = args.epochs
    DATASET = args.dataset
    NUM_LAYERS = args.num_layers
    NETWORK = args.network
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    MODELPATH_LOSS = f'./checkpoints/{DATASET}_best_loss.pt'
    MODELPATH_ED = f'./checkpoints/{DATASET}_best_ed.pt'

    LEARNING_RATE = args.lr
    BETA_1 = args.beta1
    BETA_2 = args.beta2
    EPS = args.eps
    EMBEDDING_SIZE = args.embedding_size
    DROPOUT = args.dropout
    MAX_LENGTH = 30 if 'romance' in DATASET else 15
    HIDDEN_SIZE = args.model_size
    FEEDFORWARD_DIM = args.feedforward_dim

    train_dataset, phoneme_vocab, langs = DataHandler.load_dataset(f'./data/{DATASET}/train.pickle')
    dev_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/dev.pickle')
    # special tokens in the separator embedding's vocabulary
    # TODO: create a special vocab just for the separator embeddings
    phoneme_vocab.add("<")
    phoneme_vocab.add(":")
    phoneme_vocab.add("*")
    phoneme_vocab.add(">")
    phoneme_vocab.add("<unk>")
    phoneme_vocab.add("-")
    phoneme_vocab.add("<s>")
    # treat each language as a token since each language will be included in the input sequence
    for lang in langs:
        phoneme_vocab.add(lang)
    C2I = {c: i for i, c in enumerate(sorted(phoneme_vocab))}
    C2I = defaultdict(lambda: C2I['<unk>'], C2I)

    I2C = {i: c for i, c in enumerate(sorted(phoneme_vocab))}
    L2I = {l: idx for idx, l in enumerate(langs + ['sep'])}

    model = Model(C2I, I2C,
                  num_layers=NUM_LAYERS,
                  dropout=DROPOUT,
                  feedforward_dim=FEEDFORWARD_DIM,
                  embedding_dim=EMBEDDING_SIZE,
                  model_size=HIDDEN_SIZE,
                  model_type=NETWORK,
                  langs=langs,
                  ).to(DEVICE)
    # TODO: is this what Meloni are doing?
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    # does the softmax for you
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 betas=(BETA_1, BETA_2),
                                 eps=EPS)

    train(NUM_EPOCHS, model, optimizer, loss_fn, train_dataset, dev_dataset)

    # evaluate on the model with the best loss and the one with the best edit distance
    for filepath, criterion in [(MODELPATH_LOSS, 'loss'), (MODELPATH_ED, 'edit distance')]:
        saved_info = load_model(filepath)
        args = saved_info['args']

        model = Model(C2I, I2C,
                      num_layers=NUM_LAYERS,
                      dropout=DROPOUT,
                      feedforward_dim=FEEDFORWARD_DIM,
                      embedding_dim=EMBEDDING_SIZE,
                      model_size=HIDDEN_SIZE,
                      model_type=NETWORK,
                      langs=langs,
                      ).to(DEVICE)
        model.load_state_dict(saved_info['model'])

        test_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/test.pickle')
        dev_loss, dev_ed, dev_acc, dev_preds = evaluate(model, loss_fn, dev_dataset)
        test_loss, test_ed, test_acc, test_preds = evaluate(model, loss_fn, test_dataset)

        if not os.path.isdir('predictions'):
            os.mkdir('predictions')
        if not os.path.isdir('predictions/' + DATASET):
            os.mkdir('predictions/' + DATASET)
        write_preds('predictions/' + DATASET + '/best-' + criterion + '-dev', dev_preds)
        write_preds('predictions/' + DATASET + '/best-' + criterion + '-test', test_preds)

        # TODO: remember to calculate normalized edit distance
        print(f'===== <FINAL - best {criterion}>  (epoch: {saved_info["epoch"]}) ======')
        print(f'[dev]')
        print(f'  * loss: {dev_loss}')
        print(f'  * edit distance: {dev_ed}')
        print(f'  * accuracy: {dev_acc}')
        print()
        print(f'[test]')
        print(f'  * loss: {test_loss}')
        print(f'  * edit distance: {test_ed}')
        print(f'  * accuracy: {test_acc}')
