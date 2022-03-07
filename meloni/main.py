import argparse
from model import *
from tqdm import tqdm
import os
import torch
import time
import numpy as np
import random

MODELPATH_LOSS = f'./checkpoints/{DATASET}_best_loss.pt'
MODELPATH_ED = f'./checkpoints/{DATASET}_best_ed.pt'

def get_char_batch(dataset, ipa_vocab, dialect_vocab):
    pass


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

    random.shuffle(train_data)
    good, bad = 0, 0
    total_train_loss = 0
    for cognate_set, protoform in train_data:
        optimizer.zero_grad()

        # TODO: do the C2I in the dataloader
        # TODO: do the to(device) thingy here
        # TODO: i don't think masking is needed
        scores = model(cognate_set, protoform)

        # TODO: get dims right

        loss = loss_fn(scores, protoform)
        loss.backward()
        total_train_loss += loss.item()

        optimizer.step()

        # compare indices instead of converting to string
        predicted = torch.argmax(scores)
        # TODO: one is list, one is tensor
        if predicted == protoform:
            good += 1
        else:
            bad += 1
        # TODO: do we need to calculate train accuracy?


    return total_train_loss / len(train_data), good / (good + bad)

def train(epochs, model, optimizer, loss_fn, train_data, dev_data):
    mean_train_losses, mean_dev_losses = np.zeros(epochs), np.zeros(epochs)
    best_dev_loss, best_dev_edit_distance = 0., 0.

    for epoch in tqdm(range(epochs)):
        t = time.time()

        train_loss, train_accuracy = train_once(model, optimizer, loss_fn, train_data)
        dev_loss, edit_distance, dev_accuracy = evaluate(model, loss_fn, dev_data, ipa_vocab, dialect_vocab)
        print(f'< epoch {epoch} >  (elapsed: {time.time() - t:.2f}s)')
        print(f'  * [train]  loss: {train_loss:.6f}')
        dev_result_line = f'  * [ dev ]  loss: {dev_loss:.6f}'
        if edit_distance is not None:
            dev_result_line += f'  ||  edit distance: {edit_distance}  ||  accuracy: {dev_accuracy}'
        print(dev_result_line)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_loss_epoch = epoch
            save_model(model, optimizer, args, ipa_vocab, dialect_vocab, epoch, MODELPATH_LOSS)
        if edit_distance < best_edit_distance:
            best_edit_distance = edit_distance
            best_ed_epoch = epoch
            save_model(model, optimizer, args, ipa_vocab, dialect_vocab, epoch, MODELPATH_ED)

        mean_train_losses[epoch] = train_loss
        mean_dev_losses[epoch] = dev_loss

    # TODO: be more specific in the naming
    np.save("losses/train", mean_train_losses)
    np.save("losses/dev", mean_dev_losses)


def evaluate(model, loss_fn, dataset, ipa_vocab, dialect_vocab):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        edit_distance = 0
        n_correct = 0
        # TODO: decide on the input preprocessing
        for source, dialect, target in get_char_batch(dataset, ipa_vocab, dialect_vocab):
            # calculate loss
            # TODO: why does the transformer do one forward pass thru model() and still separately do the encode/decode?
                # is it to calculate the loss?
            scores = model(cognate_set, protoform)
            loss = loss_fn(scores, protoform)
            total_loss += loss.item()

            # calculate edit distance
            encoded_state, embedded_x = model.encode(cognate_set, protoform)
            prediction = model.decode(encoded_state, embedded_x, MAX_LENGTH)
            # TODO: get the indexing / batching correct
            edit_distance += get_edit_distance(ipa_vocab.to_string(target[0]), ipa_vocab.to_string(prediction[0]))

            if ipa_vocab.to_string(target[0]) == ipa_vocab.to_string(prediction[0]):
                n_correct += 1

    accuracy = n_correct / len(dataset)
    mean_loss = total_loss / len(dataset)
    mean_edit_distance = edit_distance / len(dataset)

    return mean_loss, mean_edit_distance, accuracy


def save_model(model, optimizer, args, ipa_vocab, dialect_vocab, epoch, filepath):
    # TODO: store the vocabulary in the RNN Meloni format
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'epoch': epoch,
        'ipa_vocab': ipa_vocab,
        'dialect_vocab': dialect_vocab
    }
    torch.save(save_info, filepath)
    print(f'\t>> saved model to {filepath}')


def load_model(filepath):
    saved_info = torch.load(filepath)
    return saved_info


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='chinese/romance/austronesian')
    parser.add_argument('--network', type=str, required=True, help='lstm/gru')
    parser.add_argument('--num_layers', type=float, required=True, help='number of RNN layers')
    parser.add_argument('--model_size', type=int, required=True, help='lstm hidden layer size')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--beta1', type=float, required=True, help='beta1')
    parser.add_argument('--beta2', type=float, required=True, help='beta2')
    parser.add_argument('--embedding_size', type=int, required=True, help='embedding size')
    parser.add_argument('--feedforward_dim', type=int, required=True, help='dimension of the final MLP')
    parser.add_argument('--dropout', type=float, required=True, help='dropout value')
    parser.add_argument('--epochs', type=int, required=True)
    # TODO: batch size
    parser.add_argument('--batch_size', type=int, required=True, help='batch_size')

    TQDM_DISABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = args.epochs
    DATASET = args.dataset
    NUM_LAYERS = args.num_layers
    NETWORK = args.network

    LEARNING_RATE = args.lr
    BETA_1 = args.beta1
    BETA_2 = args.beta2
    EMBEDDING_SIZE = args.embedding_size
    DROPOUT = args.dropout
    MAX_LENGTH = 30 if 'romance' in DATASET else 15
    HIDDEN_SIZE = args.model_size
    FEEDFORWARD_DIM = args.feedforward_dim

    train_dataset, dev_dataset, test_dataset = DataHandler.get_datasets(args.dataset)
    # TODO: use our vocab class
    # TODO: use terms like dialect vocab or something
    # TODO: grab the language list
    letters, C2I, I2C = utils.create_voc(args.dataset)

    # encoder for the separator
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
            nn.init.xavier_uniform_(p)

    # does the softmax for you
    # TODO: does Meloni do padding? ignore_index=PAD_IDX
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 betas=(BETA_1, BETA_2),
                                 eps=1e-9)


    train(NUM_EPOCHS, model, optimizer, loss_fn, train_dataset, dev_dataset)

    # TODO: what is this doing?
    for filepath, criterion in [(MODELPATH_LOSS, 'loss'), (MODELPATH_ED, 'edit distance')]:
        save_info = load_model(filepath)
        saved_info = load_model(filepath)
        args = saved_info['args']
        # TODO: fix

        # TODO: don't forget
        model.load_state_dict(saved_info['model'])

        dev_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/dev.pickle')
        dev_loss, dev_ed, dev_acc = evaluate(model, loss_fn, dev_dataset, ipa_vocab, dialect_vocab)

        test_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/test.pickle')
        test_loss, test_ed, test_acc = evaluate(model, loss_fn, test_dataset, ipa_vocab, dialect_vocab)

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

# TODO: make directories if they do not exist


