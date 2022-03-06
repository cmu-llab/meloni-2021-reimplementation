import argparse
from model import *
from tqdm import tqdm
import os


MODELPATH_LOSS = f'./checkpoints/{DATASET}_best_loss.pt'
MODELPATH_ED = f'./checkpoints/{DATASET}_best_ed.pt'

def get_char_batch(dataset, ipa_vocab, dialect_vocab):
    pass


def train(epochs, model, optimizer, loss_fn, train_data, dev_data):
    pass


def evaluate(model, loss_fn, dataset, ipa_vocab, dialect_vocab, return_edit_distance):
    model.eval()
    total_loss = 0
    edit_distance = 0
    n_correct = 0
    # TODO: revise

    return total_loss, edit_distance, accuracy


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
    # TODO: is this what they are doing?
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # does the softmax for you
    # TODO: does Meloni do padding?
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 betas=(BETA_1, BETA_2),
                                 eps=1e-9)



    train(NUM_EPOCHS, model, optimizer, loss_fn, train_dataset, dev_dataset)
    for filepath, criterion in [(MODELPATH_LOSS, 'loss'), (MODELPATH_ED, 'edit distance')]:
        save_info = load_model(filepath)
        saved_info = load_model(filepath)
        args = saved_info['args']
        # TODO: fix
        model = Model(args.encoder_layers, args.decoder_layers, args.embedding_size,
                      args.nhead, len(ipa_vocab), len(dialect_vocab), args.dim_feedforward,
                      args.dropout, MAX_LENGTH).to(DEVICE)
        model.load_state_dict(saved_info['model'])

        dev_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/dev.pickle')
        dev_loss, dev_ed, dev_acc = evaluate(model, loss_fn, dev_dataset, ipa_vocab, dialect_vocab, True)

        test_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/test.pickle')
        test_loss, test_ed, test_acc = evaluate(model, loss_fn, test_dataset, ipa_vocab, dialect_vocab, True)

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


