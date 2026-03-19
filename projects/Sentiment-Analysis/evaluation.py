import pandas as pd
import torch
import argparse
import model
from configs import BasicConfigs
from process_data import transform_data, prepare_vocab

bc = BasicConfigs()
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='birnn')
args = parser.parse_args()


def evaluate(df):
    vocab, label_map, embedding_matrix = prepare_vocab(is_train=False)
    net = getattr(model, args.model_name)(
        embedding_matrix=embedding_matrix,
        num_hiddens=bc.num_hiddens,
        num_layers=bc.num_layers
    ).to(bc.device)
    net.load_state_dict(torch.load(bc.save_model_dir[args.model_name]))
    net.eval()

    result = {'correct': 0, 'wrong': 0}
    df_len = df.shape[0]

    with torch.no_grad():
        for i in range(df_len):
            record = df.loc[i, :].to_dict()
            data, label = transform_data(record, vocab, label_map)
            data, label = data.to(bc.device), label.to(bc.device)
            score = net(data)

            if score.argmax(dim=1) == label:
                result['correct'] += 1
            else:
                result['wrong'] += 1

    print(f"Classification Accuracy of Model({net.__class__.__name__}) is {result['correct'] / df_len:.4f}")


if __name__ == '__main__':
    test_data = pd.read_csv('Dataset/test.csv')
    evaluate(df=test_data)