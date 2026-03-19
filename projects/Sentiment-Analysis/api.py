import torch
import argparse
import model
from flask import Flask, request, jsonify
from process_data import transform_data, prepare_vocab
from configs import BasicConfigs

bc = BasicConfigs()
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='birnn')
args = parser.parse_args()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

vocab, label_map, embedding_matrix = prepare_vocab  (is_train=False)
itos_label = {v: k for k, v in label_map.items()}

net = getattr(model, args.model_name)(
    embedding_matrix=embedding_matrix,
    num_hiddens=bc.num_hiddens,
    num_layers=bc.num_layers
).to(bc.device)

net.load_state_dict(torch.load(bc.save_model_dir[args.model_name], map_location=bc.device))
net.eval()


@app.route('/sentiment')
def sentiment():
    sentence = request.args.get('sentence')
    record = {'text': sentence}
    data, _ = transform_data(record, vocab, label_map)
    data = data.to(bc.device)

    with torch.no_grad():
        idx = net(data).argmax(dim=1).item()

    prediction = itos_label[idx]
    result = '积极' if prediction == 'pos' else '消极'

    return jsonify({'data': result, 'status_code': 200})


if __name__ == '__main__':
    app.run(debug=False)