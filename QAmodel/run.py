import os
import sys
import torch
import argparse
import torch_geometric.data as GeoData
from model import MQAModel
from torch_geometric.data import Batch
from tqdm import tqdm
from graphs import *

parser = argparse.ArgumentParser(description="Protein Structure Quality Assessment Model")
parser.add_argument('--input', type=str, help='Directory containing multiple structures of the same target protein')
args = parser.parse_args()

def main():
    print("Initializing QA model...")
    model = MQAModel((12, 3), (100, 16), (38, 1), (38, 1), seq_in=True)
    best_dict = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best.pth')
    model.load_state_dict(torch.load(best_dict, map_location='cpu'))
    model.eval()
    poses = [t for t in os.listdir(args.input) if t.endswith('.pdb')]
    dataset = []
    for pose in tqdm(poses):
        try:
            seq, coords = load_pdb(os.path.join(args.input, pose))
        except:
            print(f"File ({pose}) format is illegal!")
            sys.exit(0)
        graph = GraphData(coords, seq, [0, 0, 0]).features()
        data = GeoData.Data(name=pose, seq=graph['seq'].long(),
                            node_s=graph['node_s'], node_v=graph['node_v'],
                            edge_s=graph['edge_s'], edge_v=graph['edge_v'],
                            edge_index=graph['edge_index'])
        dataset.append(data)
    batch = Batch.from_data_list(dataset)
    with torch.no_grad():
        names = batch.name
        nodes = (batch.node_s, batch.node_v)
        edges = (batch.edge_s, batch.edge_v)
        scores = model(nodes, batch.edge_index, edges, seq=batch.seq, batch=batch.batch)
    scores, names = zip(*sorted(zip(scores, names), reverse=True))

    with open(os.path.join(args.input, 'rank.txt'), 'w') as f:
        f.write("{:^4}\t{:^12}\t{:^8}\n".format('Rank', 'Name', 'QA-Score'))
        for i in range(len(scores)):
            f.write("{:^4}\t{:^12}\t{:^8.4f}\n".format(i+1, names[i], scores[i]))
    print(f"Evaluation result (rank.txt) is saved in {args.input}")

if __name__ == '__main__':
    main()
