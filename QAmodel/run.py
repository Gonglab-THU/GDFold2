import os
import torch
import argparse
from model import MQAModel
from torch_geometric.data import Batch
from graphs import Protein, GraphData
import torch_geometric.data as GeoData

parser = argparse.ArgumentParser(description="Protein Structure Quality Assessment")
parser.add_argument('--input', type=str, help='directory containing multiple structures generated by GDFold2')
parser.add_argument('--model', type=str, default='models/best.pth', help='trained model, default=models/best.pth')
args = parser.parse_args()

FILEPATH = os.path.dirname(os.path.realpath(__file__))
MODEL = f'{FILEPATH}/{args.model}'

def main():
    print("Loading the QA model...")
    model = MQAModel((12, 3), (100, 16), (38, 1), (38, 1), seq_in=True)
    model.load_state_dict(torch.load(MODEL, map_location='cpu'))
    model.eval()

    print("Converting proteins into graphs...")
    poses = os.listdir(args.input)
    dataset = []
    for pose in poses:
        if pose.endswith('.pdb'):
            name = pose.split('.pdb')[0]
            pose_path = f'{args.input}/{pose}'
            pro = Protein(pose_path)
            coords = pro.coords
            seq = pro.seq
            graph = GraphData(coords, seq, [0, 0, 0]).features()
            data = GeoData.Data(name=name, seq=graph['seq'].long(),
                                node_s=graph['node_s'], node_v=graph['node_v'],
                                edge_s=graph['edge_s'], edge_v=graph['edge_v'],
                                edge_index=graph['edge_index'])
            dataset.append(data)
    
    print("Evaluating...")
    batch = Batch.from_data_list(dataset)
    with torch.no_grad():
        names = batch.name
        nodes = (batch.node_s, batch.node_v)
        edges = (batch.edge_s, batch.edge_v)
        scores = model(nodes, batch.edge_index, edges, 
                       seq=batch.seq, batch=batch.batch)
    scores, names = zip(*sorted(zip(scores, names), reverse=True))

    with open(f'{args.input}/rank.txt', 'w') as f:
        f.write("{:^4}\t{:^12}\n".format('Rank', 'Name'))
        for i in range(len(scores)):
            f.write("{:^4}\t{:^12}\n".format(i+1, names[i]))
    print(f"Evaluation result is stored in {args.input}/rank.txt")

if __name__ == '__main__':
    main()
