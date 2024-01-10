import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GDFold2: a fast and parallelizable protein folding environment")
    
    parser.add_argument('fasta', type=str, help='input protein sequence (.fasta format)')
    parser.add_argument('pred', type=str, help='input predicted geometric information (.npz format)')
    parser.add_argument('output', type=str, help='output directory name')

    parser.add_argument('-n', dest='npose', type=int, default=1, help='number of structures to predict simultaneously, default=1')
    parser.add_argument('-s', dest='steps', type=int, default=400, help='number of optimization steps, default=400')
    parser.add_argument('-d', dest='device', type=str, default='cpu', help='device to run the task, default=cpu')
    parser.add_argument('-m', dest='mode', type=str, default='Cerebra', choices=['Cerebra', 'SPIRED', 'Rosetta'], 
                        help='source of the predicted geometric information, default=Cerebra')  
    args = parser.parse_args()
    return args
