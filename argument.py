import argparse

def get_args():
    parser = argparse.ArgumentParser(description="A fast and parallelizable protein folding environment")
    
    parser.add_argument('fasta', type=str, help='input protein sequence (.fasta format)')
    parser.add_argument('pred', type=str, help='input predicted geometric information (.npz format)')
    parser.add_argument('output', type=str, help='output directory path')

    parser.add_argument('-n', dest='npose', type=int, default=1, help='number of protein structures folded simultaneously (default: %(default)s)')
    parser.add_argument('-s', dest='steps', type=int, default=400, help='number of optimization steps (default: %(default)s)')
    parser.add_argument('-d', dest='device', type=str, default='cpu', help='device to run the task (default: %(default)s)')
    parser.add_argument('-m', dest='mode', type=str, default='SPIRED', choices=['SPIRED', 'Cerebra', 'Rosetta', 'Dynamics'], 
                        help='source of the geometric information (default: %(default)s)')  
    args = parser.parse_args()
    return args
