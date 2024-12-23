import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GDFold2: a fast and parallelizable protein folding environment")
    
    parser.add_argument('fasta', type=str, help='Input protein sequence (.fasta format)')
    parser.add_argument('pred', type=str, help='Input geometric information (.npz format)')
    parser.add_argument('output', type=str, help='Output directory path')

    parser.add_argument('-n', dest='npose', type=int, default=1, help='The number of protein structures folded simultaneously (default: %(default)s)')
    parser.add_argument('-s', dest='steps', type=int, default=400, help='The number of optimization steps of the first stage (default: %(default)s)')
    parser.add_argument('-d', dest='device', type=str, default='cpu', help='The device to run the task (default: %(default)s)')
    parser.add_argument('-m', dest='mode', type=str, default='SPIRED', choices=['SPIRED', 'Cerebra', 'Rosetta', 'Dynamics'], 
                        help='Mode of the geometric information (default: %(default)s)')  
    args = parser.parse_args()
    return args
