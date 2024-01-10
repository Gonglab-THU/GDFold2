# GDFold2
GDFold2 is a protein folding environment. It is designed to rapidly and parallelly fold the protein based on arbitrary predicted constraints, which could be freely integrated into the environment as user-defined loss functions. We provide three folding modes to match the different sources of predicted information, and users can also customize constraints according to the specific needs.

## Install software on Linux

1. download `GDFold2`

```bash
git clone https://github.com/Gonglab-THU/GDFold2.git
cd GDFold2
```

2. install `Anaconda` / `Miniconda` software

3. install Python packages

```bash
conda create -n gdfold2 python=3.11
conda activate gdfold2
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install biopython
```

## Usage
We provide two scripts in GDFold2:
* `fold.py`: takes protein sequence and predicted geometric information as inputs and predict protein structure (centroid model).
* `relax.py`: converts centroid model into full-atom structure and perform FastRelax.

### Folding
Run `python fold.py -h` for more detailed options.
```bash
usage: fold.py [-h] [-n NPOSE] [-s STEPS] [-d DEVICE] [-m {Cerebra,SPIRED,Rosetta}] fasta pred output

GDFold2: a fast and parallelizable protein folding environment

positional arguments:
  fasta                 input protein sequence (.fasta format)
  pred                  input predicted geometric information (.npz format)
  output                output directory name

optional arguments:
  -h, --help            show this help message and exit
  -n NPOSE              number of structures to predict simultaneously, default=1
  -s STEPS              number of optimization steps, default=400
  -d DEVICE             device to run the task, default=cpu
  -m {Cerebra,SPIRED,Rosetta}
                        source of the predicted geometric information, default=Cerebra
```

Example:
```bash
python fold.py example/test.fasta example/test.npz example -m SPIRED -d cuda:0
```

### FastRelax

Run `python relax.py -h` for more detailed options.
```bash
usage: relax.py [-h] [--input INPUT] [--output OUTPUT] [--repeat REPEAT] [--cycle CYCLE]

Perform FastRelax on predicted protein structure

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    input unrelaxed model
  --output OUTPUT  output model (.pdb format)
  --repeat REPEAT  number of repeats, default=2
  --cycle CYCLE    number of max cycles, default=200
```

:exclamation: install `PyRosetta` at [PyRosetta LICENSE](https://www.pyrosetta.org/home/licensing-pyrosetta)

Example:
```bash
python relax.py --input example/101M_1.pdb --output example/relax.pdb
```
