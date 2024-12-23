# GDFold2
GDFold2 is a protein folding environment. It is designed to rapidly and parallelly fold the protein structures based on arbitrary predicted constraints, which could be freely integrated into the environment as user-defined loss functions. We provide four folding modes to match the different geometric information. You can also customize the constraints according to your specific needs.

![Dynamics path](Dynamics/dynamics.gif "Dynamics path")

## Getting Started

### Install

```bash
git clone https://github.com/Gonglab-THU/GDFold2.git
cd GDFold2
```

### GDFold2 Environment

```bash
conda env create -f environment.yml
conda activate GDFold2
```

### Usage
#### 1. GDFold2
* `fold.py`: input protein sequence (.fasta format) and predicted geometric information (.npz format) and output protein structure(s).

  ```bash
  python fold.py example/test.fasta example/test.npz example -d cuda
  ```

#### 2. FastRelax
* Please install [PyRosetta](https://www.pyrosetta.org/home/licensing-pyrosetta) first!
* `relax.py`: perform FastRelax procedure.

  ```bash
  python relax.py --input example/101M_1.pdb --output example/relax.pdb
  ```

#### 3. QAmodel
* `QAmodel/run.py`: input a directory containing multiple protein models folded by GDFold2 and output their ranking file `rank.txt` in the input directory.

  ```bash
  python QAmodel/run.py --input QAmodel/example
  ```

#### 4. Dynamics
* Step 1: run `Dynamics/pdb2cst.py` to convert two conformational states of the same protein target into geometric constraint file (`comb.npz`).
  
  ```bash
  python Dynamics/pdb2cst.py --state1 Dynamics/1ake_A.pdb --state2 Dynamics/4ake_A.pdb --output Dynamics
  ```

* Step 2: run `fold.py` to predict the possible conformations in the transition path between the two conformational states.
  
  ```bash
  python fold.py Dynamics/comb.fasta Dynamics/comb.npz Dynamics/dynamics -n 50 -m Dynamics -d cuda
  ```

## Web Server
We provide a web sever ([GDFold2](https://structpred.life.tsinghua.edu.cn/server\_gdfold2.html)) for exploring protein structural dynamics. You can copy all the characters from `Dynamics/1ake_A.pdb` and `Dynamics/4ake_A.pdb` and paste them separately into the input box of the web server for testing.

## Citation
If you use this code in your research, please cite our paper:
```
@article
author = {Mi, Tianyu and Gong, Haipeng},
title = {GDFold2: a fast and parallelizable protein folding environment with freely defined objective functions},
year = {2024},
doi = {10.1101/2024.03.13.584741}
```
