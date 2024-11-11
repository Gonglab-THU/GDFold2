# GDFold2
GDFold2 is a protein folding environment. It is designed to rapidly and parallelly fold the protein based on arbitrary predicted constraints, which could be freely integrated into the environment as user-defined loss functions. We provide three folding modes to match the different sources of predicted information, and users can also customize constraints according to the specific needs.

![Dynamics path](Dynamics/dynamics.gif, "Dynamics path")

## Install software on Linux

1. download `GDFold2`

```bash
git clone https://github.com/Gonglab-THU/GDFold2.git
cd GDFold2
```

2. install `Anaconda` / `Miniconda` software

3. install Python packages

```bash
conda create -n GDFold2 python=3.8.20
conda activate GDFold2
conda install numpy biopython pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install scipy tqdm numba pandas
conda install biopandas -c conda-forge
conda install pyg -c pyg
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1%2Bcu113.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.1%2Bcu113.html
```

4. install `PyRosetta` at [PyRosetta LICENSE](https://www.pyrosetta.org/home/licensing-pyrosetta)

## Usage
### GDFold2
* `fold.py`: input protein sequence (`.fasta file`) and predicted geometric information (`.npz file`) and output protein structure(s).

  Example:
  ```bash
  python fold.py example/test.fasta example/test.npz example -d cuda
  ```

  Run `python fold.py -h` for more detailed options.


* `relax.py`: perform FastRelax procedure on protein models.
  
  Example:
  ```bash
  python relax.py --input example/101M_1.pdb --output example/relax.pdb
  ```
  Run `python relax.py -h` for more detailed options.

### QAmodel
`run.py`: input a directory containing multiple protein sturctures folded by GDFold2 and output the ranking file `rank.txt`.
  
Example:
```bash
python QAmodel/run.py --input QAmodel/example
```
Run `python QAmodel/run.py -h` for more detailed options.

### Dynamics
* `pdb2cst.py`: input two conformational states (`.pdb file`) of the same protein target and output the protein sequence file (`comb.fasta`) and the geometric constraint file (`comb.npz`) in the designated directory path.
  
  Example:
  ```bash
  python Dynamics/pdb2cst.py --state1 Dynamics/1ake_A.pdb --state2 Dynamics/4ake_A.pdb --output Dynamics
  ```
  Run `python Dynamics/pdb2cst.py -h` for more detailed options.

* Then, run `fold.py` to fold the possible conformations during the transition path of the two conformational states:
  ```bash
  python fold.py Dynamics/comb.fasta Dynamics/comb.npz Dynamics/dynamics -n 50 -m Dynamics -d cuda
  ```
