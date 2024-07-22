Comparison of Graph Pattern Quality Measures v1.0.0
-------------------------------------------------------------------------

# Description
This repository contains the source code and data used in article Pattern-Based Graph Classification: Comparison of Quality Measures and Importance of Preprocessing.

**Content**
* [Organization](#organization)
* [Installation](#installation)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [References](#references)

# Organization
This repository is composed of the following elements:

* `requirements.txt`: List of required Python packages.
* `src`: folder containing the source code
  * `ClusteringComparison.py`: script that reproduces the experiments of Section 5.2.
  * `KendallTauHistogram.py`: script that reproduces the experiments of Section 5.2.2.
  * `PairwiseComparisons.py`: script that reproduces the experiments of Section 5.3.
  * `GoldStandardComparison.py`: script that reproduces the experiments of Section 5.4.
* `data`: folder containing the input data. Each subfolder corresponds to a distinct dataset, cf. Section [Datasets](##Data).
* `results`: files produced by the processing.


# Installation

## Python and Packages
First, you need to install the `Python` language and the required packages:

1. Install the [`Python` language](https://www.python.org)
2. Download this project from GitHub and unzip.
3. Execute `pip install -r requirements.txt` to install the required packages (see also Section [Dependencies](#dependencies)).

## Non-Python Dependencies
Second, one of the dependencies, SPMF, is not a Python package, but rather a Java program, and therefore requires a specific installation process:

* Download its source code on [Philippe Fournier-Viger's website](https://www.philippe-fournier-viger.com/spmf/index.php?link=download.php).
* Follow the installation instructions provided on the [same website](https://www.philippe-fournier-viger.com/spmf/how_to_install.php).

Note that we use the JAR implementation of SPMF.
## Data

We retrieved the datasets from the [SPMF website](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php); they include:
* `MUTAG` : MUTAG dataset, representing chemical compounds and their mutagenic properties [[D'91](#references)]
* `NCI1` : NCI1 dataset, representing molecules and classified according to carcinogenicity [[W'06](#references)]
* `PTC` : PTC dataset, representing molecules and classified according to carcinogenicity [[T'03](#references)] 
* `DD` : DD dataset, representing amino acids and their interactions [[D'03](#references)]
* `IMDB-Binary` : IMDB-Binary dataset, representing movie collaboration graphs [[Y'15](#references)]

We retrieve two dataset from the [TU Dataset website](https://chrsmrrs.github.io/datasets/docs/datasets/):
* `AIDS` dataset, representing chemical compounds tested for AIDS inhibition [[R'08](#references)]
* `FRANKENSTEIN` dataset, representing chemical compounds tested and their mutagenic properties [[O'15](#references)]

The public procurement dataset contains graphs extracted from the FOPPA database:
* `FOPPA` : dataset extracted from FOPPA, a database of French public procurement notices [[P'23b](#references)]


# Usage
We provide two scripts to reproduces the expriments:

* `General.sh`: reproduces all experiments described in our paper.
* `OneDataset.sh` (dataset): reproduces the experiments concerning the specific dataset.

Each script extracts the data and then performs the associated experiments.


# Dependencies
Tested with `python` version 3.12.2 and the following packages:
* [`pandas`](https://pypi.org/project/pandas/): version 2.2.1
* [`numpy`](https://pypi.org/project/numpy/): version 1.26.4
* [`networkx`](https://pypi.org/project/networkx/): version 3.2.1
* [`sklearn`](https://pypi.org/project/scikit-learn/): version 1.2.2
* [`matplotlib`](https://pypi.org/project/matplotlib/): version 3.8.0
* [`tqdm`](https://pypi.org/project/tqdm/): version 4.66.4
* [`rbo`](https://pypi.org/project/rbo/): version 0.1.3
* [`shap`](https://pypi.org/project/shap/): version 0.45.0
* [`xgboost`](https://pypi.org/project/xgboost/): version 2.1.0
* [`scipy`](https://pypi.org/project/scipy/): version 1.11.4

Tested with `SPMF` version 2.62, which implements gSpan [[Y'02](#references)] (to mine frequent patterns)


# References
* **[D'91]** A. S. Debnath, R. L. Lopez, G. Debnath, A. Shusterman, C. Hansch. *Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. Correlation with molecular orbital energies and hydrophobicity*, Journal of Medicinal Chemistry 34(2):786–797, 1991. DOI: [10.1021/jm00106a046](https://doi.org/10.1021/jm00106a046)
* **[D'03]** P. D. Dobson, A. J. Doig. *Distinguishing enzyme structures from non-enzymes without alignments*, Journal of Molecular Biology 330(4):771–783, 2003. DOI: [10.1016/S0022-2836(03)00628-4](https://doi.org/10.1016/S0022-2836(03)00628-4)
* **[H'14']** M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle, M. Pickavet. *The Index-Based Subgraph Matching Algorithm with General Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph Enumeration*, PLoS ONE 9(5):e97896, 2014. DOI: [10.1371/journal.pone.0097896](https://doi.org/10.1371/journal.pone.0097896).
* **[O'15]** F. Orsini, P. Frasconi, L. De Raedt. *Graph invariant kernels*, 24th International Conference on Artificial Intelligence, pp. 3756–3762, 2015. DOI: [10.5555/2832747.2832773](http://doi.org/10.5555/2832747.2832773)
* **[P'23b]** L. Potin, V. Labatut, P. H. Morand & C. Largeron. *FOPPA: An Open Database of French Public Procurement Award Notices From 2010–2020*, Scientific Data, 2023, 10:303. DOI: [10.1038/s41597-023-02213-z](https://dx.doi.org/10.1038/s41597-023-02213-z) 
* **[T'03]** H. Toivonen, A. Srinivasan, R. D. King, S. Kramer, C. Helma. *Statistical evaluation of the predictive toxicology challenge 2000-2001*, Bioinformatics 19(10):1183–1193, 2003. DOI: [10.1093/bioinformatics/btg130](https://doi.org/10.1093/bioinformatics/btg130)
* **[W'06]** N. Wale, G. Karypis. *Comparison of descriptor spaces for chemical compound retrieval and classification*, 6th International Conference on Data Mining, pp. 678–689, 2006. DOI: [10.1007/s10115-007-0103-5](https://doi.org/10.1007/s10115-007-0103-5)
* **[Y'02]** X. Yan, J. Han. *gSpan: Graph-based substructure pattern mining*, IEEE International Conference on Data Mining, pp.721-724, 2002. DOI: [10.1109/ICDM.2002.1184038](https://doi.org/10.1109/ICDM.2002.1184038)
* **[Y'15]** P. Yanardag, S.V.N. Vishwanathan. *Deep Graph Kernels*, 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1365–1374, 2015. DOI: [10.1145/2783258.2783417](https://doi.org/10.1145/2783258.2783417)
