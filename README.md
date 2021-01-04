# Be Greedy in Multi-Armed Bandits

This repository contains the companion code for the following [paper]#(https://arxiv.org/TODO):

M. Jedor, J. Louëdec and V. Perchet. Be Greedy in Multi-Armed Bandits. arXiv preprint arXiv:TODO, 2021.

See also the bibtex reference below.

```
@article{jedor2021greedy,
  title={Be Greedy in Multi-Armed Bandits},
  author={Jedor, Matthieu and Lou{\"e}dec, Jonathan and Perchet, Vianney},
  journal={arXiv preprint arXiv:TODO},
  year={2021}
}
```

## Abstract

The Greedy algorithm is the simplest heuristic in sequential decision problem that carelessly takes the locally optimal choice at each round, disregarding any advantages of exploring and/or information gathering. Theoretically, it is known to sometimes have poor performances, for instance even a linear regret (with respect to the time horizon) in the standard multi-armed bandit problem. On the other hand, this heuristic performs reasonably well in practice and it even has sublinear, and even near-optimal, regret bounds in some very specific  linear contextual and Bayesian bandit models.

We build on a recent line of work and investigate bandit settings where the number of arms is relatively large and where simple greedy algorithms enjoy highly competitive performance, both in theory and in practice. We first provide a generic worst-case bound on the regret of the Greedy algorithm. When combined with some arms subsampling, we prove that it verifies near-optimal worst-case regret bounds in continuous, infinite and many-armed bandit problems. Moreover, for shorter time spans, the theoretical relative suboptimality of Greedy is even reduced.

As a consequence, we subversively claim that for many interesting problems and associated horizons, the best compromise between theoretical guarantees, practical performances and computational burden is definitely to follow the greedy heuristic. We support our claim by many numerical experiments that show significant improvements compared to the state-of-the-art, even for moderately long time horizon. 

## Implementation

In addition to the code reproducing the numerical experiments of the aforementioned paper, this repository contains a functional implementation of several bandit models along with state-of-the-art algorithms. Specifically, the following bandit models are implemented:

- Multi-armed bandits
- Bayesian multi-armed bandits
- Continuous-armed bandits
- Infinite-armed bandits
- Linear bandits
- Mortal bandits
- Budgeted bandits
- Cascading bandits

## Installation

This code is compatible with Python 3. 

The primary dependencies are:

* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)  
* [Scipy](https://www.scipy.org/)

## License

This code has a MIT license.
