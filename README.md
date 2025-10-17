# Information Shapes Koopman Presentation

![illustration](./figures/structure.png)

### Abstract

The Koopman operator provides a powerful framework for modeling dynamical systems and has attracted growing interest in deep learning. Yet its infinite-dimensional nature makes identifying suitable finite-dimensional subspaces challenging, especially for deep architectures. We argue these difficulties are from representation learning, where latent variables fail to balance expressivity and simplicity. This tension is closely related to the information bottleneck (IB) dilemma: constructing compressed representations that are both compact and predictive. Rethinking Koopman learning through this lens, we demonstrate that latent mutual information governs enforcing simplicity, yet an overemphasis on simplicity may induce latent space collapse to few dominant modes. In contrast, expressiveness is sustained by the von Neumann entropy, which prevents such collapse and encourage sufficient modes. This insight leads us to propose an information-theoretic Lagrangian formulation that explicitly balances this tradeoff. Also, we propose a new algorithm based on the Lagrangian formulation that encourages both simplicity and expressiveness, leading to a stable and interpretable Koopman representation. Beyond quantitative evaluations, we further visualize the learned manifolds under our representations, observing empirical results consistent with our theoretical predictions. Finally, we validate our approach across a diverse range of dynamical systems, demonstrating improved performance.

### Code Structure

```bash
INFORMATIONKOOPMAN/
│
├── data/                           # Data loading and preprocessing
│   ├── physical_simulation_Dataset.py
│   └── README.md
│
├── model/                          # Core model implementations
│   └── physical_simulation/        # Physics-based simulation models
│       ├── Cylinder/
│       │   └── cylinder_model.py
│       ├── Dam/
│       │   └── dam_model.py
│       ├── base.py                 # Base model classes
│       └── utils.py                # Utility functions
│
└── train_scripts/                  # Training scripts and configs
    └── physical_simulation/
        ├── cylinder/
        │   ├── cylinder_trainer.py
        │   └── cylinder.yaml
        ├── dam/
        │   ├── dam_trainer.py
        │   └── dam.yaml
        └── trainer.py              # Shared training utilities
```

### Install Dependencies

``` 
pip install -U -r requirements.txt
```

or 

``` 
conda env create -f environment.yml
conda activate InfoKoopman
```

*Note: You may need to install pytorch seperately for GPU support: 

``` pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121``` 


## Code Availability

Currently, this repository only contains the physical simulation experiments. The remaining experiment code will be released upon the paper’s acceptance.

## Paper Link and Citation

**Paper:** [Information Shapes Koopman Representation](https://arxiv.org/abs/2510.13025)

If you find this work useful in your research, please consider citing:

```bibtex
@misc{cheng2025informationshapeskoopmanrepresentation,
      title={Information Shapes Koopman Representation}, 
      author={Xiaoyuan Cheng and Wenxuan Yuan and Yiming Yang and Yuanzhao Zhang and Sibo Cheng and Yi He and Zhuo Sun},
      year={2025},
      eprint={2510.13025},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.13025}, 
}
```

## Contribution

This work is the result of close collaboration between all authors.

- **Xiaoyuan Cheng** led the overall project, proposed the core ideas, and guided both the theoretical development and experimental implementation.
- **Wenxuan Yuan** led the physical simulation experiments and created the key methodological figures.
- **Yiming Yang, Yuanzhao Zhang, Sibo Cheng, Yi He, and Zhuo Sun** contributed to writing and revising the manuscript.

## Contact

If you encounter any issues, please open an issue on GitHub.
