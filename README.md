# Latent Adversarial Training
This repository accompanies the paper [Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/abs/2403.05030) by *Stephen Casper, *Lennart Schulze, Oam Patel, and Dylan Hadfield-Menell. 

BibTeX:
```
@article{casper2024defending,
  title={Defending Against Unforeseen Failure Modes with Latent Adversarial Training},
  author={Casper, Stephen and Schulze, Lennart and Patel, Oam and Hadfield-Menell, Dylan},
  journal={arXiv preprint arXiv:2403.05030},
  year={2024}
}
```

If you are looking for the more recent repository for the paper [Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs](https://arxiv.org/abs/2407.15549) please see [https://github.com/aengusl/latent-adversarial-training](https://github.com/aengusl/latent-adversarial-training).

This repository contains code for performing and testing LAT on Llama-2-7b-chat with untargeted L2-norm-bounded latent adversarial perturbations applied to either the residual stream, queries, keys, or values in any layer. 

![fig1](lat_fig1.png)

## Setup

First, clone and navigate to the repo.

```
mkdir models
mkdir results
pip install -r requirements.txt
```

Then paste in your HF token to download Llama-2 in ```lat.py``` in the line saying ```TOKEN=''```

## Use

### Finetune initial model

Finetune Llama-2-7b-chat on 20k examples from the [Anthropic-HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset (a mixture of both preferred and rejected responses) with some examples poisoned to insert trojans. The model will be saved to the ```models``` folder, and info from the run will be pickled in ```results```. After running this, you will be ready to use LAT to forget trojans and OOD capabilities:

```python lat.py --epochs=2 --run_id=initial --save=True```

### Finetune with AT/LAT

There are a variety of ways to perform AT/LAT.

Finetune the model above only on 10k preferred examples from the Anthropic HH dataset:

```python lat.py --checkpoint=initial --forget=True --run_id=finetune_on_preferred --save=True```

Finetune the model with embedding space L2-norm adversarial perturbations to the text embeddings with a norm bound of 8:

```python lat.py --checkpoint=initial --forget=True --perturb_layer=0 --epsilon=8 --run_id=at_layer0_eps8 --save=True```

Finetune the model with latent space L2-norm adversarial perturbations to the residual stream at the 4th layer with a norm bound of 8:

```python lat.py --checkpoint=initial --forget=True --perturb_layer=4 --epsilon=8 --run_id=lat_layer4_eps8 --save=True```

Finetune the model with latent space L2-norm adversarial perturbations to the queries at the 4th layer with a norm bound of 8:

```python lat.py --checkpoint=initial --forget=True --perturb_layer=4 --epsilon=8 --perturb_target=queries --run_id=lat_layer4_eps8_queries --save=True```

Finetune the model with latent space L2-norm adversarial perturbations to the keys at the 4th layer with a norm bound of 8:

```python lat.py --checkpoint=initial --forget=True --perturb_layer=4 --epsilon=8 --perturb_target=keys --run_id=lat_layer4_eps8_keys --save=True```

Finetune the model with latent space L2-norm adversarial perturbations to the values at the 4th layer with a norm bound of 8:

```python lat.py --checkpoint=initial --forget=True --perturb_layer=4 --epsilon=8 --perturb_target=values --run_id=lat_layer4_eps8_values --save=True```

Finally, there are also args for controlling the dataset, learning rate, number of PGD steps, and other options in ```lat.py```.
