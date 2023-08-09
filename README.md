# A workspace for performing reproducible experiments in neural networks.

This repository contains the following:
* An experimentation setup that abstracts away busywork
  * The Experiment class allows grouping of related runs: varying parameters across a grid
  * One can define Observables, helping standardize measurement during training.
  * Configuration of runs is made clear, all parameters tracked, RNG initialization standardized for reproducibility.
  * Much of the file i/o and naming is handled automatically
* A live visualization setup to see training in real-time
* Some common model implementations: mostly toy models for experimentation, but also...
  * A stacked Transformer in PyTorch
    * Includes settings to replicate Andrey Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT)
  * A UNet with contextual embedding based on [this repo](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)
* A GPU-capable Docker environment to run Jupyterlab

A companion tool is the ["Transformer Visor"](https://github.com/dereklarson/weight_viz), which lets you visualize the training process
for a small transformer model. See a [live demo here](https://tlab.dereklarson.info)

I'm using this codebase to explore fundamental questions of neural networks, and interpreting them
in the interpretability work from [Anthropic](https://www.anthropic.com/#papers) and [Redwood Research](https://www.redwoodresearch.org/research).
It's not intended for wide use, but I do believe this project could inform a more robust framework.
