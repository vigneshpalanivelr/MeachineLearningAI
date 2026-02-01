# CS11 - Generative Adversarial Networks (GAN)

**Course:** Advanced Deep Learning (S1-25_AIMLCZG513)
**Instructor:** Prof. Sugata Ghosal
**Institution:** BITS Pilani, Pilani Campus
**Email:** sugata.ghosal@pilani.bits-pilani.ac.in

---

## Table of Contents

- [CS11 - Generative Adversarial Networks (GAN)](#cs11---generative-adversarial-networks-gan)
  - [Table of Contents](#table-of-contents)
  - [References and Links](#references-and-links)
    - [Academic Papers and Publications](#academic-papers-and-publications)
    - [Tutorials and Educational Resources](#tutorials-and-educational-resources)
    - [Interactive Demos and Visualizations](#interactive-demos-and-visualizations)
    - [Implementation Libraries and Code](#implementation-libraries-and-code)
    - [Datasets](#datasets)
    - [Video Lectures and Presentations](#video-lectures-and-presentations)
    - [Blog Posts and Articles](#blog-posts-and-articles)
    - [Research Groups and Labs](#research-groups-and-labs)
    - [Tools and Software](#tools-and-software)
    - [Additional Resources](#additional-resources)
    - [Course-Specific References](#course-specific-references)
    - [Key Blogs and Online References](#key-blogs-and-online-references)
  - [1. Introduction](#1-introduction)
    - [1.1 Motivation and Evolution](#11-motivation-and-evolution)
    - [1.2 GANs vs Other Generative Models](#12-gans-vs-other-generative-models)
  - [2. Implicit Models](#2-implicit-models)
    - [2.1 Definition](#21-definition)
    - [2.2 Key Characteristics](#22-key-characteristics)
  - [3. GAN Architecture](#3-gan-architecture)
    - [3.1 Generator](#31-generator)
    - [3.2 Discriminator](#32-discriminator)
    - [3.3 Training Process](#33-training-process)
  - [4. GAN Theory](#4-gan-theory)
    - [4.1 Loss Function](#41-loss-function)
    - [4.2 Minimax Game](#42-minimax-game)
    - [4.3 Bayes-Optimal Discriminator](#43-bayes-optimal-discriminator)
    - [4.4 Generator Objective](#44-generator-objective)
  - [5. GAN Pseudocode](#5-gan-pseudocode)
  - [6. Evaluation Metrics](#6-evaluation-metrics)
    - [6.1 Inception Score (IS)](#61-inception-score-is)
    - [6.2 Fréchet Inception Distance (FID)](#62-fréchet-inception-distance-fid)
  - [7. Deep Convolutional GAN (DCGAN)](#7-deep-convolutional-gan-dcgan)
    - [7.1 Architecture Guidelines](#71-architecture-guidelines)
      - [Generator Architecture](#generator-architecture)
      - [Discriminator Architecture](#discriminator-architecture)
    - [7.2 Key Results](#72-key-results)
  - [8. Improved Training Techniques](#8-improved-training-techniques)
    - [8.1 Feature Matching](#81-feature-matching)
    - [8.2 Minibatch Discrimination](#82-minibatch-discrimination)
    - [8.3 Historical Averaging](#83-historical-averaging)
    - [8.4 Virtual Batch Normalization](#84-virtual-batch-normalization)
    - [8.5 One-sided Label Smoothing](#85-one-sided-label-smoothing)
  - [9. Advanced GAN Variants](#9-advanced-gan-variants)
    - [9.1 Wasserstein GAN (WGAN)](#91-wasserstein-gan-wgan)
    - [9.2 WGAN with Gradient Penalty (WGAN-GP)](#92-wgan-with-gradient-penalty-wgan-gp)
    - [9.3 Spectral Normalization GAN (SNGAN)](#93-spectral-normalization-gan-sngan)
    - [9.4 Self-Attention GAN (SAGAN)](#94-self-attention-gan-sagan)
    - [9.5 Progressive GAN](#95-progressive-gan)
  - [10. Conditional GANs](#10-conditional-gans)
    - [10.1 Conditional GAN (cGAN)](#101-conditional-gan-cgan)
    - [10.2 Pix2Pix](#102-pix2pix)
    - [10.3 CycleGAN](#103-cyclegan)
  - [11. StyleGAN Series](#11-stylegan-series)
    - [11.1 StyleGAN](#111-stylegan)
    - [11.2 StyleGAN2](#112-stylegan2)
    - [11.3 StyleGAN3](#113-stylegan3)
  - [12. InfoGAN](#12-infogan)
  - [13. Projected GAN](#13-projected-gan)
  - [14. Mode Collapse and Solutions](#14-mode-collapse-and-solutions)
    - [1. Minibatch Discrimination](#1-minibatch-discrimination)
    - [2. Unrolled GAN](#2-unrolled-gan)
    - [3. Mode Regularization](#3-mode-regularization)
    - [4. Diversity-Promoting Losses](#4-diversity-promoting-losses)
    - [5. Multiple Generators](#5-multiple-generators)
    - [6. Progressive Training](#6-progressive-training)
    - [7. Improved Architectures](#7-improved-architectures)
  - [15. Comparison Tables](#15-comparison-tables)
    - [Generative Model Comparison](#generative-model-comparison)
    - [GAN Variant Comparison](#gan-variant-comparison)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Lipschitz Enforcement Methods](#lipschitz-enforcement-methods)
    - [Conditional GAN Variants](#conditional-gan-variants)
  - [Appendix: Mathematical Derivations](#appendix-mathematical-derivations)
    - [A. Jensen-Shannon Divergence in GANs](#a-jensen-shannon-divergence-in-gans)
    - [B. Wasserstein Distance via Dual Form](#b-wasserstein-distance-via-dual-form)
    - [C. Gradient Penalty Derivation](#c-gradient-penalty-derivation)
  - [Appendix: Training Tips and Best Practices](#appendix-training-tips-and-best-practices)
    - [General Tips](#general-tips)
    - [Debugging Checklist](#debugging-checklist)
    - [Architecture Guidelines (DCGAN)](#architecture-guidelines-dcgan)

---

## References and Links

### Academic Papers and Publications

| # | Title | Authors | Venue/Year | Link | Notes |
|---|-------|---------|------------|------|-------|
| 1 | Generative Adversarial Networks | I.J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio | NIPS 2014 | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) | Original GAN paper |
| 2 | Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks | A. Radford, L. Metz, S. Chintala | ICLR 2016 | [arXiv:1511.06434](https://arxiv.org/abs/1511.06434) | DCGAN architecture |
| 3 | Coupled Generative Adversarial Networks | M.-Y. Liu, O. Tuzel | NIPS 2016 | [Paper](https://papers.nips.cc/paper/2016) | CoGAN |
| 4 | Conditional Generative Adversarial Nets | M. Mirza, S. Osindero | 2014 | [arXiv:1411.1784](https://arxiv.org/abs/1411.1784) | cGAN |
| 5 | Improved Techniques for Training GANs | T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, X. Chen | NIPS 2016 | [arXiv:1606.03498](https://arxiv.org/abs/1606.03498) | Inception Score, training improvements |
| 6 | InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets | X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, P. Abbeel | NIPS 2016 | [arXiv:1606.03657](https://arxiv.org/abs/1606.03657) | Unsupervised disentanglement |
| 7 | Wasserstein GAN | M. Arjovsky, S. Chintala, L. Bottou | ICML 2017 | [arXiv:1701.07875](https://arxiv.org/abs/1701.07875) | Earth Mover's distance |
| 8 | Improved Training of Wasserstein GANs | I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, A. Courville | NIPS 2017 | [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) | WGAN-GP, gradient penalty |
| 9 | GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium | M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, S. Hochreiter | NIPS 2017 | [arXiv:1706.08500](https://arxiv.org/abs/1706.08500) | FID metric |
| 10 | Image-to-Image Translation with Conditional Adversarial Networks | P. Isola, J.-Y. Zhu, T. Zhou, A.A. Efros | CVPR 2017 | [arXiv:1611.07004](https://arxiv.org/abs/1611.07004) | Pix2Pix |
| 11 | Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks | J.-Y. Zhu, T. Park, P. Isola, A.A. Efros | ICCV 2017 | [arXiv:1703.10593](https://arxiv.org/abs/1703.10593) | CycleGAN |
| 12 | Progressive Growing of GANs for Improved Quality, Stability, and Variation | T. Karras, T. Aila, S. Laine, J. Lehtinen | ICLR 2018 | [arXiv:1710.10196](https://arxiv.org/abs/1710.10196) | Progressive training |
| 13 | Spectral Normalization for Generative Adversarial Networks | T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida | ICLR 2018 | [arXiv:1802.05957](https://arxiv.org/abs/1802.05957) | SNGAN |
| 14 | Self-Attention Generative Adversarial Networks | H. Zhang, I. Goodfellow, D. Metaxas, A. Odena | ICML 2019 | [arXiv:1805.08318](https://arxiv.org/abs/1805.08318) | SAGAN |
| 15 | A Style-Based Generator Architecture for Generative Adversarial Networks | T. Karras, S. Laine, T. Aila | CVPR 2019 | [arXiv:1812.04948](https://arxiv.org/abs/1812.04948) | StyleGAN |
| 16 | Analyzing and Improving the Image Quality of StyleGAN | T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, T. Aila | CVPR 2020 | [arXiv:1912.04958](https://arxiv.org/abs/1912.04958) | StyleGAN2 |
| 17 | Alias-Free Generative Adversarial Networks | T. Karras, M. Aittala, S. Laine, E. Härkönen, J. Hellsten, J. Lehtinen, T. Aila | NeurIPS 2021 | [arXiv:2106.12423](https://arxiv.org/abs/2106.12423) | StyleGAN3 |
| 18 | Generative Visual Manipulation on the Natural Image Manifold | J.-Y. Zhu, P. Krähenbühl, E. Shechtman, A.A. Efros | ECCV 2016 | [arXiv:1609.03552](https://arxiv.org/abs/1609.03552) | iGAN |
| 19 | Learning to Discover Cross-Domain Relations with Generative Adversarial Networks | T. Kim, M. Cha, H. Kim, J.K. Lee, J. Kim | ICML 2017 | [arXiv:1703.05192](https://arxiv.org/abs/1703.05192) | DiscoGAN |
| 20 | DualGAN: Unsupervised Dual Learning for Image-to-Image Translation | Z. Yi, H. Zhang, P. Tan, M. Gong | ICCV 2017 | [arXiv:1704.02510](https://arxiv.org/abs/1704.02510) | DualGAN |

### Tutorials and Educational Resources

| # | Resource | Author/Source | Type | Link | Description |
|---|----------|---------------|------|------|-------------|
| 1 | NIPS 2016 Tutorial: Generative Adversarial Networks | Ian Goodfellow | Tutorial Paper | [arXiv:1701.00160](https://arxiv.org/abs/1701.00160) | Comprehensive GAN tutorial |
| 2 | GAN Series by Jonathan Hui | Jonathan Hui | Blog Series | [Medium](https://jonathan-hui.medium.com/gan-gan-series-2d279f906e7b) | Detailed GAN explanations |
| 3 | Deep Learning Book - Chapter 20 | I. Goodfellow, Y. Bengio, A. Courville | Textbook | [DeepLearningBook](https://www.deeplearningbook.org/) | Generative Models chapter |
| 4 | CS236: Deep Generative Models | Stanford University | Course | [Course Website](https://deepgenerativemodels.github.io/) | Stanford DGM course |
| 5 | MIT 6.S191: Deep Learning | MIT | Course | [YouTube](https://www.youtube.com/watch?v=yFBFl1cLYx8) | GAN lecture |
| 6 | GAN Hacks | Soumith Chintala et al. | GitHub Repo | [GitHub](https://github.com/soumith/ganhacks) | Training tips and tricks |
| 7 | The GAN Zoo | Avinash Hindupur | Collection | [GitHub](https://github.com/hindupuravinash/the-gan-zoo) | Comprehensive GAN variants list |
| 8 | How to Train a GAN? Tips and tricks | Google Developers | Tutorial | [Google ML](https://developers.google.com/machine-learning/gan) | Practical training guide |

### Interactive Demos and Visualizations

| # | Demo | Description | Link | Features |
|---|------|-------------|------|----------|
| 1 | GAN Lab | Interactive GAN visualization | [GAN Lab](https://poloclub.github.io/ganlab/) | Real-time training visualization |
| 2 | GAN Playground | Interactive GAN training | [Playground](http://rll.berkeley.edu/deeprlcourse/docs/gan_playground.html) | Experiment with 2D distributions |
| 3 | This Person Does Not Exist | StyleGAN-generated faces | [Website](https://thispersondoesnotexist.com/) | Random face generation |
| 4 | Which Face is Real? | AI detection game | [Website](https://www.whichfaceisreal.com/) | Test AI-generated faces |
| 5 | Edges2Cats | Pix2Pix demo | [Demo](https://affinelayer.com/pixsrv/) | Draw cats from edges |
| 6 | Edges2Shoes | Pix2Pix demo | [Demo](https://affinelayer.com/pix2pix/) | Interactive image translation |
| 7 | GauGAN | NVIDIA Demo | [NVIDIA](https://www.nvidia.com/en-us/research/ai-playground/) | Landscape generation |

### Implementation Libraries and Code

| # | Library/Repository | Language/Framework | Link | Description |
|---|-------------------|-------------------|------|-------------|
| 1 | PyTorch-GAN | PyTorch | [GitHub](https://github.com/eriklindernoren/PyTorch-GAN) | Collection of GAN implementations |
| 2 | TensorFlow-GAN (TF-GAN) | TensorFlow | [GitHub](https://github.com/tensorflow/gan) | Official TensorFlow GAN library |
| 3 | Keras-GAN | Keras | [GitHub](https://github.com/eriklindernoren/Keras-GAN) | Keras GAN implementations |
| 4 | StyleGAN2-ADA PyTorch | PyTorch | [GitHub](https://github.com/NVlabs/stylegan2-ada-pytorch) | Official StyleGAN2-ADA |
| 5 | StyleGAN3 | PyTorch | [GitHub](https://github.com/NVlabs/stylegan3) | Official StyleGAN3 |
| 6 | Progressive GAN | TensorFlow | [GitHub](https://github.com/tkarras/progressive_growing_of_gans) | Official Progressive GAN |
| 7 | Pix2Pix | PyTorch | [GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | Pix2Pix implementation |
| 8 | CycleGAN | PyTorch | [GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | CycleGAN implementation |
| 9 | WGAN-GP | PyTorch | [GitHub](https://github.com/caogang/wgan-gp) | WGAN-GP implementation |
| 10 | BigGAN PyTorch | PyTorch | [GitHub](https://github.com/ajbrock/BigGAN-PyTorch) | BigGAN implementation |

### Datasets

| # | Dataset | Size | Resolution | Link | Common Use Cases |
|---|---------|------|------------|------|------------------|
| 1 | CelebA | 202,599 images | 178×218 | [Website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Face generation, attributes |
| 2 | CelebA-HQ | 30,000 images | 1024×1024 | [GitHub](https://github.com/tkarras/progressive_growing_of_gans) | High-quality face generation |
| 3 | FFHQ (Flickr-Faces-HQ) | 70,000 images | 1024×1024 | [GitHub](https://github.com/NVlabs/ffhq-dataset) | StyleGAN training |
| 4 | LSUN | ~59M images | Various | [Website](https://www.yf.io/p/lsun) | Scene generation (bedrooms, churches) |
| 5 | ImageNet | 14M images | Various | [Website](https://www.image-net.org/) | Large-scale conditional generation |
| 6 | CIFAR-10 | 60,000 images | 32×32 | [Website](https://www.cs.toronto.edu/~kriz/cifar.html) | Benchmarking |
| 7 | MNIST | 70,000 images | 28×28 | [Website](http://yann.lecun.com/exdb/mnist/) | Simple GAN experiments |
| 8 | Fashion-MNIST | 70,000 images | 28×28 | [GitHub](https://github.com/zalandoresearch/fashion-mnist) | Clothing generation |

### Video Lectures and Presentations

| # | Title | Presenter | Venue/Platform | Link | Duration |
|---|-------|-----------|----------------|------|----------|
| 1 | Introduction to GANs | Ian Goodfellow | NIPS 2016 | [YouTube](https://www.youtube.com/watch?v=AJVyzd0rqdc) | ~3 hours |
| 2 | Generative Adversarial Networks | Pieter Abbeel | Berkeley Deep RL | [YouTube](https://www.youtube.com/watch?v=rh4D9Kw8RYs) | ~1 hour |
| 3 | GANs for Good | Ian Goodfellow | Google I/O | [YouTube](https://www.youtube.com/watch?v=pWAc9B2zJS4) | ~40 min |
| 4 | MIT 6.S191: Deep Generative Modeling | Alexander Amini | MIT | [YouTube](https://www.youtube.com/watch?v=rZufA635dq4) | ~45 min |
| 5 | StyleGAN Explained | Two Minute Papers | YouTube | [YouTube](https://www.youtube.com/watch?v=kSLJriaOumA) | ~5 min |
| 6 | CycleGAN Explained | Two Minute Papers | YouTube | [YouTube](https://www.youtube.com/watch?v=Fkqf3dS9Cqw) | ~5 min |

### Blog Posts and Articles

| # | Title | Author | Platform | Link | Topic |
|---|-------|--------|----------|------|-------|
| 1 | Understanding Generative Adversarial Networks | Joseph Rocca | Towards Data Science | [Medium](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29) | GAN basics |
| 2 | From GAN to WGAN | Lilian Weng | Blog | [lilianweng.github.io](https://lilianweng.github.io/posts/2017-08-20-gan/) | Wasserstein GAN |
| 3 | GAN — Wasserstein GAN & WGAN-GP | Jonathan Hui | Medium | [Medium](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490) | WGAN variants |
| 4 | StyleGAN - Official NVIDIA Blog | NVIDIA | NVIDIA Blog | [NVIDIA](https://developer.nvidia.com/blog/stylegan-a-style-based-generator-architecture-for-gans/) | StyleGAN architecture |
| 5 | The GAN Landscape | GAN Lab Team | Distill | [Distill](https://distill.pub/2019/gan-open-problems/) | GAN challenges |
| 6 | How to Train a GAN | Soumith Chintala | GitHub | [GitHub](https://github.com/soumith/ganhacks) | Practical tips |

### Research Groups and Labs

| # | Institution/Group | Focus Area | Link | Notable Contributions |
|---|------------------|------------|------|----------------------|
| 1 | NVIDIA Research | Computer Graphics, AI | [Website](https://www.nvidia.com/en-us/research/) | StyleGAN series, Progressive GAN |
| 2 | Google Brain | Deep Learning | [Website](https://research.google/teams/brain/) | BigGAN, Self-Attention GAN |
| 3 | DeepMind | AI Research | [Website](https://www.deepmind.com/) | Various GAN applications |
| 4 | OpenAI | AI Research | [Website](https://openai.com/research/) | Improved training techniques |
| 5 | Berkeley AI Research (BAIR) | AI, Robotics | [Website](https://bair.berkeley.edu/) | CycleGAN, Pix2Pix |
| 6 | MIT CSAIL | Computer Science | [Website](https://www.csail.mit.edu/) | Various GAN research |
| 7 | Stanford AI Lab | AI Research | [Website](https://ai.stanford.edu/) | GAN theory and applications |

### Tools and Software

| # | Tool | Purpose | Link | Platform |
|---|------|---------|------|----------|
| 1 | TensorBoard | Visualization | [Website](https://www.tensorflow.org/tensorboard) | TensorFlow |
| 2 | Weights & Biases | Experiment tracking | [Website](https://wandb.ai/) | Framework agnostic |
| 3 | Comet.ml | ML experiment tracking | [Website](https://www.comet.ml/) | Framework agnostic |
| 4 | Neptune.ai | ML metadata store | [Website](https://neptune.ai/) | Framework agnostic |
| 5 | MLflow | ML lifecycle | [Website](https://mlflow.org/) | Framework agnostic |
| 6 | DVC | Data version control | [Website](https://dvc.org/) | Framework agnostic |

### Additional Resources

| # | Resource Type | Description | Link | Notes |
|---|--------------|-------------|------|-------|
| 1 | Papers With Code | GAN papers with implementations | [Website](https://paperswithcode.com/task/image-generation) | Community-driven |
| 2 | arXiv Sanity | Paper search and recommendations | [Website](http://arxiv-sanity.com/) | GAN paper discovery |
| 3 | Distill Pub | Visual explanations | [Website](https://distill.pub/) | Interactive articles |
| 4 | Reddit r/MachineLearning | Community discussions | [Reddit](https://www.reddit.com/r/MachineLearning/) | Latest developments |
| 5 | GAN Timeline | Historical overview | [Website](https://github.com/dongb5/GAN-Timeline) | Evolution of GANs |
| 6 | Awesome GANs | Curated list | [GitHub](https://github.com/nightrome/really-awesome-gan) | Comprehensive resources |

### Course-Specific References

| # | Topic | Reference Material | Location | Notes |
|---|-------|-------------------|----------|-------|
| 1 | Lecture Slides | CS11-12 VAE Compressed | Course Materials | Sessions 10-12 |
| 2 | Lecture Recording | January 11, 2026 | Course Platform | Prof. Sugata Ghosal |
| 3 | Course Textbook | Deep Learning (Goodfellow et al.) | Chapter 20 | Generative Models |
| 4 | Assignments | VQ-VAE and VAE | Course Platform | Upcoming assignment |

### Key Blogs and Online References

| # | URL | Description | Content Type |
|---|-----|-------------|--------------|
| 1 | https://jonathan-hui.medium.com/gan-gan-series-2d279f906e7b | Comprehensive GAN series | Tutorial |
| 2 | https://developers.google.com/machine-learning/gan | Google's GAN guide | Educational |
| 3 | https://www.whichfaceisreal.com/learn.html | Learn to detect AI faces | Interactive |
| 4 | https://affinelayer.com/pix2pix/ | Pix2Pix interactive demo | Demo |
| 5 | https://poloclub.github.io/ganlab/ | GAN Lab visualization | Interactive Tool |

---

## 1. Introduction

### 1.1 Motivation and Evolution

Generative Adversarial Networks (GANs) represent a revolutionary approach to generative modeling, introduced by Ian Goodfellow et al. in 2014. Unlike previous generative models that explicitly model probability distributions, GANs learn to generate data through an adversarial process.

**Evolution Timeline:**

| Year | Milestone | Contribution |
|------|-----------|--------------|
| 2014 | Original GAN | Goodfellow et al. - Introduced adversarial training |
| 2015 | DCGAN | Radford et al. - Deep Convolutional architecture |
| 2016 | InfoGAN, WGAN | Information maximization, Wasserstein distance |
| 2017 | WGAN-GP, Pix2Pix | Gradient penalty, paired image translation |
| 2018 | Progressive GAN, StyleGAN | Progressive growing, style-based generation |
| 2019 | StyleGAN2, BigGAN | Improved quality and resolution |
| 2020 | StyleGAN2-ADA | Data augmentation techniques |
| 2021 | StyleGAN3 | Alias-free generation |

**Quality Progression:**

```
2014: Blurry, low-resolution grayscale images (64×64)
   ↓
2015: Clearer images with DCGAN
   ↓
2016-2017: Better training stability
   ↓
2018: High-resolution (1024×1024) realistic images
   ↓
2019-2020: Photo-realistic quality
   ↓
2021: Near-perfect alias-free generation
```

### 1.2 GANs vs Other Generative Models

**Comparison with Likelihood-Based Models:**

| Feature | Autoregressive | Flow Models | VAE | GAN |
|---------|---------------|-------------|-----|-----|
| **Explicit Density** | ✓ (Exact) | ✓ (Exact) | ✓ (Approximate) | ✗ |
| **Sampling Speed** | Slow (Sequential) | Fast (Parallel) | Fast (Parallel) | Fast (Parallel) |
| **Sample Quality** | Good | Good | Blurry | Excellent |
| **Training Stability** | Stable | Stable | Stable | Challenging |
| **Mode Coverage** | Good | Good | Good | Risk of mode collapse |
| **Latent Space** | No direct latent space | Invertible | Structured | Unstructured noise |

**Key Differences:**

1. **Implicit Density Estimation**
   - GANs don't model p(x) explicitly
   - Learn through discriminator feedback
   - No direct likelihood evaluation

2. **Adversarial Training**
   - Two-player minimax game
   - Generator vs Discriminator
   - Unique optimization challenges

3. **Sample Quality**
   - Typically produces sharper images than VAE
   - Less mode averaging
   - Better perceptual quality

---

## 2. Implicit Models

### 2.1 Definition

**Implicit models** generate samples without explicitly modeling the probability distribution $p(x)$.

**Process:**
1. Sample $z \sim p(z)$ where $z$ is a noise vector from prior distribution
2. Pass through Deep Neural Network (Generator)
3. Generate $x = G(z)$ where $x$ is synthetic data

**Mathematical Formulation:**
- Given samples from data distribution: $x_1, x_2, \ldots, x_n \sim p_{\text{data}}$
- Given a sampler: $q_\phi(z) = \text{DNN}(z; \phi)$ where $z \sim p(z)$
- $x = q_\phi(z)$ induces a density function $p_{\text{model}}$
- **Goal:** Make $p_{\text{model}}$ as close to $p_{\text{data}}$ as possible

### 2.2 Key Characteristics

1. **No Explicit Form**
   - Cannot write down p_data or p_model
   - Can only draw samples

2. **Learning Through Comparison**
   - Use discriminator to compare distributions
   - Indirect learning via adversarial signal

3. **Differences from Flow/VAE:**
   - Flow: Explicit density via invertible transformations
   - VAE: Approximate density via variational inference
   - GAN: No density estimation, pure sample generation

---

## 3. GAN Architecture

### 3.1 Generator

**Purpose:** Transform random noise into realistic data samples

**Architecture:**
```
**Input:** $z \in \mathbb{R}^{d_z}$ (typically $d_z = 100-128$)
   ↓
Reshape: z → tensor (e.g., 16×16×128)
   ↓
Transposed Convolution Layers
   ↓
Batch Normalization + ReLU
   ↓
**Output:** $x \in \mathbb{R}^{H \times W \times C}$ (e.g., $64 \times 64 \times 3$)
```

**Key Components:**
- **Input:** Random noise vector $z \sim \mathcal{N}(0, I)$ or $\text{Uniform}[-1, 1]$
- **Hidden Layers:** Series of transposed convolutions (deconvolutions)
- **Activation:** ReLU in hidden layers, tanh in output
- **Normalization:** Batch Normalization (except input/output layers)

**Example Dimensions:**
```
z: [128, 1]
   ↓ Fully Connected + Reshape
[16, 16, 128]
   ↓ TransposeConv2D (stride=2)
[32, 32, 64]
   ↓ TransposeConv2D (stride=2)
[64, 64, 3]
```

### 3.2 Discriminator

**Purpose:** Distinguish between real and generated samples

**Architecture:**
```
**Input:** $x \in \mathbb{R}^{H \times W \times C}$
   ↓
Convolutional Layers (stride=2 for downsampling)
   ↓
Batch Normalization + LeakyReLU
   ↓
Flatten
   ↓
Fully Connected Layer
   ↓
**Output:** $D(x) \in [0, 1]$ (probability of being real)
```

**Key Components:**
- **Input:** Image (real or generated)
- **Hidden Layers:** Strided convolutions for downsampling
- **Activation:** LeakyReLU$(\alpha=0.2)$
- **Output:** Sigmoid activation for binary classification

### 3.3 Training Process

**Intuitive Analogy:**
```
Generator ≈ Art Forger
Discriminator ≈ Art Detective

Goal: Forger creates art so realistic that detective cannot tell fake from real
```

**Training Dynamics:**

```
Initial State:
┌─────────────────────────┐
│ Real Data Distribution  │  ← Training Data
└─────────────────────────┘

┌─────────────────────────┐
│Generated Data (Poor)    │  ← Generator Output
└─────────────────────────┘

Discriminator: Easily distinguishes ($D(\text{real}) \approx 1, D(\text{fake}) \approx 0$)
```

```
After Training:
┌─────────────────────────┐
│ Real & Generated Data   │  ← Distributions overlap
└─────────────────────────┘

Discriminator: Cannot distinguish ($D(\text{real}) \approx 0.5, D(\text{fake}) \approx 0.5$)
```

**Training Algorithm (High-Level):**

1. **Train Discriminator:**
   - Sample real data: x ~ p_data
   - Sample noise: z ~ p(z)
   - Generate fake data: x̃ = G(z)
   - Update D to maximize: log D(x) + log(1 - D(G(z)))

2. **Train Generator:**
   - Sample noise: z ~ p(z)
   - Update G to maximize: log D(G(z))
   - Equivalently minimize: log(1 - D(G(z)))

---

## 4. GAN Theory

### 4.1 Loss Function

**Complete Objective:**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

**Breakdown:**

1. **Discriminator's Objective (Maximize):**

   $$\max_D V(D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

   - First term: Correctly classify real data $(D(x) \to 1)$
   - Second term: Correctly classify fake data $(D(G(z)) \to 0)$

2. **Generator's Objective (Minimize):**

   $$\min_G V(G) = \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

   - Equivalently maximize: $\mathbb{E}_{z \sim p(z)}[\log D(G(z))]$
   - Goal: Fool discriminator $(D(G(z)) \to 1)$

### 4.2 Minimax Game

**Unique Optimization Problem:**

Unlike standard optimization:
- Not pure minimization
- Not pure maximization
- **Minimax:** One part minimizes, another maximizes

**Challenges:**
1. **No Convergence Guarantee**
   - May oscillate instead of converging
   - Nash equilibrium not always reached

2. **Training Instability**
   - Sensitive to hyperparameters
   - Requires careful balancing of G and D

3. **Mode Collapse**
   - Generator may ignore input noise
   - Produces limited variety of samples

### 4.3 Bayes-Optimal Discriminator

**Given fixed Generator G, what is optimal D?**

**Derivation:**

For fixed G, discriminator's objective:

$$\max_D V(D) = \int_x p_{\text{data}}(x) \log D(x) \, dx + \int_x p_g(x) \log(1 - D(x)) \, dx$$

Taking derivative and setting to zero:

$$\frac{\partial V}{\partial D(x)} = \frac{p_{\text{data}}(x)}{D(x)} - \frac{p_g(x)}{1 - D(x)} = 0$$

**Optimal Discriminator:**

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**Interpretation:**
- If $p_{\text{data}}(x) = p_g(x)$: $D^*(x) = 0.5$ (cannot distinguish)
- If $p_{\text{data}}(x) \gg p_g(x)$: $D^*(x) \to 1$ (confidently real)
- If $p_{\text{data}}(x) \ll p_g(x)$: $D^*(x) \to 0$ (confidently fake)

### 4.4 Generator Objective

**Given optimal $D^*$, what is G minimizing?**

Substituting $D^*$ into $V(D, G)$:

$$\begin{align}
C(G) &= \max_D V(D, G) \\
&= \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right] \\
&= -\log(4) + \text{KL}\left(p_{\text{data}} \,\|\, \frac{p_{\text{data}} + p_g}{2}\right) + \text{KL}\left(p_g \,\|\, \frac{p_{\text{data}} + p_g}{2}\right) \\
&= -\log(4) + 2 \cdot \text{JS}(p_{\text{data}} \| p_g)
\end{align}$$

**Result:** Generator minimizes **Jensen-Shannon Divergence** between $p_{\text{data}}$ and $p_g$

**Properties:**
- JS divergence is symmetric
- Always non-negative
- Equals 0 only when $p_{\text{data}} = p_g$
- Maximum value: $\log(2)$ when distributions are disjoint

---

## 5. GAN Pseudocode

**Original GAN Training Algorithm (Goodfellow et al., 2014):**

```python
for number of training iterations:
    # Train Discriminator
    for k steps:
        # Sample minibatch of m noise samples
        {z⁽¹⁾, ..., z⁽ᵐ⁾} ~ p_g(z)

        # Sample minibatch of m examples from data
        {x⁽¹⁾, ..., x⁽ᵐ⁾} ~ p_data(x)

        # Update discriminator by ascending its stochastic gradient:
        $\nabla_{\theta_d} \frac{1}{m} \sum_i[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]$

    # Train Generator
    # Sample minibatch of m noise samples
    {z⁽¹⁾, ..., z⁽ᵐ⁾} ~ p_g(z)

    # Update generator by descending its stochastic gradient:
    $\nabla_{\theta_g} \frac{1}{m} \sum_i \log(1 - D(G(z^{(i)})))$
```

**Key Parameters:**
- **k:** Number of discriminator updates per generator update
  - Original paper: k = 1
  - Common practice: k = 1-5
  - Helps discriminator stay ahead of generator

**Practical Considerations:**

1. **Alternative Generator Loss:**
   ```python
   # Instead of: minimize log(1 - D(G(z)))
   # Use: maximize log(D(G(z)))
   ```
   - Provides stronger gradients early in training
   - Same fixed point, better optimization dynamics

2. **Gradient Flow:**
   - Early training: Generator is poor, D(G(z)) ≈ 0
   - log(1 - D(G(z))) ≈ 0 → flat gradients (saturation)
   - log(D(G(z))) provides stronger signal

---

## 6. Evaluation Metrics

### 6.1 Inception Score (IS)

**Motivation:**
- Cannot report explicit likelihood (unlike VAE/Flow models)
- Need automatic metric for sample quality

**Concept:** Good generators produce samples that are:
1. **Semantically meaningful** (low conditional entropy)
2. **Diverse** (high marginal entropy)

**Mathematical Definition:**

$$\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g}[D_{\text{KL}}(p(y|x) \| p(y))]\right)$$

where:
- $p(y|x)$: Conditional class distribution from Inception-v3
- $p(y) = \mathbb{E}_{x \sim p_g}[p(y|x)]$: Marginal class distribution

**Interpretation:**

1. **p(y|x) should have low entropy:**
   - Each image should clearly belong to one class
   - Sharp predictions: [0.9, 0.05, 0.05, ...] ✓
   - Not uniform: [0.1, 0.1, 0.1, ...] ✗

2. **p(y) should have high entropy:**
   - Images should span many classes
   - Uniform over classes: [0.001, 0.001, ...] ✓
   - Concentrated: [0.9, 0.05, ...] ✗

**Score Interpretation:**
$$\text{IS} = \exp(H(Y) - H(Y|X)) = \exp\left(H(Y) - \mathbb{E}_x[H(Y|X)]\right)$$

**Range:**
- Minimum: 1 (all images identical or uniform predictions)
- Maximum: Number of classes (perfect diversity and clarity)
- For ImageNet (1000 classes): $\text{IS} \in [1, 1000]$

**Advantages:**
- Single number metric
- Correlates with human judgment
- No reference data needed

**Limitations:**
1. **Doesn't measure fidelity:**
   - 1000 images (one per class) gets perfect IS
   - Even if images are unrealistic

2. **Doesn't capture intra-class diversity:**
   - Ignores variety within each class

3. **Biased by Inception model:**
   - Pre-trained on ImageNet
   - May not work for other domains

### 6.2 Fréchet Inception Distance (FID)

**Motivation:** Address IS limitations by comparing to real data

**Concept:** Measure distance between feature distributions of real and generated images

**Mathematical Definition:**

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

where:
- $\mu_r, \Sigma_r$: Mean and covariance of real data features
- $\mu_g, \Sigma_g$: Mean and covariance of generated data features
- Features: Inception-v3 pool3 layer (2048-dimensional)

**Feature Extraction:**
```
Image → Inception-v3 → pool3 layer → 2048-dim features
```

**Computation Steps:**

1. **Extract features:**
   ```python
   real_features = inception_v3.pool3(real_images)  # [N, 2048]
   gen_features = inception_v3.pool3(gen_images)    # [N, 2048]
   ```

2. **Compute statistics:**
   ```python
   μ_r = mean(real_features)  # [2048]
   μ_g = mean(gen_features)   # [2048]
   Σ_r = cov(real_features)   # [2048, 2048]
   Σ_g = cov(gen_features)    # [2048, 2048]
   ```

3. **Calculate FID:**
   ```python
   FID = ||μ_r - μ_g||² + trace(Σ_r + Σ_g - 2*sqrt(Σ_r @ Σ_g))
   ```

**Interpretation:**

- **Lower is better** (0 = perfect match)
- Captures both quality and diversity
- $\text{FID} \approx 0$: Generated distribution matches real
- $\text{FID} \gg 0$: Significant distribution mismatch

**Effects of Distortions:**

| Distortion | Effect on FID |
|------------|---------------|
| Added noise | Increases ↑ |
| Blur | Increases ↑ |
| Mode collapse | Increases ↑ |
| Lower diversity | Increases ↑ |

**Advantages over IS:**

1. **Compares to real data**
2. **Sensitive to mode dropping**
3. **Captures both quality and diversity**
4. **More robust to artifacts**

**Limitations:**

1. **Requires many samples** (typically 10k+)
2. **Biased by Inception network**
3. **Sensitive to feature extractor choice**

**FID vs IS Comparison:**

| Metric | Range | Lower/Higher Better | Compares to Real Data | Captures Diversity |
|--------|-------|---------------------|----------------------|-------------------|
| IS | [1, ∞) | Higher ↑ | No | Partially |
| FID | [0, ∞) | Lower ↓ | Yes | Yes |

---

## 7. Deep Convolutional GAN (DCGAN)

### 7.1 Architecture Guidelines

**Key Innovation:** Stable architecture for training GANs with deep convolutional networks

**Architecture Principles (Radford et al., 2016):**

#### Generator Architecture

```
**Input:** $z \sim \mathcal{N}(0, I)$, $z \in \mathbb{R}^{100}$
   ↓ Project & Reshape
[4, 4, 1024] + ReLU + BatchNorm
   ↓ TransposeConv(stride=2, filters=512)
[8, 8, 512] + ReLU + BatchNorm
   ↓ TransposeConv(stride=2, filters=256)
[16, 16, 256] + ReLU + BatchNorm
   ↓ TransposeConv(stride=2, filters=128)
[32, 32, 128] + ReLU + BatchNorm
   ↓ TransposeConv(stride=2, filters=3)
[64, 64, 3] + tanh
```

**Design Choices:**

1. **Replace pooling layers:**
   - ✗ Max pooling / Average pooling
   - ✓ Strided convolutions (discriminator)
   - ✓ Transposed convolutions (generator)

2. **Batch Normalization:**
   - ✓ Use in both G and D
   - ✗ NOT in generator output layer
   - ✗ NOT in discriminator input layer
   - Prevents mode collapse
   - Stabilizes learning

3. **Activation Functions:**
   - Generator hidden: **ReLU**
   - Generator output: **tanh**
   - Discriminator: **LeakyReLU$(\alpha=0.2)$**

4. **No Fully Connected Layers:**
   - Except for initial projection in G
   - All-convolutional architecture

#### Discriminator Architecture

```
**Input:** $x \in \mathbb{R}^{64 \times 64 \times 3}$
   ↓ Conv(stride=2, filters=64) + LeakyReLU(0.2)
[32, 32, 64] (No BatchNorm on first layer)
   ↓ Conv(stride=2, filters=128) + LeakyReLU(0.2) + BatchNorm
[16, 16, 128]
   ↓ Conv(stride=2, filters=256) + LeakyReLU(0.2) + BatchNorm
[8, 8, 256]
   ↓ Conv(stride=2, filters=512) + LeakyReLU(0.2) + BatchNorm
[4, 4, 512]
   ↓ Flatten + Fully Connected
Output: D(x) ∈ [0, 1] (sigmoid)
```

**Training Details:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| $\beta_1$ (momentum) | 0.5 |
| $\beta_2$ | 0.999 |
| Batch Size | 128 |
| LeakyReLU slope | 0.2 |

### 7.2 Key Results

**Achievements:**

1. **High-Quality Samples:**
   - First to generate good samples on large datasets
   - 3M image datasets (CelebA, LSUN Bedrooms)
   - 64×64 resolution with good quality

2. **Smooth Latent Space Interpolations:**
   ```
   z₁ → ... → z_interpolated → ... → z₂
   ↓                                    ↓
   image₁                           image₂
   ```
   - Demonstrates learned structure
   - No sudden transitions
   - Semantically meaningful interpolations

3. **Vector Arithmetic in Latent Space:**

   **Example 1 - Smiling Manipulation:**
   ```
   smiling_woman - neutral_woman + neutral_man = smiling_man
   ```

   **Example 2 - Glasses Manipulation:**
   ```
   man_with_glasses - man_without_glasses + woman_without_glasses
   = woman_with_glasses
   ```

   **Why it works:**
   - Latent space captures semantic features
   - Linear structure for some attributes
   - Demonstrates disentanglement

4. **Representation Learning:**
   - Discriminator learns useful features
   - Can be used as feature extractor
   - Transfer learning to other tasks

**Comparison: Vector Arithmetic**

| Operation Domain | Result Quality |
|------------------|----------------|
| Pixel Space | Poor (averaging creates blur) |
| Latent Space | Excellent (semantic interpolation) |

**ImageNet Results:**
- Scaled to 1000-class ImageNet
- More challenging than single-category datasets
- Demonstrated generalizability

---

## 8. Improved Training Techniques

**Motivation:** Original GAN training is unstable and sensitive

### 8.1 Feature Matching

**Problem:** Generator over-trains on current discriminator

**Solution:** Match statistics of intermediate features

**Modified Generator Objective:**
Instead of: $\max \mathbb{E}_z[\log D(G(z))]$

Use: $\min \|\mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{z \sim p(z)}[f(G(z))]\|^2$

where $f(x)$ = intermediate layer activations in discriminator

**Benefits:**
- Prevents overfitting to current D
- Provides more stable training signal
- Reduces mode collapse

### 8.2 Minibatch Discrimination

**Problem:** Mode collapse - generator produces limited variety

**Concept:** Discriminator detects if minibatch has low diversity

**Implementation:**

```python
def minibatch_discrimination(x, batch_size):
    # x: [batch_size, features]

    # Compute similarity between samples
    similarity_matrix = compute_similarity(x)  # [batch_size, batch_size]

    # Summary statistic per sample
    o(x_i) = sum_j(similarity(x_i, x_j))

    # Concatenate with original features
    return concat([x, o(x)])
```

**How it helps:**
1. If generator collapses to few modes:
   - Generated samples become similar
   - High intra-batch similarity detected
   - Discriminator penalizes this

2. Encourages diversity:
   - Generator must produce varied samples
   - Within each minibatch

### 8.3 Historical Averaging

**Problem:** Oscillations in parameter space

**Solution:** Regularize using historical parameter values

**Modified Objective:**
$$L(\theta) = \text{Original\_Loss}(\theta) + \lambda\left\|\theta - \frac{1}{t}\sum_i \theta_i\right\|^2$$

where:
- $\theta_i$: parameters at past iterations
- $\lambda$: regularization strength

**Benefits:**
- Reduces oscillations
- Smoother convergence
- Prevents cycling

### 8.4 Virtual Batch Normalization

**Problem:** Batch normalization couples samples within minibatch

**Solution:** Use reference batch for normalization statistics

**Implementation:**
```python
# Standard Batch Norm
mean, var = compute_stats(current_batch)

# Virtual Batch Norm
reference_batch = fixed_batch  # Set once at start
combined_batch = concat([sample, reference_batch])
mean, var = compute_stats(combined_batch)
```

**Advantages:**
- Reduces dependency on minibatch composition
- More consistent gradients
- Better for small batch sizes

**Disadvantages:**
- Computationally expensive (2× forward passes)
- Typically used only in generator

### 8.5 One-sided Label Smoothing

**Problem:** Discriminator becomes too confident

**Original Labels:**
- Real: 1
- Fake: 0

**Smoothed Labels:**
- Real: 0.9 (smoothed)
- Fake: 0.0 (not smoothed)

**Why one-sided?**

```
Two-sided smoothing:
Real: 0.9, Fake: 0.1
→ D*(x) = 0.9p_data(x) + 0.1p_g(x) / (p_data(x) + p_g(x))
→ Shifts optimal D away from true ratio
```

```
One-sided smoothing:
Real: 0.9, Fake: 0
→ D*(x) ≈ p_data(x) / (p_data(x) + p_g(x))
→ Preserves optimal D structure
```

**Benefits:**
- Prevents discriminator overconfidence
- More robust to adversarial examples
- Better gradient flow to generator

---

## 9. Advanced GAN Variants

### 9.1 Wasserstein GAN (WGAN)

**Motivation:** JS divergence issues

**JS Divergence Problems:**

```
When p_data and p_g have disjoint support:
$\text{JS}(p_{\text{data}} \| p_g) = \log(2)$ (constant)
→ No gradient information
→ Training fails
```

**Earth Mover's Distance (Wasserstein-1):**

$$W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r,p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

Interpretation: Minimum "cost" to transform $p_g$ into $p_r$

**Why better?**
- Continuous everywhere
- Differentiable almost everywhere
- Provides gradients even with disjoint supports

**WGAN Objective (Kantorovich-Rubinstein Duality):**

$$W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

where $\|f\|_L \leq 1$ means $f$ is 1-Lipschitz

**Lipschitz Constraint:**
$$|f(x_1) - f(x_2)| \leq K\|x_1 - x_2\| \text{ for all } x_1, x_2$$

$K$ is the Lipschitz constant

**Implementation (Arjovsky et al., 2017):**

**Weight Clipping:**
```python
for p in critic.parameters():
    p.data.clamp_(-c, c)  # c = 0.01 typically
```

**Modified Loss:**
**Critic:**
$$\max_D \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{z \sim p(z)}[D(G(z))]$$

**Generator:**
$$\max_G \mathbb{E}_{z \sim p(z)}[D(G(z))]$$

Note: No sigmoid in D (outputs raw scores)

**Algorithm:**
```python
for iteration in training:
    for _ in range(n_critic):  # n_critic = 5 typically
        # Sample real and fake
        x_real ~ p_data
        z ~ p(z)
        x_fake = G(z)

        # Update critic
        loss_D = -mean(D(x_real)) + mean(D(x_fake))
        D.backward(loss_D)
        D.step()

        # Clip weights
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

    # Update generator
    z ~ p(z)
    loss_G = -mean(D(G(z)))
    G.backward(loss_G)
    G.step()
```

**Key Changes from Original GAN:**

| Aspect | Original GAN | WGAN |
|--------|--------------|------|
| Objective | Minimize JS divergence | Minimize Wasserstein distance |
| Discriminator output | [0, 1] (sigmoid) | ℝ (no sigmoid) |
| Loss function | log D(x) + log(1-D(G(z))) | D(x) - D(G(z)) |
| Lipschitz constraint | None | Weight clipping |
| Terminology | Discriminator | Critic |

**Advantages:**

1. **Meaningful loss metric:**
   - $W(p_r, p_g)$ correlates with sample quality
   - Can plot and monitor convergence

2. **Training stability:**
   - Works with various architectures
   - No careful balancing of G and D needed

3. **No mode collapse:**
   - Reduced (though not eliminated)

**Disadvantages:**

1. **Weight clipping issues:**
   - Can lead to capacity underutilization
   - May cause gradient exploding/vanishing
   - Suboptimal way to enforce Lipschitz

2. **Slow training:**
   - Need to train critic to convergence
   - n_critic = 5 or more

### 9.2 WGAN with Gradient Penalty (WGAN-GP)

**Motivation:** Fix weight clipping issues in WGAN

**Key Insight:** Optimal critic has gradient norm 1 almost everywhere

**Gradient Penalty:**

$$\text{GP} = \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

where $\hat{x} = \epsilon x + (1-\epsilon)G(z)$, $\epsilon \sim \text{Uniform}[0,1]$

**Complete Objective:**

$$L = \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{z \sim p(z)}[D(G(z))] + \lambda \cdot \text{GP}$$

$\lambda = 10$ typically

**Sampling x̂:**
```python
# Sample real and fake
x_real ~ p_data
x_fake = G(z)

# Random interpolation
epsilon = random.uniform(0, 1, size=batch_size)
x_hat = epsilon * x_real + (1 - epsilon) * x_fake

# Compute gradient penalty
gradients = autograd.grad(D(x_hat), x_hat)[0]
gradient_norm = sqrt(sum(gradients²))
gradient_penalty = ((gradient_norm - 1)²).mean()
```

**Algorithm:**

```python
for iteration in training:
    for _ in range(n_critic):
        # Sample batches
        x_real ~ p_data
        z ~ p(z)
        x_fake = G(z)

        # Interpolate
        ε ~ Uniform[0, 1]
        x_hat = ε*x_real + (1-ε)*x_fake

        # Compute losses
        D_real = D(x_real)
        D_fake = D(x_fake)
        D_hat = D(x_hat)

        # Gradient penalty
        gradients = grad(D_hat, x_hat)
        GP = ((||gradients||₂ - 1)²).mean()

        # Total critic loss
        loss_D = -D_real + D_fake + λ*GP

        # Update critic (no weight clipping!)
        D.backward(loss_D)
        D.step()

    # Update generator
    loss_G = -D(G(z))
    G.backward(loss_G)
    G.step()
```

**Advantages over WGAN:**

1. **No weight clipping:**
   - Better capacity utilization
   - No gradient pathologies

2. **Faster convergence:**
   - More stable gradients

3. **Better quality:**
   - Higher resolution images
   - Better FID scores

**Batch Normalization Incompatibility:**

**Problem:**
```
BatchNorm creates correlation between samples in batch
→ Gradient of D(x̂) depends on other samples
→ Violates gradient penalty assumption
```

**Solution:**
- Use **Layer Normalization** or **Instance Normalization**
- Or no normalization in critic

**Results:**

| Dataset | Resolution | FID |
|---------|-----------|-----|
| CIFAR-10 | 32×32 | ~30 |
| CelebA | 64×64 | ~20 |
| ImageNet | 128×128 | ~35 |

**Impact:**
- Became standard for GAN training
- Used in StyleGAN, Progressive GAN, etc.
- 2000+ citations

### 9.3 Spectral Normalization GAN (SNGAN)

**Motivation:** More efficient Lipschitz constraint

**Key Idea:** Constrain spectral norm of weight matrices

**Spectral Norm:**
$$\sigma(W) = \max_h \frac{\|Wh\|_2}{\|h\|_2}$$

Largest singular value of matrix $W$

**For Neural Network:**
$$\|f\|_L \leq \prod_i \sigma(W_i) \cdot \|a\|_L$$

where:
- $W_i$: weight matrix of layer i
- $a$: activation function
- Assume $\|a\|_L \approx 1$ (true for ReLU, LeakyReLU)

**Spectral Normalization:**

```python
W_SN = W / σ(W)

Effect: Forces σ(W_SN) = 1
```

**Efficient Computation (Power Iteration):**

```python
def spectral_norm(W, num_iterations=1):
    """
    W: weight matrix [out_features, in_features]
    """
    # Initialize random vector
    u = random_normal([out_features, 1])

    for _ in range(num_iterations):
        # Power iteration
        v = W.T @ u
        v = v / ||v||

        u = W @ v
        u = u / ||u||

    # Compute spectral norm
    sigma = u.T @ W @ v

    # Normalize weights
    W_normalized = W / sigma

    return W_normalized
```

**One iteration typically sufficient for stable training!**

**Implementation:**

```python
class SpectralNorm:
    def __init__(self, module):
        self.module = module
        self.u = None  # Will be initialized on first forward

    def forward(self, x):
        W = self.module.weight

        if self.u is None:
            # Initialize
            self.u = torch.randn(W.shape[0], 1)

        # Power iteration
        v = W.T @ self.u
        v = v / v.norm()

        u = W @ v
        u = u / u.norm()

        # Spectral norm
        sigma = u.T @ W @ v

        # Normalize and apply
        W_bar = W / sigma
        return F.conv2d(x, W_bar, ...)  # Or linear, etc.
```

**Advantages:**

1. **Computational efficiency:**
   - Single power iteration (vs gradient penalty backprop)
   - Minimal overhead

2. **No hyperparameters:**
   - Unlike GP which needs λ tuning

3. **Works with BatchNorm:**
   - No batch correlation issues

4. **Stable training:**
   - Can use larger learning rates

**Results:**

| Model | Dataset | Inception Score | FID |
|-------|---------|----------------|-----|
| SNGAN | CIFAR-10 | 8.22 | 21.7 |
| SNGAN | ImageNet (128×128) | 52.5 | 18.7 |

**First GAN to work well on full ImageNet!**

**Comparison: Lipschitz Enforcement:**

| Method | Mechanism | Compute Cost | Stability |
|--------|-----------|--------------|-----------|
| Weight Clipping | Hard constraint | Very low | Poor |
| Gradient Penalty | Soft constraint | High (backward pass) | Good |
| Spectral Norm | Soft constraint | Low (power iteration) | Excellent |

### 9.4 Self-Attention GAN (SAGAN)

**Motivation:** Convolutional GANs struggle with long-range dependencies

**Problem with Pure Convolution:**
```
Receptive field grows slowly with depth
→ Hard to model global structure
→ Good at textures, poor at geometric consistency
```

**Solution:** Add self-attention mechanism

**Self-Attention Module:**

```
Input: x ∈ ℝ^{C×H×W}

1. Generate queries, keys, values:
   f(x) = W_f x  → queries  [C'×HW]
   g(x) = W_g x  → keys     [C'×HW]
   h(x) = W_h x  → values   [C×HW]

2. Attention map:
   β = softmax(f(x)ᵀ g(x))  [HW×HW]

3. Output:
   o = W_o (h(x) β)

4. Residual connection:
   y = γo + x
```

**Where γ is learnable scale parameter (initialized to 0)**

**Architecture:**

```
Generator:
z → FC+Reshape → Conv+SelfAttn → Conv+SelfAttn → ... → Output

Discriminator:
Input → Conv+SelfAttn → Conv+SelfAttn → ... → FC → Output
```

**Training Tricks:**

1. **Spectral Normalization:**
   - Applied to both G and D
   - Contrary to intuition (not just D)
   - Improves stability

2. **Hinge Loss:**
   $$L_D = \mathbb{E}_x[\max(0, 1 - D(x))] + \mathbb{E}_z[\max(0, 1 + D(G(z)))]$$
   $$L_G = -\mathbb{E}_z[D(G(z))]$$

3. **TTUR (Two Time-scale Update Rule):**
   ```
   lr_D = 0.0004
   lr_G = 0.0001
   ```
   - Slower generator updates
   - Helps balance training

**Results:**

| Dataset | IS ↑ | FID ↓ |
|---------|------|-------|
| ImageNet 128×128 | 52.5 | 18.65 |
| ImageNet 256×256 | 58.5 | 15.25 |

**First unconditional GAN with good ImageNet samples!**

**Visualization:**
- Attention maps show long-range dependencies
- e.g., matching bird legs to body
- Consistent object geometry

### 9.5 Progressive GAN

**Motivation:** Direct high-resolution training is unstable

**Key Idea:** Start low-resolution, progressively add layers

**Training Progression:**

```
Stage 1: 4×4
G: z → [4×4]
D: [4×4] → score

Stage 2: 8×8
G: z → [4×4] → [8×8]
D: [8×8] → [4×4] → score

Stage 3: 16×16
G: z → [4×4] → [8×8] → [16×16]
D: [16×16] → [8×8] → [4×4] → score

...

Final: 1024×1024
```

**Smooth Transition:**

```python
def fade_in(alpha, low_res, high_res):
    """
    alpha: transition parameter [0, 1]
    low_res: upsampled previous resolution
    high_res: new layer output
    """
    return alpha * high_res + (1 - alpha) * low_res

# During transition
α starts at 0 → gradually increases to 1
```

**Architecture Details:**

**Generator Block:**
```
Input: [H, W, C]
   ↓ Upsample (nearest neighbor, 2×)
[2H, 2W, C]
   ↓ Conv 3×3
[2H, 2W, C']
   ↓ LeakyReLU
   ↓ Conv 3×3
[2H, 2W, C']
   ↓ LeakyReLU
Output: [2H, 2W, C']
```

**Discriminator Block:**
```
Input: [H, W, C]
   ↓ Conv 3×3
[H, W, C']
   ↓ LeakyReLU
   ↓ Conv 3×3
[H, W, C']
   ↓ LeakyReLU
   ↓ Downsample (average pool, 2×)
Output: [H/2, W/2, C']
```

**Additional Tricks:**

1. **Pixel Normalization (Generator):**
   ```python
   x = x / sqrt(mean(x²) + ε)
   ```
   - Per pixel, across channels
   - Prevents escalation

2. **Minibatch Standard Deviation (Discriminator):**
   - Add diversity statistic as extra channel
   - Helps detect mode collapse

3. **Equalized Learning Rate:**
   ```python
   W_runtime = W_init / sqrt(fan_in)
   ```
   - Scale weights at runtime, not initialization
   - More stable dynamics

**Training Schedule:**

| Resolution | Images Shown | Duration |
|------------|--------------|----------|
| 4×4 | 800k | ~1 day |
| 8×8 | 800k | ~1 day |
| 16×16 | 800k | ~1 day |
| 32×32 | 1600k | ~2 days |
| 64×64 | 1600k | ~2 days |
| 128×128 | 1600k | ~3 days |
| 256×256 | 3200k | ~5 days |
| 512×512 | 3200k | ~7 days |
| 1024×1024 | 3200k | ~10 days |

**Results:**

- **CelebA-HQ:** Photo-realistic 1024×1024 faces
- Smooth interpolations
- Unprecedented quality for 2017

**Impact:**
- Enabled high-resolution GAN training
- Influenced StyleGAN architecture
- Standard technique for large images

---

## 10. Conditional GANs

### 10.1 Conditional GAN (cGAN)

**Motivation:** Control what to generate

**Modification:** Condition on additional information y

**Architecture:**

```
Generator: G(z, y) → x
Discriminator: D(x, y) → [0, 1]

where y can be:
- Class label
- Text description
- Image
- Any side information
```

**Objective:**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x,y \sim p_{\text{data}}}[\log D(x, y)] + \mathbb{E}_{z \sim p(z),y \sim p(y)}[\log(1 - D(G(z, y), y))]$$

**Implementation:**

```python
# Generator
class ConditionalGenerator(nn.Module):
    def forward(self, z, y):
        # Embed label
        y_embed = self.embedding(y)  # [batch, embed_dim]

        # Concatenate with noise
        input = torch.cat([z, y_embed], dim=1)

        # Generate
        x = self.network(input)
        return x

# Discriminator
class ConditionalDiscriminator(nn.Module):
    def forward(self, x, y):
        # Embed label
        y_embed = self.embedding(y)  # [batch, embed_dim]

        # Expand to image size
        y_fill = y_embed.view(batch, embed_dim, 1, 1)
        y_fill = y_fill.expand(-1, -1, H, W)

        # Concatenate with image
        input = torch.cat([x, y_fill], dim=1)

        # Classify
        score = self.network(input)
        return score
```

**Applications:**
- Class-conditional image generation
- Text-to-image synthesis
- Attribute manipulation

### 10.2 Pix2Pix

**Goal:** Paired image-to-image translation

**Examples:**
- Edges → Photos
- Day → Night
- Segmentation map → Photo
- Sketch → Color image

**Setup:**
- Training data: Pairs (x, y)
- x: input image (e.g., edge map)
- y: output image (e.g., photo)

**Architecture:**

**Generator: U-Net**
```
Encoder:
Input → Conv → Conv → ... → Bottleneck

Decoder (with skip connections):
Bottleneck → TransposeConv + Skip → ... → Output
```

**Skip Connections:**
```
Layer 1 ─────────────────────→ Layer 7
Layer 2 ───────────────→ Layer 6
Layer 3 ─────────→ Layer 5
     ...
```

**Discriminator: PatchGAN**

**Concept:** Classify each N×N patch as real/fake

```
Instead of:
[H, W, 3] → Scalar (real/fake for entire image)

Use:
[H, W, 3] → [H', W', 1] (real/fake for each patch)
```

**Architecture:**
```
Input: [256, 256, 3]
   ↓ Conv (stride=2)
[128, 128, 64]
   ↓ Conv (stride=2)
[64, 64, 128]
   ↓ Conv (stride=2)
[32, 32, 256]
   ↓ Conv (stride=2)
[16, 16, 512]
   ↓ Conv
[14, 14, 1]  ← Each output is real/fake for 70×70 patch
```

**70×70 PatchGAN:** Each output corresponds to 70×70 receptive field

**Loss Function:**

$$L_{\text{total}} = L_{\text{GAN}} + \lambda L_{L1}$$

$$L_{\text{GAN}} = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x, z)))]$$

$$L_{L1} = \mathbb{E}_{x,y,z}[\|y - G(x, z)\|_1]$$

$\lambda = 100$ typically

**Why L1 loss?**
- Encourages correct low-frequency structure
- GAN loss handles high-frequency details

**Training:**

```python
# Train Discriminator
real_pair = (input_image, target_image)
fake_pair = (input_image, generated_image)

loss_D = -log(D(real_pair)) - log(1 - D(fake_pair))

# Train Generator
loss_G = -log(D(fake_pair)) + λ * ||target - generated||₁
```

**Results:**

| Task | Input | Output |
|------|-------|--------|
| Edges→Photos | Edge map | Realistic photo |
| BW→Color | Grayscale | Colored image |
| Labels→Facades | Segmentation | Building photo |
| Day→Night | Daytime | Nighttime |

**Advantages:**
- Learns task-specific loss
- Better than hand-crafted L1/L2
- Captures texture and structure

**Limitations:**
- Requires paired data
- Can't handle large appearance changes

### 10.3 CycleGAN

**Motivation:** Pix2Pix requires paired data (often unavailable)

**Goal:** Unpaired image-to-image translation

**Examples:**
- Horses ↔ Zebras
- Summer ↔ Winter
- Photos ↔ Monet paintings

**Key Idea: Cycle Consistency**

```
X (source) → G(X) (target) → F(G(X)) ≈ X (reconstructed)
Y (target) → F(Y) (source) → G(F(Y)) ≈ Y (reconstructed)
```

**Architecture:**

**Two Generators:**
```
G: X → Y  (e.g., Horse → Zebra)
F: Y → X  (e.g., Zebra → Horse)
```

**Two Discriminators:**
```
D_Y: Classify Y vs G(X)
D_X: Classify X vs F(Y)
```

**Loss Functions:**

**1. Adversarial Loss:**
$$L_{\text{GAN}}(G, D_Y, X, Y) = \mathbb{E}_y[\log D_Y(y)] + \mathbb{E}_x[\log(1 - D_Y(G(x)))]$$

$$L_{\text{GAN}}(F, D_X, Y, X) = \mathbb{E}_x[\log D_X(x)] + \mathbb{E}_y[\log(1 - D_X(F(y)))]$$

**2. Cycle Consistency Loss:**
$$L_{\text{cyc}}(G, F) = \mathbb{E}_x[\|F(G(x)) - x\|_1] + \mathbb{E}_y[\|G(F(y)) - y\|_1]$$

**3. Identity Loss (optional):**
$$L_{\text{identity}}(G, F) = \mathbb{E}_y[\|G(y) - y\|_1] + \mathbb{E}_x[\|F(x) - x\|_1]$$
- Encourages color preservation
- G(Y) ≈ Y (if already in target domain)

**Total Objective:**
```
L(G, F, D_X, D_Y) = L_GAN(G, D_Y, X, Y)
                   + L_GAN(F, D_X, Y, X)
                   + λ_cyc L_cyc(G, F)
                   + λ_id L_identity(G, F)

$\lambda_{\text{cyc}} = 10$, $\lambda_{\text{id}} = 5$ typically
```

**Training:**

```python
for epoch in epochs:
    for batch in data:
        x_real, y_real = batch

        # Forward cycle: X → Y → X
        y_fake = G(x_real)
        x_reconstructed = F(y_fake)

        # Backward cycle: Y → X → Y
        x_fake = F(y_real)
        y_reconstructed = G(x_fake)

        # Discriminator losses
        loss_D_Y = adversarial_loss(D_Y, y_real, y_fake)
        loss_D_X = adversarial_loss(D_X, x_real, x_fake)

        # Generator losses
        loss_adv = -log(D_Y(y_fake)) - log(D_X(x_fake))
        loss_cycle = ||x_reconstructed - x_real|| + ||y_reconstructed - y_real||
        loss_G = loss_adv + λ_cyc * loss_cycle

        # Update
        optimize_D(loss_D_X, loss_D_Y)
        optimize_G(loss_G)
```

**Generator Architecture:**

```
Encoder-Decoder with Residual Blocks

Input: [256, 256, 3]
   ↓ Conv (64 filters)
[256, 256, 64]
   ↓ Conv (stride=2, 128 filters)
[128, 128, 128]
   ↓ Conv (stride=2, 256 filters)
[64, 64, 256]
   ↓ 9× Residual Blocks
[64, 64, 256]
   ↓ TransposeConv (stride=2, 128 filters)
[128, 128, 128]
   ↓ TransposeConv (stride=2, 64 filters)
[256, 256, 64]
   ↓ Conv (3 filters, tanh)
[256, 256, 3]
```

**Discriminator: 70×70 PatchGAN**
(Same as Pix2Pix)

**Results:**

| Task | Success | Notes |
|------|---------|-------|
| Horses ↔ Zebras | ✓ | Texture change |
| Summer ↔ Winter | ✓ | Lighting/color |
| Photos ↔ Paintings | ✓ | Style transfer |
| Apples ↔ Oranges | ✓ | Color/texture |
| Object transfiguration | Partial | Needs shape change |

**Limitations:**

1. **Geometric Changes:**
   - Struggles with tasks requiring shape changes
   - e.g., Cat → Dog (different anatomy)

2. **Distribution Characteristics:**
   - Works when distributions differ in texture/color
   - Fails when differ in structure/geometry

3. **Mode Collapse:**
   - May ignore input, produce constant output
   - Cycle consistency helps but doesn't eliminate

**Comparison:**

| Method | Paired Data | Shape Changes | Typical Use |
|--------|-------------|---------------|-------------|
| Pix2Pix | Required | Can handle | Structured translation |
| CycleGAN | Not required | Struggles | Style/texture transfer |

---

## 11. StyleGAN Series

### 11.1 StyleGAN

**Key Innovation:** Style-based generator architecture

**Motivation:**
- Decouple high-level attributes (pose, identity) from stochastic variation (freckles, hair)
- Better control over image synthesis

**Architecture Changes:**

**Traditional Generator:**
```
z → Network → Image
```

**StyleGAN Generator:**
```
Mapping Network:
z → MLP(8 layers) → w ∈ W

Synthesis Network:
Constant → AdaIN(w) → Conv → AdaIN(w) → ... → Image
         ↑ Noise      ↑ Noise
```

**Adaptive Instance Normalization (AdaIN):**

$$\text{AdaIN}(x, y) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

where:
- $x_i$: feature map i
- $\mu(x_i), \sigma(x_i)$: mean and std of feature map
- $y_{s,i}, y_{b,i}$: scale and bias from style $w$

**Process:**
1. Normalize feature map to zero mean, unit variance
2. Scale and shift using style parameters

**Style Injection:**

```
At each resolution:
w → Affine(A) → [y_s, y_b]
               ↓
        AdaIN(features, [y_s, y_b])
```

**Noise Injection:**

```
Per-pixel noise added after each convolution:
features = Conv(input)
features = features + B * noise

where:
- noise: [H, W, 1], random Gaussian per pixel
- B: learnable per-channel scale
```

**Coarse vs Fine Control:**

```
Coarse styles (4×4 - 8×8):
- Pose, general hair style, face shape

Middle styles (16×16 - 32×32):
- Facial features, hair style, eyes open/closed

Fine styles (64×64 - 1024×1024):
- Color scheme, microstructure
```

**Style Mixing:**

```
Source A → w_A
Source B → w_B

Use w_A for layers 1-4 (coarse)
Use w_B for layers 5-18 (fine)

Result: Pose from A, appearance from B
```

**Generator Architecture:**

```
Mapping Network:
z [512] → FC → FC → ... (8 layers) → w [512]

Synthesis Network:
Learned Constant [4, 4, 512]
   ↓ AdaIN(w) + Noise + Conv
[4, 4, 512]
   ↓ AdaIN(w) + Noise + Conv + Upsample
[8, 8, 512]
   ↓ AdaIN(w) + Noise + Conv
[8, 8, 512]
   ↓ ... (continue to 1024×1024)
[1024, 1024, 3]
```

**Truncation Trick:**

$$w_{\text{truncated}} = \bar{w} + \psi(w - \bar{w})$$

where:
- $\bar{w}$: average $w$ in training set
- $\psi \in [0, 1]$: truncation parameter

$\psi = 1$: Full diversity, may be lower quality
$\psi = 0.7$: Reduced diversity, higher quality

**Results:**
- 1024×1024 photo-realistic faces
- Smooth interpolations
- Disentangled representations
- FID: $\approx 4.4$ on FFHQ

### 11.2 StyleGAN2

**Motivation:** Fix artifacts in StyleGAN

**Problems in StyleGAN:**

1. **Water Droplet Artifacts:**
   - Blob-like artifacts
   - Caused by AdaIN

2. **Phase Artifacts:**
   - Image features "stick" to pixel coordinates
   - Not shift-equivariant

**Solutions:**

**1. Redesigned Normalization:**

```
StyleGAN (AdaIN):
x → Normalize → Scale/Shift (from style)

StyleGAN2 (Weight Demodulation):
Normalize weights, not activations
```

**Weight Demodulation:**

```python
# Modulate weights
w'_ijk = s_i * w_ijk

# Demodulate
w''_ijk = w'_ijk / sqrt(Σ_jk w'²_ijk + ε)

# Then convolve
output = Conv(input, w'')
```

**2. No Progressive Growing:**
- Directly train at target resolution
- Use skip connections instead

**3. Path Length Regularization:**

Encourage smooth mapping from $w$ to images:

$$L_{\text{path}} = \mathbb{E}_{w,y}[\|J_w^T y\|^2 - a]^2$$

where:
- $J_w$: Jacobian of generator w.r.t. $w$
- $y$: random image-space directions
- $a$: exponential moving average

**Architecture:**

```
Mapping Network: (same as StyleGAN)
z → 8× FC layers → w

Synthesis Network:
Constant [4, 4, 512]
   ↓ Modulated Conv + Noise + Activation
[4, 4, 512]
   ↓ Modulated Conv + Noise + Activation + Upsample
[8, 8, 512]
   ↓ ... (skip connections from each resolution to RGB)
[1024, 1024, 3]
```

**Skip Connection Architecture:**

```
Each resolution outputs to RGB:
Features → ToRGB → Sum with previous RGB (upsampled)
```

**Results:**
- FID: $\approx 2.8$ on FFHQ (vs 4.4 for StyleGAN)
- No water droplet artifacts
- Better quality overall
- Cleaner images

### 11.3 StyleGAN3

**Motivation:** Perfect equivariance to translation and rotation

**Problem in StyleGAN2:**
- "Texture sticking" to pixel grid
- Not truly shift-invariant
- Aliasing issues

**Root Cause:** Aliasing in:
1. Upsampling
2. Non-linearities
3. Downsampling

**Solution: Alias-Free Operations**

**1. Alias-Free Upsampling:**
```
Traditional: Nearest neighbor → Conv
StyleGAN3: Filtered upsampling
```

**2. Alias-Free Activation:**
```
Apply activation in continuous domain
Filter before and after
```

**3. Rotation Equivariance:**
- Fourier-based operations
- Rotational symmetry in frequency domain

**Architecture:**

```
Config-T (Translation equivariant):
- Alias-free upsampling
- Filtered non-linearities

Config-R (Rotation equivariant):
- Rotation-equivariant convolutions
- Fourier features
```

**Results:**
- Perfect shift equivariance
- Rotation equivariance (Config-R)
- Natural motion in videos
- FID comparable to StyleGAN2

**Comparison:**

| Feature | StyleGAN | StyleGAN2 | StyleGAN3 |
|---------|----------|-----------|-----------|
| Artifacts | Water droplets | Minimal | None |
| FID (FFHQ) | ~4.4 | ~2.8 | ~2.8 |
| Equivariance | Poor | Better | Perfect |
| Training speed | Fast | Fast | Slower |

---

## 12. InfoGAN

**Motivation:** Unsupervised disentanglement of latent factors

**Key Idea:** Maximize mutual information between latent codes and observations

**Architecture:**

```
Latent variables:
z: Incompressible noise
c: Interpretable latent codes

Generator: G(z, c) → x
Discriminator: D(x) → real/fake
Auxiliary network: Q(c | x) → predict c from x
```

**Objective:**

Standard GAN loss + Information regularization:

$$\min_{G,Q} \max_D V_{\text{InfoGAN}}(D, G, Q) = V_{\text{GAN}}(D, G) - \lambda I(c; G(z, c))$$

where $I(c; G(z, c))$ is mutual information

**Mutual Information:**

$$I(c; X) = H(c) - H(c | X) = H(X) - H(X | c)$$

Want: High MI $\rightarrow$ knowing $X$ tells us about $c$

**Variational Lower Bound:**

$$I(c; G(z, c)) \geq \mathbb{E}_{c \sim P(c),x \sim G(z,c)}[\log Q(c | x)] + H(c)$$

$$L_I = \mathbb{E}_{c,x}[\log Q(c | x)] \text{  (maximize this)}$$

**Implementation:**

```python
class InfoGAN:
    def __init__(self):
        self.G = Generator()
        self.D = Discriminator()
        self.Q = AuxiliaryNetwork()  # Shares weights with D

    def train_step(self):
        # Sample
        z = sample_noise()
        c = sample_codes()  # e.g., categorical + continuous

        # Generate
        x_fake = self.G(z, c)

        # Discriminator loss (standard GAN)
        loss_D = GAN_loss(self.D, x_real, x_fake)

        # Generator loss
        loss_G_adv = -log(self.D(x_fake))

        # Information loss
        c_pred = self.Q(x_fake)
        loss_info = -log_likelihood(c, c_pred)

        loss_G = loss_G_adv + λ * loss_info

        return loss_D, loss_G
```

**Latent Code Types:**

**1. Categorical (for discrete factors):**
```
c ~ Categorical(K)  # K classes

Example: Digit type in MNIST (0-9)
Loss: Cross-entropy
```

**2. Continuous (for continuous factors):**
```
c ~ Uniform[-1, 1] or Gaussian

Example: Rotation angle, width
Loss: Gaussian likelihood or L2
```

**Results on MNIST:**

| Code | Discovered Factor |
|------|------------------|
| c₁ (categorical, 10 classes) | Digit type (0-9) |
| c₂ (continuous) | Rotation |
| c₃ (continuous) | Width |

**Without any labels!**

**Results on 3D Faces:**

| Code | Discovered Factor |
|------|------------------|
| c₁ (continuous) | Azimuth (left-right) |
| c₂ (continuous) | Elevation (up-down) |
| c₃ (continuous) | Lighting |

**Advantages:**

1. **Unsupervised learning** of interpretable features
2. **Controllable generation**
3. **Automatic discovery** of data structure

**Limitations:**

1. **Not guaranteed** to find all factors
2. **Entanglement** still possible
3. **Requires careful** choice of code types

---

## 13. Projected GAN

**Motivation:** Leverage pretrained features for faster, better training

**Key Idea:** Train discriminator in pretrained feature space

**Architecture:**

```
Traditional GAN Discriminator:
Image → CNN (train from scratch) → Real/Fake

Projected GAN Discriminator:
Image → Pretrained Features → Multi-scale D → Real/Fake
```

**Multi-Scale Discriminator:**

```
Image → Pretrained Network (e.g., EfficientNet)
         ↓            ↓             ↓
      Layer 2      Layer 4       Layer 6
         ↓            ↓             ↓
      Random       Random        Random
      Projection   Projection    Projection
         ↓            ↓             ↓
      Conv → D₁    Conv → D₂     Conv → D₃
         ↓            ↓             ↓
      Real/Fake    Real/Fake     Real/Fake
```

**Random Projections:**

```python
# Project high-dim features to lower dim
features_projected = RandomMatrix @ features

Benefits:
- Reduces computation
- Increases diversity of discriminator
```

**Cross-Channel Mixing (CCM):**
```
Mix information across channels
Helps discriminator see broader patterns
```

**Cross-Scale Mixing (CSM):**
```
Mix information across scales
Helps with multi-scale consistency
```

**Training:**

```python
# Freeze pretrained network
pretrained_net.requires_grad = False

# Only train projection and discriminator layers
for image in batch:
    # Extract features (no grad)
    with torch.no_grad():
        features = pretrained_net(image)

    # Train discriminator on features
    loss_D = discriminator_loss(features)
    loss_D.backward()
```

**Advantages:**

1. **Faster Training:**
   - 2-4× speedup
   - Fewer iterations to convergence

2. **Better Sample Efficiency:**
   - Works with fewer images
   - Leverages pretrained knowledge

3. **Higher Quality:**
   - Better FID scores
   - Especially on smaller datasets

**Results:**

| Dataset | Standard GAN FID | Projected GAN FID | Speedup |
|---------|------------------|-------------------|---------|
| FFHQ | ~15 | ~8 | 3× |
| LSUN Churches | ~20 | ~12 | 4× |
| Small dataset (1k images) | ~50 | ~25 | 5× |

**Choice of Pretrained Network:**

| Network | Performance | Notes |
|---------|-------------|-------|
| EfficientNet | Best | Recommended |
| ResNet | Good | Widely available |
| VGG | Poor | Too old |

---

## 14. Mode Collapse and Solutions

**Mode Collapse:** Generator produces limited variety of samples

**Types:**

**1. Complete Mode Collapse:**
```
All samples identical
$G(z_1) = G(z_2) = \ldots = G(z_n)$
```

**2. Partial Mode Collapse:**
```
Few distinct modes covered
Missing many modes of p_data
```

**Why It Happens:**

```
If D can't distinguish between generated samples:
→ G has no incentive to diversify
→ Produces safest samples that fool D
```

**Detection:**

1. **Visual Inspection:**
   - Look at generated samples
   - Check for repeated patterns

2. **Metrics:**
   - Low Inception Score (poor diversity)
   - High FID (missing modes)
   - Birthday paradox test

**Solutions:**

### 1. Minibatch Discrimination
(Described in Section 8.2)

### 2. Unrolled GAN

**Idea:** Optimize G against future D, not current

```python
# Standard GAN
loss_G = -log D(G(z))

# Unrolled GAN (k steps)
D_future = D.copy()
for _ in range(k):
    D_future.step()  # Update towards optimum

loss_G = -log D_future(G(z))
```

**Benefits:**
- G considers D's response
- Reduces cycling
- More stable

**Cost:**
- Computationally expensive
- Need to backprop through D updates

### 3. Mode Regularization

**Add penalty for low diversity:**

```python
# Sample multiple z with same target
z1, z2 ~ p(z)
x1, x2 = G(z1), G(z2)

# Penalize if different z produce similar x
loss_mode = -λ * ||x1 - x2||₂ / ||z1 - z2||₂
```

### 4. Diversity-Promoting Losses

**DPP-GAN (Determinantal Point Process):**
```
Encourage diverse samples via DPP
Kernel measures sample similarity
```

### 5. Multiple Generators

**MAD-GAN (Multi-Agent Diverse GAN):**
```
Train K generators simultaneously
Each specializes in different modes
```

### 6. Progressive Training

**Start simple, add complexity:**
```
Early: Easy to cover all modes
Later: Maintain coverage while improving quality
```

### 7. Improved Architectures

- **Spectral Normalization:** Stabilizes training
- **Self-Attention:** Helps global consistency
- **Progressive GAN:** Natural mode coverage

**Mode Coverage vs. Sample Quality:**

```
Trade-off:
├─ Full mode coverage → May sacrifice quality
└─ High quality → Risk mode collapse

Balance needed!
```

---

## 15. Comparison Tables

### Generative Model Comparison

| Model | Explicit Density | Fast Sampling | High Quality | Stable Training | Latent Space |
|-------|-----------------|---------------|--------------|----------------|--------------|
| Autoregressive | ✓ | ✗ | Good | ✓ | ✗ |
| Flow | ✓ | ✓ | Good | ✓ | Invertible |
| VAE | Approx | ✓ | Moderate | ✓ | Structured |
| GAN | ✗ | ✓ | Excellent | ✗ | Unstructured |

### GAN Variant Comparison

| Variant | Key Innovation | Training Stability | Sample Quality | Computational Cost |
|---------|---------------|-------------------|----------------|-------------------|
| Original GAN | Adversarial training | Poor | Moderate | Low |
| DCGAN | Convolutional architecture | Better | Good | Low |
| WGAN | Wasserstein distance | Good | Good | Medium |
| WGAN-GP | Gradient penalty | Excellent | Very Good | High |
| SNGAN | Spectral normalization | Excellent | Very Good | Low |
| SAGAN | Self-attention | Good | Excellent | High |
| Progressive GAN | Progressive growing | Good | Excellent | Very High |
| StyleGAN | Style-based generation | Good | Outstanding | Very High |

### Evaluation Metrics

| Metric | Measures | Range | Better | Requires Reference | Limitations |
|--------|----------|-------|--------|-------------------|-------------|
| Inception Score | Quality & Diversity | [1, ∞) | Higher ↑ | No | Doesn't compare to real |
| FID | Distribution distance | [0, ∞) | Lower ↓ | Yes | Requires many samples |
| Precision | Quality | [0, 1] | Higher ↑ | Yes | May miss diversity |
| Recall | Diversity | [0, 1] | Higher ↑ | Yes | May miss quality |

### Lipschitz Enforcement Methods

| Method | Mechanism | Compute | Effectiveness | Issues |
|--------|-----------|---------|---------------|--------|
| Weight Clipping | Hard constraint | Very Low | Poor | Gradient problems |
| Gradient Penalty | Soft constraint | High | Excellent | Computational cost |
| Spectral Norm | Normalization | Low | Excellent | None significant |

### Conditional GAN Variants

| Variant | Input | Training Data | Use Case |
|---------|-------|---------------|----------|
| cGAN | Class label | Labeled | Class-conditional generation |
| Pix2Pix | Image | Paired images | Image-to-image translation |
| CycleGAN | Image | Unpaired images | Unsupervised translation |
| StyleGAN | Style vector | Images | Controllable generation |

---

## Appendix: Mathematical Derivations

### A. Jensen-Shannon Divergence in GANs

**Starting from optimal discriminator:**
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**Substitute into objective:**
```
C(G) = 𝔼_x~p_data[log D*(x)] + 𝔼_x~p_g[log(1 - D*(x))]

     = 𝔼_x~p_data[log p_data(x)/(p_data(x) + p_g(x))] +
       𝔼_x~p_g[log p_g(x)/(p_data(x) + p_g(x))]
```

**Define mixture distribution:**
$$p_m(x) = \frac{p_{\text{data}}(x) + p_g(x)}{2}$$

**Rewrite:**
```
C(G) = 𝔼_x~p_data[log p_data(x)/2p_m(x)] + 𝔼_x~p_g[log p_g(x)/2p_m(x)]

     = -log 4 + KL(p_data || p_m) + KL(p_g || p_m)

     = -log 4 + 2 · JS(p_data || p_g)
```

**Where JS divergence is:**
$$\text{JS}(p \| q) = \frac{\text{KL}(p \| m) + \text{KL}(q \| m)}{2} \text{ with } m = \frac{p + q}{2}$$

### B. Wasserstein Distance via Dual Form

**Primal form (intractable):**
$$W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r,p_g)} \int\int \|x - y\| d\gamma(x,y)$$

**Kantorovich-Rubinstein duality:**
$$W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \left(\mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]\right)$$

**Proof sketch:**
1. Define cost function c(x,y) = ||x - y||
2. Apply Kantorovich duality theorem
3. For L1 cost, Lipschitz constant K = 1
4. Result: supremum over 1-Lipschitz functions

### C. Gradient Penalty Derivation

**Goal:** Enforce 1-Lipschitz constraint

**Necessary condition:** Optimal critic has ||∇D(x)|| = 1 almost everywhere

**Why?**
If $D$ is optimal and 1-Lipschitz:

$$\forall x: |D(x_r) - D(x_g)| \leq \|x_r - x_g\|$$

At optimum, equality holds:

$$|D(x_r) - D(x_g)| = \|x_r - x_g\|$$

This implies $\|\nabla D(x)\| = 1$ on line from $x_r$ to $x_g$

**Penalty:**
$$\text{GP} = \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\| - 1)^2]$$

where $\hat{x}$ sampled uniformly on lines between real and fake

---

## Appendix: Training Tips and Best Practices

### General Tips

1. **Learning Rates:**
   - Start with lr = 0.0002 (Adam)
   - β₁ = 0.5 or 0.0 (not 0.9)
   - $\beta_2$ = 0.999

2. **Batch Normalization:**
   - Use in both G and D (DCGAN)
   - Or use Layer/Instance Norm (WGAN-GP)
   - Don't use in first D layer or last G layer

3. **Initialization:**
   - Xavier/He initialization
   - Or Normal(0, 0.02) for DCGAN

4. **Monitoring:**
   - Plot discriminator and generator loss
   - Visualize samples regularly
   - Track FID/IS if possible

### Debugging Checklist

**If discriminator loss → 0 (too strong):**
- Reduce discriminator learning rate
- Decrease discriminator capacity
- Add noise to discriminator inputs

**If generator loss → ∞ (too weak):**
- Increase generator capacity
- Use different loss (non-saturating)
- Check for vanishing gradients

**If mode collapse:**
- Try minibatch discrimination
- Use WGAN-GP or Spectral Normalization
- Reduce generator capacity
- Add diversity regularization

**If training unstable:**
- Use Spectral Normalization
- Try WGAN-GP
- Reduce learning rates
- Use gradient clipping

**If poor sample quality:**
- Increase model capacity
- Train longer
- Try progressive training
- Use better architecture (DCGAN guidelines)

### Architecture Guidelines (DCGAN)

**Do:**
- ✓ Use strided convolutions (no pooling)
- ✓ Use BatchNorm in G and D
- ✓ Use ReLU in G (except output: tanh)
- ✓ Use LeakyReLU in D

**Don't:**
- ✗ Use fully connected layers (except first G layer)
- ✗ Use max/average pooling
- ✗ Use vanilla ReLU in D
- ✗ Use sigmoid in G hidden layers

---

**End of Lecture Notes**

---

**Course Information:**
- **Course Code:** AIMLCZG513
- **Semester:** S1-25
- **Institution:** BITS Pilani, Pilani Campus
- **Instructor:** Prof. Sugata Ghosal
- **Contact:** sugata.ghosal@pilani.bits-pilani.ac.in

**Document Version:** 1.0
**Last Updated:** February 2026
**Total Pages:** ~50

---