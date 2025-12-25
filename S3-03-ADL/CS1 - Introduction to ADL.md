# **CS1 - Introduction to Advanced Deep Learning**

## **Table of Contents**

0. **CS0 - Important Links**
# **CS0 - Important Links (Computer Vision)**

| Category | Resource | Link | Description |
|----------|----------|------|-------------|
| **CV Libraries** | OpenCV | https://opencv.org/ | Comprehensive computer vision library |
| **CV Libraries** | PyTorch Vision | https://pytorch.org/vision/stable/index.html | PyTorch library for computer vision |
| **Deep Learning** | Hugging Face | https://huggingface.co/ | Pre-trained vision models and transformers |
| **Deep Learning** | TensorFlow | https://www.tensorflow.org/ | End-to-end ML platform with CV capabilities |
| **Object Detection** | Ultralytics YOLO | https://github.com/ultralytics/ultralytics | State-of-the-art real-time object detection |
| **Object Detection** | Detectron2 | https://github.com/facebookresearch/detectron2 | Facebook's object detection framework |
| **Segmentation** | Segment Anything (SAM) | https://github.com/facebookresearch/segment-anything | Meta's universal segmentation model |
| **Image Processing** | Pillow (PIL) | https://pillow.readthedocs.io/ | Python Imaging Library for basic operations |
| **Augmentation** | Albumentations | https://albumentations.ai/ | Fast image augmentation library |
| **Annotation Tools** | Roboflow | https://roboflow.com/ | Dataset management and annotation |
| **Annotation Tools** | CVAT | https://www.cvat.ai/ | Computer Vision Annotation Tool |
| **Datasets** | ImageNet | https://www.image-net.org/ | Large-scale image dataset |
| **Datasets** | COCO | https://cocodataset.org/ | Object detection and segmentation dataset |
| **Development** | Google Colab | https://colab.research.google.com/ | Free Jupyter notebook environment with GPU |
| **Courses** | Stanford CS231n | http://cs231n.stanford.edu/ | CNN for Visual Recognition course |
| **Papers** | Papers With Code | https://paperswithcode.com/area/computer-vision | CV papers with implementations |
| **Textbooks** | Deep Learning Book | https://www.deeplearningbook.org/ | Goodfellow, Bengio, Courville |

1. **CS1 - Introduction to ADL**
   - 1.1 Deep Unsupervised Learning
   - 1.2 Generative Modeling
   - 1.3 Principal Component Analysis (PCA)
   - 1.4 Autoencoders
   - 1.5 Autoregressive Models
   - 1.6 Normalizing Flow Models
   - 1.7 Generative Adversarial Networks (GANs)
   - 1.8 Progress in GANs
   - 1.9 Score-Based and Denoising Diffusion Models
   - 1.10 Multimodal Pretraining

---

## **CS0 - Important Links**

### **0.1 Official Documentation & Resources**

| Resource | Link | Description |
|----------|------|-------------|
| **PyTorch Documentation** | https://pytorch.org/docs/stable/index.html | Official PyTorch docs for deep learning |
| **TensorFlow Documentation** | https://www.tensorflow.org/api_docs | TensorFlow API and guides |
| **Anthropic Claude API** | https://docs.claude.com | Claude API documentation |
| **Hugging Face Hub** | https://huggingface.co/docs | Models, datasets, and tutorials |
| **Papers with Code** | https://paperswithcode.com | Research papers with implementations |

### **0.2 Research Papers & Publications**

#### **Autoencoders & VAE**
- **Auto-Encoding Variational Bayes** (Kingma & Welling, 2013): https://arxiv.org/abs/1312.6114
- **Tutorial on VAE**: https://arxiv.org/abs/1606.05908

#### **GANs**
- **Original GAN Paper** (Goodfellow et al., 2014): https://arxiv.org/abs/1406.2661
- **DCGAN Paper**: https://arxiv.org/abs/1511.06434
- **Progressive GAN**: https://arxiv.org/abs/1710.10196
- **StyleGAN**: https://arxiv.org/abs/1812.04948
- **StyleGAN2**: https://arxiv.org/abs/1912.04958
- **Wasserstein GAN**: https://arxiv.org/abs/1701.07875

#### **Normalizing Flows**
- **NICE**: https://arxiv.org/abs/1410.8516
- **RealNVP**: https://arxiv.org/abs/1605.08803
- **Glow**: https://arxiv.org/abs/1807.03039

#### **Diffusion Models**
- **Denoising Diffusion Probabilistic Models (DDPM)**: https://arxiv.org/abs/2006.11239
- **Improved DDPM**: https://arxiv.org/abs/2102.09672
- **DDIM (Fast Sampling)**: https://arxiv.org/abs/2010.02502
- **Score-Based Generative Models**: https://arxiv.org/abs/2011.13456
- **Diffusion Models Survey**: https://arxiv.org/abs/2209.00796

#### **Multimodal & Text-to-Image**
- **CLIP**: https://arxiv.org/abs/2103.00020
- **DALL-E**: https://arxiv.org/abs/2102.12092
- **DALL-E 2**: https://arxiv.org/abs/2204.06125
- **Stable Diffusion**: https://arxiv.org/abs/2112.10752
- **GLIDE**: https://arxiv.org/abs/2112.10741

#### **Autoregressive Models**
- **Attention Is All You Need (Transformers)**: https://arxiv.org/abs/1706.03762
- **GPT-3**: https://arxiv.org/abs/2005.14165
- **PixelCNN**: https://arxiv.org/abs/1606.05328

### **0.3 Tutorials & Courses**

| Resource | Link | Topic Coverage |
|----------|------|----------------|
| **Stanford CS236: Deep Generative Models** | https://deepgenerativemodels.github.io | Comprehensive generative models course |
| **Berkeley Deep Unsupervised Learning** | https://sites.google.com/view/berkeley-cs294-158-sp20 | Full course on unsupervised learning |
| **MIT 6.S191: Intro to Deep Learning** | http://introtodeeplearning.com | General deep learning concepts |
| **Fast.ai Practical Deep Learning** | https://course.fast.ai | Hands-on deep learning |
| **DeepLearning.AI Specialization** | https://www.deeplearning.ai | Coursera deep learning courses |
| **Lil'Log Blog (Lilian Weng)** | https://lilianweng.github.io | Excellent explanations of generative models |
| **Distill.pub** | https://distill.pub | Visual explanations of ML concepts |

### **0.4 Implementation Libraries**

#### **General Deep Learning**
| Library | Link | Use Case |
|---------|------|----------|
| **PyTorch** | https://pytorch.org | Primary deep learning framework |
| **TensorFlow/Keras** | https://www.tensorflow.org | Alternative deep learning framework |
| **JAX** | https://github.com/google/jax | High-performance ML research |
| **Lightning** | https://lightning.ai | PyTorch training framework |

#### **Generative Models**
| Library | Link | Models Supported |
|---------|------|------------------|
| **Hugging Face Diffusers** | https://github.com/huggingface/diffusers | Diffusion models (Stable Diffusion, DDPM) |
| **Stability AI SDK** | https://github.com/Stability-AI/stablediffusion | Stable Diffusion implementations |
| **PyTorch-GAN** | https://github.com/eriklindernoren/PyTorch-GAN | Various GAN implementations |
| **StyleGAN3** | https://github.com/NVlabs/stylegan3 | NVIDIA's StyleGAN |
| **Normalizing Flows** | https://github.com/kamenbliznashki/normalizing_flows | Flow-based models |

#### **Multimodal**
| Library | Link | Capability |
|---------|------|-----------|
| **CLIP** | https://github.com/openai/CLIP | Image-text embeddings |
| **OpenCLIP** | https://github.com/mlfoundations/open_clip | Open-source CLIP |
| **Transformers** | https://github.com/huggingface/transformers | Pre-trained multimodal models |

### **0.5 Community Resources**

| Resource | Link | Description |
|----------|------|-------------|
| **Reddit - r/MachineLearning** | https://www.reddit.com/r/MachineLearning | ML research discussions |
| **Reddit - r/StableDiffusion** | https://www.reddit.com/r/StableDiffusion | Diffusion model community |
| **Two Minute Papers (YouTube)** | https://www.youtube.com/@TwoMinutePapers | Visual paper summaries |
| **Yannic Kilcher (YouTube)** | https://www.youtube.com/@YannicKilcher | Detailed paper explanations |
| **Awesome Deep Learning** | https://github.com/ChristosChristofidis/awesome-deep-learning | Curated resource list |
| **Awesome Diffusion Models** | https://github.com/heejkoo/Awesome-Diffusion-Models | Diffusion-specific resources |
| **Awesome GANs** | https://github.com/nightrome/really-awesome-gan | GAN papers and code |

### **0.6 Interactive Tools & Demos**

| Tool | Link | What You Can Do |
|------|------|-----------------|
| **Hugging Face Spaces** | https://huggingface.co/spaces | Try models in browser |
| **Google Colab** | https://colab.research.google.com | Free GPU for experiments |
| **Kaggle Kernels** | https://www.kaggle.com/code | Datasets + compute |
| **Replicate** | https://replicate.com | Run models via API |
| **Stability AI Playground** | https://beta.dreamstudio.ai | Stable Diffusion interface |
| **OpenAI Playground** | https://platform.openai.com/playground | GPT models |

### **0.7 Datasets**

| Dataset | Link | Type |
|---------|------|------|
| **ImageNet** | https://www.image-net.org | Large-scale image dataset |
| **COCO** | https://cocodataset.org | Image captioning |
| **LAION-5B** | https://laion.ai/blog/laion-5b | Large image-text pairs |
| **CelebA** | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | Face attributes |
| **MNIST/Fashion-MNIST** | http://yann.lecun.com/exdb/mnist | Beginner datasets |

---

## **1.1 Deep Unsupervised Learning**

**Definition:** Capturing rich patterns in raw data with deep networks in a **label-free** way.

**Goal:** Learn useful properties of dataset structure without explicit labels.

| Category | Purpose | Examples |
|----------|---------|----------|
| **Generative Models** | Recreate raw data distribution | VAE, GAN, Diffusion Models |
| **Self-supervised Learning** | Design "puzzle" tasks for representations | Contrastive learning, Masked prediction |

**Applications:** Novel data generation • Conditional synthesis • Data compression • Pre-training for downstream tasks

---

## **1.2 Generative Modeling**

**Core Idea:** *"What I understand, I can create"*

**Process:** Data → Learn P(x) → Generate new samples

**Challenge:** Make P_θ (model) ≈ P_data (real distribution)

### **Three Pillars**

| Pillar | Focus | Question Answered |
|--------|-------|-------------------|
| **Representation** | Model joint distribution compactly | How to represent? |
| **Learning** | Minimize distance between distributions | How to train? |
| **Inference** | Learn latent variables/hidden features | What's learned? |

---

## **1.3 Principal Component Analysis (PCA)**

**Purpose:** Dimensionality reduction (n-dims → k-dims)

**Objective:** Minimize projection error • Find axes of maximum variability

**Nature:** Linear technique

**Key Insight:** Undercomplete autoencoder with linear decoder + MSE loss = PCA
→ **Nonlinear autoencoder = Generalized PCA**

---

## **1.4 Autoencoders**

**Function:** Input → Encoder → **Bottleneck** → Decoder → Reconstructed Output

```
Original Data → [Encoder] → Latent Space (compressed) → [Decoder] → Reconstructed Data
```

### **Comparison: Standard AE vs VAE**

| Feature | Autoencoder (AE) | Variational AE (VAE) |
|---------|------------------|----------------------|
| **Latent Space** | Unstructured | Gaussian (regularized) |
| **Type** | Deterministic | Probabilistic |
| **Generation** | Not designed for it | Fast sampling from latent |
| **Use Case** | Feature learning | Generative modeling |

---

## **1.5 Autoregressive Models**

**Core Mechanism:** Chain rule of probability

**Formula:** P(x₁, x₂, ..., x_N) = P(x₁) × ∏ P(x_n | x₁, ..., x_{n-1})

**Key Property:** Future depends on past (sequential)

| Model Type | Example | Characteristic |
|------------|---------|----------------|
| **RNN-based** | LSTM, GRU | Sequential memory |
| **Transformer** | GPT | Self-attention + sequential generation |

**Limitation:** Both training and generation are **sequential** (cannot parallelize)

---

## **1.6 Normalizing Flow Models**

**Mechanism:** Simple base distribution (Gaussian) → Invertible transformations → Complex data distribution

**Key Requirement:** Every layer must be **invertible**

**Advantages:**
- ✓ Exact likelihood computation
- ✓ Efficient inference (both directions)
- ✓ Stable training

**Variants:** NICE • RealNVP • Glow • AR/IAR flows

---

## **1.7 Generative Adversarial Networks (GANs)**

**Setup:** Zero-sum game between two networks

| Network | Role | Goal |
|---------|------|------|
| **Generator (G)** | Noise → Fake samples | Fool discriminator |
| **Discriminator (D)** | Real vs Fake classifier | Detect fakes |

**Training:** Min(G) Max(D) value function

**Major Challenges:**
- Training instability
- **Mode collapse** (limited sample diversity)

---

## **1.8 Progress in GANs**

| Technique | Innovation | Benefit |
|-----------|-----------|----------|
| **Progressive Growing** | Start 4×4 → gradually increase resolution | Stable training, high-res images |
| **StyleGAN** | Separate style from noise injection | Fine-grained control over generation |
| **WGAN** | Wasserstein distance loss | More stable training |
| **DCGAN** | Deep convolutional architecture | Better image synthesis |

---

## **1.9 Score-Based and Denoising Diffusion Models**

### **Two-Phase Process**

| Phase | Process | Nature |
|-------|---------|--------|
| **Forward (Diffusion)** | Data → Add noise gradually → Pure noise | Prespecified |
| **Reverse (Denoising)** | Pure noise → Remove noise gradually → Data | Learned |

```
Forward:  x → z₁ → z₂ → ... → z_T (noise)
Reverse:  z_T → ... → z₂ → z₁ → x (data)
```

**Pros:** Exceptional quality • Stable training
**Cons:** Slow generation (many sequential steps)
**Solution:** DDIM (skip steps for faster sampling)

---

## **1.10 Multimodal Pretraining**

**Goal:** Relate information across modalities (image ↔ text)

**Applications:**

| Direction | Task | Example |
|-----------|------|---------|
| Image → Text | Caption generation | Describe this image |
| Text → Image | Image synthesis | "A red car on a beach" → Image |
| Cross-modal | Retrieval, search | Find images matching text query |

**Examples:** GLIDE • DALL-E • Stable Diffusion • CLIP

---

## **Model Comparison Table**

| Model Type | Likelihood | Generation Speed | Sample Quality | Training Stability | Use Case |
|------------|-----------|------------------|----------------|-------------------|----------|
| **VAE** | Approximate | Fast | Good | Stable | Quick generation |
| **Autoregressive** | Exact | Slow (sequential) | Good | Stable | Text, audio |
| **Normalizing Flow** | Exact | Fast | Good | Stable | Density estimation |
| **GAN** | Implicit (none) | Fast | Excellent* | Unstable | Image synthesis |
| **Diffusion** | Tractable | Slow | Excellent | Very stable | High-quality images |

*When training succeeds

---

## **Visual Analogy: Diamond Creation**

Think of creating a perfect synthetic diamond (real data):

| Model | Analogy |
|-------|---------|
| **Normalizing Flow** | Clear reversible chemistry: simple elements → precise steps → diamond (can verify each step) |
| **GAN** | Forger vs Gemologist: make it *look* real without knowing underlying physics |
| **Diffusion** | Reverse corrosion: pure rust → gradually remove damage → pristine object |

---

## **Quick Reference Guide**

### **When to Use Which Model?**

- **Need exact likelihood?** → Normalizing Flow or Autoregressive
- **Need fast generation?** → GAN or VAE
- **Need best quality?** → Diffusion Models
- **Need stable training?** → Diffusion or VAE
- **Working with text/sequential?** → Autoregressive
- **Need interpretable latent space?** → VAE

### **Key Trade-offs**

| Aspect | Fast but Unstable | Slow but Stable |
|--------|-------------------|-----------------|
| **Training** | GAN | Diffusion, VAE |
| **Generation** | GAN, Flow, VAE | Autoregressive, Diffusion |
| **Quality** | GAN (when works) | Diffusion |

---

**End of Document**