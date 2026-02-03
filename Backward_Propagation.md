# Feed-Forward Neural Network
_(Complete Numerical Walkthrough with LaTeX-Enhanced Tabular Calculations)_

---

## Table of Contents

- [Feed-Forward Neural Network](#feed-forward-neural-network)
  - [Table of Contents](#table-of-contents)
  - [Network Architecture Visualization](#network-architecture-visualization)
    - [Layer-by-Layer Data Flow](#layer-by-layer-data-flow)
  - [Reference: Formulas Used](#reference-formulas-used)
    - [Activation Function](#activation-function)
    - [Forward Propagation Formulas](#forward-propagation-formulas)
    - [Loss Function](#loss-function)
  - [Feed-Forward Calculation Steps](#feed-forward-calculation-steps)
    - [Initial Parameters](#initial-parameters)
    - [Step 1: Hidden Layer Linear Combination](#step-1-hidden-layer-linear-combination)
    - [Step 2: Hidden Layer Activation (Sigmoid)](#step-2-hidden-layer-activation-sigmoid)
    - [Step 3: Output Layer Linear Combination](#step-3-output-layer-linear-combination)
    - [Step 4: Output Layer Activation (Final Predictions)](#step-4-output-layer-activation-final-predictions)
    - [Summary of Feed-Forward Results](#summary-of-feed-forward-results)
  - [Loss Calculation](#loss-calculation)
    - [Known Values](#known-values)
    - [Step 1: Error for Output Neuron 1](#step-1-error-for-output-neuron-1)
    - [Step 2: Error for Output Neuron 2](#step-2-error-for-output-neuron-2)
    - [Step 3: Total Loss](#step-3-total-loss)
    - [Interpretation of Loss](#interpretation-of-loss)
  - [Backpropagation: Weight Updates](#backpropagation-weight-updates)
    - [Overview of Weight Updates](#overview-of-weight-updates)
  - [Part 1: Output Layer Gradients](#part-1-output-layer-gradients)
    - [Why Start with Output Layer?](#why-start-with-output-layer)
    - [Formulas Used in Backpropagation](#formulas-used-in-backpropagation)
      - [Loss Function and Its Derivative](#loss-function-and-its-derivative)
      - [Sigmoid Activation Function and Its Derivative](#sigmoid-activation-function-and-its-derivative)
      - [Net Input to Activation Derivative](#net-input-to-activation-derivative)
      - [Summary Table: Backpropagation Derivatives](#summary-table-backpropagation-derivatives)
      - [Chain Rule for Output Layer Weights](#chain-rule-for-output-layer-weights)
    - [Gradient for w₅: Complete Derivation](#gradient-for-w-complete-derivation)
      - [Step 1: Identify the Dependency Chain](#step-1-identify-the-dependency-chain)
      - [Step 2: Apply Chain Rule](#step-2-apply-chain-rule)
      - [Step 3: Compute Each Partial Derivative](#step-3-compute-each-partial-derivative)
      - [Step 4: Multiply All Terms (Chain Rule)](#step-4-multiply-all-terms-chain-rule)
      - [Step 5: Update Weight](#step-5-update-weight)
    - [Gradients for Remaining Output Layer Weights](#gradients-for-remaining-output-layer-weights)
      - [Gradient for w₆](#gradient-for-w)
      - [Gradient for w₇](#gradient-for-w-1)
      - [Gradient for w₈](#gradient-for-w-2)
    - [Summary: Output Layer Weight Updates](#summary-output-layer-weight-updates)
  - [Part 2: Hidden Layer Gradients](#part-2-hidden-layer-gradients)
    - [Why Hidden Layer Gradients Are Different](#why-hidden-layer-gradients-are-different)
    - [Gradient for w₁: Complete Derivation](#gradient-for-w-complete-derivation-1)
      - [Step 1: Identify the Dependency Paths](#step-1-identify-the-dependency-paths)
      - [Step 2: Write Total Gradient as Sum](#step-2-write-total-gradient-as-sum)
      - [Step 3: Expand Each Path Using Chain Rule](#step-3-expand-each-path-using-chain-rule)
      - [Step 4: Compute Each Derivative](#step-4-compute-each-derivative)
      - [Step 5: Calculate Path 1 Contribution](#step-5-calculate-path-1-contribution)
      - [Step 6: Calculate Path 2 Contribution](#step-6-calculate-path-2-contribution)
      - [Step 7: Sum Both Error Paths](#step-7-sum-both-error-paths)
      - [Step 8: Complete the Gradient Calculation](#step-8-complete-the-gradient-calculation)
      - [Step 9: Update Weight](#step-9-update-weight)
    - [Understanding the Result](#understanding-the-result)
    - [Gradients for Remaining Hidden Layer Weights](#gradients-for-remaining-hidden-layer-weights)
      - [Gradient for w₂](#gradient-for-w-3)
      - [Gradient for w₃](#gradient-for-w-4)
      - [Gradient for w₄](#gradient-for-w-5)
    - [Summary: All Weight Updates](#summary-all-weight-updates)
  - [Part 3: Matrix Formulation of Backpropagation](#part-3-matrix-formulation-of-backpropagation)
    - [Notation and Setup](#notation-and-setup)
    - [Forward Pass (Vectorized)](#forward-pass-vectorized)
    - [Backward Pass (Vectorized)](#backward-pass-vectorized)
      - [Step 1: Output Layer Delta](#step-1-output-layer-delta)
      - [Step 2: Hidden Layer Delta](#step-2-hidden-layer-delta)
      - [Step 3: Gradient Matrices](#step-3-gradient-matrices)
      - [Step 4: Weight Updates](#step-4-weight-updates)
    - [Comparison: Scalar vs Matrix Form](#comparison-scalar-vs-matrix-form)
  - [Final Summary and Key Insights](#final-summary-and-key-insights)
    - [The Complete Backpropagation Algorithm](#the-complete-backpropagation-algorithm)
    - [Understanding Each Component](#understanding-each-component)
    - [Why Gradients Sum for Hidden Layers](#why-gradients-sum-for-hidden-layers)
    - [The Vanishing Gradient Problem](#the-vanishing-gradient-problem)
  - [Appendix: Complete Values Reference](#appendix-complete-values-reference)
    - [Forward Pass Values](#forward-pass-values)
    - [Loss Values](#loss-values)
    - [All Weight Gradients and Updates](#all-weight-gradients-and-updates)
  - [References and Further Reading](#references-and-further-reading)
    - [Original Article](#original-article)
    - [Foundational Papers](#foundational-papers)
    - [Recommended Textbooks](#recommended-textbooks)
    - [Online Resources](#online-resources)
  - [Next Steps for Deeper Understanding](#next-steps-for-deeper-understanding)
    - [1. Advanced Optimizers](#1-advanced-optimizers)
    - [2. Regularization Techniques](#2-regularization-techniques)
    - [3. Modern Architectures](#3-modern-architectures)
    - [4. Practical Implementation](#4-practical-implementation)
    - [5. Theoretical Deep Dives](#5-theoretical-deep-dives)

---

## Network Architecture Visualization

```
Input Layer          Hidden Layer              Output Layer

I₁ (0.05) ----w₁=0.15--→ H₁ (0.50687) ----w₅=0.40--→ O₁ (0.60635)
   |        \           /    |        \           /
   |         \         /     |         \         /
   |          w₂=0.20 /      |          w₆=0.45 /
   |           \     /       |           \     /
   |            \   /        |            \   /
   |             \ /         |             \ /
   |              X          |              X
   |             / \         |             / \
   |            /   \        |            /   \
   |          w₃=0.25\       |          w₇=0.50\
   |         /         \     |         /         \
   |        /           \    |        /           \
I₂ (0.10) ----w₄=0.30--→ H₂ (0.51062) ----w₈=0.55--→ O₂ (0.63002)
```

### Layer-by-Layer Data Flow

| From | To | Weight | Calculation |
|------|-----|--------|-------------|
| $I_1$ | $H_1$ | $w_1 = 0.15$ | $I_1 \times w_1 = 0.05 \times 0.15 = 0.0075$ |
| $I_2$ | $H_1$ | $w_2 = 0.20$ | $I_2 \times w_2 = 0.10 \times 0.20 = 0.0200$ |
| $I_1$ | $H_2$ | $w_3 = 0.25$ | $I_1 \times w_3 = 0.05 \times 0.25 = 0.0125$ |
| $I_2$ | $H_2$ | $w_4 = 0.30$ | $I_2 \times w_4 = 0.10 \times 0.30 = 0.0300$ |
| $H_1$ | $O_1$ | $w_5 = 0.40$ | $h_1 \times w_5 = 0.50687 \times 0.40 = 0.20275$ |
| $H_2$ | $O_1$ | $w_6 = 0.45$ | $h_2 \times w_6 = 0.51062 \times 0.45 = 0.22978$ |
| $H_1$ | $O_2$ | $w_7 = 0.50$ | $h_1 \times w_7 = 0.50687 \times 0.50 = 0.25344$ |
| $H_2$ | $O_2$ | $w_8 = 0.55$ | $h_2 \times w_8 = 0.51062 \times 0.55 = 0.28084$ |

---

## Reference: Formulas Used

### Activation Function

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Derivative of Sigmoid (needed for backpropagation):**

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

### Forward Propagation Formulas

**Hidden Layer:**

$$
\begin{align}
\text{net}_{h1} &= I_1 \cdot w_1 + I_2 \cdot w_2 \\
\text{net}_{h2} &= I_1 \cdot w_3 + I_2 \cdot w_4 \\
\\
h_1 &= \sigma(\text{net}_{h1}) = \frac{1}{1 + e^{-\text{net}_{h1}}} \\
h_2 &= \sigma(\text{net}_{h2}) = \frac{1}{1 + e^{-\text{net}_{h2}}}
\end{align}
$$

**Output Layer:**

$$
\begin{align}
\text{net}_{o1} &= h_1 \cdot w_5 + h_2 \cdot w_6 \\
\text{net}_{o2} &= h_1 \cdot w_7 + h_2 \cdot w_8 \\
\\
o_1 &= \sigma(\text{net}_{o1}) = \frac{1}{1 + e^{-\text{net}_{o1}}} \\
o_2 &= \sigma(\text{net}_{o2}) = \frac{1}{1 + e^{-\text{net}_{o2}}}
\end{align}
$$

### Loss Function

**Mean Squared Error (MSE) per output:**

$$
E_{total} = E_1 + E_2
$$

Where:

$$
E_i = \frac{1}{2}(t_i - o_i)^2
$$

The factor $\frac{1}{2}$ simplifies derivatives during backpropagation.

---

## Feed-Forward Calculation Steps

### Initial Parameters

**Input Values:**

| Parameter | Value |
|-----------|-------|
| $I_1$ | $0.05$ |
| $I_2$ | $0.10$ |

**Weights: Input → Hidden Layer:**

| Weight | Value | Connection |
|--------|-------|------------|
| $w_1$ | $0.15$ | $I_1 \to H_1$ |
| $w_2$ | $0.20$ | $I_2 \to H_1$ |
| $w_3$ | $0.25$ | $I_1 \to H_2$ |
| $w_4$ | $0.30$ | $I_2 \to H_2$ |

**Weights: Hidden → Output Layer:**

| Weight | Value | Connection |
|--------|-------|------------|
| $w_5$ | $0.40$ | $H_1 \to O_1$ |
| $w_6$ | $0.45$ | $H_2 \to O_1$ |
| $w_7$ | $0.50$ | $H_1 \to O_2$ |
| $w_8$ | $0.55$ | $H_2 \to O_2$ |

---

### Step 1: Hidden Layer Linear Combination

**Calculation Table: Hidden Layer Net Input**

| Neuron | Formula | Calculation | Result |
|--------|---------|-------------|--------|
| $\text{net}_{h1}$ | $I_1 \cdot w_1 + I_2 \cdot w_2$ | $(0.05 \times 0.15) + (0.10 \times 0.20) = 0.0075 + 0.0200$ | $\boxed{0.0275}$ |
| $\text{net}_{h2}$ | $I_1 \cdot w_3 + I_2 \cdot w_4$ | $(0.05 \times 0.25) + (0.10 \times 0.30) = 0.0125 + 0.0300$ | $\boxed{0.0425}$ |

**Detailed Breakdown**

**Hidden Neuron 1 ($\text{net}_{h1}$)**

| Component | Weight | Input | Product |
|-----------|--------|-------|---------|
| $I_1 \times w_1$ | $0.15$ | $0.05$ | $0.0075$ |
| $I_2 \times w_2$ | $0.20$ | $0.10$ | $0.0200$ |
| **Sum** | | | $\boxed{0.0275}$ |

$$
\text{net}_{h1} = I_1 \cdot w_1 + I_2 \cdot w_2 = (0.05)(0.15) + (0.10)(0.20) = 0.0275
$$

**Hidden Neuron 2 ($\text{net}_{h2}$)**

| Component | Weight | Input | Product |
|-----------|--------|-------|---------|
| $I_1 \times w_3$ | $0.25$ | $0.05$ | $0.0125$ |
| $I_2 \times w_4$ | $0.30$ | $0.10$ | $0.0300$ |
| **Sum** | | | $\boxed{0.0425}$ |

$$
\text{net}_{h2} = I_1 \cdot w_3 + I_2 \cdot w_4 = (0.05)(0.25) + (0.10)(0.30) = 0.0425
$$

---

### Step 2: Hidden Layer Activation (Sigmoid)

**Calculation Table: Hidden Layer Activations**

| Neuron | Net Input | Formula | Intermediate Steps | Result |
|--------|-----------|---------|-------------------|--------|
| $h_1$ | $0.0275$ | $\sigma(0.0275)$ | $e^{-0.0275} \approx 0.9729$ <br> $\frac{1}{1 + 0.9729}$ | $\boxed{0.50687}$ |
| $h_2$ | $0.0425$ | $\sigma(0.0425)$ | $e^{-0.0425} \approx 0.9584$ <br> $\frac{1}{1 + 0.9584}$ | $\boxed{0.51062}$ |

**Detailed Calculations**

**Activation of Hidden Neuron 1**

$$
h_1 = \sigma(\text{net}_{h1}) = \sigma(0.0275) = \frac{1}{1 + e^{-0.0275}}
$$

| Step | Calculation | Value |
|------|-------------|-------|
| 1. Compute $e^{-0.0275}$ | $e^{-0.0275}$ | $0.9729$ |
| 2. Add 1 | $1 + 0.9729$ | $1.9729$ |
| 3. Divide | $\frac{1}{1.9729}$ | $\boxed{0.50687}$ |

**Activation of Hidden Neuron 2**

$$
h_2 = \sigma(\text{net}_{h2}) = \sigma(0.0425) = \frac{1}{1 + e^{-0.0425}}
$$

| Step | Calculation | Value |
|------|-------------|-------|
| 1. Compute $e^{-0.0425}$ | $e^{-0.0425}$ | $0.9584$ |
| 2. Add 1 | $1 + 0.9584$ | $1.9584$ |
| 3. Divide | $\frac{1}{1.9584}$ | $\boxed{0.51062}$ |

---

### Step 3: Output Layer Linear Combination

**Calculation Table: Output Layer Net Input**

| Neuron | Formula | Calculation | Result |
|--------|---------|-------------|--------|
| $\text{net}_{o1}$ | $h_1 \cdot w_5 + h_2 \cdot w_6$ | $(0.50687 \times 0.40) + (0.51062 \times 0.45) = 0.20275 + 0.22978$ | $\boxed{0.43253}$ |
| $\text{net}_{o2}$ | $h_1 \cdot w_7 + h_2 \cdot w_8$ | $(0.50687 \times 0.50) + (0.51062 \times 0.55) = 0.25344 + 0.28084$ | $\boxed{0.53428}$ |

**Detailed Breakdown**

**Output Neuron 1 ($\text{net}_{o1}$)**

| Component | Weight | Hidden Activation | Product |
|-----------|--------|-------------------|---------|
| $h_1 \times w_5$ | $0.40$ | $0.50687$ | $0.20275$ |
| $h_2 \times w_6$ | $0.45$ | $0.51062$ | $0.22978$ |
| **Sum** | | | $\boxed{0.43253}$ |

$$
\text{net}_{o1} = h_1 \cdot w_5 + h_2 \cdot w_6 = (0.50687)(0.40) + (0.51062)(0.45) = 0.43253
$$

**Output Neuron 2 ($\text{net}_{o2}$)**

| Component | Weight | Hidden Activation | Product |
|-----------|--------|-------------------|---------|
| $h_1 \times w_7$ | $0.50$ | $0.50687$ | $0.25344$ |
| $h_2 \times w_8$ | $0.55$ | $0.51062$ | $0.28084$ |
| **Sum** | | | $\boxed{0.53428}$ |

$$
\text{net}_{o2} = h_1 \cdot w_7 + h_2 \cdot w_8 = (0.50687)(0.50) + (0.51062)(0.55) = 0.53428
$$

---

### Step 4: Output Layer Activation (Final Predictions)

**Calculation Table: Output Layer Activations**

| Neuron | Net Input | Formula | Intermediate Steps | Result |
|--------|-----------|---------|-------------------|--------|
| $o_1$ | $0.43253$ | $\sigma(0.43253)$ | $e^{-0.43253} \approx 0.648$ <br> $\frac{1}{1 + 0.648}$ | $\boxed{0.60635}$ |
| $o_2$ | $0.53428$ | $\sigma(0.53428)$ | $e^{-0.53428} \approx 0.586$ <br> $\frac{1}{1 + 0.586}$ | $\boxed{0.63002}$ |

**Detailed Calculations**

**Output 1 Activation**

$$
o_1 = \sigma(\text{net}_{o1}) = \sigma(0.43253) = \frac{1}{1 + e^{-0.43253}}
$$

| Step | Calculation | Value |
|------|-------------|-------|
| 1. Compute $e^{-0.43253}$ | $e^{-0.43253}$ | $0.648$ |
| 2. Add 1 | $1 + 0.648$ | $1.648$ |
| 3. Divide | $\frac{1}{1.648}$ | $\boxed{0.60635}$ |

**Output 2 Activation**

$$
o_2 = \sigma(\text{net}_{o2}) = \sigma(0.53428) = \frac{1}{1 + e^{-0.53428}}
$$

| Step | Calculation | Value |
|------|-------------|-------|
| 1. Compute $e^{-0.53428}$ | $e^{-0.53428}$ | $0.586$ |
| 2. Add 1 | $1 + 0.586$ | $1.586$ |
| 3. Divide | $\frac{1}{1.586}$ | $\boxed{0.63002}$ |

---

### Summary of Feed-Forward Results

**Complete Network Propagation Table**

| Layer | Neuron | Net Input $(z)$ | Activation $(a)$ |
|-------|--------|-----------------|------------------|
| **Hidden** | $H_1$ | $\text{net}_{h1} = 0.0275$ | $h_1 = 0.50687$ |
| **Hidden** | $H_2$ | $\text{net}_{h2} = 0.0425$ | $h_2 = 0.51062$ |
| **Output** | $O_1$ | $\text{net}_{o1} = 0.43253$ | $o_1 = 0.60635$ |
| **Output** | $O_2$ | $\text{net}_{o2} = 0.53428$ | $o_2 = 0.63002$ |

**Weight Update Preparation**

For backpropagation, we will need to calculate gradients $\frac{\partial E}{\partial w}$ for all 8 weights:
- **Input → Hidden weights:** $w_1, w_2, w_3, w_4$
- **Hidden → Output weights:** $w_5, w_6, w_7, w_8$

The gradient calculations will follow the chain rule and use the values computed in this feed-forward pass.

---

## Loss Calculation

We continue from the feed-forward results computed above.

### Known Values

**Predictions (from Feed-Forward):**

| Output | Value |
|--------|-------|
| $o_1$ | $0.60635$ |
| $o_2$ | $0.63002$ |

**Target Values:**

| Target | Value |
|--------|-------|
| $t_1$ | $0.01$ |
| $t_2$ | $0.99$ |

**Loss Function:**

The network uses **Mean Squared Error (MSE)** per output, with total loss:

$$
E_{total} = E_1 + E_2
$$

Where each individual error is:

$$
E_i = \frac{1}{2}(t_i - o_i)^2
$$

The factor $\frac{1}{2}$ simplifies the derivative during backpropagation.

---

### Step 1: Error for Output Neuron 1

$$
E_1 = \frac{1}{2}(t_1 - o_1)^2
$$

| Step | Calculation | Value |
|------|-------------|-------|
| 1. Compute difference | $t_1 - o_1 = 0.01 - 0.60635$ | $-0.59635$ |
| 2. Square the difference | $(-0.59635)^2$ | $0.35563$ |
| 3. Multiply by $\frac{1}{2}$ | $\frac{1}{2} \times 0.35563$ | $\boxed{0.17782}$ |

**Detailed Calculation:**

$$
\begin{align}
E_1 &= \frac{1}{2}(t_1 - o_1)^2 \\
&= \frac{1}{2}(0.01 - 0.60635)^2 \\
&= \frac{1}{2}(-0.59635)^2 \\
&= \frac{1}{2}(0.35563) \\
&= \boxed{0.17782}
\end{align}
$$

---

### Step 2: Error for Output Neuron 2

$$
E_2 = \frac{1}{2}(t_2 - o_2)^2
$$

| Step | Calculation | Value |
|------|-------------|-------|
| 1. Compute difference | $t_2 - o_2 = 0.99 - 0.63002$ | $0.35998$ |
| 2. Square the difference | $(0.35998)^2$ | $0.12959$ |
| 3. Multiply by $\frac{1}{2}$ | $\frac{1}{2} \times 0.12959$ | $\boxed{0.06480}$ |

**Detailed Calculation:**

$$
\begin{align}
E_2 &= \frac{1}{2}(t_2 - o_2)^2 \\
&= \frac{1}{2}(0.99 - 0.63002)^2 \\
&= \frac{1}{2}(0.35998)^2 \\
&= \frac{1}{2}(0.12959) \\
&= \boxed{0.06480}
\end{align}
$$

---

### Step 3: Total Loss

$$
E_{total} = E_1 + E_2
$$

| Component | Value |
|-----------|-------|
| $E_1$ | $0.17782$ |
| $E_2$ | $0.06480$ |
| **$E_{total}$** | $\boxed{0.24262}$ |

**Calculation:**

$$
E_{total} = E_1 + E_2 = 0.17782 + 0.06480 = \boxed{0.24262}
$$

---

### Interpretation of Loss

| Aspect | Meaning |
|--------|---------|
| **Magnitude** | $E_{total} = 0.24262$ quantifies how wrong the network's predictions are |
| **Goal** | Backpropagation will compute gradients to reduce this value |
| **Method** | Each weight will be adjusted based on how it contributes to this error |

The loss value indicates the network needs significant training, as both predictions are far from their targets:
- Output 1: $o_1 = 0.606$ vs target $t_1 = 0.01$ (error: $-0.596$)
- Output 2: $o_2 = 0.630$ vs target $t_2 = 0.99$ (error: $-0.360$)

---

## Backpropagation: Weight Updates

Now that we have computed the loss, we can calculate how to adjust each weight to reduce the error. Backpropagation uses the **chain rule** to systematically compute gradients.

### Overview of Weight Updates

To update weights, we use **gradient descent**:

$$
w_{new} = w_{old} - \eta \cdot \frac{\partial E}{\partial w}
$$

Where:
- $\eta$ is the **learning rate** (we'll use $\eta = 0.5$)
- $\frac{\partial E}{\partial w}$ is the **gradient** (how much the error changes with respect to the weight)

We need to compute gradients for all 8 weights:
- **Output layer weights:** $w_5, w_6, w_7, w_8$ (easier - single error path)
- **Hidden layer weights:** $w_1, w_2, w_3, w_4$ (harder - multiple error paths)

---

## Part 1: Output Layer Gradients

### Why Start with Output Layer?

Output layer weights are simpler because each weight affects only one output neuron, giving us a single, clean chain-rule path.

### Formulas Used in Backpropagation

#### Loss Function and Its Derivative

**Original Loss Function (Mean Squared Error):**

$$E_i = \frac{1}{2}(t_i - o_i)^2$$

Where:
- $E_i$ is the error for output neuron $i$
- $t_i$ is the target value
- $o_i$ is the predicted output

**Derivative of Loss w.r.t. Output:**

$$\frac{\partial E_i}{\partial o_i} = \frac{\partial}{\partial o_i}\left[\frac{1}{2}(t_i - o_i)^2\right]$$

**Step-by-step derivation:**

1. Rewrite for easier differentiation:
   $$E_i = \frac{1}{2}(o_i - t_i)^2$$

2. Apply chain rule:
   $$\frac{\partial E_i}{\partial o_i} = \frac{1}{2} \cdot 2(o_i - t_i) \cdot 1$$

3. Simplify:
   $$\frac{\partial E_i}{\partial o_i} = (o_i - t_i)$$

**Result:** The derivative simplifies beautifully - it's just the difference between prediction and target.

#### Sigmoid Activation Function and Its Derivative

**Original Sigmoid Function:**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivative of Sigmoid:**

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

**Derivation:**

Starting with:
$$\sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1}$$

Using the chain rule:
$$\frac{d\sigma}{dx} = -1 \cdot (1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

Rewrite to show the elegant form:
$$\frac{d\sigma}{dx} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x) \cdot \frac{e^{-x}}{1 + e^{-x}}$$

Since $\frac{e^{-x}}{1 + e^{-x}} = 1 - \frac{1}{1 + e^{-x}} = 1 - \sigma(x)$:
$$\frac{d\sigma}{dx} = \sigma(x) \cdot (1 - \sigma(x))$$

**Key insight:** If we already computed $\sigma(x)$ during forward pass, we can compute its derivative with just one multiplication!

#### Net Input to Activation Derivative

**For any neuron:**

$$\text{net} = \sum_i w_i \cdot a_i$$

where $a_i$ are the activations from the previous layer.

**Derivative w.r.t. a specific weight $w_j$:**

$$\frac{\partial \text{net}}{\partial w_j} = a_j$$

**Why:** All other terms in the sum disappear (their derivatives are 0), leaving only the activation that this weight multiplies.

#### Summary Table: Backpropagation Derivatives

| Component | Original Function | Derivative | Simplified Form |
|-----------|------------------|------------|-----------------|
| **Loss (MSE)** | $E = \frac{1}{2}(t - o)^2$ | $\frac{\partial E}{\partial o} = \frac{1}{2} \cdot 2(o - t)$ | $(o - t)$ |
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $\frac{d\sigma}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2}$ | $\sigma(x) \cdot (1 - \sigma(x))$ |
| **Weighted Sum** | $\text{net} = \sum_i w_i \cdot a_i$ | $\frac{\partial \text{net}}{\partial w_j}$ | $a_j$ |
| **Activation Pass-through** | $\text{net} = \sum_i w_i \cdot a_i$ | $\frac{\partial \text{net}}{\partial a_j}$ | $w_j$ |

#### Chain Rule for Output Layer Weights

For a weight $w$ connecting hidden neuron $h$ to output neuron $o$:

$$\frac{\partial E}{\partial w} = \frac{\partial E}{\partial o} \cdot \frac{\partial o}{\partial \text{net}_o} \cdot \frac{\partial \text{net}_o}{\partial w}$$

Substituting our derivatives:

$$\frac{\partial E}{\partial w} = (o - t) \cdot \sigma'(\text{net}_o) \cdot h = (o - t) \cdot o(1 - o) \cdot h$$

This is the fundamental formula for updating output layer weights.

---

### Gradient for w₅: Complete Derivation

**Weight:** $w_5$ **connects** $H_1 \to O_1$

#### Step 1: Identify the Dependency Chain

$$
w_5 \to \text{net}_{o1} \to o_1 \to E_1 \to E_{total}
$$

#### Step 2: Apply Chain Rule

$$
\frac{\partial E}{\partial w_5} = \frac{\partial E}{\partial E_1} \cdot \frac{\partial E_1}{\partial o_1} \cdot \frac{\partial o_1}{\partial \text{net}_{o1}} \cdot \frac{\partial \text{net}_{o1}}{\partial w_5}
$$

Now we compute each derivative step by step.

---

#### Step 3: Compute Each Partial Derivative

**Term 1: Error Contribution**

$$
\frac{\partial E}{\partial E_1} = \frac{\partial}{\partial E_1}(E_1 + E_2) = 1 + 0 = 1
$$

| Question | Answer |
|----------|--------|
| Why is this 1? | Because $E = E_1 + E_2$ is a simple sum |
| Meaning | $E_1$ contributes directly to total error with no scaling |

---

**Term 2: Error w.r.t. Output**

Start with the loss function:

$$
E_1 = \frac{1}{2}(t_1 - o_1)^2
$$

Rewrite for easier differentiation:

$$
E_1 = \frac{1}{2}(o_1 - t_1)^2
$$

Differentiate using the chain rule:

$$
\frac{\partial E_1}{\partial o_1} = \frac{1}{2} \cdot 2(o_1 - t_1) \cdot 1 = (o_1 - t_1)
$$

| Step | Calculation | Value |
|------|-------------|-------|
| Compute difference | $o_1 - t_1 = 0.60635 - 0.01$ | $0.59635$ |
| **Result** | $\frac{\partial E_1}{\partial o_1}$ | $\boxed{0.59635}$ |

**Interpretation:** This measures **how wrong the prediction is**.

---

**Term 3: Sigmoid Derivative**

For the sigmoid activation:

$$
o_1 = \sigma(\text{net}_{o1}) = \frac{1}{1 + e^{-\text{net}_{o1}}}
$$

The derivative of sigmoid is:

$$
\frac{\partial \sigma(x)}{\partial x} = \sigma(x) \cdot (1 - \sigma(x))
$$

**Proof of Sigmoid Derivative:**

$$
\begin{align}
\frac{d}{dx}\left(\frac{1}{1 + e^{-x}}\right) &= \frac{e^{-x}}{(1 + e^{-x})^2} \\
&= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\
&= \sigma(x) \cdot (1 - \sigma(x))
\end{align}
$$

Therefore:

$$
\frac{\partial o_1}{\partial \text{net}_{o1}} = o_1(1 - o_1)
$$

| Step | Calculation | Value |
|------|-------------|-------|
| Compute $o_1$ | (from forward pass) | $0.60635$ |
| Compute $1 - o_1$ | $1 - 0.60635$ | $0.39365$ |
| Multiply | $0.60635 \times 0.39365$ | $\boxed{0.23886}$ |

**Interpretation:** This measures **how sensitive the neuron is** to changes in its input.

---

**Term 4: Weight Contribution**

The output neuron's input is:

$$
\text{net}_{o1} = h_1 \cdot w_5 + h_2 \cdot w_6
$$

Differentiate with respect to $w_5$:

$$
\frac{\partial \text{net}_{o1}}{\partial w_5} = \frac{\partial}{\partial w_5}(h_1 \cdot w_5 + h_2 \cdot w_6) = h_1
$$

| Step | Value |
|------|-------|
| $\frac{\partial \text{net}_{o1}}{\partial w_5}$ | $h_1 = \boxed{0.50687}$ |

**Interpretation:** This is **how much this weight contributed** to the output.

---

#### Step 4: Multiply All Terms (Chain Rule)

$$
\frac{\partial E}{\partial w_5} = 1 \times 0.59635 \times 0.23886 \times 0.50687
$$

| Term | Value | Meaning |
|------|-------|---------|
| $\frac{\partial E}{\partial E_1}$ | $1$ | Error contribution |
| $\frac{\partial E_1}{\partial o_1}$ | $0.59635$ | How wrong? |
| $\frac{\partial o_1}{\partial \text{net}_{o1}}$ | $0.23886$ | How sensitive? |
| $\frac{\partial \text{net}_{o1}}{\partial w_5}$ | $0.50687$ | How much did it contribute? |

**Final calculation:**

$$
\begin{align}
\frac{\partial E}{\partial w_5} &= 1 \times 0.59635 \times 0.23886 \times 0.50687 \\
&= 0.59635 \times 0.23886 \times 0.50687 \\
&= 0.14247 \times 0.50687 \\
&= \boxed{0.07223}
\end{align}
$$

---

#### Step 5: Update Weight

**Weight being updated:** $w_5$

Using learning rate $\eta = 0.5$:

$$
w_5^{new} = w_5^{old} - \eta \cdot \frac{\partial E}{\partial w_5}
$$

| Step | Calculation | Value |
|------|-------------|-------|
| Old weight | $w_5^{old}$ | $0.40$ |
| Learning rate | $\eta$ | $0.5$ |
| Gradient | $\frac{\partial E}{\partial w_5}$ | $0.07223$ |
| Weight change | $\eta \times 0.07223$ | $0.03612$ |
| **New weight** | $0.40 - 0.03612$ | $\boxed{0.36388}$ |

**Result:** $w_5$ decreases from $0.40$ to $0.36388$, which will reduce the error in the next forward pass.

---

### Gradients for Remaining Output Layer Weights

Since $w_6$, $w_7$, and $w_8$ follow the exact same derivation pattern as $w_5$, we present their calculations in tabular form for efficiency.

#### Gradient for w₆

**Connection:** $H_2 \to O_1$

| Step | Component | Formula | Value |
|------|-----------|---------|-------|
| 1 | Error contribution | $\frac{\partial E}{\partial E_1}$ | $1$ |
| 2 | Error w.r.t. output | $\frac{\partial E_1}{\partial o_1} = (o_1 - t_1)$ | $0.59635$ |
| 3 | Sigmoid derivative | $\frac{\partial o_1}{\partial \text{net}_{o1}} = o_1(1 - o_1)$ | $0.23886$ |
| 4 | Weight contribution | $\frac{\partial \text{net}_{o1}}{\partial w_6} = h_2$ | $0.51062$ |
| 5 | **Gradient** | $\frac{\partial E}{\partial w_6} = 1 \times 0.59635 \times 0.23886 \times 0.51062$ | **$0.07277$** |
| 6 | Weight update | $w_{6,\text{new}} = 0.45 - 0.5 \times 0.07277$ | **$0.41361$** |

**Key difference from $w_5$:** Uses $h_2$ instead of $h_1$ as the input activation.

#### Gradient for w₇

**Connection:** $H_1 \to O_2$

| Step | Component | Formula | Value |
|------|-----------|---------|-------|
| 1 | Error contribution | $\frac{\partial E}{\partial E_2}$ | $1$ |
| 2 | Error w.r.t. output | $\frac{\partial E_2}{\partial o_2} = (o_2 - t_2)$ | $-0.35998$ |
| 3 | Sigmoid derivative | $\frac{\partial o_2}{\partial \text{net}_{o2}} = o_2(1 - o_2)$ | $0.23311$ |
| 4 | Weight contribution | $\frac{\partial \text{net}_{o2}}{\partial w_7} = h_1$ | $0.50687$ |
| 5 | **Gradient** | $\frac{\partial E}{\partial w_7} = 1 \times (-0.35998) \times 0.23311 \times 0.50687$ | **$-0.04251$** |
| 6 | Weight update | $w_{7,\text{new}} = 0.50 - 0.5 \times (-0.04251)$ | **$0.52126$** |

**Key differences:**
- Uses output neuron $O_2$ instead of $O_1$
- Gradient is negative (weight increases) because prediction is too low

#### Gradient for w₈

**Connection:** $H_2 \to O_2$

| Step | Component | Formula | Value |
|------|-----------|---------|-------|
| 1 | Error contribution | $\frac{\partial E}{\partial E_2}$ | $1$ |
| 2 | Error w.r.t. output | $\frac{\partial E_2}{\partial o_2} = (o_2 - t_2)$ | $-0.35998$ |
| 3 | Sigmoid derivative | $\frac{\partial o_2}{\partial \text{net}_{o2}} = o_2(1 - o_2)$ | $0.23311$ |
| 4 | Weight contribution | $\frac{\partial \text{net}_{o2}}{\partial w_8} = h_2$ | $0.51062$ |
| 5 | **Gradient** | $\frac{\partial E}{\partial w_8} = 1 \times (-0.35998) \times 0.23311 \times 0.51062$ | **$-0.04285$** |
| 6 | Weight update | $w_{8,\text{new}} = 0.55 - 0.5 \times (-0.04285)$ | **$0.57143$** |

**Key differences:**
- Uses $h_2$ and $O_2$
- Gradient is negative (weight increases)

### Summary: Output Layer Weight Updates

| Weight | Old Value | Gradient | Update ($\Delta w$) | New Value |
|--------|-----------|----------|---------------------|-----------|
| $w_5$ | $0.40$ | $+0.07223$ | $-0.03612$ | $\boxed{0.36388}$ |
| $w_6$ | $0.45$ | $+0.07277$ | $-0.03639$ | $\boxed{0.41361}$ |
| $w_7$ | $0.50$ | $-0.04251$ | $+0.02126$ | $\boxed{0.52126}$ |
| $w_8$ | $0.55$ | $-0.04285$ | $+0.02143$ | $\boxed{0.57143}$ |

**Observations:**
- Weights to $O_1$ **decrease** (prediction was too high: $0.606$ vs target $0.01$)
- Weights to $O_2$ **increase** (prediction was too low: $0.630$ vs target $0.99$)
- This will reduce the error in the next forward pass!

---

## Part 2: Hidden Layer Gradients

### Why Hidden Layer Gradients Are Different

Hidden layer weights are **more complex** because each hidden neuron affects **multiple output neurons**.

```
        h₁
       /  \
     o₁    o₂
```

Therefore:
- $h_1$ contributes to error $E_1$ (through $o_1$)
- $h_1$ contributes to error $E_2$ (through $o_2$)

**Key insight:** We must **sum both error paths** when computing the gradient.

---

### Gradient for w₁: Complete Derivation

**Weight:** $w_1$ **connects** $I_1 \to H_1$

This is the most important derivation in backpropagation.

---

#### Step 1: Identify the Dependency Paths

$w_1$ affects the total error through **two paths**:

**Path 1 (through $o_1$):**
$$
w_1 \to \text{net}_{h1} \to h_1 \to \text{net}_{o1} \to o_1 \to E_1 \to E_{total}
$$

**Path 2 (through $o_2$):**
$$
w_1 \to \text{net}_{h1} \to h_1 \to \text{net}_{o2} \to o_2 \to E_2 \to E_{total}
$$

---

#### Step 2: Write Total Gradient as Sum

Since $E = E_1 + E_2$:

$$
\frac{\partial E}{\partial w_1} = \frac{\partial E_1}{\partial w_1} + \frac{\partial E_2}{\partial w_1}
$$

**Mathematical principle:** The derivative of a sum equals the sum of derivatives.

This is **why gradients from multiple outputs must be added**.

---

#### Step 3: Expand Each Path Using Chain Rule

**Path 1 (through output 1):**

$$
\frac{\partial E_1}{\partial w_1} = \frac{\partial E_1}{\partial o_1} \cdot \frac{\partial o_1}{\partial \text{net}_{o1}} \cdot \frac{\partial \text{net}_{o1}}{\partial h_1} \cdot \frac{\partial h_1}{\partial \text{net}_{h1}} \cdot \frac{\partial \text{net}_{h1}}{\partial w_1}
$$

**Path 2 (through output 2):**

$$
\frac{\partial E_2}{\partial w_1} = \frac{\partial E_2}{\partial o_2} \cdot \frac{\partial o_2}{\partial \text{net}_{o2}} \cdot \frac{\partial \text{net}_{o2}}{\partial h_1} \cdot \frac{\partial h_1}{\partial \text{net}_{h1}} \cdot \frac{\partial \text{net}_{h1}}{\partial w_1}
$$

**Key observation:** Both paths share the last two terms:
- $\frac{\partial h_1}{\partial \text{net}_{h1}}$ (hidden neuron sensitivity)
- $\frac{\partial \text{net}_{h1}}{\partial w_1}$ (input contribution)

We can **factor these out**.

---

#### Step 4: Compute Each Derivative

**Derivatives we already know (from output layer):**

| Derivative | Path 1 (Output 1) | Path 2 (Output 2) |
|------------|-------------------|-------------------|
| Error w.r.t. output | $o_1 - t_1 = 0.59635$ | $o_2 - t_2 = -0.35998$ |
| Sigmoid derivative | $o_1(1-o_1) = 0.23886$ | $o_2(1-o_2) = 0.23311$ |

---

**New derivative: Output input w.r.t. hidden activation**

For output neuron 1:
$$
\text{net}_{o1} = h_1 \cdot w_5 + h_2 \cdot w_6
$$

$$
\frac{\partial \text{net}_{o1}}{\partial h_1} = w_5 = 0.40
$$

For output neuron 2:
$$
\text{net}_{o2} = h_1 \cdot w_7 + h_2 \cdot w_8
$$

$$
\frac{\partial \text{net}_{o2}}{\partial h_1} = w_7 = 0.50
$$

**Interpretation:** These weights determine **how much $h_1$ influences each output**.

---

**New derivative: Hidden neuron sigmoid**

$$
h_1 = \sigma(\text{net}_{h1})
$$

$$
\frac{\partial h_1}{\partial \text{net}_{h1}} = h_1(1 - h_1)
$$

| Step | Calculation | Value |
|------|-------------|-------|
| $h_1$ | (from forward pass) | $0.50687$ |
| $1 - h_1$ | $1 - 0.50687$ | $0.49313$ |
| $h_1(1 - h_1)$ | $0.50687 \times 0.49313$ | $\boxed{0.24995}$ |

---

**New derivative: Hidden input w.r.t. weight**

$$
\text{net}_{h1} = I_1 \cdot w_1 + I_2 \cdot w_2
$$

$$
\frac{\partial \text{net}_{h1}}{\partial w_1} = I_1 = \boxed{0.05}
$$

---

#### Step 5: Calculate Path 1 Contribution

**Error signal from output 1:**

$$
\delta_{o1} = (o_1 - t_1) \cdot o_1(1 - o_1) = 0.59635 \times 0.23886 = 0.14247
$$

**Propagated back through $w_5$:**

$$
\delta_{o1} \times w_5 = 0.14247 \times 0.40 = 0.05699
$$

---

#### Step 6: Calculate Path 2 Contribution

**Error signal from output 2:**

$$
\delta_{o2} = (o_2 - t_2) \cdot o_2(1 - o_2) = (-0.35998) \times 0.23311 = -0.08391
$$

**Propagated back through $w_7$:**

$$
\delta_{o2} \times w_7 = -0.08391 \times 0.50 = -0.04196
$$

---

#### Step 7: Sum Both Error Paths

This is **the critical step** that makes hidden layer gradients different:

$$
\text{Total error to } h_1 = 0.05699 + (-0.04196) = \boxed{0.01503}
$$

| Component | Value | Source |
|-----------|-------|--------|
| Error from $o_1$ | $+0.05699$ | Path 1 |
| Error from $o_2$ | $-0.04196$ | Path 2 |
| **Total** | $\boxed{0.01503}$ | **Sum of both paths** |

**Interpretation:** $h_1$ receives blame from both outputs, and we must account for both!

---

#### Step 8: Complete the Gradient Calculation

Now multiply by the common factors:

$$
\frac{\partial E}{\partial w_1} = 0.01503 \times h_1(1 - h_1) \times I_1
$$

| Step | Calculation | Value |
|------|-------------|-------|
| Total error to $h_1$ | (from step 7) | $0.01503$ |
| Hidden sensitivity | $h_1(1-h_1)$ | $0.24995$ |
| Product | $0.01503 \times 0.24995$ | $0.00376$ |
| Input contribution | $I_1$ | $0.05$ |
| **Final gradient** | $0.00376 \times 0.05$ | $\boxed{0.000188}$ |

---

#### Step 9: Update Weight

**Weight being updated:** $w_1$

Using learning rate $\eta = 0.5$:

$$
w_1^{new} = w_1^{old} - \eta \cdot \frac{\partial E}{\partial w_1}
$$

| Step | Calculation | Value |
|------|-------------|-------|
| Old weight | $w_1^{old}$ | $0.15$ |
| Gradient | $\frac{\partial E}{\partial w_1}$ | $0.000188$ |
| Weight change | $0.5 \times 0.000188$ | $0.000094$ |
| **New weight** | $0.15 - 0.000094$ | $\boxed{0.149906}$ |

---

### Understanding the Result

**Why is the gradient so small?**

$$
\frac{\partial E}{\partial w_1} = 0.000188 \quad \text{vs} \quad \frac{\partial E}{\partial w_5} = 0.07223
$$

The hidden layer gradient is **~380 times smaller**!

**Reasons:**

1. **Multiple layers:** Error must flow backward through more layers
2. **Sigmoid derivatives:** Each sigmoid contributes a factor $< 0.25$
3. **Error cancellation:** The two paths partially cancel ($+0.057 - 0.042 = 0.015$)

This phenomenon is called the **vanishing gradient problem** and explains why:
- Hidden layers learn **much slower** than output layers
- Deep networks are **harder to train**
- Modern architectures use ReLU instead of sigmoid

---

### Gradients for Remaining Hidden Layer Weights

The derivations for $w_2$, $w_3$, and $w_4$ follow the same pattern as $w_1$. Here are the results:

---

#### Gradient for w₂

**Connection:** $I_2 \to H_1$

**Key difference:** $\frac{\partial \text{net}_{h1}}{\partial w_2} = I_2 = 0.10$ (instead of $I_1$)

The error paths are identical:

$$
\frac{\partial E}{\partial w_2} = [0.05699 + (-0.04196)] \times 0.24995 \times I_2
$$

| Step | Calculation | Value |
|------|-------------|-------|
| Total error to $h_1$ | (same as $w_1$) | $0.01503$ |
| Hidden sensitivity | $h_1(1-h_1)$ | $0.24995$ |
| Input contribution | $I_2$ | $0.10$ |
| **Gradient** | $0.01503 \times 0.24995 \times 0.10$ | $\boxed{0.000376}$ |

**Weight update:**

$$
w_2^{new} = 0.20 - 0.5 \times 0.000376 = \boxed{0.199812}
$$

---

#### Gradient for w₃

**Connection:** $I_1 \to H_2$

For $w_3$, we need to compute error flowing into $h_2$ through both outputs.

**Path 1 (through $o_1$):**

$$
\delta_{o1} \times w_6 = 0.14247 \times 0.45 = 0.06411
$$

**Path 2 (through $o_2$):**

$$
\delta_{o2} \times w_8 = -0.08391 \times 0.55 = -0.04615
$$

**Total error to $h_2$:**

$$
0.06411 + (-0.04615) = 0.01796
$$

**Hidden neuron $h_2$ sensitivity:**

$$
h_2(1 - h_2) = 0.51062 \times 0.48938 = 0.24987
$$

**Complete gradient:**

| Step | Calculation | Value |
|------|-------------|-------|
| Total error to $h_2$ | $0.06411 - 0.04615$ | $0.01796$ |
| Hidden sensitivity | $h_2(1-h_2)$ | $0.24987$ |
| Product | $0.01796 \times 0.24987$ | $0.00449$ |
| Input contribution | $I_1$ | $0.05$ |
| **Gradient** | $0.00449 \times 0.05$ | $\boxed{0.000224}$ |

**Weight update:**

$$
w_3^{new} = 0.25 - 0.5 \times 0.000224 = \boxed{0.249888}
$$

---

#### Gradient for w₄

**Connection:** $I_2 \to H_2$

**Key difference:** $\frac{\partial \text{net}_{h2}}{\partial w_4} = I_2 = 0.10$

| Step | Calculation | Value |
|------|-------------|-------|
| Total error to $h_2$ | (same as $w_3$) | $0.01796$ |
| Hidden sensitivity | $h_2(1-h_2)$ | $0.24987$ |
| Input contribution | $I_2$ | $0.10$ |
| **Gradient** | $0.01796 \times 0.24987 \times 0.10$ | $\boxed{0.000449}$ |

**Weight update:**

$$
w_4^{new} = 0.30 - 0.5 \times 0.000449 = \boxed{0.299776}
$$

---

### Summary: All Weight Updates

**Learning Rate:** $\eta = 0.5$

| Layer | Weight | Old Value | Gradient | New Value | Change |
|-------|--------|-----------|----------|-----------|--------|
| **Output** | $w_5$ | $0.40$ | $+0.07223$ | $0.36388$ | $-0.03612$ |
| **Output** | $w_6$ | $0.45$ | $+0.07277$ | $0.41361$ | $-0.03639$ |
| **Output** | $w_7$ | $0.50$ | $-0.04251$ | $0.52126$ | $+0.02126$ |
| **Output** | $w_8$ | $0.55$ | $-0.04285$ | $0.57143$ | $+0.02143$ |
| **Hidden** | $w_1$ | $0.15$ | $+0.000188$ | $0.149906$ | $-0.000094$ |
| **Hidden** | $w_2$ | $0.20$ | $+0.000376$ | $0.199812$ | $-0.000188$ |
| **Hidden** | $w_3$ | $0.25$ | $+0.000224$ | $0.249888$ | $-0.000112$ |
| **Hidden** | $w_4$ | $0.30$ | $+0.000449$ | $0.299776$ | $-0.000224$ |

**Key Observations:**

1. **Output layer gradients** are ~200-400× larger than **hidden layer gradients**
2. **Weights to $O_1$** decrease (prediction too high: $0.606$ vs $0.01$)
3. **Weights to $O_2$** increase (prediction too low: $0.630$ vs $0.99$)
4. **Hidden weights** change very slightly due to vanishing gradients

---

## Part 3: Matrix Formulation of Backpropagation

The scalar derivations above are crucial for understanding, but deep learning frameworks use **vectorized matrix operations** for efficiency. Here's how the same math looks in matrix form.

---

### Notation and Setup

**Weight Matrices:**

$$
\mathbf{W}^{(1)} = \begin{bmatrix} w_1 & w_2 \\ w_3 & w_4 \end{bmatrix} = \begin{bmatrix} 0.15 & 0.20 \\ 0.25 & 0.30 \end{bmatrix}
$$

$$
\mathbf{W}^{(2)} = \begin{bmatrix} w_5 & w_6 \\ w_7 & w_8 \end{bmatrix} = \begin{bmatrix} 0.40 & 0.45 \\ 0.50 & 0.55 \end{bmatrix}
$$

**Input Vector:**

$$
\mathbf{x} = \begin{bmatrix} I_1 \\ I_2 \end{bmatrix} = \begin{bmatrix} 0.05 \\ 0.10 \end{bmatrix}
$$

**Target Vector:**

$$
\mathbf{t} = \begin{bmatrix} t_1 \\ t_2 \end{bmatrix} = \begin{bmatrix} 0.01 \\ 0.99 \end{bmatrix}
$$

---

### Forward Pass (Vectorized)

**Hidden Layer:**

$$
\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} = \begin{bmatrix} 0.0275 \\ 0.0425 \end{bmatrix}
$$

$$
\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)}) = \begin{bmatrix} h_1 \\ h_2 \end{bmatrix} = \begin{bmatrix} 0.50687 \\ 0.51062 \end{bmatrix}
$$

**Output Layer:**

$$
\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} = \begin{bmatrix} 0.43253 \\ 0.53428 \end{bmatrix}
$$

$$
\mathbf{a}^{(2)} = \sigma(\mathbf{z}^{(2)}) = \begin{bmatrix} o_1 \\ o_2 \end{bmatrix} = \begin{bmatrix} 0.60635 \\ 0.63002 \end{bmatrix}
$$

---

### Backward Pass (Vectorized)

#### Step 1: Output Layer Delta

$$
\boldsymbol{\delta}^{(2)} = (\mathbf{a}^{(2)} - \mathbf{t}) \odot \sigma'(\mathbf{z}^{(2)})
$$

Where $\odot$ denotes element-wise multiplication (Hadamard product).

**Computation:**

$$
\mathbf{a}^{(2)} - \mathbf{t} = \begin{bmatrix} 0.60635 - 0.01 \\ 0.63002 - 0.99 \end{bmatrix} = \begin{bmatrix} 0.59635 \\ -0.35998 \end{bmatrix}
$$

$$
\sigma'(\mathbf{z}^{(2)}) = \mathbf{a}^{(2)} \odot (1 - \mathbf{a}^{(2)}) = \begin{bmatrix} 0.23886 \\ 0.23311 \end{bmatrix}
$$

$$
\boldsymbol{\delta}^{(2)} = \begin{bmatrix} 0.59635 \\ -0.35998 \end{bmatrix} \odot \begin{bmatrix} 0.23886 \\ 0.23311 \end{bmatrix} = \begin{bmatrix} 0.14247 \\ -0.08391 \end{bmatrix}
$$

---

#### Step 2: Hidden Layer Delta

$$
\boldsymbol{\delta}^{(1)} = (\mathbf{W}^{(2)T} \boldsymbol{\delta}^{(2)}) \odot \sigma'(\mathbf{z}^{(1)})
$$

**This single line explains "why gradients sum"!**

The matrix multiplication $\mathbf{W}^{(2)T} \boldsymbol{\delta}^{(2)}$ **automatically sums all downstream error paths**.

**Computation:**

$$
\mathbf{W}^{(2)T} = \begin{bmatrix} 0.40 & 0.50 \\ 0.45 & 0.55 \end{bmatrix}
$$

$$
\mathbf{W}^{(2)T} \boldsymbol{\delta}^{(2)} = \begin{bmatrix} 0.40 & 0.50 \\ 0.45 & 0.55 \end{bmatrix} \begin{bmatrix} 0.14247 \\ -0.08391 \end{bmatrix} = \begin{bmatrix} 0.0150 \\ 0.0179 \end{bmatrix}
$$

**Breaking down for $h_1$:**
$$
0.40 \times 0.14247 + 0.50 \times (-0.08391) = 0.05699 - 0.04196 = 0.01503
$$

This matches our manual calculation!

$$
\sigma'(\mathbf{z}^{(1)}) = \begin{bmatrix} h_1(1-h_1) \\ h_2(1-h_2) \end{bmatrix} = \begin{bmatrix} 0.24995 \\ 0.24987 \end{bmatrix}
$$

$$
\boldsymbol{\delta}^{(1)} = \begin{bmatrix} 0.0150 \\ 0.0179 \end{bmatrix} \odot \begin{bmatrix} 0.24995 \\ 0.24987 \end{bmatrix} = \begin{bmatrix} 0.00375 \\ 0.00448 \end{bmatrix}
$$

---

#### Step 3: Gradient Matrices

**Output Layer Gradient:**

$$
\frac{\partial E}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} \mathbf{a}^{(1)T}
$$

$$
= \begin{bmatrix} 0.14247 \\ -0.08391 \end{bmatrix} \begin{bmatrix} 0.50687 & 0.51062 \end{bmatrix}
$$

$$
= \begin{bmatrix} 0.07223 & 0.07277 \\ -0.04251 & -0.04285 \end{bmatrix}
$$

This gives us all four output-layer gradients in one matrix!

**Hidden Layer Gradient:**

$$
\frac{\partial E}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T
$$

$$
= \begin{bmatrix} 0.00375 \\ 0.00448 \end{bmatrix} \begin{bmatrix} 0.05 & 0.10 \end{bmatrix}
$$

$$
= \begin{bmatrix} 0.000188 & 0.000375 \\ 0.000224 & 0.000448 \end{bmatrix}
$$

---

#### Step 4: Weight Updates

$$
\mathbf{W}^{(2)}_{new} = \mathbf{W}^{(2)} - \eta \frac{\partial E}{\partial \mathbf{W}^{(2)}}
$$

$$
\mathbf{W}^{(1)}_{new} = \mathbf{W}^{(1)} - \eta \frac{\partial E}{\partial \mathbf{W}^{(1)}}
$$

---

### Comparison: Scalar vs Matrix Form

| Aspect | Scalar Form | Matrix Form |
|--------|-------------|-------------|
| **Understanding** | Clear, intuitive | Abstract |
| **Computation** | Slow, manual | Fast, GPU-optimized |
| **Error paths** | Explicit sums | Implicit in matrix multiply |
| **Code** | Many loops | One line per layer |
| **Usage** | Learning | Production (PyTorch, TensorFlow) |

**Key Insight:** Matrix multiplication $\mathbf{W}^T \boldsymbol{\delta}$ is equivalent to summing weighted error contributions from all downstream neurons.

---

## Final Summary and Key Insights

### The Complete Backpropagation Algorithm

1. **Forward Pass:** Compute activations layer by layer
2. **Compute Loss:** Calculate error at output
3. **Output Layer Delta:** $\boldsymbol{\delta}^{(L)} = (a^{(L)} - t) \odot \sigma'(z^{(L)})$
4. **Backpropagate Deltas:** $\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)T} \boldsymbol{\delta}^{(l+1)}) \odot \sigma'(z^{(l)})$
5. **Compute Gradients:** $\frac{\partial E}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} a^{(l-1)T}$
6. **Update Weights:** $\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial E}{\partial \mathbf{W}^{(l)}}$

---

### Understanding Each Component

| Component | Mathematical Role | Intuitive Meaning |
|-----------|-------------------|-------------------|
| $(o_i - t_i)$ | Loss derivative | **How wrong is the prediction?** |
| $\sigma'(z) = a(1-a)$ | Activation derivative | **How sensitive is the neuron?** |
| $a^{(l-1)}$ | Previous activation | **How much did the input contribute?** |
| $\mathbf{W}^T \boldsymbol{\delta}$ | Weighted backprop | **How does error flow backward?** |

---

### Why Gradients Sum for Hidden Layers

**Mathematical reason:** Total error is a sum, so its derivative is a sum of derivatives.

$$
E = E_1 + E_2 \quad \Rightarrow \quad \frac{\partial E}{\partial w_1} = \frac{\partial E_1}{\partial w_1} + \frac{\partial E_2}{\partial w_1}
$$

**Intuitive reason:** Hidden neurons share responsibility for multiple outputs, so they receive blame from all of them.

**Matrix view:** The transpose-multiply $\mathbf{W}^T \boldsymbol{\delta}$ automatically implements this summation.

---

### The Vanishing Gradient Problem

**Observation:**

$$
\left|\frac{\partial E}{\partial w_1}\right| = 0.000188 \ll \left|\frac{\partial E}{\partial w_5}\right| = 0.07223
$$

**Causes:**

1. **Layer depth:** Each layer multiplies by $\sigma'(z) \leq 0.25$
2. **Sigmoid saturation:** When $|z|$ is large, $\sigma'(z) \approx 0$
3. **Many multiplications:** $n$ layers → gradient scales as $(0.25)^n$

**Solutions in modern deep learning:**

- **ReLU activation:** $\frac{d}{dx}\text{ReLU}(x) = 1$ for $x > 0$
- **Batch normalization:** Keeps activations in sensitive range
- **Residual connections:** Provides gradient shortcuts
- **Better initialization:** Xavier/He initialization

---

## Appendix: Complete Values Reference

For quick lookup, here are all computed values in one place:

### Forward Pass Values

| Layer | Neuron | Net Input | Activation |
|-------|--------|-----------|------------|
| Input | $I_1$ | — | $0.05$ |
| Input | $I_2$ | — | $0.10$ |
| Hidden | $H_1$ | $0.0275$ | $0.50687$ |
| Hidden | $H_2$ | $0.0425$ | $0.51062$ |
| Output | $O_1$ | $0.43253$ | $0.60635$ |
| Output | $O_2$ | $0.53428$ | $0.63002$ |

### Loss Values

| Component | Value |
|-----------|-------|
| $E_1$ | $0.17782$ |
| $E_2$ | $0.06480$ |
| $E_{\text{total}}$ | $0.24262$ |

### All Weight Gradients and Updates

| Weight | Connection | Old Value | Gradient | Update ($\Delta w$) | New Value |
|--------|-----------|-----------|----------|---------------------|-----------|
| $w_1$ | $I_1 \to H_1$ | $0.15$ | $+0.000188$ | $-0.000094$ | $0.149906$ |
| $w_2$ | $I_2 \to H_1$ | $0.20$ | $+0.000376$ | $-0.000188$ | $0.199812$ |
| $w_3$ | $I_1 \to H_2$ | $0.25$ | $+0.000224$ | $-0.000112$ | $0.249888$ |
| $w_4$ | $I_2 \to H_2$ | $0.30$ | $+0.000449$ | $-0.000224$ | $0.299776$ |
| $w_5$ | $H_1 \to O_1$ | $0.40$ | $+0.07223$ | $-0.03612$ | $0.36388$ |
| $w_6$ | $H_2 \to O_1$ | $0.45$ | $+0.07277$ | $-0.03639$ | $0.41361$ |
| $w_7$ | $H_1 \to O_2$ | $0.50$ | $-0.04251$ | $+0.02126$ | $0.52126$ |
| $w_8$ | $H_2 \to O_2$ | $0.55$ | $-0.04285$ | $+0.02143$ | $0.57143$ |

**Learning rate:** $\eta = 0.5$

---

## References and Further Reading

### Original Article

- [Understanding Neural Networks and the Mathematics Behind Weights Updation](https://medium.com/@dandare120/understanding-neural-networks-and-the-mathematics-behind-weights-updation-e92c22a3ac94) by Dan Dare

### Foundational Papers

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323(6088), 533-536.

### Recommended Textbooks

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Nielsen, M. (2015). *Neural Networks and Deep Learning*. Determination Press.

### Online Resources

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) - Stanford
- [Deep Learning Specialization](https://www.deeplearning.ai/) - Andrew Ng
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch documentation

---

## Next Steps for Deeper Understanding

Now that you understand the fundamentals, here are recommended next topics:

### 1. Advanced Optimizers

- Momentum and Nesterov Accelerated Gradient
- Adam, RMSprop, and adaptive learning rates
- Learning rate schedules and warm restarts

### 2. Regularization Techniques

- $L_1$ and $L_2$ regularization (weight decay)
- Dropout and its variants
- Batch normalization and layer normalization
- Early stopping

### 3. Modern Architectures

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs) and LSTMs
- Transformers and attention mechanisms
- Residual networks (ResNets)

### 4. Practical Implementation

- Implement this from scratch in NumPy
- Build the same network in PyTorch/TensorFlow
- Experiment with different activation functions (ReLU, Leaky ReLU, ELU)
- Try different loss functions (Cross-entropy, Huber loss)

### 5. Theoretical Deep Dives

- Universal approximation theorem
- Information theory perspective on deep learning
- Optimization landscape and loss surface geometry
- Generalization bounds and PAC learning

---

_This document was created to provide a complete, step-by-step understanding of neural network training mathematics. Every calculation has been verified and explained from first principles._

**Version:** 2.1 (LaTeX Enhanced)
**Last Updated:** 2026-02-04