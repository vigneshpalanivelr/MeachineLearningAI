# Complete Machine Learning Course Guide
## CS1 • CS2 • CS3 • CS5: Linear Models & Regression + Classification

---

## MAIN TABLE OF CONTENTS

### [CS1 - Machine Learning Fundamentals](#cs1---machine-learning-fundamentals)
- [Definition](#cs1-definition)
- [Core Concepts](#cs1-core-concepts)
- [Target Function and Hypothesis](#cs1-target-function-and-hypothesis)
- [The Model](#cs1-the-model)
- [Generalization](#cs1-generalization)

### [CS2 - Machine Learning Workflow](#cs2---machine-learning-workflow)
- [Types of Attributes](#cs2-types-of-attributes)
- [Discrete vs Continuous Attributes](#cs2-discrete-vs-continuous-attributes)
- [Data Types](#cs2-data-types)
- [Data Pre-processing Steps](#cs2-data-pre-processing-steps)
- [Feature Engineering](#cs2-feature-engineering)

### [CS3 - Linear Regression & Regularization](#cs3---linear-regression--regularization)
- [What is Linear Regression?](#cs3-what-is-linear-regression)
- [Linear Regression Formulas](#cs3-linear-regression-formulas)
- [Types of Regression Models](#cs3-types-of-regression-models)
- [Least Squares Regression](#cs3-least-squares-regression)
- [Hypothesis Function](#cs3-hypothesis-function)
- [Ridge Regression (L2)](#cs3-ridge-regression-l2)
- [Lasso Regression (L1)](#cs3-lasso-regression-l1)
- [Ridge vs Lasso Comparison](#cs3-ridge-vs-lasso-comparison)
- [Elastic Net](#cs3-elastic-net)
- [Numerical Examples](#cs3-numerical-examples)

### [CS5 - Linear Models for Classification](#cs5---linear-models-for-classification)
- [What is Classification?](#cs5-what-is-classification)
- [Decision Theory](#cs5-decision-theory)
- [Generative vs Discriminative Models](#cs5-generative-vs-discriminative-models)
- [Discriminant Functions](#cs5-discriminant-functions)
- [Logistic Regression](#cs5-logistic-regression)
- [Cost Function (Cross-Entropy)](#cs5-cost-function-cross-entropy)
- [Gradient Descent for Classification](#cs5-gradient-descent-for-classification)
- [Multi-Class Classification](#cs5-multi-class-classification)
- [Regularization in Classification](#cs5-regularization-in-classification)
- [Evaluation Metrics](#cs5-evaluation-metrics)
- [Numerical Examples](#cs5-numerical-examples)

---

# CS1 - Machine Learning Fundamentals

## CS1 TABLE OF CONTENTS
1. [Definition](#cs1-definition)
2. [Core Concepts](#cs1-core-concepts)
3. [Target Function and Hypothesis](#cs1-target-function-and-hypothesis)
4. [The Model](#cs1-the-model)
5. [Generalization](#cs1-generalization)

---

## CS1 Definition

Machine Learning (ML) is a field of study about creating programs that automatically improve by learning from experience and data.

Instead of writing explicit rules, ML systems learn patterns from examples and use those patterns to make decisions.

### ML Draws From Multiple Disciplines

- **Statistics** - Probability, distributions, statistical inference
- **Artificial Intelligence (AI)** - Learning algorithms and reasoning methods
- **Information Theory** - Measuring and processing information
- **Control Theory** - Adaptive systems and feedback mechanisms
- **Biology** - Inspiration for neural networks and adaptive systems

---

## CS1 Core Concepts

### Performance-Task-Experience Triplet

ML algorithms improve performance (P) at some task (T) with experience (E) as **⟨P, T, E⟩**

| Component | Meaning | Example |
|-----------|---------|---------|
| **Experience (E)** | The dataset or historical examples used for training | 1000 labeled emails (spam/not spam) |
| **Task (T)** | The goal or problem being solved | Email spam classification |
| **Performance (P)** | The metric that measures success | Accuracy, precision, recall |

**Example:**
- E: Historical house sales data
- T: Predict house prices
- P: Mean Squared Error, R²

---

## CS1 Target Function and Hypothesis

### Key Definitions

| Term | Symbol | Meaning |
|------|--------|---------|
| **Target Function** | $f$ | The hidden rule that connects inputs to correct outputs in the real world |
| **Input Data** | $x$ | The input data we provide |
| **Real Answer** | $f(x)$ | The correct output (ground truth) |
| **Hypothesis** | $h$ | Our ML model's attempt to replicate the hidden rule |
| **Prediction** | $h(x)$ | The prediction our model makes |

### The Learning Challenge

$$\text{Goal: Adjust } h \text{ until } h(x) \approx f(x)$$

$$\text{Success: When our predictions match what actually happened}$$

### Visual Representation

```
Real World           Our Model
    ↓                    ↓
 f(x) = Real Answer   h(x) = Our Prediction
    │                    │
    └────────────────────┘
       Minimize Gap
```

---

## CS1 The Model

### What is a Model?

A **model** is what you get after training an algorithm on data:
- Captures the patterns and relationships learned from data
- Encodes the knowledge gained from data
- Learned representation that makes predictions

### Examples of Models

**Linear Regression Model:**
$$y = wx + b$$
Learns: How input changes affect output linearly

**Decision Tree Model:**
```
if feature_1 > threshold:
    if feature_2 < other_threshold:
        predict A
    else:
        predict B
else:
    predict C
```
Learns: Rule-based tree structure for decisions

**Neural Network Model:**
Learns: Millions of parameters (weights, biases) that transform inputs to outputs

### Model Output

The final model encodes:
- Learned parameters/weights
- Relationships between features and target
- Decision boundaries or prediction functions

---

## CS1 Generalization

### Central Goal of ML

**Generalization** = Performing well on unseen or new data (unobserved examples)

Not just memorizing training data!

### The Trade-off: Three Scenarios

#### 1. Overfitting (Bad)
```
Training Performance: ✅ Excellent (95% accuracy)
Test Performance:     ❌ Poor (60% accuracy)
Problem:              Model memorized training data, doesn't generalize
Cause:                Too complex, high variance
```

#### 2. Underfitting (Bad)
```
Training Performance: ❌ Poor (70% accuracy)
Test Performance:     ❌ Poor (68% accuracy)
Problem:              Model too simple to capture patterns
Cause:                Too simple, high bias
```

#### 3. Good Generalization (Good)
```
Training Performance: ✅ Good (85% accuracy)
Test Performance:     ✅ Good (84% accuracy)
Problem:              None
Cause:                Balanced complexity, learns underlying patterns
```

### Bias-Variance Trade-off

| Aspect | High Complexity | Low Complexity |
|--------|-----------------|-----------------|
| **Bias** | Low (fits data well) | High (too rigid) |
| **Variance** | High (sensitive to data) | Low (stable) |
| **Result** | Overfitting | Underfitting |
| **Example** | Degree 10 polynomial | Linear model |

### Achieving Good Generalization

✅ Use proper training/test split  
✅ Apply regularization (Ridge, Lasso)  
✅ Use cross-validation  
✅ Monitor both training and validation error  
✅ Keep model complexity balanced  

---

# CS2 - Machine Learning Workflow

## CS2 TABLE OF CONTENTS
1. [Types of Attributes](#cs2-types-of-attributes)
2. [Discrete vs Continuous Attributes](#cs2-discrete-vs-continuous-attributes)
3. [Data Types](#cs2-data-types)
4. [Data Pre-processing Steps](#cs2-data-pre-processing-steps)
5. [Feature Engineering](#cs2-feature-engineering)

---

## CS2 Types of Attributes

### Numerical Attributes
- **Definition:** Attributes with numeric values
- **Example:** Age, salary, temperature, distance

### Categorical Attributes
- **Definition:** Attributes with distinct categories (no order)
- **Example:** Color (red, blue, green), gender (male, female), city names

### Ordinal Attributes
- **Definition:** Categorical attributes with meaningful order
- **Example:** Education level (high school < bachelor < master), ratings (1 star < 5 stars)

### Binary Attributes
- **Definition:** Attributes with only two possible values
- **Example:** Yes/No, True/False, Present/Absent, 0/1

---

## CS2 Discrete vs Continuous Attributes

### Discrete Attributes
- **Definition:** Takes non-floating point values
- **Characteristics:** Countable, distinct values, gaps between values
- **Example:** 
  - Number of students (5, 10, 15, not 5.5)
  - Product ratings (1, 2, 3, 4, 5)
  - Age in years (25, 26, 27)

### Continuous Attributes
- **Definition:** Takes any real value in a range
- **Characteristics:** Infinite possible values, can have decimal values
- **Example:**
  - Temperature (98.6°F, 20.3°C)
  - Height (5.8 meters, 5.81 meters, 5.812 meters)
  - Salary ($50,000, $50,000.50, $50,000.75)

---

## CS2 Data Types

### Relational/Object Data
- **Structure:** Rows = objects/samples, Columns = attributes/features
- **Use Case:** Standard format for statistical data
- **Example:** Student database with rows=students, columns=name, age, GPA
- **Visual:**
```
|  Name   | Age | GPA  |
|---------|-----|------|
| Alice   | 20  | 3.8  |
| Bob     | 21  | 3.6  |
| Charlie | 19  | 3.9  |
```

### Transactional Data
- **Structure:** Each record represents items purchased in a transaction
- **Use Case:** Market basket analysis, recommendation systems
- **Example:** Transaction ID, item list, purchase date
- **Visual:**
```
| TransactionID | Items                      | Date       |
|---------------|----------------------------|------------|
| T001          | Bread, Milk, Butter        | 2025-01-15 |
| T002          | Apple, Orange, Banana      | 2025-01-15 |
```

### Document Data
- **Structure:** Data objects represented using vectors (words or terms)
- **Use Case:** Text analysis, Natural Language Processing
- **Example:** Document-term matrix, bag of words
- **Visual:**
```
| Doc | word1 | word2 | word3 |
|-----|-------|-------|-------|
| D1  |   2   |   1   |   0   |
| D2  |   0   |   3   |   2   |
```

### Sequence Data
- **Structure:** Attributes with relationships involving order (time or space)
- **Use Case:** Time or spatial ordered analysis
- **Example:** DNA sequences (ATCG...), event logs, action sequences
- **Visual:**
```
Sequence: A → T → C → G → A → T → C
Position: 1   2   3   4   5   6   7
```

### Time Series Data
- **Structure:** Sequences arising through measurement of time
- **Use Case:** Temporal pattern analysis, forecasting
- **Example:**
  - Daily rainfall measurements
  - Stock prices over time
  - Acoustic features for speech recognition
- **Visual:**
```
Date       | Price |
|---------|--------|
2025-01-01| $100   |
2025-01-02| $102   |
2025-01-03| $101   |
```

### Spatial and Spatio-Temporal Data
- **Structure:** Objects with attributes related to positions/areas
- **Use Case:** Geographic/location analysis
- **Example:** Weather maps, GPS coordinates, location-based services

### Web & Social Network Data
- **Structure:** Network relationships and connections
- **Use Case:** Social analysis, recommendation systems
- **Example:** Twitter connections, Facebook links, citation networks

---

## CS2 Data Pre-processing Steps

### Step 1: Data Cleaning

**Goal:** Detection and correction of data quality problems

**Why Important:** Poor quality data leads to highly inaccurate models

#### 1.1 Handling Missing Values

**Options:**

| Option | When to Use | Pros | Cons |
|--------|------------|------|------|
| Delete Column | >10% missing | Simple | Loss of data |
| Replace with Zero | Sparse data | Simple | May introduce bias |
| Replace with Mean/Median | Few missing | Preserves stats | Loses variability |
| Last Known Value | Time series | Maintains trend | May not be accurate |
| ML Prediction | Few missing | Accurate | Complex |

**Decision Tree:**
```
Missing values?
    ├─ >10% → Delete column
    ├─ Few (1-5) → Replace with mean/median
    ├─ Time series → Forward/backward fill
    └─ Predictable → Use ML imputation
```

#### 1.2 Handling Outliers

- **Definition:** Data points unusual relative to the rest of the dataset
- **Example:** Age = 999, Salary = -50000
- **Detection:** Values > 3 standard deviations from mean
- **Action:** Remove if identified as noise/error

#### 1.3 Addressing Inconsistency

- **Definition:** Contradictory attributes within same record
- **Example:** Age = 25 but Birth Year = 1990 (would be 34-35)
- **Action:** Correct or remove inconsistent records

#### 1.4 Removing Duplicates

- **Definition:** Identical or near-identical records
- **Action:** Remove duplicate entries

### Step 2: Data Aggregation

- **Definition:** Combine/summarize data based on common identifier
- **Example:** 
  - Calculate total sales for each transaction ID
  - Sum monthly revenue by store
  - Average customer spending by region
- **Benefit:** Reduces data size, creates meaningful groups

### Step 3: Sampling

- **Definition:** Select subset of data objects for analysis
- **Key Point:** Sample size must represent population
- **Types:**
  - Simple random sampling: Each record has equal probability
  - Stratified sampling: Sample from each group proportionally

**Risk:** 
- Too small sample → Misses patterns in original dataset
- Too large sample → Computational inefficiency

### Step 4: Scaling and Normalization

#### 4.1 Normalization

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

- **Range:** [0,1] or [-1,1]
- **Use:** Non-assumption based algorithms (KNN, Neural Networks)
- **Effect:** All features on same scale

**Example:**
- Original: Age ∈ [18, 80], Salary ∈ [30000, 200000]
- Normalized: Both ∈ [0, 1]

#### 4.2 Standardization (Z-score)

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

- **Mean:** 0, **Std Dev:** 1
- **Use:** Algorithms assuming Gaussian distribution
- **Advantage:** Less affected by outliers than normalization

**Example:**
- Original: Age = 30 (mean=40, σ=10)
- Standardized: (30-40)/10 = -1.0

---

## CS2 Feature Engineering

Feature engineering is the art of creating or selecting features to improve model performance.

### Step 1: Feature Extraction

#### 1.1 Dimensionality Reduction

**Goal:** Reduce number of attributes

**Why:** 
- Too many features → overfitting, slow computation
- Curse of dimensionality

**Technique - Principal Component Analysis (PCA):**
- Combines multiple features into fewer uncorrelated features
- Example: 100 features → 10 principal components

#### 1.2 Transformation Examples

**Density Calculation:**
$$\text{Density} = \frac{\text{Mass}}{\text{Volume}}$$
Create new feature from existing features

**Fourier Transform for Time Series:**
- Convert time domain to frequency domain
- Extract frequency components as features

---

### Step 2: Feature Selection

**Definition:** Select subset of existing features

**Goal:** Discard redundant or irrelevant features

#### 2.1 Filter Methods

- **How:** Use heuristics independent of learning algorithm
- **Example:** Correlation scores, mutual information
- **Advantage:** Fast, independent of model
- **Disadvantage:** Ignores feature interactions

#### 2.2 Wrapper Methods

- **How:** Use learning algorithm itself as "black box"
- **Process:** Try different feature subsets, evaluate performance
- **Advantage:** Considers feature interactions
- **Disadvantage:** Computationally expensive

#### 2.3 Embedded Methods

- **How:** Feature selection automatic during algorithm operation
- **Example:** Decision tree algorithms, Lasso regression
- **Advantage:** Efficient, integrated with learning
- **Disadvantage:** Algorithm-specific

---

### Step 3: Feature Construction

**Definition:** Creating genuinely new features, often by combining existing ones

#### 3.1 Polynomial Expansion

**Technique:** Use univariate mathematical functions

**Example:**
- Original feature: $x$
- New features: $x^2, x^3, \sqrt{x}, \log(x)$

#### 3.2 Feature Crossing

**Technique:** Combine features to capture interactions

**Example:**
- Features: Age, Income
- New feature: Age × Income (interaction effect)

**Use Case:** Age and income together might be more predictive than separately

---

### Step 4: Transformation and Encoding

#### 4.1 Discretization (Binning/Bucketing)

**Definition:** Convert continuous to discrete attributes

**Example:**
- Original: Age ∈ [18, 80]
- Binned: Youth (18-30), Adult (31-50), Senior (51-80)

**Methods:**

| Method | Approach | Example |
|--------|----------|---------|
| Equal-width | Divide into equal-width bins | [0-25), [25-50), [50-75), [75-100] |
| Equal-depth | Divide so each bin has same count | Each bin has same number of records |

#### 4.2 Binarization

**Definition:** Convert attributes to one or more binary variables

**Example: One-Hot Encoding for Categorical Features**

Original: Color ∈ {Red, Blue, Green}
```
Original               Binarized
Red      →    Red=1, Blue=0, Green=0
Blue     →    Red=0, Blue=1, Green=0
Green    →    Red=0, Blue=0, Green=1
```

---

# CS3 - Linear Regression & Regularization

## CS3 TABLE OF CONTENTS
1. [What is Linear Regression?](#cs3-what-is-linear-regression)
2. [Linear Regression Formulas](#cs3-linear-regression-formulas)
3. [Types of Regression Models](#cs3-types-of-regression-models)
4. [Least Squares Regression](#cs3-least-squares-regression)
5. [Hypothesis Function](#cs3-hypothesis-function)
6. [Ridge Regression (L2)](#cs3-ridge-regression-l2)
7. [Lasso Regression (L1)](#cs3-lasso-regression-l1)
8. [Ridge vs Lasso Comparison](#cs3-ridge-vs-lasso-comparison)
9. [Elastic Net](#cs3-elastic-net)
10. [Numerical Examples](#cs3-numerical-examples)

---

## CS3 What is Linear Regression?

### Simple Definition

Linear regression is a method to predict a continuous value based on one or more input variables by finding the best-fitting straight line (or plane in higher dimensions).

### Key Points

- **Input variables** (features/independent variables) → $X$
- **Output variable** (target/dependent variable) → $y$
- **Goal:** Find the relationship between them
- **Assumption:** Relationship is linear

### Real-world Applications

| Application | Input | Output | Goal |
|-------------|-------|--------|------|
| House Price | Size (sq ft) | Price | Predict price from size |
| Student Performance | Hours studied | Exam score | Predict score from hours |
| Employee Salary | Years experience | Salary | Predict salary from experience |
| Car Fuel | Weight | MPG | Predict efficiency from weight |

### Why Linear Regression?

✅ **Simple and Interpretable** - Easy to understand and explain  
✅ **Computationally Efficient** - Fast to train and predict  
✅ **Good Baseline** - Often works well as starting point  
✅ **Theoretical Foundation** - Well-established math  

---

## CS3 Linear Regression Formulas

### Simple Linear Regression (One Input Variable)

#### Formula

$$y = w_0 + w_1 x$$

Or: $$\hat{y} = b + mx$$

#### Parameters

| Symbol | Name | Meaning |
|--------|------|---------|
| $y$ or $\hat{y}$ | Predicted Output | The value we predict |
| $x$ | Input Variable | Independent variable |
| $w_0$ or $b$ | Intercept/Bias | y-axis crossing point |
| $w_1$ or $m$ | Slope/Weight | Change in $y$ per unit change in $x$ |

#### Real Example: House Price

$$\text{Price} = 50,000 + 200 \times \text{Size}$$

- $w_0 = 50,000$: Base price
- $w_1 = 200$: Each sq ft adds $200
- 1000 sq ft house: $50,000 + 200(1000) = 250,000
- 2000 sq ft house: $50,000 + 200(2000) = 450,000

---

### Multiple Linear Regression (Many Input Variables)

#### Formula

$$y = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

Or in vector form: $$y = \mathbf{w}^T \mathbf{x}$$

#### Real Example: House Price with Multiple Features

$$\text{Price} = 50,000 + 200(\text{size}) + 5,000(\text{bedrooms}) - 100(\text{age})$$

**Interpretation:**
- Base price: $50,000
- Each sq ft adds: $200
- Each bedroom adds: $5,000
- Each year of age reduces by: $100

**Calculation:** 1500 sq ft, 3 bedrooms, 10 years old
$$\text{Price} = 50,000 + 200(1500) + 5,000(3) - 100(10)$$
$$= 50,000 + 300,000 + 15,000 - 1,000 = 364,000$$

---

## CS3 Types of Regression Models

### Simple Linear Regression

$$y = w_0 + w_1 x$$

- **Characteristics:** One input, straight line, simplest form
- **When to use:** Single feature, linear relationship
- **Pros:** Simple, interpretable
- **Cons:** Limited expressiveness

---

### Multiple Linear Regression

$$y = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

- **Characteristics:** Multiple inputs, hyperplane in high dimensions
- **When to use:** Many features, linear relationships
- **Pros:** More expressive, interpretable
- **Cons:** May underfit complex relationships

---

### Polynomial Regression

$$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + ... + w_d x^d$$

- **Characteristic:** Linear in parameters but nonlinear in input
- **Examples:** 
  - Degree 2: $y = 1 + 2x + 3x^2$
  - Degree 3: $y = 5 + 2x - 3x^2 + 0.5x^3$
- **When to use:** Curved relationships, non-linear patterns
- **Pros:** More flexible, fits complex patterns
- **Cons:** Risk of overfitting, can become unstable

---

## CS3 Least Squares Regression

### The Goal

Find the line that **minimizes sum of squared errors** between predictions and actual values.

### Cost Function

$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N (y_n - \hat{y}_n)^2$$

Or: $$J(\mathbf{w}) = \frac{1}{N}\sum_{n=1}^N (y_n - \hat{y}_n)^2$$ (Mean Squared Error)

### Why Least Squares?

✅ Mathematically convenient (differentiable)  
✅ Penalizes large errors more (quadratic)  
✅ Has analytical solution  
✅ Well-established theory  

### Key Notations

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of training examples |
| $x_i$ | $i$-th input value |
| $y_i$, $t_i$ | Actual $i$-th output |
| $\hat{y}_i$ | Predicted $i$-th output |
| $e_i$ | Error: $e_i = y_i - \hat{y}_i$ |
| $\mathbf{\Phi}$ | Design/Feature matrix |
| $\mathbf{w}$ | Weight vector |
| $\bar{x}$, $\bar{y}$ | Mean of x, y values |

### Optimal Weights for Simple Linear Regression

**Slope:**
$$w_1 = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}$$

**Intercept:**
$$w_0 = \bar{y} - w_1 \bar{x}$$

### Numerical Example

**Data:** (1,2), (2,4), (3,5), (4,4), (5,5)

**Step 1:** Calculate means
$$\bar{x} = 3, \quad \bar{y} = 4$$

**Step 2:** Deviation table

| $i$ | $x_i$ | $y_i$ | $(x_i-\bar{x})$ | $(y_i-\bar{y})$ | Product | Square |
|-----|-------|-------|----------------|----------------|---------|--------|
| 1 | 1 | 2 | -2 | -2 | 4 | 4 |
| 2 | 2 | 4 | -1 | 0 | 0 | 1 |
| 3 | 3 | 5 | 0 | 1 | 0 | 0 |
| 4 | 4 | 4 | 1 | 0 | 0 | 1 |
| 5 | 5 | 5 | 2 | 1 | 2 | 4 |
| Sum | | | | | **6** | **10** |

**Step 3:** Calculate weights
$$w_1 = \frac{6}{10} = 0.6$$
$$w_0 = 4 - 0.6(3) = 2.2$$

**Final Equation:** $\hat{y} = 2.2 + 0.6x$

---

### Solution Methods

#### Method 1: Closed-Form (Normal Equations)

$$\mathbf{w} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$$

**Advantages:** Direct solution, guaranteed optimal, fast for small data  
**Disadvantages:** Slow for large N/M, numerically unstable, memory intensive

#### Method 2: Gradient Descent (Iterative)

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

Where:
$$\frac{\partial J}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N (h(\mathbf{x}_i) - y_i) x_j^{(i)}$$

**Advantages:** Scalable to large data, memory efficient, flexible  
**Disadvantages:** Needs hyperparameter tuning, many iterations, slower per iteration

---

## CS3 Hypothesis Function

### Definition

A **hypothesis function** is our current model for the relationship between inputs and outputs.

**Different Names:**
- Hypothesis: $h(\mathbf{x})$
- Model: $f(\mathbf{x})$
- Predictor: $\hat{y}(\mathbf{x})$
- All represent the same concept

### Simple Linear Regression

$$h(x) = w_0 + w_1 x$$

**Example:** If $h(x) = 3 + 2x$:
- When $x=0$: $h(0)=3$
- When $x=5$: $h(5)=13$
- When $x=10$: $h(10)=23$

### Multiple Linear Regression

$$h(\mathbf{x}) = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

**Vector form:** $h(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$

**Example:**
$$h(x_1, x_2, x_3) = 10 + 2x_1 + 3x_2 - 0.5x_3$$

For $(x_1=5, x_2=4, x_3=2)$:
$$h = 10 + 2(5) + 3(4) - 0.5(2) = 31$$

---

## CS3 Ridge Regression (L2)

### What is Ridge Regression?

Ridge Regression is a regularized linear regression that **prevents overfitting** by adding a **penalty on weight magnitudes**.

Also called **Tikhonov regularization** or **L2 regularization**.

### The Problem It Solves

**Without Regularization:**
- Model can fit training data perfectly (including noise)
- Creates large weights → overfitting
- Poor performance on new data

**With Ridge Regularization:**
- Balances fit to data with model simplicity
- Keeps weights small → simpler model
- Better generalization

### Ridge Cost Function

$$J(\mathbf{w}) = \underbrace{\frac{1}{2N}\sum_{n=1}^N (y_n - \hat{y}_n)^2}_{\text{Data Fit}} + \underbrace{\frac{\lambda}{2N}\sum_{j=1}^p w_j^2}_{\text{L2 Penalty}}$$

Or simply:
$$J(\mathbf{w}) = \text{MSE} + \lambda \sum w_j^2$$

### Parameters

| Parameter | Meaning | Effect |
|-----------|---------|--------|
| **$\lambda$** | Regularization strength | Controls penalty magnitude |
| **$\sum w_j^2$** | Sum of squared weights | Penalizes large weights |

### The Role of Lambda ($\lambda$)

| $\lambda$ | Behavior | Weights | Overfitting | Underfitting |
|---------|----------|---------|-------------|-------------|
| $0$ | Standard regression | Large | High | Low |
| Small | Weak penalty | Medium | Medium | Low |
| Medium | Balanced | Small-Medium | Low | Medium |
| Large | Strong penalty | Very small | Minimal | High |

### Ridge Solution

$$\mathbf{w}_{\text{Ridge}} = (\lambda \mathbf{I} + \mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$$

**Key Advantage:** Adding $\lambda \mathbf{I}$ makes matrix invertible (more stable)

### Ridge Characteristics

| Characteristic | Ridge |
|---|---|
| **Penalty Type** | L2 (squared weights) |
| **Weights** | Shrink, but rarely reach zero |
| **Feature Selection** | No (keeps all) |
| **Solution** | Closed-form available |
| **Speed** | Fast |
| **Multicollinearity** | Handles well |
| **Interpretability** | Medium (all features used) |

### When to Use Ridge

✅ Want to keep all features  
✅ Model overfits  
✅ Features are correlated  
✅ Need stable solution  
✅ Need fast computation  

---

## CS3 Lasso Regression (L1)

### What is Lasso Regression?

**Lasso** = **L**east **A**bsolute **S**hrinkage and **S**election **O**perator

Regularized regression that can:
1. **Shrink** weights towards zero
2. **Force** some weights to **exactly zero** (automatic feature selection)
3. Reduce model complexity

Also called **L1 regularization**.

### Lasso Cost Function

$$J(\mathbf{w}) = \frac{1}{2N}\sum_{n=1}^N (y_n - \hat{y}_n)^2 + \frac{\lambda}{N}\sum_{j=1}^p |w_j|$$

Or simply:
$$J(\mathbf{w}) = \text{MSE} + \lambda \sum |w_j|$$

### Key Difference: Absolute Value vs Squared

**Ridge:** $\lambda \sum w_j^{\mathbf{2}}$ (quadratic, smooth)

**Lasso:** $\lambda \sum |w_j|$ (linear, sharp corners)

### Why Lasso Creates Sparse Solutions

The absolute value penalty creates a non-smooth optimization surface with sharp corners. When the optimal point lands on a corner (axis), that weight becomes exactly zero.

```
Ridge (L2) Constraint:       Lasso (L1) Constraint:
    w2                           w2
    |   ○○○                      |
    |  ○   ○ (circle)            | ◇◇◇
    | ○●  ○                      | ◇●◇ (diamond)
    |_____w1                      |____w1
    
Smooth (rarely zero)         Sharp corners (can be zero)
```

### Automatic Feature Selection

**Example:**

Given 10 features, Lasso produces:
$$\hat{y} = 5 + 2x_1 + 0x_2 + 0.5x_3 + 0x_4 + ... + 0x_{10}$$

Features 2, 4, 6, 7, 9, 10 are removed (coefficient = 0)

Only uses: $\hat{y} = 5 + 2x_1 + 0.5x_3$

### Why No Closed-Form Solution?

The absolute value in the gradient is non-differentiable at zero:

$$\frac{\partial}{\partial w_j}|w_j| = \begin{cases} 1 & w_j > 0 \\ -1 & w_j < 0 \\ \text{undefined} & w_j = 0 \end{cases}$$

This non-differentiability **causes exact zeros**!

Must use iterative methods:
- Coordinate Descent
- Proximal Gradient Descent
- Shooting Algorithm

### Lasso Characteristics

| Characteristic | Lasso |
|---|---|
| **Penalty Type** | L1 (absolute values) |
| **Weights** | Can be exactly zero |
| **Feature Selection** | Yes (automatic) |
| **Solution** | No closed-form (iterative) |
| **Speed** | Slower (iterative) |
| **Multicollinearity** | Less stable |
| **Interpretability** | High (sparse) |

### When to Use Lasso

✅ Need automatic feature selection  
✅ Have many features, only some matter  
✅ Want interpretable model (fewer features)  
✅ Suspect feature redundancy  
✅ Want sparse solution  

---

## CS3 Ridge vs Lasso Comparison

### Side-by-Side Comparison

| Aspect | Ridge | Lasso | Elastic Net |
|--------|-------|-------|-----------|
| **Full Name** | Tikhonov | LASSO | Combination |
| **Penalty** | $\lambda \sum w_j^2$ | $\lambda \sum \|w_j\|$ | Both |
| **Weights** | Shrink, rarely zero | Can be exactly zero | Some zero |
| **Feature Selection** | ❌ No | ✅ Yes | ✅ Yes |
| **Closed Form** | ✅ Yes | ❌ No | ❌ No |
| **Speed** | ⚡ Fast | 🐢 Slow | 🐢 Slow |
| **Multicollinearity** | ✅ Good | ⚠️ Poor | ✅ Good |
| **Sparsity** | Dense | Sparse | Sparse |
| **Stability** | ✅ Stable | ❌ Unstable | ✅ Stable |
| **Hyperparameters** | 1 ($\lambda$) | 1 ($\lambda$) | 2 ($\lambda_1$, $\lambda_2$) |

### Geometric Interpretation

**Constraint Regions:**

```
Ridge (L2):              Lasso (L1):              Elastic Net:
  w2                       w2                       w2
  |  ○○○                   |  ◇◇◇                   |  □-○
  | ○   ○●                 | ◇●◇●                  | □   □●
  |○     ○                 |◇   ◇                  |□  □□  
  ├───w1                   ├───w1                  ├───w1
  
Circle constraint      Diamond constraint      Combined constraint
(smooth)              (sharp corners)         (corners + smoothness)
```

### Handling Correlated Features

**Ridge with Correlated Features:**
```
Features x₁ and x₂ highly correlated
Ridge: w₁ = 2.5, w₂ = 2.3
Effect: Distributes weights
Both features kept
Stable solution ✅
```

**Lasso with Correlated Features:**
```
Features x₁ and x₂ highly correlated
Lasso: w₁ = 5, w₂ = 0 (or opposite)
Effect: Arbitrarily picks one
May remove important feature
Unstable ⚠️
```

### Coefficient Path Visualization

As $\lambda$ increases:

**Ridge:**
```
Coefficient
    |  w₁ ────────
    | w₂  ──────── 
    | w₃   ─────   
    | w₄    ────   Smooth decrease
    |__________→ λ Never reaches zero
```

**Lasso:**
```
Coefficient
    |  w₁ ──────────
    | w₂ ────╱─────  (becomes zero)
    | w₃ ──╱────     (becomes zero)
    | w₄●──────────  (becomes zero early)
    |__________→ λ Can reach exactly zero
```

---

## CS3 Elastic Net

### What is Elastic Net?

Elastic Net combines both **L1 (Lasso)** and **L2 (Ridge)** penalties.

Gets the best of both worlds:
- Feature selection from Lasso
- Stability from Ridge
- Better handling of correlated features

### Elastic Net Cost Function

$$J(\mathbf{w}) = \frac{1}{2N}\sum_{n=1}^N (y_n - \hat{y}_n)^2 + \lambda_1 \sum |w_j| + \lambda_2 \sum w_j^2$$

Or with mixing parameter $\alpha$ (where $0 \leq \alpha \leq 1$):

$$J(\mathbf{w}) = \text{MSE} + \lambda[\alpha \sum |w_j| + (1-\alpha) \sum w_j^2]$$

### Parameters

- **$\lambda_1$:** Controls L1 (Lasso) strength
- **$\lambda_2$:** Controls L2 (Ridge) strength
- **$\alpha = 0$:** Pure Ridge
- **$\alpha = 1$:** Pure Lasso
- **$0 < \alpha < 1$:** Balanced combination

### Elastic Net Characteristics

| Characteristic | Elastic Net |
|---|---|
| **Feature Selection** | ✅ Yes (L1) |
| **Stability** | ✅ High (L2) |
| **Correlated Features** | ✅ Handles well |
| **Sparse Solutions** | ✅ Yes |
| **Hyperparameters** | 2 ($\lambda_1$, $\lambda_2$) |
| **Computation** | Iterative |

### When to Use Elastic Net

✅ Have many correlated features  
✅ Want feature selection AND stability  
✅ Need robust solution  
✅ Unsure between Ridge and Lasso  

### Elastic Net vs Pure Approaches

```
Ridge (α=0)          Elastic Net (α=0.5)      Lasso (α=1)

y = 2x₁ + 1.5x₂      y = 2x₁ + 0x₂            y = 2x₁ + 0x₂
    + 0.5x₃              + 0.3x₃                   + 0x₃
    + 0.2x₄              + 0x₄                     + 0x₄

All features used   Some removed             Only important
                    Balanced approach        features remain
```

---

## CS3 Numerical Examples

### Example 1: Simple Linear Regression

**Problem:** Predict exam scores from study hours

**Data:**
| Hours ($x$) | Score ($y$) |
|-----------|---------|
| 2 | 50 |
| 3 | 60 |
| 4 | 65 |
| 5 | 70 |
| 6 | 85 |

**Solution:**
- $\bar{x} = 4$, $\bar{y} = 66$
- $w_1 = 0.8$, $w_0 = 34$
- **Equation:** $\hat{y} = 34 + 8x$

**Interpretation:** Each hour adds 8 points

---

### Example 2: Multiple Linear Regression

**Problem:** Predict house price from size and age

**Data (5 houses):**
| Size (sqft) | Age (years) | Price ($K) |
|-----------|-----------|----------|
| 1200 | 10 | 250 |
| 1800 | 5 | 350 |
| 1500 | 15 | 280 |
| 2000 | 2 | 400 |
| 1400 | 20 | 260 |

**Solution:**
- **Equation:** $\text{Price} = 100 + 0.15(\text{Size}) - 2(\text{Age})$

**Interpretation:**
- Base price: $100K
- Size adds: $150/sqft
- Age reduces: $2K/year

**New House (1600 sqft, 8 years):**
$$\text{Price} = 100 + 0.15(1600) - 2(8) = 324K$$

---

### Example 3: Ridge vs Lasso on Correlated Features

**Problem:** Predict price with correlated features (size & bedrooms)

**Standard LR Result:**
```
Price = 100 + 0.20*size + 50*bedrooms - 2*age
Issues: Large, unstable coefficients
```

**Ridge Result (λ=1):**
```
Price = 95 + 0.12*size + 30*bedrooms - 1.8*age
Effect: All weights shrunk, both features used, stable
```

**Lasso Result (λ=0.5):**
```
Price = 90 + 0.15*size + 0*bedrooms - 1.5*age
Effect: Bedrooms eliminated (redundant), simpler model
```

**Elastic Net Result (λ=0.5, α=0.5):**
```
Price = 92 + 0.14*size + 5*bedrooms - 1.7*age
Effect: All kept (Ridge) but bedrooms weight small
```

---

## Summary & Selection Guide

### Quick Decision Tree

```
                    Start
                      |
              Many features?
                   /    \
                YES      NO
                /          \
            Correlated?     Use Simple
              /    \        Linear
            YES    NO       Regression
            /       \
        Elastic    Ridge
        Net        
                    
OR: Need feature
selection?
    /      \
  YES      NO
  /         \
Lasso    Ridge/Elastic
```

### Selection Criteria

**Use Ridge When:**
- ✅ Want to keep all features
- ✅ Features are correlated
- ✅ Need stable, fast solution
- ✅ Model overfits slightly

**Use Lasso When:**
- ✅ Need feature selection
- ✅ Have many features, few matter
- ✅ Want interpretable model
- ✅ Suspect redundancy

**Use Elastic Net When:**
- ✅ Have correlated features AND need selection
- ✅ Unsure between Ridge and Lasso
- ✅ Want robust, balanced solution

---

## Key Formulas Reference

| Method | Cost Function | Solution |
|--------|---|---|
| **Standard LR** | $\text{MSE}$ | $\mathbf{w} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$ |
| **Ridge** | $\text{MSE} + \lambda \sum w_j^2$ | $\mathbf{w} = (\lambda \mathbf{I} + \mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$ |
| **Lasso** | $\text{MSE} + \lambda \sum \|w_j\|$ | No closed form (iterative) |
| **Elastic Net** | $\text{MSE} + \lambda_1 \sum \|w_j\| + \lambda_2 \sum w_j^2$ | Iterative methods |

---

# CS5 - Linear Models for Classification

## CS5 TABLE OF CONTENTS
1. [What is Classification?](#cs5-what-is-classification)
2. [Decision Theory](#cs5-decision-theory)
3. [Generative vs Discriminative Models](#cs5-generative-vs-discriminative-models)
4. [Discriminant Functions](#cs5-discriminant-functions)
5. [Logistic Regression](#cs5-logistic-regression)
6. [Cost Function (Cross-Entropy)](#cs5-cost-function-cross-entropy)
7. [Gradient Descent for Classification](#cs5-gradient-descent-for-classification)
8. [Multi-Class Classification](#cs5-multi-class-classification)
9. [Regularization in Classification](#cs5-regularization-in-classification)
10. [Evaluation Metrics](#cs5-evaluation-metrics)
11. [Numerical Examples](#cs5-numerical-examples)

---

## CS5 What is Classification?

### Simple Definition

Classification is predicting a **discrete class label** (category) rather than a continuous value.

### Regression vs Classification

| Aspect | Regression | Classification |
|--------|-----------|----------------|
| **Output Type** | Continuous (numbers) | Discrete (categories) |
| **Example Output** | $150,000, 2.5, 98.6 | Spam/Ham, Dog/Cat, Yes/No |
| **Goal** | Predict quantity | Predict category |
| **Example Task** | House price, temperature | Email spam, disease diagnosis |
| **Evaluation** | MSE, RMSE, R² | Accuracy, precision, recall |

### Types of Classification

**Binary Classification:**
- Two classes: {0, 1} or {Positive, Negative}
- Examples: Spam detection, disease diagnosis, pass/fail

**Multi-Class Classification:**
- More than two classes: {Class 1, Class 2, ..., Class K}
- Examples: Digit recognition (0-9), animal species, product categories

### Real-world Applications

| Application | Input | Output | Classes |
|-------------|-------|--------|---------|
| Email Filtering | Email text | Spam/Ham | 2 |
| Medical Diagnosis | Patient symptoms | Disease/Healthy | 2 |
| Image Recognition | Image pixels | Cat/Dog/Bird | 3+ |
| Credit Approval | Financial data | Approve/Deny | 2 |
| Student Admission | GPA, test scores | Accept/Reject | 2 |

---

## CS5 Decision Theory

### The Goal

Given input **x**, predict which class it belongs to by learning $p(C_k|x)$ (probability of class $C_k$ given input $x$).

### Key Probability Concepts

**Joint Probability:**
$$p(x, C_k) = p(C_k|x) \times p(x)$$

**Bayes' Theorem:**
$$p(C_k|x) = \frac{p(x|C_k) \times p(C_k)}{p(x)}$$

Where:
- $p(C_k|x)$ = **Posterior:** Probability of class given input
- $p(x|C_k)$ = **Likelihood:** Probability of input given class
- $p(C_k)$ = **Prior:** Probability of class (before seeing data)
- $p(x)$ = **Evidence:** Probability of input (normalization)

### Decision Rule

**Assign** $x$ **to class** $C_k$ **if:**
$$p(C_k|x) > p(C_j|x) \text{ for all } j \neq k$$

This is called the **Bayes Classifier** - theoretically optimal decision rule.

### Example: Medical Diagnosis

**Problem:** Diagnose disease from biomarker level

**Given:**
- $p(\text{Disease}) = 0.01$ (1% prevalence - rare disease)
- $p(\text{Healthy}) = 0.99$
- Test results: biomarker level $x$

**Decision boundary:** Where posteriors are equal
$$p(x|\text{Disease}) \times 0.01 = p(x|\text{Healthy}) \times 0.99$$

**Key Insight:** Likelihood ratio must be **99:1** to overcome low prior!

Even if test is "positive", might still be healthy due to low base rate.

---

## CS5 Generative vs Discriminative Models

### Two Approaches to Classification

| Approach | What it Models | Steps |
|----------|---------------|-------|
| **Generative** | $p(x\|C_k)$ and $p(C_k)$ | 1. Model class distributions<br>2. Learn priors<br>3. Apply Bayes' theorem |
| **Discriminative** | $p(C_k\|x)$ directly | 1. Learn decision boundary<br>2. Done! |

---

### Generative Models

**Three-Step Process:**

**Step 1:** Learn class-conditional densities $p(x|C_k)$
- For each class, model how features are distributed
- Example: Assume Gaussian distributions

**Step 2:** Learn prior probabilities $p(C_k)$
- Count frequency of each class in training data

**Step 3:** Apply Bayes' theorem
$$p(C_k|x) = \frac{p(x|C_k) \times p(C_k)}{p(x)}$$

**Example: Job Hiring**

**Data:**
| CGPA | IQ | Job Offered |
|------|-----|-------------|
| 5.5 | 6.7 | 1 |
| 8.0 | 6.0 | 1 |
| 9.0 | 7.0 | 1 |
| 5.0 | 7.0 | 0 |

**For Job=1:** $\mu_1 = [7.5, 6.57]$, learn $\Sigma_1$  
**For Job=0:** $\mu_2 = [5.0, 7.0]$, learn $\Sigma_2$  
**Priors:** $p(\text{Job}=1) = 0.75$, $p(\text{Job}=0) = 0.25$

**Challenge:** Need to estimate many parameters
- Mean vector: $d$ parameters
- Covariance matrix: $\frac{d(d+1)}{2}$ parameters
- For 100 features: ~5,000 parameters per class!

---

### Discriminative Models

**Direct Approach:** Learn $p(C_k|x)$ or decision boundary directly

**Example: Logistic Regression**
$$p(y=1|x) = \sigma(w^T x)$$

Only need $(d+1)$ parameters (weights + bias)!

**Advantages:**
- **Fewer parameters** - more efficient
- **Better with limited data** - focuses on boundary
- **Faster training** - simpler optimization

**Example Comparison:**

| Model Type | Parameters for 100 features |
|------------|---------------------------|
| Generative (Gaussian) | ~5,000 per class |
| Discriminative (Logistic) | 101 total |

---

### When to Use Each

| Use Generative When | Use Discriminative When |
|---------------------|------------------------|
| ✅ Need to generate new samples | ✅ Focus is only on classification |
| ✅ Have missing data | ✅ Have large datasets |
| ✅ Have strong prior knowledge about distributions | ✅ Want better performance (usually) |
| ✅ Classes are well-separated | ✅ Limited computational resources |
| ✅ Want to model full data distribution | ✅ Need faster training |
| ✅ Multiple related prediction tasks | ✅ Complex decision boundaries |  

---

## CS5 Discriminant Functions

### Linear Discriminant (Two Classes)

**Decision Function:**
$$y(x) = w^T x + w_0$$

**Classification Rule:**
- If $y(x) > 0$: Predict Class 1
- If $y(x) < 0$: Predict Class 0
- If $y(x) = 0$: On the boundary

### Geometric Interpretation

**Weight Vector w:**
- Direction perpendicular to decision boundary
- Points toward Class 1

**Bias Term** $w_0$:
- Shifts boundary away from origin
- Controls threshold

**Distance from point to boundary:**
$$r = \frac{w^T x + w_0}{||w||}$$

### Example: Student Admission

**Features:** GPA ($x_1$), Test Score ($x_2$)

**Learned Model:**
$$y(x) = -200 + 20 \times \text{GPA} + 1.5 \times \text{TestScore}$$

**Decision Boundary:** (where $y(x) = 0$)
$$20 \times \text{GPA} + 1.5 \times \text{TestScore} = 200$$

**Interpretation:**
- Each GPA point is "worth" $\frac{20}{1.5} = 13.33$ test points
- If GPA = 3.0: Need TestScore $\geq 93.33$
- If GPA = 4.0: Need TestScore $\geq 80$

**Test Examples:**

**Student A:** GPA=3.2, TestScore=85
$$y = -200 + 20(3.2) + 1.5(85) = -8.5 < 0 \rightarrow \text{REJECT}$$

**Student B:** GPA=3.5, TestScore=90
$$y = -200 + 20(3.5) + 1.5(90) = 5 > 0 \rightarrow \text{ACCEPT}$$

---

## CS5 Logistic Regression

### Why Not Linear Regression for Classification?

**Problems with linear regression:**
- Outputs can be $< 0$ or $> 1$ (invalid probabilities)
- Sensitive to outliers
- Not designed for classification

**Solution:** Use sigmoid function to map to [0,1]

### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- Range: $(0, 1)$ - perfect for probabilities
- Smooth and differentiable
- Monotonic (always increasing)

**Behavior:**

| z | σ(z) | Interpretation |
|---|------|----------------|
| -∞ | 0 | Definitely Class 0 |
| -5 | 0.007 | Very unlikely Class 1 |
| -2 | 0.12 | Unlikely Class 1 |
| 0 | 0.5 | 50-50 |
| 2 | 0.88 | Likely Class 1 |
| 5 | 0.993 | Very likely Class 1 |
| +∞ | 1 | Definitely Class 1 |

### Logistic Regression Model

$$h(x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$

**Interpretation:**
- $h(x)$ = Probability that $y = 1$
- $1 - h(x)$ = Probability that $y = 0$

**Decision Rule:**
- If $h(x) \geq 0.5$: Predict Class 1
- If $h(x) < 0.5$: Predict Class 0

### Log-Odds (Logit)

$$\log\left[\frac{p}{1-p}\right] = w^T x$$

**Interpretation:**
- Log-odds is linear in features
- This is why it's called "Logistic Regression"
- $w_j > 0$: Feature increases odds of Class 1
- $w_j < 0$: Feature decreases odds of Class 1

### Complete Example: Email Spam

**Features:**
- $x_1$ = Number of exclamation marks
- $x_2$ = Contains "free" (1/0)
- $x_3$ = Email length (words)

**Learned Weights:**
```
w₀ = -2.0
w₁ = 0.5  (exclamation marks)
w₂ = 3.0  (contains "free")
w₃ = -0.01 (length)
```

**New Email:** 2 exclamation marks, contains "free", 100 words

**Step 1: Compute z**
$$z = -2.0 + 0.5(2) + 3.0(1) + (-0.01)(100)$$
$$z = -2.0 + 1.0 + 3.0 - 1.0 = 1.0$$

**Step 2: Apply sigmoid**
$$h(x) = \frac{1}{1 + e^{-1}} = 0.731$$

**Result:** 73.1% probability of spam → Classify as SPAM

**Interpretation:**
- "free" has largest impact (weight = 3.0)
- Each exclamation mark increases spam probability
- Longer emails slightly less likely spam

---

## CS5 Cost Function (Cross-Entropy)

### Why Not Mean Squared Error?

MSE is **non-convex** for logistic regression:
- Multiple local minima
- Gradient descent may get stuck
- Slow convergence

### Cross-Entropy Loss

**For single example:**
$$\text{Cost}(h(x), y) = -[y \log(h(x)) + (1-y) \log(1-h(x))]$$

**For entire dataset:**
$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)} \log(h(x^{(i)})) + (1-y^{(i)}) \log(1-h(x^{(i)}))\right]$$

### Why Cross-Entropy?

✅ **Convex** - single global minimum  
✅ **Penalizes wrong confident predictions heavily**  
✅ Comes from **maximum likelihood estimation**  
✅ Works well with sigmoid function  

### Understanding the Penalty

**When y = 1 (actual class is positive):**

| h(x) | Cost | Interpretation |
|------|------|----------------|
| 0.99 | 0.01 | Low penalty - good! |
| 0.9 | 0.11 | Small penalty |
| 0.5 | 0.69 | Medium penalty |
| 0.1 | 2.30 | Large penalty! |
| 0.01 | 4.61 | Huge penalty! |
| → 0 | → ∞ | Infinite penalty |

**When y = 0 (actual class is negative):**

| h(x) | Cost | Interpretation |
|------|------|----------------|
| 0.01 | 0.01 | Low penalty - good! |
| 0.1 | 0.11 | Small penalty |
| 0.5 | 0.69 | Medium penalty |
| 0.9 | 2.30 | Large penalty! |
| 0.99 | 4.61 | Huge penalty! |
| → 1 | → ∞ | Infinite penalty |

**Key Property:** Being confident AND wrong is heavily penalized!

### Example Calculation

**3 Predictions:**
- Prediction 1: $h(x)=0.9$, $y=1$ → Cost = $-\log(0.9) = 0.105$
- Prediction 2: $h(x)=0.2$, $y=1$ → Cost = $-\log(0.2) = 1.609$
- Prediction 3: $h(x)=0.1$, $y=0$ → Cost = $-\log(0.9) = 0.105$

**Average Cost:** $\frac{0.105 + 1.609 + 0.105}{3} = 0.606$

Prediction 2 dominates (was very wrong)!

---

## CS5 Gradient Descent for Classification

### Update Rule

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

Where:
$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})x_j^{(i)}$$

**Beautiful Result:** Same form as linear regression, but $h(x) = \sigma(w^T x)$!

### Algorithm Steps

**1. Initialize weights** (e.g., all zeros or small random values)

**2. Repeat until convergence:**
   - Compute predictions: $h(x^{(i)}) = \sigma(w^T x^{(i)})$ for all examples
   - Compute gradients: $\frac{\partial J}{\partial w_j}$ for each weight
   - Update weights: $w_j := w_j - \alpha \times \text{gradient}$
   - Compute cost $J(w)$ to monitor progress

**3. Stop when:**
   - Cost stops decreasing
   - Gradients become very small
   - Maximum iterations reached

### Simple Example

**Data:** Pass/Fail based on study hours

| Hours (x) | Pass (y) |
|-----------|----------|
| 1 | 0 |
| 2 | 0 |
| 3 | 1 |
| 4 | 1 |

**Hyperparameters:** $\alpha = 0.1$, initial $w_0 = 0, w_1 = 0$

**Iteration 1:**

For all examples with $w=[0, 0]$:
```
z = 0, h = σ(0) = 0.5 for all
Errors: [0.5, 0.5, -0.5, -0.5]
```

**Gradients:**
$$\frac{\partial J}{\partial w_0} = \frac{1}{4}[0.5 + 0.5 - 0.5 - 0.5] = 0$$
$$\frac{\partial J}{\partial w_1} = \frac{1}{4}[0.5(1) + 0.5(2) - 0.5(3) - 0.5(4)] = -0.5$$

**Update:**
$$w_0 := 0 - 0.1(0) = 0$$
$$w_1 := 0 - 0.1(-0.5) = 0.05$$

**After many iterations:**
```
w₀ ≈ -5.5
w₁ ≈ 2.5
Decision boundary: -5.5 + 2.5(hours) = 0
                   hours = 2.2
```

**Interpretation:** Need ~2.2 hours to have 50% pass probability

### Variants

| Method | Description | Batch Size |
|--------|-------------|-----------|
| **Batch GD** | Use all examples per update | m |
| **Stochastic GD** | Use 1 example per update | 1 |
| **Mini-batch GD** | Use small batch per update | 32, 64, 128 |

**Most Common:** Mini-batch (balances speed and stability)

### Learning Rate Selection

| α | Effect | Problem |
|---|--------|---------|
| Too small | Slow convergence | Many iterations needed |
| Just right | Steady decrease | Efficient learning |
| Too large | Oscillation/divergence | Cost increases! |

**Typical values:** 0.001, 0.01, 0.1

---

## CS5 Multi-Class Classification

### Two Main Approaches

---

### 1. One-vs-All (One-vs-Rest)

**Strategy:** Train K binary classifiers (one per class)

**Example:** Classify {Sports, Politics, Technology}

**Classifier 1:** Sports vs. {Politics, Technology}
- Label Sports as 1, others as 0
- Learn weights $w^{(1)}$

**Classifier 2:** Politics vs. {Sports, Technology}
- Label Politics as 1, others as 0
- Learn weights $w^{(2)}$

**Classifier 3:** Technology vs. {Sports, Technology}
- Label Technology as 1, others as 0
- Learn weights $w^{(3)}$

**Prediction:**
$$\hat{y} = \arg\max_k h_k(x)$$

Choose class with highest probability.

**Example:**
```
h₁(x) = 0.85  (Sports)
h₂(x) = 0.10  (Politics)
h₃(x) = 0.30  (Technology)

Prediction: Sports
```

**Characteristics:**
- ✅ Simple to implement
- ✅ Parallelizable
- ❌ Probabilities don't sum to 1
- ❌ Class imbalance during training

---

### 2. Softmax Regression

**Direct multi-class approach**

**Model:**
$$p(C_k|x) = \frac{\exp(w_k^T x)}{\sum_{j=1}^{K}\exp(w_j^T x)}$$

**Properties:**
- All probabilities positive
- Sum to 1: $\sum_k p(C_k|x) = 1$
- Reduces to logistic for K=2

**Example:** Image classification {Cat, Dog, Bird}

**Scores (logits):**
```
a₁ = 5.0  (Cat)
a₂ = 2.0  (Dog)
a₃ = 1.0  (Bird)
```

**Step 1: Exponentiate**
```
exp(5.0) = 148.4
exp(2.0) = 7.4
exp(1.0) = 2.7
Sum = 158.5
```

**Step 2: Normalize**
```
p(Cat) = 148.4 / 158.5 = 0.936  (93.6%)
p(Dog) = 7.4 / 158.5 = 0.047    (4.7%)
p(Bird) = 2.7 / 158.5 = 0.017   (1.7%)
```

**Prediction: Cat** (highest probability)

### Cross-Entropy for Softmax

$$J(W) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y_k^{(i)} \log(p_k^{(i)})$$

where $y_k$ is one-hot encoded (1 for true class, 0 otherwise)

### Comparison

| Aspect | One-vs-All | Softmax |
|--------|------------|---------|
| **Training** | K separate models | Single joint model |
| **Probabilities** | Don't sum to 1 | Always sum to 1 |
| **Calibration** | Poor | Better |
| **Implementation** | Simpler | More complex |
| **Use Case** | Quick baseline | Need probabilities |

---

## CS5 Regularization in Classification

### Why Regularization?

**Problem:** Overfitting in classification
- Model memorizes training data
- Poor generalization to new data
- Large weight magnitudes

**Solution:** Add penalty to cost function

---

### L2 Regularization (Ridge)

$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)} \log h(x^{(i)}) + (1-y^{(i)}) \log(1-h(x^{(i)}))\right] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$

**Effect:**
- Shrinks all weights toward zero
- Keeps all features
- Reduces overfitting

**Gradient:**
$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}w_j$$

---

### L1 Regularization (Lasso)

$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)} \log h(x^{(i)}) + (1-y^{(i)}) \log(1-h(x^{(i)}))\right] + \frac{\lambda}{m}\sum_{j=1}^{n}|w_j|$$

**Effect:**
- Can set weights to exactly zero
- Performs feature selection
- Creates sparse models

---

### Choosing Lambda (λ)

| λ | Effect | Training Acc | Test Acc |
|---|--------|--------------|----------|
| 0 | No regularization | 99% | 65% (overfit) |
| 0.01 | Weak | 95% | 88% |
| 0.1 | Medium | 92% | 91% (best!) |
| 1.0 | Strong | 85% | 86% |
| 10.0 | Too strong | 75% | 76% (underfit) |

**Best λ:** Use cross-validation to select

---

## CS5 Evaluation Metrics

### Beyond Accuracy

**Why accuracy isn't enough:**

Example: Disease detection (1% prevalence)
```
Model predicts "Healthy" for everyone
Accuracy = 99% ← Seems great!
But catches 0% of diseases ← Useless!
```

---

### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actually Negative** | True Negative (TN) | False Positive (FP) |
| **Actually Positive** | False Negative (FN) | True Positive (TP) |

**Example:** Email Spam Filter (2000 emails)

|  | Predicted Ham | Predicted Spam |
|--|---------------|----------------|
| **Actually Ham** | 950 (TN) | 50 (FP) |
| **Actually Spam** | 30 (FN) | 970 (TP) |

---

### Key Metrics

**Accuracy:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{1920}{2000} = 0.96$$

**Precision (Positive Predictive Value):**
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{970}{1020} = 0.951$$

**Question:** Of predicted spam, how many are actually spam?

**Recall (Sensitivity, True Positive Rate):**
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{970}{1000} = 0.970$$

**Question:** Of actual spam, how many did we catch?

**F1 Score (Harmonic Mean):**
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 0.960$$

**Specificity (True Negative Rate):**
$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{950}{1000} = 0.950$$

---

### Precision-Recall Tradeoff

**Threshold Effect:**

| Threshold | Precision | Recall | F1 | Use Case |
|-----------|-----------|--------|-----|----------|
| 0.9 | 0.98 | 0.60 | 0.74 | High confidence needed |
| 0.7 | 0.92 | 0.85 | 0.88 | Balanced |
| 0.5 | 0.85 | 0.92 | 0.88 | Standard |
| 0.3 | 0.70 | 0.98 | 0.82 | Don't miss positives |

**Tradeoff:**
- High threshold → High precision, low recall
- Low threshold → Low precision, high recall

---

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic):**
- X-axis: False Positive Rate = $\frac{FP}{FP+TN}$
- Y-axis: True Positive Rate (Recall) = $\frac{TP}{TP+FN}$

**AUC (Area Under ROC Curve):**
- Single number summarizing performance
- AUC = 1.0: Perfect classifier
- AUC = 0.9-1.0: Excellent
- AUC = 0.8-0.9: Good
- AUC = 0.5: Random guessing

**Advantage:** Threshold-independent metric

---

### Choosing the Right Metric

| Scenario | Best Metric | Why |
|----------|-------------|-----|
| Balanced classes | Accuracy | Simple, interpretable |
| Imbalanced classes | F1, Precision, Recall | Accounts for imbalance |
| False positives costly | Precision | Minimize FP |
| False negatives costly | Recall | Minimize FN |
| Need threshold-free | AUC-ROC | All thresholds |
| Medical screening | Recall | Don't miss sick |
| Spam filter | Precision | Don't mark good as spam |

---

## CS5 Numerical Examples

### Example 1: Binary Classification - Student Admission

**Problem:** Predict admission from GPA and test score

**Data:**
| GPA | Test | Admitted |
|-----|------|----------|
| 3.5 | 85 | 1 |
| 3.2 | 75 | 0 |
| 3.8 | 90 | 1 |
| 2.9 | 70 | 0 |
| 3.6 | 88 | 1 |

**Learned Model:**
$$z = -20 + 5 \times \text{GPA} + 0.2 \times \text{Test}$$
$$h(x) = \sigma(z)$$

**New Student:** GPA=3.4, Test=82

**Step 1:** Compute z
$$z = -20 + 5(3.4) + 0.2(82) = -20 + 17 + 16.4 = 13.4$$

**Step 2:** Apply sigmoid
$$h(x) = \frac{1}{1 + e^{-13.4}} \approx 0.9999$$

**Result:** 99.99% probability of admission → **ADMIT**

---

### Example 2: Multi-Class - News Classification

**Classes:** {Sports, Politics, Technology}

**New Article Features:** word counts

**One-vs-All Results:**
```
h₁(x) = σ(2.5) = 0.924  (Sports)
h₂(x) = σ(-0.5) = 0.378 (Politics)
h₃(x) = σ(0.8) = 0.689  (Technology)
```

**Prediction:** Sports (highest at 0.924)

**Softmax Results:**
```
Logits: a₁=2.5, a₂=-0.5, a₃=0.8

p(Sports) = exp(2.5) / [exp(2.5) + exp(-0.5) + exp(0.8)]
         = 12.18 / [12.18 + 0.61 + 2.23]
         = 12.18 / 15.02 = 0.811

p(Politics) = 0.61 / 15.02 = 0.041
p(Technology) = 2.23 / 15.02 = 0.148

Prediction: Sports (81.1%)
```

---

### Example 3: Evaluation Metrics

**Spam Filter Results (1000 emails):**

|  | Predicted Ham | Predicted Spam |
|--|---------------|----------------|
| **Actually Ham** | 870 | 30 |
| **Actually Spam** | 20 | 80 |

**Calculate Metrics:**

**Accuracy:**
$$\frac{870 + 80}{1000} = 0.95 \text{ (95\%)}$$

**Precision:**
$$\frac{80}{80 + 30} = 0.727 \text{ (72.7\%)}$$
Of emails marked spam, 72.7% are actually spam

**Recall:**
$$\frac{80}{80 + 20} = 0.80 \text{ (80\%)}$$
Catch 80% of actual spam

**F1 Score:**
$$2 \times \frac{0.727 \times 0.80}{0.727 + 0.80} = 0.762$$

**Interpretation:**
- Good accuracy (95%)
- But precision only 72.7% → Many false positives
- Missing 20% of spam (recall = 80%)
- Could adjust threshold to improve recall

---

### Example 4: Regularization Effect

**Problem:** Predict credit default with 50 features

**Without Regularization:**
```
Training Accuracy: 98%
Test Accuracy: 72%
Issue: Overfitting!
```

**With L2 (λ=0.1):**
```
Training Accuracy: 91%
Test Accuracy: 89%
Result: Better generalization
```

**With L1 (λ=0.1):**
```
Training Accuracy: 90%
Test Accuracy: 88%
Non-zero weights: 12 out of 50
Result: Sparse model, feature selection
```

**Key Finding:** Only 12 features actually matter!

---

## Summary & Selection Guide

### Classification vs Regression

| Use Classification When | Use Regression When |
|------------------------|---------------------|
| Output is category | Output is quantity |
| Discrete labels | Continuous values |
| Example: Spam/Ham | Example: House price |
| Metrics: Accuracy, F1 | Metrics: MSE, R² |

### Model Selection

**Use Logistic Regression When:**
✅ Need interpretable model  
✅ Linear decision boundary works  
✅ Want probability outputs  
✅ Fast training needed  
✅ Good baseline  

**Use Softmax When:**
✅ Multi-class problem (K > 2)  
✅ Need calibrated probabilities  
✅ Classes are mutually exclusive  

**Add Regularization When:**
✅ Model overfits training data  
✅ Have many features  
✅ Features are correlated (L2)  
✅ Need feature selection (L1)  

### Evaluation Metric Selection

```
                Start
                  |
        Balanced classes?
           /          \
         YES           NO
         /              \
    Use Accuracy    Use F1/Precision/Recall
                         |
                  What's costly?
                   /          \
         False Positives   False Negatives
              /                  \
        Use Precision        Use Recall
```

---

## Key Formulas Reference

| Component | Formula |
|-----------|---------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| **Logistic Model** | $h(x) = \sigma(w^T x)$ |
| **Cross-Entropy** | $J(w) = -\frac{1}{m}\sum[y\log h(x) + (1-y)\log(1-h(x))]$ |
| **Gradient** | $\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum(h(x^{(i)}) - y^{(i)})x_j^{(i)}$ |
| **L2 Regularization** | $J(w) = \text{CE} + \frac{\lambda}{2m}\sum w_j^2$ |
| **L1 Regularization** | $J(w) = \text{CE} + \frac{\lambda}{m}\sum \|w_j\|$ |
| **Softmax** | $p(C_k\|x) = \frac{\exp(w_k^T x)}{\sum_j \exp(w_j^T x)}$ |
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ |
| **Precision** | $\frac{TP}{TP+FP}$ |
| **Recall** | $\frac{TP}{TP+FN}$ |
| **F1 Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |

---

**Document Version:** Complete ML Guide with Classification  
**Covers:** CS1 (Fundamentals), CS2 (Workflow), CS3 (Regression), CS5 (Classification)  
**Suitable for:** Students, Reference Material, Exam Prep

**Last Updated:** Enhanced with CS5 content
