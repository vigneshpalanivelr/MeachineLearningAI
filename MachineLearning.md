# Complete Machine Learning Course Guide
## CS1 • CS2 • CS3: Linear Models & Regression

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

✅ Many features, want to keep all  
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

**Document Version:** Complete ML Guide  
**Covers:** CS1 (Fundamentals), CS2 (Workflow), CS3 (Linear Models & Regularization)  
**Suitable for:** Students, Reference Material, Exam Prep
