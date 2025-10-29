# CS1 - Machine Learning (ML)

## Definition
- Machine Learning (ML) is a field of study about creating programs that automatically improve by learning from experience and data
- Instead of writing explicit rules, ML systems learn patterns from examples and use those patterns to make decisions
- ML draws from multiple disciplines
  - Statistics
  - Artificial Intelligence (AI)
  - Information Theory
  - Biology
  - Control Theory

## Core Concepts and Definitions

### Performance-Task–Experience Triplet
- ML algorithms improve performance (P) at some task (T) with experience (E) as <P, T, E>
  - Experience (E): The dataset or historical examples used for training
  - Task (T): The goal or problem being solved, e.g., recognizing handwriting or filtering spam
  - Performance (P): The metric that measures success, e.g., accuracy, error rate, precision, recall

### Target Function and Hypothesis
- f = The hidden rule that connects inputs to correct outputs in the real world
- x = The input data we provide
- f(x) = The real answer (the correct output)
- h = Our ML model's attempt to replicate that hidden rule
- h(x) = The prediction our model makes (what we calculated)
- Challenge: We need to adjust h until h(x) matches f(x)
- Success : When our predictions match what actually happened

### The Model
- A model is what you get after training an algorithm on data (captures the patterns and relationships learned from data)
- The model encodes the knowledge gained from data (learned representation that makes predictions)
- Examples:
  - Linear Regression Learns: y = w * x + b
  - Decision Tree Learns rules like tree structure: if Action-1 else Action-2
  - Neural Network Learns millions of parameters (weights, biases)

### Generalization
- A central goal of ML is generalization — performing well on unseen or new data (unobserved examples)
- Goal
  - Overfitting: Model memorizes training data; great on training, poor on new data
  - Underfitting: Model too simple; poor on both training and new data
  - Good Generalization: Model learns underlying patterns; performs well on both

# CS2 - Machine Learning Workflow

## Types of Attributes
### Numerical Attributes
  - Definition: Attributes with numeric values
  - Example: Age, salary, temperature

### Categorical Attributes
  - Definition: Attributes with distinct categories
  - Example: Color (red, blue, green), gender (male, female)

### Ordinal Attributes
  - Definition: Categorical with meaningful order
  - Example: Education level (high school, bachelor, master)

### Binary Attributes
  - Definition: Attributes with only two values
  - Example: Yes/No, True/False

## Discrete vs Continuous Attributes
### Discrete Attributes
  - Definition: Takes non-floating point values
  - Characteristics: Countable, distinct values
  - Example: Number of students, ratings (1-5), age in years

### Continuous Attributes
  - Definition: Takes any real value in a range
  - Characteristics: Infinite possible values, decimal values
  - Example: Temperature (98.6°F), height (5.8 meters), salary ($50,000.50)

## Data Types
### Relational/Object Data
  - Structure: Rows = objects, Columns = attributes
  - Use: Standard format for statistical data
  - Example: Student database (rows=students, columns=name, age, GPA)

### Transactional Data
  - Structure: Records represent purchased items
  - Use: Market basket analysis
  - Example: Transaction ID, item list, purchase date

### Document Data
  - Structure: Vectors of words/terms
  - Use: Text analysis, NLP
  - Example: Document-term matrix, bag of words

### Sequence Data
  - Structure: Attributes with order relationships
  - Use: Time or spatial ordered data
  - Example: DNA sequences, event logs

### Time Series Data
  - Structure: Measurements over successive time points
  - Use: Temporal pattern analysis
  - Example: Daily rainfall, stock prices, temperature readings

### Spatial and Spatio-Temporal Data
  - Structure: Attributes related to positions/areas
  - Use: Geographic analysis
  - Example: Weather maps, location-based data

### Web & Social Network Data
  - Structure: Network relationships
  - Use: Social analysis
  - Example: Twitter connections, Facebook links

## Data Pre-processing Steps
### Data Cleaning
  - Goal:
    - Detection and correction of data quality problems.
    - Poor quality data (e.g., noisy data, wrong data, missing values) can lead to highly inaccurate models.
  - Handling Missing Values
    - Option 1: Delete column (if >10% missing)
    - Option 2: Replace with zero
    - Option 3: Replace with last known value
    - Option 4: Use ML to predict missing values
  - Handling Outliers
    - Definition: Data points unusual relative to rest
    - Action: Remove if identified as noise
  - Addressing Inconsistency
    - Definition: Contradictory attributes (age vs birth year)
    - Action: Correct or remove duplicates

### Data Aggregation
  - Definition: Combine data based on common identifier
  - Example: Calculate total sales for transaction ID

### Sampling
  - Definition: Select subset of data for analysis
  - Key Point: Sample size must represent population
  - Types: Simple random sampling, stratified sampling
  - Risk: Too small sample misses patterns

### Scaling and Normalization
  - Normalization
    - Range: [0,1] or [-1,1]
    - Use: Non-assumption based algorithms (KNN, Neural Networks)
  - Standardization
    - Use: Algorithms assuming Gaussian distribution
    - Advantage: Less affected by outliers

## Feature Engineering Steps
### Feature Extraction Techniques
  - Definition: Create lower-dimension features from raw data
  - Dimensionality Reduction
    - Goal: Reduce number of attributes
    - Example: Principal Component Analysis (PCA)
  - Transformation Examples
    - Density = mass / volume
    - Fourier transform for time series

### Feature Selection Techniques
  - Definition: Select subset of existing features
  - Goal: Discard redundant/irrelevant features
  - Techniques
    - Filter Methods: Use heuristics (like correlation scores) independent of the learning algorithm to assess feature relevance
    - Wrapper Methods: Use the target learning algorithm itself as a "black box" to evaluate how well a subset of features performs
    - Embedded Methods: Perform feature selection automatically during the operation of the data mining algorithm (e.g., decision tree algorithms).

### Feature Construction Techniques
  - Definition: Creating genuinely new features, often based on combining existing ones by utilizing domain knowledge
  - Techniques
    - Polynomial Expansion: Univariate mathematical functions
    - Feature Crossing: Combine features to capture interactions

### Transformation and Encoding Techniques
  - Discretization (Binning/Bucketing)
    - Convert: Continuous to discrete (age to youth, adult)
    - Techniques
      - Equal-width partitioning
      - Equal-depth (frequency) partitioning
  - Binarization
    - Convert: Attributes to one or more binary variables
    - Methods: One Hot Encoding (categorical features)
   
# CS3 - Linear Regression

## Table of Contents
1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Linear Regression Interpretation and Formula](#2-linear-regression-interpretation-and-formula)
3. [Types of Regression Models and Formulas](#3-types-of-regression-models-and-formulas)
4. [Least Square Regression Line and Notations](#4-least-square-regression-line-and-notations)
5. [Linear Regression – Hypothesis Function and Formulas](#5-linear-regression--hypothesis-function-and-formulas)
6. [Quick Reference Table](#6-quick-reference-table)
7. [Examples and Numerical Solutions](#7-examples-and-numerical-solutions)

---

## 1. What is Linear Regression?

### Simple Definition

Linear regression is a method to predict a continuous value based on one or more input variables by finding the best-fitting straight line (or plane in higher dimensions).

### Key Points

- We have **input variables** (features/independent variables) → $X$
- We have **output variable** (target/dependent variable) → $y$
- We want to find the **relationship** between them
- The relationship is assumed to be **linear**

### Real-world Examples

**Example 1: House Price Prediction**
- Input ($X$): House size (sq ft)
- Output ($y$): House price ($)
- Goal: Find the line that best describes this relationship

**Example 2: Student Performance**
- Input ($X$): Hours studied
- Output ($y$): Exam score
- Goal: Predict score based on study hours

**Example 3: Employee Salary**
- Input ($X$): Years of experience
- Output ($y$): Salary
- Goal: Predict salary based on experience

### Why Linear Regression?

- **Simple and Interpretable**: Easy to understand and explain
- **Computationally Efficient**: Fast to train and predict
- **Good Baseline**: Often works well as a starting point
- **Theoretical Foundation**: Well-established mathematical theory

---

## 2. Linear Regression Interpretation and Formula

### Simple Linear Regression (One Input Variable)

#### Formula

$$y = w_0 + w_1 x$$

Or equivalently:

$$\hat{y} = b + mx$$

#### Parameters Explanation

| Symbol | Name | Meaning |
|--------|------|---------|
| $y$ or $\hat{y}$ | Predicted Output | The value we predict (the "hat" means it's a prediction) |
| $x$ | Input Variable | Independent variable (feature) |
| $w_0$ or $b$ | Intercept/Bias | Where the line crosses the y-axis (value when $x=0$) |
| $w_1$ or $m$ | Slope/Weight | How much $y$ changes when $x$ increases by 1 unit |

#### Visual Representation

```
      y
      |     •(actual point)
      |    /|
      |   / |error
      |  /• (predicted point)
      | /
      |/_____ x
      
The line y = w_0 + w_1*x is fitted to minimize total error
```

#### Real Example: House Price Prediction

If we predict house price: $y = 50,000 + 200x$

- $w_0 = 50,000$ → Base price (intercept)
- $w_1 = 200$ → Price increases $200 per sq ft (slope)
- If house is 1000 sq ft: $\hat{y} = 50,000 + 200(1000) = 250,000$
- If house is 2000 sq ft: $\hat{y} = 50,000 + 200(2000) = 450,000$

**Interpretation:**
- Every additional square foot adds $200 to the price
- A house with 0 sq ft would cost $50,000 (theoretical base)

---

### Multiple Linear Regression (Many Input Variables)

#### Formula

$$y = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

Or in vector form:

$$y = \mathbf{w}^T \mathbf{x}$$

#### Vector Form Explanation

$$\mathbf{w}^T \mathbf{x} = \begin{bmatrix} w_0 & w_1 & w_2 & \cdots & w_n \end{bmatrix} \cdot \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

$$= w_0(1) + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

#### Parameters Explanation

- $x_1, x_2, ..., x_n$ = multiple input features
- $w_0$ = intercept/bias term
- $w_1, w_2, ..., w_n$ = weights for each feature
- Each weight $w_i$ shows how much feature $x_i$ contributes to the output

#### Real Example: House Price with Multiple Features

$$\text{Price} = 50,000 + 200(\text{size}) + 5,000(\text{bedrooms}) - 100(\text{age})$$

**Interpretation:**
- Base price: $50,000
- Each sq ft adds: $200
- Each bedroom adds: $5,000
- Each year of age reduces price by: $100

**Sample Calculation:**
For a house with 1500 sq ft, 3 bedrooms, and 10 years old:
$$\text{Price} = 50,000 + 200(1500) + 5,000(3) - 100(10)$$
$$= 50,000 + 300,000 + 15,000 - 1,000$$
$$= 364,000$$

---

## 3. Types of Regression Models and Formulas

### A. Simple Linear Regression

#### Formula
$$y = w_0 + w_1 x$$

#### Characteristics
- One input variable
- Simplest form of regression
- Produces a straight line
- Easy to visualize and interpret

#### When to Use
- When you have only one feature
- When relationship appears linear
- For initial exploration and baseline

---

### B. Multiple Linear Regression

#### Formula
$$y = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

Or in vector form:
$$y = \mathbf{w}^T \mathbf{x}$$

#### Characteristics
- Multiple input variables
- Still linear in parameters ($w$)
- Produces a plane (in 3D) or hyperplane (in higher dimensions)
- More expressive than simple linear regression

#### Example with 3 Features
$$\text{Salary} = 30,000 + 2,000(\text{experience}) + 500(\text{education}) + 100(\text{skills})$$

---

### C. Polynomial Regression

#### Formula
$$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + ... + w_d x^d$$

#### Characteristics
- **Linear in parameters** ($w$) but **nonlinear in input** ($x$)
- Can fit curved relationships
- Produces curved lines/surfaces
- Higher degree = more complex curves

#### Examples

**Degree 2 (Quadratic):**
$$y = 1 + 2x + 3x^2$$

**Degree 3 (Cubic):**
$$y = 5 + 2x - 3x^2 + 0.5x^3$$

#### Visual Difference
```
Linear:        Quadratic:     Cubic:
   /              ∩            ∧∨
  /              / \          / \
 /              /   \        /   \
```

#### When to Use
- When data shows curved patterns
- Non-linear relationships
- Physics/engineering problems with natural curves

#### Caution
- Higher degree = risk of over-fitting
- Can become unstable (oscillates wildly)

---

### D. Ridge Regression (L2 Regularization)

#### Formula
$$y = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

#### Cost Function with Penalty
$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N (y_n - t_n)^2 + \frac{\lambda}{2} \sum_{j=1}^n w_j^2$$

Or:
$$E(\mathbf{w}) = \text{Error} + \lambda \cdot \text{Penalty}$$

#### Parameters
- **First term** = data fit error (wants this small)
- **Second term** = penalty for large weights (wants this small)
- **$\lambda$** = regularization parameter (balances the two)

#### Interpretation
- Large $\lambda$ → penalize big weights heavily → simpler model (more bias, less variance)
- Small $\lambda$ → penalize less → more complex model (less bias, more variance)
- $\lambda = 0$ → regular linear regression (no penalty)

#### Closed-Form Solution
$$\mathbf{w} = (\lambda \mathbf{I} + \mathbf{\Phi}^T\mathbf{\Phi})^{-1} \mathbf{\Phi}^T\mathbf{t}$$

Where $\mathbf{I}$ is the identity matrix.

#### Advantages
- Prevents over-fitting
- Handles multicollinearity (correlated features)
- All features retained
- Stable solution

#### When to Use
- When model over-fits training data
- With many features
- When you want to keep all features

---

### E. Lasso Regression (L1 Regularization)

#### Formula
$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N (y_n - t_n)^2 + \lambda \sum_{j=1}^n |w_j|$$

#### Key Difference from Ridge
- Uses **absolute value** ($|w_j|$) instead of **square** ($w_j^2$)
- Can shrink some weights to **exactly zero**
- Performs **automatic feature selection**

#### Advantages
- Automatic feature selection (some weights become 0)
- Interpretable (only important features remain)
- Sparse solutions
- Simpler final model

#### Disadvantages
- No closed-form solution (needs iterative methods)
- Can be unstable with correlated features
- Slower to compute than Ridge

#### When to Use
- When you want feature selection
- With many features, only some relevant
- Want a sparse, interpretable model

---

### F. Elastic Net (Combination of L1 and L2)

#### Formula
$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N (y_n - t_n)^2 + \lambda_1 \sum_{j=1}^n |w_j| + \lambda_2 \sum_{j=1}^n w_j^2$$

Or:
$$E(\mathbf{w}) = \text{Error} + \lambda_1 \cdot \text{L1 Penalty} + \lambda_2 \cdot \text{L2 Penalty}$$

#### Parameters
- $\lambda_1$ = controls L1 (Lasso) regularization
- $\lambda_2$ = controls L2 (Ridge) regularization

#### Advantages
- Combines benefits of both Lasso and Ridge
- Better with correlated features than Lasso
- Performs feature selection like Lasso
- More stable than Lasso alone

#### When to Use
- When you have many correlated features
- Want both regularization and feature selection
- Best general-purpose regularized regression

---

## 4. Least Square Regression Line and Notations

### The Least Squares Method

#### Goal
Find the line that **minimizes the sum of squared errors** (differences between actual and predicted values).

#### Why Least Squares?
- Mathematically convenient (differentiable)
- Penalizes large errors more (quadratic penalty)
- Has analytical solution
- Well-established theory

---

### Key Notations

| Symbol | Meaning | Example |
|--------|---------|---------|
| $N$ | Number of training examples | 100 data points |
| $n$ | Index of example | $n = 1, 2, ..., N$ |
| $x_i$ or $x_{(i)}$ | $i$-th input value | 1500 (sq ft) |
| $y_i$, $t_i$, or $y_{(i)}$ | Actual $i$-th output value | 350,000 (price) |
| $\hat{y}_i$ or $\hat{t}_i$ | Predicted $i$-th output value | 348,000 |
| $e_i$ | Error for $i$-th example | $e_i = y_i - \hat{y}_i = 2,000$ |
| $\mathbf{X}$ | Design matrix (all inputs) | $N \times M$ matrix |
| $\mathbf{y}$ or $\mathbf{t}$ | Vector of actual outputs | $N \times 1$ vector |
| $\mathbf{\hat{y}}$ | Vector of predictions | $N \times 1$ vector |
| $\mathbf{w}$ | Weight vector | Contains $w_0, w_1, ..., w_n$ |
| $\mathbf{\Phi}$ | Basis/Feature matrix | $N \times M$ matrix of features |
| $\bar{x}$ | Mean of $x$ values | $\frac{1}{N}\sum x_i$ |
| $\bar{y}$ | Mean of $y$ values | $\frac{1}{N}\sum y_i$ |

---

### Cost Function (Sum of Squared Errors)

#### Definition

$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N (y_n - \hat{y}_n)^2$$

Or equivalently:

$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N (y_n - \mathbf{w}^T \mathbf{x}_n)^2$$

Or with Mean Squared Error:

$$J(\mathbf{w}) = \frac{1}{N}\sum_{n=1}^N (y_n - \hat{y}_n)^2$$

#### What This Means

1. Calculate error for each example: $e_i = y_i - \hat{y}_i$
2. Square each error: $e_i^2$
3. Sum all squared errors: $\sum e_i^2$
4. Divide by 2 (or N for MSE) - normalization

#### Why Square Errors?

- Positive and negative errors don't cancel out
- Large errors are penalized more (quadratic growth)
- Mathematically convenient (differentiable)
- Makes sense for Gaussian noise assumption

#### Example with 3 Data Points

| $n$ | $x$ | Actual $y$ | Predicted $\hat{y}$ | Error $e$ | Squared Error $e^2$ |
|-----|-----|-----------|-------------------|-----------|------------------|
| 1 | 1 | 3 | 2.5 | 0.5 | 0.25 |
| 2 | 2 | 5 | 5.2 | -0.2 | 0.04 |
| 3 | 3 | 7 | 7.1 | -0.1 | 0.01 |

$$E(\mathbf{w}) = \frac{1}{2}(0.25 + 0.04 + 0.01) = 0.15$$

$$J(\mathbf{w}) = \frac{1}{3}(0.25 + 0.04 + 0.01) = 0.10$$

---

### Finding the Optimal Line (Simple Linear Regression)

#### For Model: $y = w_0 + w_1 x$

The optimal weights that minimize squared error are:

#### Formula for Slope ($w_1$)

$$w_1 = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2}$$

Or equivalently:

$$w_1 = \frac{\text{Covariance}(x, y)}{\text{Variance}(x)}$$

#### Formula for Intercept ($w_0$)

$$w_0 = \bar{y} - w_1 \bar{x}$$

#### Interpretation

- **$w_1$** = measures how much $y$ changes when $x$ changes
- **$w_0$** = ensures line passes through point $(\bar{x}, \bar{y})$
- The line is "balanced" through the center of the data

---

### Numerical Example: Finding the Least Squares Line

#### Data Points
```
(1, 2), (2, 4), (3, 5), (4, 4), (5, 5)
```

#### Step 1: Calculate Means
$$\bar{x} = \frac{1+2+3+4+5}{5} = \frac{15}{5} = 3$$

$$\bar{y} = \frac{2+4+5+4+5}{5} = \frac{20}{5} = 4$$

#### Step 2: Create Deviation Tables

| $i$ | $x_i$ | $y_i$ | $(x_i - \bar{x})$ | $(y_i - \bar{y})$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|-----|-------|-------|------------------|------------------|----------------------------------|------------------|
| 1 | 1 | 2 | -2 | -2 | 4 | 4 |
| 2 | 2 | 4 | -1 | 0 | 0 | 1 |
| 3 | 3 | 5 | 0 | 1 | 0 | 0 |
| 4 | 4 | 4 | 1 | 0 | 0 | 1 |
| 5 | 5 | 5 | 2 | 1 | 2 | 4 |
| **Sum** | | | | | **6** | **10** |

#### Step 3: Calculate $w_1$ (Slope)

$$w_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{6}{10} = 0.6$$

**Interpretation:** For every 1 unit increase in $x$, $y$ increases by 0.6 units.

#### Step 4: Calculate $w_0$ (Intercept)

$$w_0 = \bar{y} - w_1 \bar{x} = 4 - 0.6(3) = 4 - 1.8 = 2.2$$

**Interpretation:** When $x = 0$, $y = 2.2$.

#### Step 5: Final Equation

$$\hat{y} = 2.2 + 0.6x$$

#### Step 6: Make Predictions

| $x$ | Predicted $\hat{y}$ |
|-----|-------------------|
| 1 | 2.2 + 0.6(1) = 2.8 |
| 2 | 2.2 + 0.6(2) = 3.4 |
| 3 | 2.2 + 0.6(3) = 4.0 |
| 4 | 2.2 + 0.6(4) = 4.6 |
| 5 | 2.2 + 0.6(5) = 5.2 |

---

### Matrix Notation (Multiple Linear Regression)

#### Design Matrix ($\mathbf{\Phi}$ or $\mathbf{X}$)

For data with $N$ samples and $M$ features:

$$\mathbf{\Phi} = \begin{bmatrix}
\phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \phi_2(\mathbf{x}_1) & \cdots & \phi_{M-1}(\mathbf{x}_1) \\
\phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \phi_2(\mathbf{x}_2) & \cdots & \phi_{M-1}(\mathbf{x}_2) \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \phi_2(\mathbf{x}_N) & \cdots & \phi_{M-1}(\mathbf{x}_N)
\end{bmatrix}$$

Dimensions: $N \times M$ (rows = samples, columns = features)

#### Weight Vector

$$\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_{M-1} \end{bmatrix}$$

Dimensions: $M \times 1$

#### Target Vector

$$\mathbf{t} = \begin{bmatrix} t_1 \\ t_2 \\ \vdots \\ t_N \end{bmatrix}$$

Dimensions: $N \times 1$

#### Predictions in Matrix Form

$$\mathbf{\hat{y}} = \mathbf{\Phi} \mathbf{w}$$

#### Cost Function in Matrix Form

$$E(\mathbf{w}) = \frac{1}{2}(\mathbf{\Phi}\mathbf{w} - \mathbf{t})^T(\mathbf{\Phi}\mathbf{w} - \mathbf{t})$$

#### Optimal Solution (Normal Equations)

$$\mathbf{w}_{\text{opt}} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$$

**Components:**
- $\mathbf{\Phi}^T$ = transpose of design matrix
- $\mathbf{\Phi}^T\mathbf{\Phi}$ = Gram matrix (symmetric, positive definite)
- $(\cdot)^{-1}$ = matrix inverse
- $\mathbf{\Phi}^T\mathbf{t}$ = cross-correlation vector

---

## 5. Linear Regression – Hypothesis Function and Formulas

### What is a Hypothesis Function?

A **hypothesis function** is our current model/guess for the relationship between inputs and outputs. It's the formula we use to make predictions.

**Analogy:** A hypothesis is like a proposed "law of nature" that we're testing to see if it fits our data.

#### Different Names for Same Concept

- Hypothesis function: $h(\mathbf{x})$
- Model function: $f(\mathbf{x})$
- Prediction function: $\hat{y}(\mathbf{x})$
- Learner: machine learning model
- All represent the same idea

---

### Hypothesis Function Notations

#### Notation 1: Machine Learning Style

$$h_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$$

Subscript shows dependence on parameters $\mathbf{w}$

#### Notation 2: Statistical Style

$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{\phi}(\mathbf{x})$$

With explicit basis functions

#### Notation 3: Simple Style

$$y = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

Expanded form, most intuitive

#### All Three Are Equivalent

All represent: **weighted linear combination of inputs**

---

### Simple Linear Regression Hypothesis

#### Formula

$$h(x) = w_0 + w_1 x$$

Or with $\theta$ notation (common in Andrew Ng's courses):

$$h_\theta(x) = \theta_0 + \theta_1 x$$

#### Parameters

| Symbol | Meaning |
|--------|---------|
| $h(x)$ | Hypothesis function (prediction) |
| $\theta_0$ | Intercept parameter |
| $\theta_1$ | Slope parameter |
| $x$ | Input feature |

#### Example

If $h(x) = 3 + 2x$, then:
- When $x = 0$: $h(0) = 3$
- When $x = 5$: $h(5) = 3 + 2(5) = 13$
- When $x = 10$: $h(10) = 3 + 2(10) = 23$

---

### Multiple Linear Regression Hypothesis

#### Formula

$$h(\mathbf{x}) = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

Or in vector form:

$$h(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$$

#### Vector Form Details

$$h(\mathbf{x}) = \begin{bmatrix} w_0 & w_1 & w_2 & \cdots & w_n \end{bmatrix} \cdot \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

**Note:** $x_0 = 1$ is a dummy variable for the bias term.

#### Example with 3 Features

$$h(x_1, x_2, x_3) = 10 + 2x_1 + 3x_2 - 0.5x_3$$

For inputs $(x_1=5, x_2=4, x_3=2)$:
$$h = 10 + 2(5) + 3(4) - 0.5(2) = 10 + 10 + 12 - 1 = 31$$

---

### Cost Function for Hypothesis

#### Purpose

Measure how good our hypothesis is. Lower cost = better hypothesis.

#### Squared Error Cost Function

$$J(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (h(\mathbf{x}_i) - y_i)^2$$

Or equivalently:

$$J(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

#### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (h(\mathbf{x}_i) - y_i)^2$$

Note: MSE = $2 \times$ cost function with $\frac{1}{2}$

#### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (h(\mathbf{x}_i) - y_i)^2}$$

**Advantage:** In same units as target variable (more interpretable)

#### Example

| $i$ | $\mathbf{x}_i$ | Actual $y_i$ | Prediction $h(\mathbf{x}_i)$ | Error | Squared Error |
|-----|-----------|----------|-------------|-------|---------------|
| 1 | ... | 10 | 11 | -1 | 1 |
| 2 | ... | 20 | 19 | 1 | 1 |
| 3 | ... | 30 | 31 | -1 | 1 |

$$J(\mathbf{w}) = \frac{1}{2(3)}(1 + 1 + 1) = \frac{3}{6} = 0.5$$

$$\text{MSE} = \frac{1}{3}(1 + 1 + 1) = 1.0$$

---

### Goal: Minimize Cost Function

#### Objective

Find the weights $\mathbf{w}$ that **minimize** $J(\mathbf{w})$:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} J(\mathbf{w})$$

Read as: "$\mathbf{w}^*$ is the value of $\mathbf{w}$ that minimizes $J(\mathbf{w})$"

#### Geometric Interpretation

- Think of cost function as a bowl-shaped surface
- Optimal weights are at the bottom of the bowl
- We want to find the lowest point

```
High cost
    |
    ↑  ╱╲
    │ ╱  ╲
    │╱    ╲
Cost│     ╱╲ ← Minimum (optimal weights)
    │    ╱  ╲
    │   ╱    ╲
    └─────────────
      w1 → w2 →
```

---

### Two Methods to Find Optimal Weights

#### Method 1: Analytical Solution (Closed-Form)

**When to use:** Small to medium datasets

##### Formula (Normal Equations)

$$\mathbf{w} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$$

##### Components

- $\mathbf{\Phi}$ = Design matrix ($N \times M$)
- $\mathbf{\Phi}^T$ = Transpose
- $\mathbf{\Phi}^T\mathbf{\Phi}$ = Gram matrix ($M \times M$)
- $(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}$ = Matrix inverse
- $\mathbf{\Phi}^T\mathbf{t}$ = Cross-product with targets

##### Advantages
- Direct solution (one calculation)
- No tuning needed
- Guaranteed to find global minimum
- Faster for small problems

##### Disadvantages
- Computationally expensive for large $N$ or $M$
- Matrix inversion can be numerically unstable
- Memory intensive
- Doesn't work if matrix is singular

---

#### Method 2: Iterative Solution (Gradient Descent)

**When to use:** Large datasets, online learning

##### General Update Rule

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla J(\mathbf{w}^{(t)})$$

##### Component Interpretation

- $\mathbf{w}^{(t)}$ = weights at iteration $t$
- $\mathbf{w}^{(t+1)}$ = weights at iteration $t+1$
- $\alpha$ = learning rate (step size)
- $\nabla J$ = gradient (direction of steepest ascent)

##### For Each Parameter $w_j$

$$w_j^{(t+1)} = w_j^{(t)} - \alpha \frac{\partial J}{\partial w_j}$$

##### Gradient of MSE Cost Function

$$\frac{\partial J}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N (h(\mathbf{x}_i) - y_i) x_j^{(i)}$$

##### Full Update Equation

$$w_j := w_j - \alpha \frac{1}{N} \sum_{i=1}^N (h(\mathbf{x}_i) - y_i) x_j^{(i)}$$

Or in vector form:

$$\mathbf{w} := \mathbf{w} - \alpha \mathbf{\Phi}^T(\mathbf{\Phi}\mathbf{w} - \mathbf{t})$$

##### How Gradient Descent Works

1. **Start** with random weights
2. **Calculate** error for each example
3. **Find gradient** (direction of steepest increase)
4. **Move opposite** to gradient by step $\alpha$
5. **Repeat** until convergence

##### Algorithm Pseudocode

```
Initialize w randomly
for iteration = 1 to max_iterations:
    Calculate predictions: y_pred = Φ * w
    Calculate errors: e = y_pred - y_actual
    Calculate gradient: g = (1/N) * Φ^T * e
    Update weights: w = w - α * g
    If convergence: break
```

##### Learning Rate $\alpha$ Intuition

- **Too small $\alpha$**: Very slow convergence, takes many iterations
- **Too large $\alpha$**: May overshoot, diverge, or oscillate
- **Good $\alpha$**: Smooth, steady descent to minimum

```
Loss curve over iterations:
        |
Loss    |  Too large α (diverges)
        | /
        |/  Good α (converges smoothly)
        |\  
        | \___  Too small α (very slow)
        |______ 
        Iteration →
```

##### Advantages
- Works for large datasets
- Memory efficient
- Can be used for online learning
- Flexible (works for many algorithms)

##### Disadvantages
- Need to tune learning rate $\alpha$
- Many iterations needed
- May get stuck in local minimum (for non-convex problems)
- Slower per iteration than closed-form

---

## 6. Quick Reference Table

### Key Formulas

| Concept | Formula | Purpose |
|---------|---------|---------|
| **Simple Linear Regression** | $\hat{y} = w_0 + w_1 x$ | Predict with one input |
| **Multiple Linear Regression** | $\hat{y} = \mathbf{w}^T \mathbf{x}$ | Predict with multiple inputs |
| **Hypothesis Function** | $h(x) = w_0 + w_1 x$ | Our prediction model |
| **Cost Function** | $J(w) = \frac{1}{2N}\sum(h(x_i)-y_i)^2$ | Measure prediction error |
| **Slope (Simple)** | $w_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$ | Optimal slope for simple regression |
| **Intercept (Simple)** | $w_0 = \bar{y} - w_1 \bar{x}$ | Optimal intercept for simple regression |
| **Normal Equations** | $\mathbf{w} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$ | Analytical solution for multiple regression |
| **Gradient** | $\frac{\partial J}{\partial w_j} = \frac{1}{N}\sum(h(\mathbf{x}_i)-y_i)x_j^{(i)}$ | Direction of steepest increase |
| **Gradient Descent Update** | $w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$ | Iterative weight update |
| **Ridge Regression Cost** | $J = \frac{1}{2}\sum e_i^2 + \frac{\lambda}{2}\sum w_j^2$ | Regularized cost |
| **Ridge Solution** | $\mathbf{w} = (\lambda \mathbf{I} + \mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$ | Regularized optimal weights |
| **Lasso Cost** | $J = \frac{1}{2}\sum e_i^2 + \lambda \sum \|w_j\|$ | Sparsity-inducing cost |
| **Elastic Net Cost** | $J = \frac{1}{2}\sum e_i^2 + \lambda_1 \sum \|w_j\| + \lambda_2 \sum w_j^2$ | Combined L1 and L2 |

### Regression Types Comparison

| Type | Linear Parameters | Linear Inputs | Use Case | Pros | Cons |
|------|------------------|--------------|----------|------|------|
| Simple Linear | Yes | Yes | 1 feature, linear relationship | Simple, interpretable | Limited expressiveness |
| Multiple Linear | Yes | Yes | Many features, linear | Interpretable, efficient | May underfit |
| Polynomial | Yes | No | Curved relationships | More expressive | Risk of overfitting |
| Ridge | Yes | Yes/No | Many features, overfitting | Prevents overfitting | Must tune λ |
| Lasso | Yes | Yes/No | Feature selection needed | Sparse solutions | Unstable with correlations |
| Elastic Net | Yes | Yes/No | Many correlated features | Best of both | Must tune λ₁, λ₂ |

### Optimization Methods Comparison

| Method | Formula | When to Use | Advantages | Disadvantages |
|--------|---------|------------|------------|---------------|
| **Closed-Form** | $(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$ | Small datasets | Direct, guaranteed global min | Slow for large data, numerically unstable |
| **Gradient Descent** | $w := w - \alpha \nabla J$ | Large datasets | Scalable, works with many algorithms | Needs learning rate tuning |
| **Stochastic GD** | Update with 1 sample | Very large datasets | Most scalable | Noisy, may oscillate |
| **Mini-batch GD** | Update with batch | Moderate-large datasets | Balanced trade-off | Still needs hyperparameter tuning |

---

## 7. Examples and Numerical Solutions

### Complete Example 1: Simple Linear Regression

#### Problem Statement
Predict student exam scores based on hours studied.

#### Given Data

| Hours ($x$) | Score ($y$) |
|-----------|---------|
| 2 | 50 |
| 3 | 60 |
| 4 | 65 |
| 5 | 70 |
| 6 | 85 |

#### Step 1: Calculate Means

$$\bar{x} = \frac{2+3+4+5+6}{5} = 4$$

$$\bar{y} = \frac{50+60+65+70+85}{5} = 66$$

#### Step 2: Create Deviation Table

| $i$ | $x_i$ | $y_i$ | $(x_i - \bar{x})$ | $(y_i - \bar{y})$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|-----|-------|-------|------------------|------------------|----------------------------------|------------------|
| 1 | 2 | 50 | -2 | -16 | 32 | 4 |
| 2 | 3 | 60 | -1 | -6 | 6 | 1 |
| 3 | 4 | 65 | 0 | -1 | 0 | 0 |
| 4 | 5 | 70 | 1 | 4 | 4 | 1 |
| 5 | 6 | 85 | 2 | 19 | 38 | 4 |
| **Sum** | | | | | **80** | **10** |

#### Step 3: Calculate $w_1$ (Slope)

$$w_1 = \frac{80}{10} = 8$$

**Interpretation:** Each additional hour of study increases score by 8 points on average.

#### Step 4: Calculate $w_0$ (Intercept)

$$w_0 = 66 - 8(4) = 66 - 32 = 34$$

**Interpretation:** A student who studies 0 hours would score around 34 (hypothetical).

#### Step 5: Regression Equation

$$\hat{y} = 34 + 8x$$

#### Step 6: Make Predictions

| Hours ($x$) | Predicted Score | Actual Score | Error |
|-----------|-----------------|-------------|-------|
| 2 | 34 + 8(2) = 50 | 50 | 0 |
| 3 | 34 + 8(3) = 58 | 60 | -2 |
| 4 | 34 + 8(4) = 66 | 65 | 1 |
| 5 | 34 + 8(5) = 74 | 70 | 4 |
| 6 | 34 + 8(6) = 82 | 85 | -3 |

#### Step 7: Calculate Cost

$$J(w) = \frac{1}{2(5)}(0^2 + 2^2 + 1^2 + 4^2 + 3^2)$$

$$= \frac{1}{10}(0 + 4 + 1 + 16 + 9) = \frac{30}{10} = 3$$

#### Prediction for New Data

If student studies 7 hours:
$$\hat{y} = 34 + 8(7) = 90$$

Expected score: 90

---

### Complete Example 2: Multiple Linear Regression

#### Problem Statement
Predict house price based on size and age.

#### Given Data (5 houses)

| Size (sqft) | Age (years) | Price ($1000) |
|-----------|-----------|------------|
| 1200 | 10 | 250 |
| 1800 | 5 | 350 |
| 1500 | 15 | 280 |
| 2000 | 2 | 400 |
| 1400 | 20 | 260 |

#### Step 1: Create Design Matrix

$$\mathbf{\Phi} = \begin{bmatrix}
1 & 1200 & 10 \\
1 & 1800 & 5 \\
1 & 1500 & 15 \\
1 & 2000 & 2 \\
1 & 1400 & 20
\end{bmatrix}, \quad \mathbf{t} = \begin{bmatrix} 250 \\ 350 \\ 280 \\ 400 \\ 260 \end{bmatrix}$$

#### Step 2: Using Normal Equations (Simplified)

Using computational tools (or matrix calculations):

$$\mathbf{\Phi}^T\mathbf{\Phi} = \begin{bmatrix} 5 & 7900 & 52 \\ 7900 & 12,740,000 & 91,000 \\ 52 & 91,000 & 852 \end{bmatrix}$$

$$\mathbf{\Phi}^T\mathbf{t} = \begin{bmatrix} 1540 \\ 2,376,000 \\ 11,620 \end{bmatrix}$$

#### Step 3: Solve for Weights

After matrix inversion and multiplication:

$$\mathbf{w} = \begin{bmatrix} 100 \\ 0.15 \\ -2 \end{bmatrix}$$

#### Step 4: Regression Equation

$$\text{Price} = 100 + 0.15(\text{Size}) - 2(\text{Age})$$

#### Step 5: Interpretation

- **$w_0 = 100$**: Base price is $100,000
- **$w_1 = 0.15$**: Each additional sq ft adds $150 to price
- **$w_2 = -2$**: Each year of age reduces price by $2,000

#### Step 6: Predictions

**House 1:** 1200 sqft, 10 years
$$\text{Price} = 100 + 0.15(1200) - 2(10) = 100 + 180 - 20 = 260$$
(Actual: 250, Error: 10)

**House 3:** 1500 sqft, 15 years
$$\text{Price} = 100 + 0.15(1500) - 2(15) = 100 + 225 - 30 = 295$$
(Actual: 280, Error: 15)

**New House:** 1600 sqft, 8 years
$$\text{Price} = 100 + 0.15(1600) - 2(8) = 100 + 240 - 16 = 324$$
Predicted price: $324,000

---

### Example 3: Polynomial Regression

#### Problem Statement
Fit a quadratic (degree 2) polynomial to data showing curved relationship.

#### Data

| $x$ | $y$ |
|-----|-----|
| 0 | 1 |
| 1 | 2 |
| 2 | 5 |
| 3 | 10 |

#### Model
$$y = w_0 + w_1 x + w_2 x^2$$

#### Design Matrix (with basis functions $1, x, x^2$)

$$\mathbf{\Phi} = \begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 1 \\
1 & 2 & 4 \\
1 & 3 & 9
\end{bmatrix}, \quad \mathbf{t} = \begin{bmatrix} 1 \\ 2 \\ 5 \\ 10 \end{bmatrix}$$

#### Solution (Using Normal Equations)

After solving: $\mathbf{w} = [1, 0.5, 0.5]$

#### Equation

$$\hat{y} = 1 + 0.5x + 0.5x^2$$

#### Predictions

| $x$ | Predicted $\hat{y}$ | Actual $y$ | Error |
|-----|------------------|----------|-------|
| 0 | 1 + 0.5(0) + 0.5(0) = 1 | 1 | 0 |
| 1 | 1 + 0.5(1) + 0.5(1) = 2 | 2 | 0 |
| 2 | 1 + 0.5(2) + 0.5(4) = 4 | 5 | -1 |
| 3 | 1 + 0.5(3) + 0.5(9) = 7.5 | 10 | -2.5 |

Note: Quadratic model fits the increasing curvature better than a line would.

---

## Summary

### Key Takeaways

1. **Linear Regression** predicts continuous values using linear models
2. **Simple** (1 input) and **Multiple** (many inputs) forms exist
3. **Polynomial** regression extends to curved relationships
4. **Least Squares** minimizes sum of squared errors
5. **Normal Equations** provide analytical solution for small data
6. **Gradient Descent** is iterative method for large data
7. **Regularization** (Ridge, Lasso, Elastic Net) prevents overfitting
8. **Bias-Variance Tradeoff** governs model complexity

### When to Use Each Approach

- **Simple Linear:** 1 feature, linear relationship, interpretability critical
- **Multiple Linear:** Many features, linear relationships
- **Polynomial:** Curved relationships, feature engineering
- **Ridge:** Overfitting, all features important
- **Lasso:** Many features, want automatic selection
- **Elastic Net:** Many correlated features

### Implementing Steps

1. Prepare data (clean, normalize if needed)
2. Split into training and test sets
3. Choose model type and complexity
4. Select optimization method (closed-form or gradient descent)
5. Train model (find optimal weights)
6. Evaluate on test set
7. Tune hyperparameters if needed
8. Make predictions on new data

---
