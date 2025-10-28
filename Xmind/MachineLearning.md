# Machine Learning (ML)

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
