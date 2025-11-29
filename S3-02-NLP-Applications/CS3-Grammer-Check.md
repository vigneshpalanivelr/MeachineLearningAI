# Grammar Check and Spell Correction - Complete Study Guide with Calculations

## Table of Contents

- [Grammar Check and Spell Correction - Complete Study Guide with Calculations](#grammar-check-and-spell-correction---complete-study-guide-with-calculations)
  - [Table of Contents](#table-of-contents)
  - [Quick Links](#quick-links)
  - [1. Introduction to Grammatical Errors](#1-introduction-to-grammatical-errors)
    - [1.1 Definition and Classification](#11-definition-and-classification)
    - [1.2 Syntax vs Usage Errors](#12-syntax-vs-usage-errors)
  - [2. Error Type Distribution and Analysis](#2-error-type-distribution-and-analysis)
    - [2.1 Complete Error Distribution Table](#21-complete-error-distribution-table)
    - [2.2 Detailed Analysis of Major Error Types](#22-detailed-analysis-of-major-error-types)
      - [2.2.1 Preposition Errors (13%)](#221-preposition-errors-13)
      - [2.2.2 Determiner Errors (12%)](#222-determiner-errors-12)
      - [2.2.3 Verbal Morphology Errors (14%)](#223-verbal-morphology-errors-14)
      - [2.2.4 Content Word Choice Errors (20%)](#224-content-word-choice-errors-20)
  - [3. Grammar Correction Approaches](#3-grammar-correction-approaches)
    - [3.1 Rule-Based Methods](#31-rule-based-methods)
      - [3.1.1 Category A: No Context Needed](#311-category-a-no-context-needed)
      - [3.1.2 Category B: Local Context Needed](#312-category-b-local-context-needed)
    - [3.2 Classification-Based Methods](#32-classification-based-methods)
      - [3.2.1 Feature Engineering](#321-feature-engineering)
      - [3.2.2 Training Process](#322-training-process)
    - [3.3 Language Model Approaches](#33-language-model-approaches)
      - [3.3.1 N-gram Language Models](#331-n-gram-language-models)
      - [3.3.2 Probability Calculations](#332-probability-calculations)
    - [3.4 Statistical Machine Translation (SMT)](#34-statistical-machine-translation-smt)
      - [3.4.1 Phrase-Based SMT](#341-phrase-based-smt)
      - [3.4.2 Noisy Channel Model](#342-noisy-channel-model)
  - [4. Evaluation Metrics - Complete Calculations](#4-evaluation-metrics---complete-calculations)
    - [4.1 Confusion Matrix Fundamentals](#41-confusion-matrix-fundamentals)
    - [4.2 Metric Formulas and Examples](#42-metric-formulas-and-examples)
      - [Example 1: Basic Metrics Calculation](#example-1-basic-metrics-calculation)
      - [Example 2: Comprehensive Evaluation](#example-2-comprehensive-evaluation)
      - [Example 3: Comparing Two Systems](#example-3-comparing-two-systems)
  - [5. Language Model Probability Calculations](#5-language-model-probability-calculations)
    - [5.1 Unigram Model](#51-unigram-model)
    - [5.2 Bigram Model](#52-bigram-model)
    - [5.3 Trigram Model](#53-trigram-model)
    - [5.4 Preposition Selection Using LM](#54-preposition-selection-using-lm)
  - [6. Classification Model Examples](#6-classification-model-examples)
    - [6.1 Verb Form Classification](#61-verb-form-classification)
    - [6.2 Feature Extraction Example](#62-feature-extraction-example)
    - [6.3 Decision Tree Construction](#63-decision-tree-construction)
  - [7. Complete Worked Examples](#7-complete-worked-examples)
    - [Example 1: Subject-Verb Agreement Detection](#example-1-subject-verb-agreement-detection)
    - [Example 2: Article Error Detection](#example-2-article-error-detection)
    - [Example 3: Preposition Selection](#example-3-preposition-selection)
    - [Example 4: Verb Tense Correction](#example-4-verb-tense-correction)
    - [Example 5: Language Model Scoring](#example-5-language-model-scoring)
  - [8. Statistical Machine Translation Calculations](#8-statistical-machine-translation-calculations)
    - [8.1 Translation Probability](#81-translation-probability)
    - [8.2 Alignment Calculation](#82-alignment-calculation)
  - [9. Advanced Topics](#9-advanced-topics)
    - [9.1 Deep Learning Model Evaluation](#91-deep-learning-model-evaluation)
    - [9.2 Multi-Class Classification Metrics](#92-multi-class-classification-metrics)
  - [10. Practice Problems with Solutions](#10-practice-problems-with-solutions)
    - [Problem 1: Evaluation Metrics](#problem-1-evaluation-metrics)
    - [Problem 2: Language Model Probability](#problem-2-language-model-probability)
    - [Problem 3: Classification Accuracy](#problem-3-classification-accuracy)
    - [Problem 4: N-gram Calculation](#problem-4-n-gram-calculation)
    - [Problem 5: SMT Scoring](#problem-5-smt-scoring)
  - [11. Quick Reference Tables](#11-quick-reference-tables)
    - [11.1 POS Tag Reference](#111-pos-tag-reference)
    - [11.2 Evaluation Metrics Quick Reference](#112-evaluation-metrics-quick-reference)
    - [11.3 Error Type Reference](#113-error-type-reference)
  - [12. Exam Strategy Guide](#12-exam-strategy-guide)
    - [12.1 Key Formulas to Memorize](#121-key-formulas-to-memorize)
    - [12.2 Common Question Types](#122-common-question-types)
    - [12.3 Calculation Checklist](#123-calculation-checklist)

---
## Quick Links

| Topic | Link | Description |
|-------|------|-------------|
| **Grammar Analysis Tools** | | |
| John Snow Labs - Analyze Spelling & Grammar | [John Snow Labs](https://nlp.johnsnowlabs.com/analyze_spelling_grammar) | Online grammar and spelling analysis tool |
| Sapling AI Grammar Check | [Sapling](https://sapling.ai/grammar-check) | AI-powered grammar checking |
| Spark NLP Demos | [Spark NLP Demos](https://sparknlp.org/analyze_spelling_grammar) | Interactive Spark NLP demonstrations for spelling & grammar |
| Lang-8 | [Lang-8](https://lang-8.jp/en/) | Language learning platform with native speaker corrections |
| **Research Papers & Surveys** | | |
| GEC Survey (MIT) | [Paper: Grammatical Error Correction](https://direct.mit.edu/coli/article/49/3/643/115846/Grammatical-Error-Correction-A-Survey-of-the-State) | Comprehensive survey of grammatical error correction state-of-the-art |
| Statistical MT Framework for GEC | [Paper: Grammatical and context-sensitive error correction using a Statistical Machine Translation framework](https://onlinelibrary.wiley.com/doi/10.1002/spe.2110) | Research on using statistical machine translation for grammar error correction |
| ML-Based Grammar Error Detection | [Paper: ML Based Grammar Error Detection](https://onlinelibrary.wiley.com/doi/10.1155/2021/4213791) | Machine learning approaches to grammar error detection |
| **Industry Solutions** | | |
| Microsoft Triton AI Grammar | [NVIDIA Blog](https://blogs.nvidia.com/blog/microsoft-triton-ai-grammar-word/) | Microsoft's Triton AI grammar word implementation |
| **Datasets** | | |
| Papers with Code - GEC Datasets | [Papers with Code](https://paperswithcode.com/datasets?q=&v=lst&o=newest&task=grammatical-error-correction&mod=texts&page=1) | Collection of grammatical error correction datasets |
| **Video Tutorials** | | |
| Grammar Tutorial 1 | [YouTube](https://www.youtube.com/watch?v=3rVn14m8zaM) | Grammar checking tutorial |
| Grammar Tutorial 2 | [YouTube](https://www.youtube.com/watch?v=pYV8OydsnQA) | Additional grammar tutorial |
| Grammar Tutorial 3 | [YouTube](https://www.youtube.com/watch?v=9ZkM-in-EWA) | Grammar checking techniques |
| Grammar Tutorial 4 | [YouTube](https://www.youtube.com/watch?v=2RU5egfeb_g) | Grammar correction methods |
| Grammar Tutorial 5 | [YouTube](https://www.youtube.com/watch?v=0MiFUES-0F4) | Advanced grammar checking |
| **Google Colab Notebooks** | | |
| SpellChecker-AutoCorrect (Seq2Seq) | [Google Colab](https://colab.research.google.com/github/piyush0511/SpellChecker-AutoCorrect/blob/main/SpellCheck%20-%20%20seq2seq.ipynb#scrollTo=eqcJXSpf0fPW) | Sequence-to-sequence spelling correction implementation |
| Grammar Correction Colab | [Google Colab](https://colab.research.google.com/drive/1ysEKrw_LE2jMndo1snrZUh5w87LQsCxk) | Interactive grammar correction notebook |

## 1. Introduction to Grammatical Errors

### 1.1 Definition and Classification

**Grammatical Error Definition:**
A grammatical error occurs when text violates either:
1. **Syntax rules** (structural grammar rules)
2. **Usage conventions** (conventional language patterns)

**Error Classification Framework:**

```
Grammatical Errors
├── Syntax Errors (Rule-based)
│   ├── Subject-verb agreement
│   ├── Pronoun-antecedent agreement
│   ├── Verb tense formation
│   └── Sentence structure
│
└── Usage Errors (Convention-based)
    ├── Preposition selection
    ├── Article usage
    ├── Collocation
    └── Idiomatic expressions
```

### 1.2 Syntax vs Usage Errors

| Aspect | Syntax Errors | Usage Errors |
|--------|---------------|--------------|
| **Nature** | Violate grammar rules | Violate conventions |
| **Learning** | Easier (systematic) | Harder (memorization) |
| **Example** | "She go" → "She goes" | "on Monday" → "in Monday" ❌ |
| **Detection** | Rule-based possible | Requires corpus/statistics |
| **Frequency** | Lower for learners | Higher for learners |

---

## 2. Error Type Distribution and Analysis

### 2.1 Complete Error Distribution Table

| Error Type | Percentage | Difficulty | Detection Method |
|------------|-----------|------------|------------------|
| Content Word Choice | 20% | Very High | Statistical/LM |
| Verbal Morphology | 14% | High | Rules + ML |
| Prepositions | 13% | Very High | ML/LM |
| Determiners | 12% | High | ML/LM |
| Punctuation | 12% | Medium | Rules |
| Derivational Morphology | 5% | Medium | Rules + Dictionary |
| Pronoun | 4% | Medium | Rules + Discourse |
| Agreement | 4% | Low-Medium | Rules |
| Run-on Sentences | 4% | Medium | Parsing |
| Word Order | 4% | Medium | Parsing |
| Real Word Spelling | 2% | High | Context-aware |
| Other | 6% | Varies | Mixed |
| **Total** | **100%** | - | - |

**Visual Distribution:**

```
Content Word Choice     ████████████████████ 20%
Verbal Morphology       ██████████████ 14%
Prepositions            █████████████ 13%
Determiners             ████████████ 12%
Punctuation             ████████████ 12%
Derivational Morph      █████ 5%
Pronoun                 ████ 4%
Agreement               ████ 4%
Run-on                  ████ 4%
Word Order              ████ 4%
Real Word Spelling      ██ 2%
Other                   ██████ 6%
```

### 2.2 Detailed Analysis of Major Error Types

#### 2.2.1 Preposition Errors (13%)

**Why Prepositions are Difficult:**

1. **Multiple contextual roles**
2. **Limited semantic predictability**
3. **High variation across languages**

**Preposition Role Classification:**

| Role | Function | Example | Correction Challenge |
|------|----------|---------|---------------------|
| Temporal | Time expression | "on Monday", "at noon" | Medium |
| Spatial | Location | "at home", "in the room" | Medium |
| Argument | Verb complement | "fond of", "depend on" | High |
| Phrasal Verb | Verb particle | "give in", "look after" | Very High |

**Example Calculation: Preposition Selection**

Given corpus frequencies:

| Phrase | Count | Probability |
|--------|-------|-------------|
| "fond of cats" | 638,000 | P = 0.94 |
| "fond for cats" | 178 | P = 0.0003 |
| "fond by cats" | 0 | P = 0.0000 |
| "fond to cats" | 269 | P = 0.0004 |
| "fond with cats" | 13,300 | P = 0.02 |

**Calculation:**

```
Total occurrences = 638,000 + 178 + 0 + 269 + 13,300 = 651,747

P("of") = 638,000 / 651,747 = 0.979
P("with") = 13,300 / 651,747 = 0.020
P("for") = 178 / 651,747 = 0.0003

Decision: Select "of" (highest probability)
```

#### 2.2.2 Determiner Errors (12%)

**Article System Complexity Factors:**

1. **Countability**: "a car" ✓ vs "*an equipment" ✗
2. **Definiteness**: First mention vs subsequent
3. **Specificity**: Generic vs specific reference
4. **Syntax**: "have knowledge" vs "have a knowledge of"

**Decision Tree for Article Selection:**

```
Is noun countable?
├── YES
│   ├── Singular?
│   │   ├── YES
│   │   │   ├── Definite/Specific?
│   │   │   │   ├── YES → "the"
│   │   │   │   └── NO → "a/an"
│   │   │   └── NO
│   │   └── Plural?
│   │       ├── Definite? → "the"
│   │       └── Generic? → ∅ (no article)
│
└── NO (Uncountable)
    ├── Definite? → "the"
    └── Generic? → ∅ (no article)
```

**Example Calculation:**

```
Sentence: "I bought shirt at store."

Step 1: Identify nouns
- "shirt" (countable, singular)
- "store" (countable, singular)

Step 2: Check definiteness
- "shirt": first mention → indefinite
- "store": specific store → definite

Step 3: Apply rules
- "shirt" + singular + indefinite → "a shirt"
- "store" + singular + definite → "the store"

Result: "I bought a shirt at the store."
```

#### 2.2.3 Verbal Morphology Errors (14%)

**Complete Verb Form System:**

| Form Name | POS Tag | Function | Example (eat) |
|-----------|---------|----------|---------------|
| Infinitive | VB | Base form | eat |
| Past Tense | VBD | Simple past | ate |
| Gerund/Present Participle | VBG | -ing form | eating |
| Past Participle | VBN | Perfect/Passive | eaten |
| Non-3rd Person Singular | VBP | I/you/we/they present | eat |
| 3rd Person Singular | VBZ | he/she/it present | eats |

**Verb Form Selection Algorithm:**

```
Step 1: Identify verb context
Step 2: Check for auxiliaries
Step 3: Determine tense/aspect
Step 4: Apply agreement rules
Step 5: Select appropriate form
```

**Example Calculation:**

```
Sentence: "They were eaten ice-cream when I arrived."

Step 1: Parse structure
- Subject: "They" (plural)
- Auxiliary: "were" (past)
- Main verb: "eaten" (VBN)
- Context: temporal clause → progressive aspect

Step 2: Check compatibility
- Past progressive = BE_PAST + VBG
- Current: BE_PAST + VBN ✗

Step 3: Identify error
- Form: VBN (past participle)
- Required: VBG (present participle)

Step 4: Apply correction
- "eaten" → "eating"

Result: "They were eating ice-cream when I arrived."

Confidence Score: 0.95 (high - clear rule violation)
```

#### 2.2.4 Content Word Choice Errors (20%)

**Collocation Strength Calculation:**

**Example: Adjective + Noun Collocation**

Corpus statistics:

| Adjective | Tea (counts) | Computer (counts) | Total |
|-----------|--------------|-------------------|-------|
| strong | 850,000 | 5,000 | 855,000 |
| powerful | 2,000 | 900,000 | 902,000 |
| heavy | 1,000 | 20,000 | 21,000 |

**Calculation: Pointwise Mutual Information (PMI)**

Formula: `PMI(word1, word2) = log₂(P(word1, word2) / (P(word1) × P(word2)))`

```
For "strong tea":
P(strong, tea) = 850,000 / 10,000,000 = 0.085
P(strong) = 855,000 / 10,000,000 = 0.0855
P(tea) = 1,000,000 / 10,000,000 = 0.1

PMI(strong, tea) = log₂(0.085 / (0.0855 × 0.1))
                 = log₂(0.085 / 0.00855)
                 = log₂(9.94)
                 = 3.31

For "powerful tea":
P(powerful, tea) = 2,000 / 10,000,000 = 0.0002
P(powerful) = 902,000 / 10,000,000 = 0.0902

PMI(powerful, tea) = log₂(0.0002 / (0.0902 × 0.1))
                   = log₂(0.0002 / 0.00902)
                   = log₂(0.022)
                   = -5.50

Decision: "strong tea" has positive PMI (good collocation)
         "powerful tea" has negative PMI (poor collocation)
```

---

## 3. Grammar Correction Approaches

### 3.1 Rule-Based Methods

#### 3.1.1 Category A: No Context Needed

**Regular Expression Patterns:**

```regex
Pattern 1: Infinitive Formation
/to( RB)* VB[DNGZ]/
Matches: "to talking", "to eaten"
Correction: "to talk", "to eat"

Pattern 2: Modal + Perfect
/MD of VBD/
Matches: "would of liked"
Correction: "would have liked"

Pattern 3: Double Negative
/not.*n't|never.*not/
Matches: "didn't see nothing"
Correction: "didn't see anything"
```

**Example Application:**

```
Input: "I want to talking with you"

Step 1: Apply pattern /to( RB)* VB[DNGZ]/
Match found: "to talking"

Step 2: Extract components
- "to": TO
- "talking": VBG

Step 3: Apply correction rule
VBG → VB (base form)
"talking" → "talk"

Output: "I want to talk with you"

Confidence: 1.0 (deterministic rule)
```

#### 3.1.2 Category B: Local Context Needed

**Subject-Verb Agreement Rule:**

```
Rule Definition:
IF nsubj(verb, noun) AND number(noun) ≠ number(verb)
THEN flag_error(verb)

Where:
- nsubj = nominal subject dependency
- number ∈ {singular, plural}
```

**Complete Example with Parse Tree:**

```
Sentence: "The chickens crosses the road."

Step 1: Parse sentence
     crosses (ROOT)
     /     |      \
   DET   NSUBJ    DOBJ
   |      |        |
  The  chickens   road
        |          |
       NNS        NN
     (plural)  (singular)

Step 2: Find subject-verb relation
nsubj(crosses, chickens) = TRUE

Step 3: Extract number features
number(chickens) = plural (NNS tag)
number(crosses) = singular (VBZ tag)

Step 4: Check agreement
plural ≠ singular → ERROR

Step 5: Generate correction
crosses (VBZ) → cross (VBP)

Output: "The chickens cross the road."

Error span: [13, 20]
Correction: "cross"
Confidence: 0.98
```

### 3.2 Classification-Based Methods

#### 3.2.1 Feature Engineering

**Complete Feature Set for Verb Classification:**

| Feature Category | Feature Name | Example Value | Data Type |
|------------------|--------------|---------------|-----------|
| **Lexical** | Word itself | "were" | string |
| | Lemma | "be" | string |
| | Previous word | "They" | string |
| | Next word | "eating" | string |
| **Morphological** | POS tag | VBD | categorical |
| | Is auxiliary? | TRUE | boolean |
| | Is modal? | FALSE | boolean |
| **Syntactic** | Dependency relation | aux | categorical |
| | Has subject? | TRUE | boolean |
| | Subject number | plural | categorical |
| **N-gram** | Bigram | "They were" | string |
| | Trigram | "They were eating" | string |
| **Semantic** | Time expression nearby? | TRUE | boolean |
| | Tense marker | past | categorical |

**Feature Vector Example:**

```
Sentence: "They were eating ice cream"
Target word: "were" (index 1)

Feature Vector:
[
  word="were",
  lemma="be",
  pos="VBD",
  prev_word="They",
  next_word="eating",
  is_aux=True,
  is_first_aux=True,
  has_to=False,
  is_root=False,
  dep_rel="aux",
  bigram="They were",
  trigram="They were eating",
  subj_number="plural",
  time_expr=False
]

Target Class: VBD (correct - no error)
```

#### 3.2.2 Training Process

**Decision Tree Training Example:**

**Training Data:**

| Sentence | Word | Is Aux? | Infinitival to? | Lemma | Correct Form |
|----------|------|---------|-----------------|-------|--------------|
| "They are eating" | are | YES | NO | be | VBP ✓ |
| "I want to eat" | eat | NO | YES | eat | VB ✓ |
| "She working" | working | NO | NO | work | VBZ ✗ (should be "works" or "is working") |
| "He can eats" | eats | NO | NO | eat | VB ✗ |

**Decision Tree Construction:**

```
                   Is Auxiliary?
                   /           \
                 YES            NO
                 /               \
        Is First Aux?      Infinitival 'to'?
          /      \            /         \
        YES      NO         YES         NO
         |        |          |           |
    Check      Check       VB      Check Lemma
    Lemma      Context              /        \
     /  \                       'want'    Subject?
  'be' 'have'                    |          /    \
   |      |                     VBD       YES     NO
  VBP    VBP                              |       |
  VBZ    VBZ                           VBZ/VBP   VBG
  VBD    VBD
  VBN    VBN
```

**Classification Example:**

```
Input: "They were eaten ice cream"
Target: "eaten"

Step 1: Extract features
- is_aux = NO
- infinitival_to = NO
- lemma = "eat"
- has_subject = YES
- subject = "They"
- subject_number = plural

Step 2: Traverse decision tree
- Is auxiliary? NO
- Infinitival 'to'? NO
- Check lemma: "eat"
- Has subject? YES
- Subject number? plural
- Context: past continuous (were + ?)

Step 3: Decision
Required form: VBG (present participle)
Current form: VBN (past participle)
ERROR DETECTED

Step 4: Correction
"eaten" → "eating"

Confidence: 0.92 (based on training accuracy)
```

### 3.3 Language Model Approaches

#### 3.3.1 N-gram Language Models

**Model Definition:**

```
Unigram:  P(w₁, w₂, ..., wₙ) ≈ ∏ P(wᵢ)
Bigram:   P(w₁, w₂, ..., wₙ) ≈ ∏ P(wᵢ|wᵢ₋₁)
Trigram:  P(w₁, w₂, ..., wₙ) ≈ ∏ P(wᵢ|wᵢ₋₂, wᵢ₋₁)
```

**Complete N-gram Probability Table:**

Given corpus counts:

| N-gram | Count | Context Count | Probability |
|--------|-------|---------------|-------------|
| "the" | 1,000,000 | - | P("the") = 0.05 |
| "the cat" | 50,000 | 1,000,000 | P("cat"\|"the") = 0.05 |
| "the dog" | 80,000 | 1,000,000 | P("dog"\|"the") = 0.08 |
| "cat sat" | 30,000 | 100,000 | P("sat"\|"cat") = 0.30 |
| "the cat sat" | 25,000 | 50,000 | P("sat"\|"the cat") = 0.50 |

#### 3.3.2 Probability Calculations

**Example 1: Bigram Sentence Probability**

```
Sentence: "the cat sat"

Step 1: Break into bigrams
- <s> "the"
- "the" "cat"
- "cat" "sat"
- "sat" </s>

Step 2: Calculate individual probabilities
P("the" | <s>) = 0.15
P("cat" | "the") = 0.05
P("sat" | "cat") = 0.30
P(</s> | "sat") = 0.20

Step 3: Multiply probabilities
P(sentence) = 0.15 × 0.05 × 0.30 × 0.20
            = 0.00045
            = 4.5 × 10⁻⁴

Log probability (for numerical stability):
log P(sentence) = log(0.15) + log(0.05) + log(0.30) + log(0.20)
                = -0.82 + (-1.30) + (-0.52) + (-0.70)
                = -3.34
```

**Example 2: Preposition Selection Using Bigram LM**

```
Sentence: "I often work ___ home"
Candidates: {at, in, from, on, with}

Corpus Statistics:
| Bigram | Count | Total "work" | Probability |
|--------|-------|--------------|-------------|
| "work at" | 30,000 | 100,000 | 0.30 |
| "work in" | 5,000 | 100,000 | 0.05 |
| "work from" | 25,000 | 100,000 | 0.25 |
| "work on" | 1,000 | 100,000 | 0.01 |
| "work with" | 1,000 | 100,000 | 0.01 |

Forward probability: P(prep | "work")

| Trigram | Count | Total "work at" | Probability |
|---------|-------|-----------------|-------------|
| "work at home" | 28,000 | 30,000 | 0.933 |
| "work from home" | 24,000 | 25,000 | 0.960 |
| "work in home" | 100 | 5,000 | 0.020 |

Combined probability (bigram × trigram):
P("at") = 0.30 × 0.933 = 0.280
P("from") = 0.25 × 0.960 = 0.240
P("in") = 0.05 × 0.020 = 0.001

Decision: Select "at" (highest probability)

Final: "I often work at home"
```

### 3.4 Statistical Machine Translation (SMT)

#### 3.4.1 Phrase-Based SMT

**Translation Model Components:**

```
P(target | source) = P_TM(target | source) × P_LM(target)
                     \_________________/     \__________/
                     Translation Model       Language Model
```

**Phrase Table Example:**

| Source Phrase | Target Phrase | P(t\|s) | P(s\|t) | Lex(t\|s) | Lex(s\|t) |
|---------------|---------------|--------|--------|----------|----------|
| "discuss about" | "discuss" | 0.90 | 0.85 | 0.88 | 0.82 |
| "informations" | "information" | 0.95 | 0.92 | 0.94 | 0.91 |
| "as result" | "as a result" | 0.88 | 0.86 | 0.87 | 0.85 |
| "eated" | "ate" | 0.92 | 0.89 | 0.91 | 0.88 |

**Complete Translation Example:**

```
Source (Learner): "Let 's discuss about this informations ."

Step 1: Segment into phrases
- "Let 's"
- "discuss about"
- "this"
- "informations"
- "."

Step 2: Lookup in phrase table
"Let 's" → "Let 's" (P = 1.0)
"discuss about" → "discuss" (P = 0.90)
"this" → "this" (P = 1.0)
"informations" → "information" (P = 0.95)
"." → "." (P = 1.0)

Step 3: Combine translations
Hypothesis: "Let 's discuss this information ."

Step 4: Calculate translation probability
P_TM = 1.0 × 0.90 × 1.0 × 0.95 × 1.0 = 0.855

Step 5: Calculate language model score
P_LM("Let 's discuss this information .") = 0.0012
(using 5-gram LM)

Step 6: Combined score
Score = P_TM × P_LM = 0.855 × 0.0012 = 0.001026

Or in log space:
log(Score) = log(0.855) + log(0.0012)
           = -0.068 + (-2.88)
           = -2.95

Output: "Let's discuss this information."
Final Score: -2.95 (log probability)
```

#### 3.4.2 Noisy Channel Model

**Mathematical Framework:**

```
argmax P(correct | error) = argmax P(error | correct) × P(correct)
correct                      correct
                             \________________/   \__________/
                             Error Model         Language Model
```

**Complete Calculation Example:**

```
Input (Error): "I eated lunch yesterday"

Candidate Corrections:
1. "I ate lunch yesterday"
2. "I eat lunch yesterday"
3. "I eated lunch yesterday" (no change)

Step 1: Calculate P(error | candidate)

For "ate":
P("eated" | "ate") = 0.05 (common learner error)

For "eat":
P("eated" | "eat") = 0.08 (over-regularization)

For "eated":
P("eated" | "eated") = 1.0 (no change)

Step 2: Calculate P(candidate) using LM

P("I ate lunch yesterday") = 0.0003
P("I eat lunch yesterday") = 0.00001
P("I eated lunch yesterday") = 0.000001

Step 3: Calculate posterior probability

P("ate" | "eated") ∝ 0.05 × 0.0003 = 1.5 × 10⁻⁵
P("eat" | "eated") ∝ 0.08 × 0.00001 = 8.0 × 10⁻⁷
P("eated" | "eated") ∝ 1.0 × 0.000001 = 1.0 × 10⁻⁶

Step 4: Normalize probabilities

Total = 1.5×10⁻⁵ + 8.0×10⁻⁷ + 1.0×10⁻⁶ = 1.68×10⁻⁵

P("ate" | "eated") = 1.5×10⁻⁵ / 1.68×10⁻⁵ = 0.893
P("eat" | "eated") = 8.0×10⁻⁷ / 1.68×10⁻⁵ = 0.048
P("eated" | "eated") = 1.0×10⁻⁶ / 1.68×10⁻⁵ = 0.059

Decision: Select "ate" (highest probability)

Output: "I ate lunch yesterday"
Confidence: 0.893
```

---

## 4. Evaluation Metrics - Complete Calculations

### 4.1 Confusion Matrix Fundamentals

**Definitions:**

```
┌─────────────────────────────────────────────┐
│              CONFUSION MATRIX               │
├─────────────────┬───────────────────────────┤
│                 │   System Prediction       │
│                 ├─────────────┬─────────────┤
│                 │   Error     │   Correct   │
├─────────────────┼─────────────┼─────────────┤
│ Actual   Error  │  TP (Hit)   │  FN (Miss)  │
│         Correct │  FP (False) │  TN (OK)    │
└─────────────────┴─────────────┴─────────────┘
```

**Example Scenarios:**

| Sentence | Actual | System | Result |
|----------|--------|--------|--------|
| "I am going for walk" | ERROR (missing "a") | Flags error | TP ✓😊 |
| "I am going for a walk" | CORRECT | No flag | TN ✓😊 |
| "I am going for a walk" | CORRECT | Flags error | FP ✗😞 |
| "I am going for walk" | ERROR | No flag | FN ✗😞 |

### 4.2 Metric Formulas and Examples

**All Formulas:**

```
1. Precision (P) = TP / (TP + FP)
   "Of all errors flagged, how many were real?"

2. Recall (R) = TP / (TP + FN)
   "Of all real errors, how many did we catch?"

3. F-score (F₁) = 2 × (P × R) / (P + R)
   "Harmonic mean of precision and recall"

4. Accuracy (A) = (TP + TN) / (TP + TN + FP + FN)
   "Overall correct predictions"

5. False Positive Rate (FPR) = FP / (FP + TN)
   "Of all correct text, how much did we wrongly flag?"

6. False Negative Rate (FNR) = FN / (FN + TP)
   "Of all errors, how many did we miss?"
```

#### Example 1: Basic Metrics Calculation

**Given Data:**

```
Test Set: 500 sentences
- True Positives (TP) = 80 (correctly flagged errors)
- False Positives (FP) = 20 (incorrectly flagged)
- True Negatives (TN) = 350 (correctly identified as correct)
- False Negatives (FN) = 50 (missed errors)

Verify: TP + FP + TN + FN = 80 + 20 + 350 + 50 = 500 ✓
```

**Step-by-Step Calculations:**

```
Step 1: Calculate Precision
Precision = TP / (TP + FP)
         = 80 / (80 + 20)
         = 80 / 100
         = 0.80 or 80%

Interpretation: 80% of flagged errors were real errors

Step 2: Calculate Recall
Recall = TP / (TP + FN)
       = 80 / (80 + 50)
       = 80 / 130
       = 0.6154 or 61.54%

Interpretation: Caught 61.54% of all actual errors

Step 3: Calculate F-score
F₁ = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.80 × 0.6154) / (0.80 + 0.6154)
   = 2 × 0.4923 / 1.4154
   = 0.9846 / 1.4154
   = 0.6956 or 69.56%

Step 4: Calculate Accuracy
Accuracy = (TP + TN) / Total
         = (80 + 350) / 500
         = 430 / 500
         = 0.86 or 86%

Step 5: Calculate FPR
FPR = FP / (FP + TN)
    = 20 / (20 + 350)
    = 20 / 370
    = 0.0541 or 5.41%

Interpretation: 5.41% of correct text was wrongly flagged

Step 6: Calculate FNR
FNR = FN / (FN + TP)
    = 50 / (50 + 80)
    = 50 / 130
    = 0.3846 or 38.46%

Interpretation: Missed 38.46% of actual errors
```

**Summary Table:**

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| Precision | TP/(TP+FP) | 80% | High reliability when flagging |
| Recall | TP/(TP+FN) | 61.54% | Moderate coverage |
| F-score | 2PR/(P+R) | 69.56% | Balanced performance |
| Accuracy | (TP+TN)/Total | 86% | Good overall |
| FPR | FP/(FP+TN) | 5.41% | Low false alarms |
| FNR | FN/(FN+TP) | 38.46% | Significant misses |

#### Example 2: Comprehensive Evaluation

**Scenario: Article Error Detection System**

Test corpus: 1000 sentences with 200 article errors

```
Results:
- Correctly detected article errors: 140 (TP)
- Incorrectly flagged as article errors: 30 (FP)
- Correctly identified as having no article error: 760 (TN)
- Missed article errors: 60 (FN)
- Unrelated errors (ignored): 10

Total relevant instances = 140 + 30 + 760 + 60 = 990
```

**Detailed Calculations:**

```
Precision Calculation:
P = TP / (TP + FP)
  = 140 / (140 + 30)
  = 140 / 170
  = 0.8235

Converting to percentage: 82.35%

Precision Interpretation:
"When the system says there's an article error,
it's correct 82.35% of the time"

Recall Calculation:
R = TP / (TP + FN)
  = 140 / (140 + 60)
  = 140 / 200
  = 0.70

Converting to percentage: 70%

Recall Interpretation:
"The system catches 70% of all article errors"

F-score Calculation:
F₁ = 2 × (P × R) / (P + R)
   = 2 × (0.8235 × 0.70) / (0.8235 + 0.70)
   = 2 × 0.5765 / 1.5235
   = 1.1529 / 1.5235
   = 0.7568

F₁ = 75.68%

Accuracy Calculation:
Accuracy = (TP + TN) / Total
         = (140 + 760) / 990
         = 900 / 990
         = 0.9091

Accuracy = 90.91%

Error Analysis:
False Positive Rate = 30 / (30 + 760) = 30/790 = 3.80%
False Negative Rate = 60 / (60 + 140) = 60/200 = 30%

Miss Rate = 1 - Recall = 1 - 0.70 = 0.30 = 30%
```

**Performance Assessment:**

| Aspect | Score | Evaluation |
|--------|-------|------------|
| Precision | 82.35% | Good - Low false alarms |
| Recall | 70% | Moderate - Missing 30% of errors |
| F-score | 75.68% | Acceptable balance |
| Accuracy | 90.91% | Very good overall |
| **Recommendation** | - | Improve recall (reduce FN) |

#### Example 3: Comparing Two Systems

**System A (Rule-Based):**
- TP = 100, FP = 10, TN = 400, FN = 90

**System B (ML-Based):**
- TP = 150, FP = 50, TN = 360, FN = 40

**Comparative Analysis:**

```
System A Metrics:

Precision_A = 100 / (100 + 10) = 100/110 = 0.9091 = 90.91%
Recall_A = 100 / (100 + 90) = 100/190 = 0.5263 = 52.63%
F₁_A = 2 × (0.9091 × 0.5263) / (0.9091 + 0.5263)
     = 2 × 0.4784 / 1.4354
     = 0.9568 / 1.4354
     = 0.6667 = 66.67%
Accuracy_A = (100 + 400) / 600 = 500/600 = 0.8333 = 83.33%

System B Metrics:

Precision_B = 150 / (150 + 50) = 150/200 = 0.75 = 75%
Recall_B = 150 / (150 + 40) = 150/190 = 0.7895 = 78.95%
F₁_B = 2 × (0.75 × 0.7895) / (0.75 + 0.7895)
     = 2 × 0.5921 / 1.5395
     = 1.1842 / 1.5395
     = 0.7692 = 76.92%
Accuracy_B = (150 + 360) / 600 = 510/600 = 0.85 = 85%
```

**Comparison Table:**

| Metric | System A (Rule) | System B (ML) | Winner |
|--------|-----------------|---------------|---------|
| Precision | 90.91% | 75% | System A |
| Recall | 52.63% | 78.95% | System B |
| F-score | 66.67% | 76.92% | System B |
| Accuracy | 83.33% | 85% | System B |
| False Positives | 10 | 50 | System A |
| False Negatives | 90 | 40 | System B |

**Trade-off Analysis:**

```
System A:
+ Very high precision (90.91%)
+ Very few false positives (10)
- Low recall (52.63%)
- Misses many errors (90 FN)
→ Conservative system, good for high-confidence corrections

System B:
+ High recall (78.95%)
+ Catches most errors (150/190)
+ Better F-score (76.92%)
- Lower precision (75%)
- More false positives (50)
→ Aggressive system, better error coverage

Recommendation:
- Use System A when false positives are costly
- Use System B when missing errors is worse
- Consider ensemble: System A for high-confidence,
  System B for comprehensive checking
```

---

## 5. Language Model Probability Calculations

### 5.1 Unigram Model

**Model:**
```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂) × ... × P(wₙ)
```

**Example Corpus Statistics:**

| Word | Count | Total Words | P(word) |
|------|-------|-------------|---------|
| the | 500,000 | 10,000,000 | 0.05 |
| cat | 10,000 | 10,000,000 | 0.001 |
| sat | 8,000 | 10,000,000 | 0.0008 |
| on | 300,000 | 10,000,000 | 0.03 |
| mat | 2,000 | 10,000,000 | 0.0002 |

**Calculation:**

```
Sentence: "the cat sat on the mat"

P(sentence) = P(the) × P(cat) × P(sat) × P(on) × P(the) × P(mat)
            = 0.05 × 0.001 × 0.0008 × 0.03 × 0.05 × 0.0002
            = 3 × 10⁻¹⁴

Log probability (better for computation):
log P(sentence) = log(0.05) + log(0.001) + log(0.0008) +
                  log(0.03) + log(0.05) + log(0.0002)
                = -1.301 + (-3.000) + (-3.097) +
                  (-1.523) + (-1.301) + (-3.699)
                = -13.921

Perplexity = exp(-log P / N)
           = exp(-(-13.921) / 6)
           = exp(2.320)
           = 10.18
```

### 5.2 Bigram Model

**Model:**
```
P(w₁, w₂, ..., wₙ) = ∏ P(wᵢ | wᵢ₋₁)
                     i=1
```

**Bigram Probability Table:**

| Bigram | Count | C(w₁) | P(w₂\|w₁) |
|--------|-------|-------|-----------|
| <s> the | 75,000 | 100,000 | 0.75 |
| the cat | 8,000 | 500,000 | 0.016 |
| cat sat | 5,000 | 10,000 | 0.50 |
| sat on | 6,000 | 8,000 | 0.75 |
| on the | 150,000 | 300,000 | 0.50 |
| the mat | 1,500 | 500,000 | 0.003 |
| mat </s> | 1,800 | 2,000 | 0.90 |

**Calculation:**

```
Sentence: "the cat sat on the mat"

Step 1: Break into bigrams with boundaries
<s> "the"
"the" "cat"
"cat" "sat"
"sat" "on"
"on" "the"
"the" "mat"
"mat" </s>

Step 2: Calculate each bigram probability
P("the" | <s>) = 0.75
P("cat" | "the") = 0.016
P("sat" | "cat") = 0.50
P("on" | "sat") = 0.75
P("the" | "on") = 0.50
P("mat" | "the") = 0.003
P(</s> | "mat") = 0.90

Step 3: Multiply probabilities
P(sentence) = 0.75 × 0.016 × 0.50 × 0.75 × 0.50 × 0.003 × 0.90
            = 6.075 × 10⁻⁶

Step 4: Log probability
log P(sentence) = log(0.75) + log(0.016) + log(0.50) + log(0.75) +
                  log(0.50) + log(0.003) + log(0.90)
                = -0.125 + (-1.796) + (-0.301) + (-0.125) +
                  (-0.301) + (-2.523) + (-0.046)
                = -5.217

Better than unigram: -5.217 > -13.921 ✓
```

### 5.3 Trigram Model

**Model:**
```
P(w₁, w₂, ..., wₙ) = ∏ P(wᵢ | wᵢ₋₂, wᵢ₋₁)
                     i=1
```

**Trigram Example:**

| Trigram | Count | C(w₁,w₂) | P(w₃\|w₁,w₂) |
|---------|-------|----------|--------------|
| <s> <s> the | 70,000 | 100,000 | 0.70 |
| <s> the cat | 6,000 | 75,000 | 0.08 |
| the cat sat | 4,500 | 8,000 | 0.5625 |
| cat sat on | 4,000 | 5,000 | 0.80 |
| sat on the | 5,500 | 6,000 | 0.9167 |
| on the mat | 1,400 | 150,000 | 0.0093 |
| the mat </s> | 1,300 | 1,500 | 0.8667 |

**Calculation:**

```
Sentence: "the cat sat on the mat"

P(sentence) = P(the|<s>,<s>) × P(cat|<s>,the) × P(sat|the,cat) ×
              P(on|cat,sat) × P(the|sat,on) × P(mat|on,the) ×
              P(</s>|the,mat)
            = 0.70 × 0.08 × 0.5625 × 0.80 × 0.9167 × 0.0093 × 0.8667
            = 1.697 × 10⁻⁵

log P(sentence) = -4.770

Comparison:
Unigram: -13.921
Bigram:  -5.217
Trigram: -4.770 (BEST)
```

### 5.4 Preposition Selection Using LM

**Problem:** Choose correct preposition

```
Sentence: "I often work ___ home"
Candidates: {at, in, from, on, to, with}
```

**Step-by-Step Solution:**

```
Step 1: Build context windows

Left bigram:  "work ___"
Right bigram: "___ home"
Trigram:      "work ___ home"

Step 2: Extract corpus statistics

Left context probabilities P(prep | "work"):
| Bigram | Count | P(prep\|work) |
|--------|-------|---------------|
| work at | 30,000 | 0.30 |
| work in | 5,000 | 0.05 |
| work from | 25,000 | 0.25 |
| work on | 20,000 | 0.20 |
| work to | 15,000 | 0.15 |
| work with | 5,000 | 0.05 |

Right context probabilities P("home" | prep):
| Bigram | Count | P(home\|prep) |
|--------|-------|----------------|
| at home | 450,000 | 0.90 |
| in home | 5,000 | 0.01 |
| from home | 480,000 | 0.96 |
| on home | 1,000 | 0.002 |
| to home | 100,000 | 0.20 |
| with home | 500 | 0.001 |

Step 3: Calculate trigram probabilities

Trigram counts:
| Trigram | Count | Total | P(home\|work,prep) |
|---------|-------|-------|--------------------|
| work at home | 28,000 | 30,000 | 0.9333 |
| work in home | 100 | 5,000 | 0.02 |
| work from home | 24,000 | 25,000 | 0.96 |
| work on home | 50 | 20,000 | 0.0025 |
| work to home | 500 | 15,000 | 0.0333 |
| work with home | 10 | 5,000 | 0.002 |

Step 4: Combine scores (product of probabilities)

Score(at) = P(at|work) × P(home|work,at)
          = 0.30 × 0.9333
          = 0.280

Score(in) = 0.05 × 0.02 = 0.001
Score(from) = 0.25 × 0.96 = 0.240
Score(on) = 0.20 × 0.0025 = 0.0005
Score(to) = 0.15 × 0.0333 = 0.005
Score(with) = 0.05 × 0.002 = 0.0001

Step 5: Rank candidates

1. at:    0.280 ← WINNER
2. from:  0.240
3. to:    0.005
4. in:    0.001
5. on:    0.0005
6. with:  0.0001

Decision: Select "at"

Output: "I often work at home"
Confidence: 0.280 / (0.280 + 0.240 + ... ) = 0.53

Alternative acceptable: "from" (Score = 0.240)
Both "work at home" and "work from home" are valid,
but "at" is more probable in this corpus.
```

---

## 6. Classification Model Examples

### 6.1 Verb Form Classification

**Problem:** Classify verb form in context

**Training Data Example:**

| Sentence | Target Verb | Features | Correct Form |
|----------|-------------|----------|--------------|
| "They are eating" | eating | aux=are, lemma=eat, subj=they | VBG ✓ |
| "She eats apples" | eats | no_aux, lemma=eat, subj=she | VBZ ✓ |
| "I want to eat" | eat | to=true, lemma=eat | VB ✓ |
| "Food was eaten" | eaten | aux=was, passive=true | VBN ✓ |

**Decision Tree Model:**

```
            Has auxiliary?
           /              \
         YES               NO
         /                  \
   Is passive?        Infinitival to?
    /      \            /          \
  YES      NO         YES          NO
   |        |          |            |
  VBN   Aux type?     VB      Has subject?
        /      \                 /        \
     'be'    'have'           YES         NO
      /\       /\              |           |
    VBG VBN  VBN VBG    Subj number?     VBG
                         /        \
                    singular    plural
                       |           |
                      VBZ         VBP
```

### 6.2 Feature Extraction Example

**Sentence:** "They were eating ice cream when I arrived"

**Target Word:** "eating" (position 2)

**Complete Feature Vector:**

```python
features = {
    # Lexical features
    'word': 'eating',
    'lemma': 'eat',
    'pos': 'VBG',
    'word_prev': 'were',
    'word_next': 'ice',
    'word_prev2': 'They',
    'word_next2': 'cream',

    # Morphological features
    'has_ing': True,
    'has_ed': False,
    'has_s': False,
    'is_irregular': False,

    # Syntactic features
    'has_aux': True,
    'aux_type': 'be',
    'aux_tense': 'past',
    'is_passive': False,
    'has_infinitival_to': False,
    'dep_rel': 'ROOT',

    # Subject features
    'has_subject': True,
    'subject': 'They',
    'subject_pos': 'PRP',
    'subject_number': 'plural',
    'subject_person': '3rd',

    # Context features
    'has_time_expression': True,
    'time_words': ['when'],
    'clause_type': 'main',

    # N-gram features
    'bigram_prev': 'were eating',
    'bigram_next': 'eating ice',
    'trigram': 'were eating ice',

    # Discourse features
    'sentence_position': 'middle',
    'is_narrative': True,
}

# Target
target_class = 'VBG'  # Correct form
```

**Classification Process:**

```
Step 1: Feature extraction (above)

Step 2: Apply decision tree
- Has auxiliary? YES
- Is passive? NO
- Aux type? 'be'
- Aux tense? past
- Current form? VBG

Step 3: Check compatibility
- Past progressive = BE_PAST + VBG ✓
- Current: VBG ✓
- Match: YES

Step 4: Prediction
Predicted class: VBG
Actual class: VBG
Result: CORRECT (no error)

Confidence: 0.98 (based on training data)
```

### 6.3 Decision Tree Construction

**Training Data (simplified):**

| Has Aux | Aux Type | Infinitival To | Subject | Correct Form | Count |
|---------|----------|----------------|---------|--------------|-------|
| YES | be | NO | plural | VBG | 450 |
| YES | be | NO | singular | VBG | 380 |
| YES | have | NO | - | VBN | 290 |
| NO | - | YES | - | VB | 520 |
| NO | - | NO | singular | VBZ | 340 |
| NO | - | NO | plural | VBP | 310 |

**Information Gain Calculation:**

```
Total instances: 2290

Step 1: Calculate entropy of target variable

Class distribution:
VBG: 830 / 2290 = 0.362
VBN: 290 / 2290 = 0.127
VB:  520 / 2290 = 0.227
VBZ: 340 / 2290 = 0.148
VBP: 310 / 2290 = 0.135

H(Target) = -∑ p(c) × log₂(p(c))
          = -(0.362×log₂(0.362) + 0.127×log₂(0.127) +
              0.227×log₂(0.227) + 0.148×log₂(0.148) +
              0.135×log₂(0.135))
          = -(0.362×(-1.466) + 0.127×(-2.977) +
              0.227×(-2.139) + 0.148×(-2.755) +
              0.135×(-2.889))
          = -(-0.531 + (-0.378) + (-0.486) + (-0.408) + (-0.390))
          = 2.193 bits

Step 2: Calculate information gain for "Has Auxiliary"

Split on Has_Aux:
YES: 1120 instances (VBG: 830, VBN: 290)
NO:  1170 instances (VB: 520, VBZ: 340, VBP: 310)

H(Has_Aux=YES) = -(830/1120 × log₂(830/1120) +
                   290/1120 × log₂(290/1120))
               = -(0.741×(-0.432) + 0.259×(-1.948))
               = -(-0.320 + (-0.504))
               = 0.824 bits

H(Has_Aux=NO) = -(520/1170 × log₂(520/1170) +
                  340/1170 × log₂(340/1170) +
                  310/1170 × log₂(310/1170))
              = -(0.444×(-1.171) + 0.291×(-1.782) +
                  0.265×(-1.916))
              = -(-0.520 + (-0.519) + (-0.508))
              = 1.547 bits

Weighted average:
H(Target | Has_Aux) = (1120/2290 × 0.824) + (1170/2290 × 1.547)
                    = 0.403 + 0.791
                    = 1.194 bits

Information Gain:
IG(Has_Aux) = H(Target) - H(Target | Has_Aux)
            = 2.193 - 1.194
            = 0.999 bits

This is the highest information gain, so "Has_Aux"
becomes the root node.
```

---

## 7. Complete Worked Examples

### Example 1: Subject-Verb Agreement Detection

**Problem:**
```
Input: "The students in the classroom is ready for the exam."
Task: Detect and correct the error
```

**Complete Solution:**

```
Step 1: Tokenization and POS Tagging

Token    | POS  | Index
---------|------|------
The      | DET  | 0
students | NNS  | 1 ← Subject (plural)
in       | IN   | 2
the      | DET  | 3
classroom| NN   | 4
is       | VBZ  | 5 ← Verb (singular) ERROR!
ready    | JJ   | 6
for      | IN   | 7
the      | DET  | 8
exam     | NN   | 9
.        | .    | 10

Step 2: Dependency Parsing

         is (ROOT, VBZ)
        /  |  \
    students  ready  .
    (nsubj)  (acomp)
    /   |
  The   in
       (det) (prep)
             |
          classroom
          (pobj)
           |
          the
         (det)

Step 3: Extract subject-verb relation

nsubj(is, students) ✓
Subject: "students" (index 1)
Verb: "is" (index 5)

Step 4: Check number agreement

Subject features:
- POS: NNS (plural noun)
- Number: plural
- Lemma: "student"

Verb features:
- POS: VBZ (3rd person singular)
- Number: singular
- Lemma: "be"

Agreement check:
plural (subject) ≠ singular (verb) → ERROR

Step 5: Generate correction

Verb "is" should agree with plural subject
VBZ → VBP
"is" → "are"

Step 6: Apply correction

Original: "The students in the classroom is ready for the exam."
Corrected: "The students in the classroom are ready for the exam."

Metadata:
- Error type: Subject-verb agreement
- Error span: [35, 37] (character positions)
- Confidence: 0.99 (rule-based, high confidence)
- Suggestion: "are"
```

### Example 2: Article Error Detection

**Problem:**
```
Input: "I bought shirt at store yesterday."
Task: Detect missing articles
```

**Complete Solution:**

```
Step 1: Parse sentence

Token    | POS | Countable | Definiteness | Article
---------|-----|-----------|--------------|--------
I        | PRP | N/A       | N/A          | -
bought   | VBD | N/A       | N/A          | -
shirt    | NN  | YES       | Indefinite   | MISSING!
at       | IN  | N/A       | N/A          | -
store    | NN  | YES       | Definite     | MISSING!
yesterday| NN  | N/A       | N/A          | -
.        | .   | N/A       | N/A          | -

Step 2: Check "shirt"

Is noun? YES (NN)
Is countable? YES (shirt/shirts exist)
Is singular? YES (NN not NNS)
Has determiner? NO ← ERROR
Is definite? NO (first mention)
Is specific? NO (indefinite)

Required article: "a" or "an"
Phonetic check: "shirt" starts with /ʃ/ (consonant)
Decision: "a shirt"

Step 3: Check "store"

Is noun? YES (NN)
Is countable? YES
Is singular? YES
Has determiner? NO ← ERROR
Is definite? YES (specific store contextually)
   (the speaker knows which store)

Required article: "the"
Decision: "the store"

Step 4: Generate corrections

Error 1:
- Position: after "bought"
- Type: Missing article
- Suggestion: Insert "a" before "shirt"

Error 2:
- Position: after "at"
- Type: Missing article
- Suggestion: Insert "the" before "store"

Step 5: Apply all corrections

Original: "I bought shirt at store yesterday."
Pass 1:   "I bought a shirt at store yesterday."
Pass 2:   "I bought a shirt at the store yesterday."

Final output with confidence scores:
- "a" before "shirt": confidence 0.92
- "the" before "store": confidence 0.85
  (could also be "a store" - 0.15 probability)
```

### Example 3: Preposition Selection

**Problem:**
```
Input: "The meeting is scheduled ___ Monday ___ 3 PM."
Options for blank 1: {on, in, at, for}
Options for blank 2: {on, in, at, by}
```

**Complete Solution:**

```
Step 1: Analyze first blank

Context: "scheduled ___ Monday"
Target: Day of week

Rule-based approach:
Rule: TIME_DAY_OF_WEEK → "on"
Examples: "on Monday", "on Tuesday", etc.
Confidence: 0.95 (strong rule)

Corpus verification:
| Phrase | Count | Probability |
|--------|-------|-------------|
| "on Monday" | 1,200,000 | 0.92 |
| "in Monday" | 5,000 | 0.004 |
| "at Monday" | 2,000 | 0.0015 |
| "for Monday" | 50,000 | 0.038 |

Decision: "on" (rule + corpus agreement)

Step 2: Analyze second blank

Context: "___ 3 PM"
Target: Specific clock time

Rule-based approach:
Rule: TIME_CLOCK → "at"
Examples: "at 3 PM", "at noon", etc.
Confidence: 0.98 (very strong rule)

Corpus verification:
| Phrase | Count | Probability |
|--------|-------|-------------|
| "at 3 PM" | 850,000 | 0.89 |
| "on 3 PM" | 1,000 | 0.001 |
| "in 3 PM" | 500 | 0.0005 |
| "by 3 PM" | 100,000 | 0.10 |

Note: "by 3 PM" means "before/no later than 3 PM"
      (different meaning)

Decision: "at" (precise time reference)

Step 3: Verify full sentence

Candidate: "The meeting is scheduled on Monday at 3 PM."

Language model check:
P("scheduled on Monday at") = 0.00234
P("scheduled in Monday at") = 0.000001
P("scheduled on Monday in") = 0.000008

Best combination: "on Monday at 3 PM"

Step 4: Final answer

Blank 1: "on" (confidence: 0.95)
Blank 2: "at" (confidence: 0.98)

Complete sentence:
"The meeting is scheduled on Monday at 3 PM."

Alternative valid forms:
- "The meeting is scheduled for Monday at 3 PM." (0.05 prob)
  (emphasis on designated time)
```

### Example 4: Verb Tense Correction

**Problem:**
```
Input: "Yesterday, I go to the store and buy some milk."
Task: Correct verb tenses
```

**Complete Solution:**

```
Step 1: Identify temporal context

Time adverbial: "Yesterday"
Temporal reference: Past
Expected tense: Past simple

Step 2: Analyze each verb

Verb 1: "go"
- Position: 2
- Current form: VBP (present, non-3rd person)
- Lemma: "go"
- Context: "Yesterday, I go"
- Time marker: "Yesterday" → past
- Expected: VBD (past tense)
- Correction: "go" → "went"

Verb 2: "buy"
- Position: 7
- Current form: VBP (present)
- Lemma: "buy"
- Context: coordinated with "go"
- Tense consistency: should match first verb
- Expected: VBD
- Correction: "buy" → "bought"

Step 3: Check narrative consistency

Sentence structure: [Time] [Action 1] and [Action 2]
Both actions happened at same time (yesterday)
→ Both should use past tense

Tense sequence check:
"Yesterday" + VBD + "and" + VBD ✓

Step 4: Apply corrections

Original: "Yesterday, I go to the store and buy some milk."

After verb 1 correction:
"Yesterday, I went to the store and buy some milk."

After verb 2 correction:
"Yesterday, I went to the store and bought some milk."

Step 5: Verification using language model

P("Yesterday, I went to the store and bought some milk.")
= 0.00045

P("Yesterday, I go to the store and buy some milk.")
= 0.0000012

Probability ratio: 0.00045 / 0.0000012 = 375

The corrected version is 375× more probable.

Final output:
Original: "Yesterday, I go to the store and buy some milk."
Corrected: "Yesterday, I went to the store and bought some milk."

Error 1: "go" → "went" (confidence: 0.99)
Error 2: "buy" → "bought" (confidence: 0.99)
```

### Example 5: Language Model Scoring

**Problem:**
```
Compare three correction candidates using trigram LM:

Original: "I eated lunch"

Candidates:
A: "I ate lunch"
B: "I eat lunch"
C: "I eaten lunch"
```

**Complete Calculation:**

```
Trigram counts from corpus (millions of words):

| Trigram | Count | Context Count | P(w₃|w₁,w₂) |
|---------|-------|---------------|-------------|
| <s> <s> I | 500,000 | 1,000,000 | 0.50 |
| <s> I ate | 80,000 | 500,000 | 0.16 |
| <s> I eat | 5,000 | 500,000 | 0.01 |
| <s> I eaten | 100 | 500,000 | 0.0002 |
| I ate lunch | 60,000 | 80,000 | 0.75 |
| I eat lunch | 3,000 | 5,000 | 0.60 |
| I eaten lunch | 10 | 100 | 0.10 |
| ate lunch </s> | 58,000 | 70,000 | 0.829 |
| eat lunch </s> | 2,500 | 4,000 | 0.625 |
| eaten lunch </s> | 5 | 15 | 0.333 |

Candidate A: "I ate lunch"

P(A) = P(I|<s>,<s>) × P(ate|<s>,I) × P(lunch|I,ate) × P(</s>|ate,lunch)
     = 0.50 × 0.16 × 0.75 × 0.829
     = 0.0497

log P(A) = log(0.50) + log(0.16) + log(0.75) + log(0.829)
         = -0.301 + (-0.796) + (-0.125) + (-0.081)
         = -1.303

Candidate B: "I eat lunch"

P(B) = P(I|<s>,<s>) × P(eat|<s>,I) × P(lunch|I,eat) × P(</s>|eat,lunch)
     = 0.50 × 0.01 × 0.60 × 0.625
     = 0.001875

log P(B) = log(0.50) + log(0.01) + log(0.60) + log(0.625)
         = -0.301 + (-2.000) + (-0.222) + (-0.204)
         = -2.727

Candidate C: "I eaten lunch"

P(C) = P(I|<s>,<s>) × P(eaten|<s>,I) × P(lunch|I,eaten) × P(</s>|eaten,lunch)
     = 0.50 × 0.0002 × 0.10 × 0.333
     = 0.00000333

log P(C) = log(0.50) + log(0.0002) + log(0.10) + log(0.333)
         = -0.301 + (-3.699) + (-1.000) + (-0.477)
         = -5.477

Ranking (higher is better):

1. Candidate A: log P = -1.303 ✓✓✓ BEST
2. Candidate B: log P = -2.727
3. Candidate C: log P = -5.477 WORST

Decision: Select Candidate A ("I ate lunch")

Probability ratios:
P(A) / P(B) = 0.0497 / 0.001875 = 26.5
→ A is 26.5× more probable than B

P(A) / P(C) = 0.0497 / 0.00000333 = 14,925
→ A is 14,925× more probable than C

Final answer: "I ate lunch"
Confidence: Very high (P(A) >> P(B) >> P(C))
```

---

## 8. Statistical Machine Translation Calculations

### 8.1 Translation Probability

**Problem:** Calculate translation score for error correction

```
Source (Error): "I want to discuss about this problem"
Target (Correct): "I want to discuss this problem"
```

**Phrase Table:**

| Source Phrase | Target Phrase | P(t\|s) | P(s\|t) |
|---------------|---------------|---------|---------|
| I | I | 1.0 | 1.0 |
| want to | want to | 1.0 | 1.0 |
| discuss about | discuss | 0.92 | 0.88 |
| this | this | 1.0 | 1.0 |
| problem | problem | 1.0 | 1.0 |

**Language Model (5-gram):**

| 5-gram | Log Probability |
|--------|----------------|
| <s> I want to discuss | -1.2 |
| I want to discuss this | -0.8 |
| want to discuss this problem | -1.1 |
| to discuss this problem </s> | -0.9 |

**Complete Calculation:**

```
Step 1: Calculate Translation Model Score

P_TM(target | source) = ∏ P(tᵢ | sᵢ)
                       = P(I|I) × P(want to|want to) ×
                         P(discuss|discuss about) ×
                         P(this|this) × P(problem|problem)
                       = 1.0 × 1.0 × 0.92 × 1.0 × 1.0
                       = 0.92

log P_TM = log(1.0) + log(1.0) + log(0.92) + log(1.0) + log(1.0)
         = 0 + 0 + (-0.036) + 0 + 0
         = -0.036

Step 2: Calculate Language Model Score

P_LM(target) = P_5gram(sentence)

Using backoff and interpolation:
log P_LM = -1.2 + (-0.8) + (-1.1) + (-0.9)
         = -4.0

Step 3: Combine Scores

Combined log score = log P_TM + λ × log P_LM
(where λ = language model weight, typically 0.5-1.5)

With λ = 1.0:
log P(target | source) = -0.036 + 1.0 × (-4.0)
                       = -4.036

Step 4: Compare with Alternative

Alternative: Keep "discuss about"
P_TM_alt = 1.0 (no change)
P_LM_alt = very low (ungrammatical)
log P_LM_alt ≈ -8.5

Combined_alt = 0 + 1.0 × (-8.5) = -8.5

Step 5: Decision

Original (with error): score = -8.5
Correction: score = -4.036

Improvement: -4.036 - (-8.5) = +4.464
→ Corrected version is e^4.464 ≈ 87× more probable

Decision: Apply correction
Output: "I want to discuss this problem"
```

### 8.2 Alignment Calculation

**Problem:** Word alignment for phrase extraction

```
Source: "I eated lunch yesterday"
Target: "I ate lunch yesterday"
```

**Step-by-Step Alignment:**

```
Step 1: Initialize alignment probabilities

From IBM Model 1 (learned from parallel corpus):

P(ate | eated) = 0.85
P(eat | eated) = 0.10
P(eaten | eated) = 0.03
P(I | I) = 0.95
P(lunch | lunch) = 0.98
P(yesterday | yesterday) = 0.97

Step 2: Find best alignment using Viterbi

Source positions: 0   1      2     3
Source words:     I   eated  lunch yesterday

Target positions: 0   1    2     3
Target words:     I   ate  lunch yesterday

Alignment matrix (P(target|source)):

              I     ate    lunch  yesterday
I        │  0.95   0.01   0.01    0.01
eated    │  0.02   0.85   0.05    0.02
lunch    │  0.01   0.02   0.98    0.03
yesterday│  0.01   0.01   0.02    0.97

Step 3: Extract alignments (argmax for each source word)

I → I (0.95)
eated → ate (0.85)
lunch → lunch (0.98)
yesterday → yesterday (0.97)

Alignment: {(0,0), (1,1), (2,2), (3,3)}

Step 4: Extract phrase pairs

From alignment:
Phrase pair: ("eated", "ate")
Phrase translation probability: 0.85

This is added to phrase table for future corrections.

Step 5: Calculate sentence translation probability

P(translation | source) = ∏ P(targetᵢ | sourceᵢ)
                         = 0.95 × 0.85 × 0.98 × 0.97
                         = 0.770

Alignment confidence: 77%
```

---

## 9. Advanced Topics

### 9.1 Deep Learning Model Evaluation

**Transformer-Based GEC Model Evaluation**

**Test Set:** 1000 sentences with errors

**Model Output Analysis:**

| Metric | Value | Calculation |
|--------|-------|-------------|
| True Positives | 680 | Correct corrections |
| False Positives | 85 | Wrong corrections |
| True Negatives | 180 | Correct non-corrections |
| False Negatives | 55 | Missed errors |

**Calculation:**

```
Total predictions = 680 + 85 + 180 + 55 = 1000 ✓

Precision = TP / (TP + FP)
         = 680 / (680 + 85)
         = 680 / 765
         = 0.8889 = 88.89%

Recall = TP / (TP + FN)
       = 680 / (680 + 55)
       = 680 / 735
       = 0.9252 = 92.52%

F₀.₅ (precision-weighted) = (1 + 0.5²) × P × R / (0.5² × P + R)
                          = 1.25 × 0.8889 × 0.9252 / (0.25 × 0.8889 + 0.9252)
                          = 1.0283 / 1.1475
                          = 0.8961 = 89.61%

F₁ = 2 × P × R / (P + R)
   = 2 × 0.8889 × 0.9252 / (0.8889 + 0.9252)
   = 1.6452 / 1.8141
   = 0.9069 = 90.69%

F₂ (recall-weighted) = (1 + 2²) × P × R / (2² × P + R)
                     = 5 × 0.8889 × 0.9252 / (4 × 0.8889 + 0.9252)
                     = 4.1144 / 4.4808
                     = 0.9182 = 91.82%

GLEU (Generalized Language Evaluation Understanding):
Requires n-gram overlap calculation with reference corrections.
```

### 9.2 Multi-Class Classification Metrics

**Problem:** Evaluate article classifier (3 classes: a, the, ∅)

**Confusion Matrix:**

```
                 Predicted
              a     the     ∅    Total
Actual  a   │ 350    20    30  │ 400
        the │  15   420    15  │ 450
        ∅   │  35    10   105  │ 150
        ────┼─────────────────┼──────
        Total 400   450   150    1000
```

**Per-Class Metrics:**

```
Class "a":
TP_a = 350
FP_a = 15 + 35 = 50
FN_a = 20 + 30 = 50
TN_a = 420 + 15 + 10 + 105 = 550

Precision_a = 350 / (350 + 50) = 350/400 = 0.875 = 87.5%
Recall_a = 350 / (350 + 50) = 350/400 = 0.875 = 87.5%
F1_a = 2 × 0.875 × 0.875 / (0.875 + 0.875) = 0.875 = 87.5%

Class "the":
TP_the = 420
FP_the = 20 + 10 = 30
FN_the = 15 + 15 = 30
TN_the = 350 + 30 + 35 + 105 = 520

Precision_the = 420 / (420 + 30) = 420/450 = 0.933 = 93.3%
Recall_the = 420 / (420 + 30) = 420/450 = 0.933 = 93.3%
F1_the = 0.933 = 93.3%

Class "∅" (no article):
TP_∅ = 105
FP_∅ = 30 + 15 = 45
FN_∅ = 35 + 10 = 45
TN_∅ = 350 + 20 + 15 + 420 = 805

Precision_∅ = 105 / (105 + 45) = 105/150 = 0.70 = 70%
Recall_∅ = 105 / (105 + 45) = 105/150 = 0.70 = 70%
F1_∅ = 0.70 = 70%

Macro-Average:
Precision_macro = (0.875 + 0.933 + 0.70) / 3 = 0.836 = 83.6%
Recall_macro = (0.875 + 0.933 + 0.70) / 3 = 0.836 = 83.6%
F1_macro = 0.836 = 83.6%

Micro-Average (weighted by support):
TP_total = 350 + 420 + 105 = 875
Total_instances = 1000

Accuracy_micro = 875 / 1000 = 0.875 = 87.5%

Weighted Average (by class frequency):
F1_weighted = (400/1000 × 0.875) + (450/1000 × 0.933) + (150/1000 × 0.70)
            = 0.35 + 0.420 + 0.105
            = 0.875 = 87.5%
```

---

## 10. Practice Problems with Solutions

### Problem 1: Evaluation Metrics

**Question:**
A grammar checker processes 2000 sentences. Results:
- 300 sentences have errors
- System flags 350 sentences as having errors
- Of the flagged sentences, 250 actually have errors
- Calculate: Precision, Recall, F-score, Accuracy

**Solution:**

```
Step 1: Organize given information

Total sentences: 2000
Actual errors: 300
System flags: 350
Correct flags (TP): 250

Step 2: Calculate confusion matrix values

TP (True Positives) = 250
   (correctly flagged errors)

FP (False Positives) = 350 - 250 = 100
   (flagged but no error)

FN (False Negatives) = 300 - 250 = 50
   (errors that were missed)

TN (True Negatives) = 2000 - 300 - 100 = 1600
   (correctly identified as correct)

Verify: 250 + 100 + 50 + 1600 = 2000 ✓

Step 3: Calculate Precision

Precision = TP / (TP + FP)
         = 250 / (250 + 100)
         = 250 / 350
         = 0.7143
         = 71.43%

Meaning: When system flags an error, it's right 71.43% of the time

Step 4: Calculate Recall

Recall = TP / (TP + FN)
       = 250 / (250 + 50)
       = 250 / 300
       = 0.8333
       = 83.33%

Meaning: System catches 83.33% of all errors

Step 5: Calculate F-score

F₁ = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.7143 × 0.8333) / (0.7143 + 0.8333)
   = 2 × 0.5952 / 1.5476
   = 1.1905 / 1.5476
   = 0.7692
   = 76.92%

Step 6: Calculate Accuracy

Accuracy = (TP + TN) / Total
         = (250 + 1600) / 2000
         = 1850 / 2000
         = 0.925
         = 92.5%

Final Answers:
- Precision: 71.43%
- Recall: 83.33%
- F-score: 76.92%
- Accuracy: 92.5%
```

### Problem 2: Language Model Probability

**Question:**
Given bigram probabilities, calculate sentence probability:

```
P(the | <s>) = 0.20
P(cat | the) = 0.05
P(sat | cat) = 0.40
P(on | sat) = 0.60
P(mat | on) = 0.02
P(</s> | mat) = 0.80
```

Calculate: P("the cat sat on mat")

**Solution:**

```
Step 1: Write sentence with boundaries

Sentence: <s> the cat sat on mat </s>

Step 2: Identify all bigrams

1. <s> → the
2. the → cat
3. cat → sat
4. sat → on
5. on → mat
6. mat → </s>

Step 3: Apply bigram formula

P(sentence) = P(the|<s>) × P(cat|the) × P(sat|cat) ×
              P(on|sat) × P(mat|on) × P(</s>|mat)

Step 4: Substitute values

P(sentence) = 0.20 × 0.05 × 0.40 × 0.60 × 0.02 × 0.80

Step 5: Calculate step by step

0.20 × 0.05 = 0.01
0.01 × 0.40 = 0.004
0.004 × 0.60 = 0.0024
0.0024 × 0.02 = 0.000048
0.000048 × 0.80 = 0.0000384

P(sentence) = 3.84 × 10⁻⁵

Step 6: Calculate log probability (for practical use)

log P(sentence) = log(0.20) + log(0.05) + log(0.40) +
                  log(0.60) + log(0.02) + log(0.80)

log₁₀ values:
log(0.20) = -0.699
log(0.05) = -1.301
log(0.40) = -0.398
log(0.60) = -0.222
log(0.02) = -1.699
log(0.80) = -0.097

Sum = -4.416

P(sentence) = 10^(-4.416) = 3.84 × 10⁻⁵ ✓

Final Answers:
- P(sentence) = 3.84 × 10⁻⁵ = 0.0000384
- log₁₀ P(sentence) = -4.416
- Perplexity = 10^(4.416/6) = 4.92
```

### Problem 3: Classification Accuracy

**Question:**
A verb form classifier is tested on 800 instances:

| Actual Class | VB | VBD | VBG | VBN | VBZ |
|--------------|-----|-----|-----|-----|-----|
| VB | 140 | 5 | 0 | 0 | 5 |
| VBD | 2 | 180 | 3 | 10 | 5 |
| VBG | 1 | 2 | 145 | 2 | 0 |
| VBN | 0 | 8 | 1 | 135 | 1 |
| VBZ | 7 | 5 | 1 | 3 | 134 |

Calculate overall accuracy and per-class precision/recall.

**Solution:**

```
Step 1: Extract data from confusion matrix

Total instances = 800

Diagonal (correct predictions):
TP_VB = 140
TP_VBD = 180
TP_VBG = 145
TP_VBN = 135
TP_VBZ = 134

Total correct = 140 + 180 + 145 + 135 + 134 = 734

Step 2: Calculate Overall Accuracy

Accuracy = Total Correct / Total Instances
         = 734 / 800
         = 0.9175
         = 91.75%

Step 3: Calculate Per-Class Metrics

Class VB:
Actual VB instances = 140 + 5 + 0 + 0 + 5 = 150
Predicted VB = 140 + 2 + 1 + 0 + 7 = 150

Precision_VB = 140 / 150 = 0.9333 = 93.33%
Recall_VB = 140 / 150 = 0.9333 = 93.33%
F1_VB = 0.9333 = 93.33%

Class VBD (Past Tense):
Actual VBD = 2 + 180 + 2 + 8 + 5 = 197
   (Note: Row sum, not column)

Wait, let me recalculate correctly:

Reading confusion matrix:
Rows = Predicted
Columns = Actual

Actual VBD (column sum):
= 5 + 180 + 2 + 8 + 5 = 200

Predicted VBD (row sum):
= 2 + 180 + 3 + 10 + 5 = 200

Precision_VBD = 180 / 200 = 0.90 = 90%
Recall_VBD = 180 / 200 = 0.90 = 90%
F1_VBD = 0.90 = 90%

Class VBG (Gerund):
Actual VBG = 0 + 3 + 145 + 1 + 1 = 150
Predicted VBG = 1 + 2 + 145 + 2 + 0 = 150

Precision_VBG = 145 / 150 = 0.9667 = 96.67%
Recall_VBG = 145 / 150 = 0.9667 = 96.67%
F1_VBG = 0.9667 = 96.67%

Class VBN (Past Participle):
Actual VBN = 0 + 10 + 2 + 135 + 3 = 150
Predicted VBN = 0 + 8 + 1 + 135 + 1 = 145

Precision_VBN = 135 / 145 = 0.9310 = 93.10%
Recall_VBN = 135 / 150

= 0.90 = 90%
F1_VBN = 2 × (0.9310 × 0.90) / (0.9310 + 0.90)
       = 1.6758 / 1.8310
       = 0.9153 = 91.53%

Class VBZ (3rd Person Singular):
Actual VBZ = 5 + 5 + 0 + 1 + 134 = 145
   (Note: This doesn't sum to expected)

Let me recalculate row/column totals:

Actually, looking at the matrix structure:
Rows should represent ACTUAL (ground truth)
Columns represent PREDICTED

Correcting the interpretation:

For VB (row 1):
Actual VB instances = 140+5+0+0+5 = 150
Correctly predicted as VB = 140

Predicted as VB (column 1):
= 140+2+1+0+7 = 150

Precision_VB = 140/150 = 93.33%
Recall_VB = 140/150 = 93.33%

Summary Table:

Class | Precision | Recall | F1-Score
------|-----------|--------|----------
VB    | 93.33%   | 93.33% | 93.33%
VBD   | 90.00%   | 90.00% | 90.00%
VBG   | 96.67%   | 96.67% | 96.67%
VBN   | 93.10%   | 90.00% | 91.53%
VBZ   | 89.93%   | 92.41% | 91.16%

Macro-average F1 = (93.33 + 90.00 + 96.67 + 91.53 + 91.16) / 5
                 = 92.54%

Overall Accuracy = 91.75%
```

### Problem 4: N-gram Calculation

**Question:**
Calculate trigram probability with smoothing:

Corpus: "the cat sat on the mat . the dog sat on the rug ."

Calculate: P("the cat sat") using:
a) Maximum Likelihood Estimation (MLE)
b) Add-one (Laplace) smoothing

**Solution:**

```
Step 1: Extract corpus statistics

Corpus tokens: [the, cat, sat, on, the, mat, ., the, dog, sat, on, the, rug, .]
Vocabulary size V = unique words = {the, cat, sat, on, mat, ., dog, rug} = 8

Trigram "the cat sat":
Count(the cat sat) = 1

Bigram "the cat":
Count(the cat) = 1

Step 2: Calculate MLE Probability

P_MLE(sat | the, cat) = Count(the cat sat) / Count(the cat)
                       = 1 / 1
                       = 1.0

This seems wrong because of data sparsity!

Actually, in such a small corpus, we need to be more careful.

Let's count properly:

Trigrams in corpus:
1. <s> <s> the
2. <s> the cat
3. the cat sat
4. cat sat on
5. sat on the
6. on the mat
7. the mat .
8. mat . <s>
9. <s> <s> the
10. <s> the dog
11. the dog sat
12. dog sat on
13. sat on the
14. on the rug
15. the rug .
16. rug . <s>

Count(the cat sat) = 1
Count(the cat) = 1
Count(the dog sat) = 1
Count(the dog) = 1

P_MLE(sat | the cat) = 1/1 = 1.0

Step 3: Calculate with Add-One Smoothing

P_add1(sat | the cat) = (Count(the cat sat) + 1) / (Count(the cat) + V)
                       = (1 + 1) / (1 + 8)
                       = 2 / 9
                       = 0.2222

Where V = vocabulary size = 8

Step 4: Calculate full trigram probability for "the cat sat"

Using MLE (chain rule):
P(the cat sat) = P(the|<s>,<s>) × P(cat|<s>,the) × P(sat|the,cat)

Count(<s> <s> the) = 2 (appears twice in corpus)
Count(<s> <s>) = 2

P(the|<s>,<s>) = 2/2 = 1.0

Count(<s> the cat) = 1
Count(<s> the) = 2

P(cat|<s>,the) = 1/2 = 0.5

P(sat|the,cat) = 1/1 = 1.0

P_MLE(the cat sat) = 1.0 × 0.5 × 1.0 = 0.5

Using Add-One Smoothing:

P(the|<s>,<s>) = (2+1)/(2+8) = 3/10 = 0.3
P(cat|<s>,the) = (1+1)/(2+8) = 2/10 = 0.2
P(sat|the,cat) = (1+1)/(1+8) = 2/9 = 0.2222

P_add1(the cat sat) = 0.3 × 0.2 × 0.2222
                    = 0.0133

Final Answers:
a) MLE: P("the cat sat") = 0.5
b) Add-one smoothing: P("the cat sat") = 0.0133

Note: MLE gives higher probability but is prone to overfitting.
Add-one smoothing is more conservative and handles unseen n-grams better.
```

### Problem 5: SMT Scoring

**Question:**
Score three correction candidates using SMT:

Original: "I want discuss this"

Candidates:
A: "I want to discuss this"
B: "I want discuss this" (no change)
C: "I want discussing this"

Phrase table scores:
P("to discuss" | "discuss") = 0.75
P("discuss" | "discuss") = 1.0
P("discussing" | "discuss") = 0.15

Language model scores (log probabilities):
P_LM("I want to discuss this") = -3.2
P_LM("I want discuss this") = -7.5
P_LM("I want discussing this") = -6.8

Calculate combined scores with λ = 1.0

**Solution:**

```
Step 1: Understand scoring formula

Score(translation) = log P_TM(target|source) + λ × log P_LM(target)

Where λ = language model weight = 1.0

Step 2: Score Candidate A ("I want to discuss this")

Translation model:
Phrase mappings:
- "I" → "I" (P = 1.0)
- "want" → "want" (P = 1.0)
- "discuss" → "to discuss" (P = 0.75)
- "this" → "this" (P = 1.0)

P_TM(A|source) = 1.0 × 1.0 × 0.75 × 1.0 = 0.75
log P_TM(A) = log(0.75) = -0.125

Language model:
log P_LM(A) = -3.2 (given)

Combined score:
Score(A) = -0.125 + 1.0 × (-3.2)
         = -0.125 + (-3.2)
         = -3.325

Step 3: Score Candidate B ("I want discuss this")

Translation model:
All phrases unchanged:
P_TM(B|source) = 1.0 × 1.0 × 1.0 × 1.0 = 1.0
log P_TM(B) = log(1.0) = 0

Language model:
log P_LM(B) = -7.5 (given)

Combined score:
Score(B) = 0 + 1.0 × (-7.5)
         = -7.5

Step 4: Score Candidate C ("I want discussing this")

Translation model:
- "discuss" → "discussing" (P = 0.15)

P_TM(C|source) = 1.0 × 1.0 × 0.15 × 1.0 = 0.15
log P_TM(C) = log(0.15) = -0.824

Language model:
log P_LM(C) = -6.8 (given)

Combined score:
Score(C) = -0.824 + 1.0 × (-6.8)
         = -0.824 + (-6.8)
         = -7.624

Step 5: Rank candidates (higher score is better)

Ranking:
1. Candidate A: -3.325 ✓✓✓ BEST
2. Candidate B: -7.5
3. Candidate C: -7.624 WORST

Score differences:
A vs B: -3.325 - (-7.5) = +4.175
→ A is e^4.175 = 65.1× better than B

A vs C: -3.325 - (-7.624) = +4.299
→ A is e^4.299 = 73.4× better than C

Step 6: Interpretation

Candidate A wins because:
- Moderate translation score (0.75)
- STRONG language model score (-3.2)
- The phrase "to discuss" is much more grammatical

Candidate B has:
- Perfect translation score (1.0) - no change
- WEAK language model score (-7.5)
- "want discuss" is ungrammatical in English

Candidate C has:
- Weak translation score (0.15)
- Weak language model score (-6.8)
- "want discussing" is ungrammatical

Final Decision: Select Candidate A
Output: "I want to discuss this"
Confidence: Very high (65× better than alternatives)
```

---

## 11. Quick Reference Tables

### 11.1 POS Tag Reference

| Tag | Description | Example |
|-----|-------------|---------|
| **Verb Forms** |
| VB | Base form | eat, go, be |
| VBD | Past tense | ate, went, was |
| VBG | Gerund/Present participle | eating, going, being |
| VBN | Past participle | eaten, gone, been |
| VBP | Non-3rd person singular present | eat, go, are |
| VBZ | 3rd person singular present | eats, goes, is |
| MD | Modal | can, will, should |
| **Nouns** |
| NN | Singular noun | cat, house, idea |
| NNS | Plural noun | cats, houses, ideas |
| NNP | Proper singular noun | John, London |
| NNPS | Proper plural noun | Americans |
| **Determiners** |
| DT | Determiner | the, a, an, this |
| PDT | Predeterminer | all, both |
| **Pronouns** |
| PRP | Personal pronoun | I, you, he, she |
| PRP$ | Possessive pronoun | my, your, his, her |
| WP | Wh-pronoun | who, what, which |
| **Adjectives & Adverbs** |
| JJ | Adjective | big, old, green |
| JJR | Comparative adjective | bigger, older |
| JJS | Superlative adjective | biggest, oldest |
| RB | Adverb | quickly, very |
| RBR | Comparative adverb | faster |
| RBS | Superlative adverb | fastest |
| **Prepositions & Conjunctions** |
| IN | Preposition | in, of, at, on |
| CC | Coordinating conjunction | and, or, but |
| **Others** |
| TO | "to" | to (infinitive marker) |
| . | Punctuation | . , ! ? ; : |

### 11.2 Evaluation Metrics Quick Reference

| Metric | Formula | Range | Interpretation | When to Use |
|--------|---------|-------|----------------|-------------|
| **Precision** | TP/(TP+FP) | [0,1] | Accuracy of positive predictions | When false positives are costly |
| **Recall** | TP/(TP+FN) | [0,1] | Coverage of actual positives | When false negatives are costly |
| **F₁-score** | 2PR/(P+R) | [0,1] | Harmonic mean of P and R | Balance precision and recall |
| **F₀.₅** | 1.25PR/(0.25P+R) | [0,1] | Precision-weighted | Emphasize precision |
| **F₂** | 5PR/(4P+R) | [0,1] | Recall-weighted | Emphasize recall |
| **Accuracy** | (TP+TN)/Total | [0,1] | Overall correctness | Balanced datasets |
| **FPR** | FP/(FP+TN) | [0,1] | False alarm rate | Measure over-flagging |
| **FNR** | FN/(FN+TP) | [0,1] | Miss rate | Measure under-detection |

**Decision Matrix:**

```
┌───────────────────────────────────────────────┐
│ If False Positives are worse → Maximize Precision
│ If False Negatives are worse → Maximize Recall
│ If both equally important → Maximize F₁
│ If balanced classes → Use Accuracy
│ If imbalanced classes → Use F₁ or PR-AUC
└───────────────────────────────────────────────┘
```

### 11.3 Error Type Reference

| Error Type | % | Detection Method | Correction Difficulty | Example |
|------------|---|------------------|----------------------|---------|
| Content Word Choice | 20% | LM, Embeddings | Very High | "powerful tea" → "strong tea" |
| Verbal Morphology | 14% | Rules + ML | High | "were eaten" → "were eating" |
| Prepositions | 13% | ML, LM | Very High | "on Monday" vs "in Monday" |
| Determiners | 12% | ML, LM | High | "bought shirt" → "bought a shirt" |
| Punctuation | 12% | Rules | Medium | Missing commas |
| Derivational Morph | 5% | Rules, Dictionary | Medium | "admiration" → "admire" |
| Pronoun | 4% | Rules, Discourse | Medium | Gender/number agreement |
| Agreement | 4% | Rules | Low | Subject-verb agreement |
| Run-on | 4% | Parsing | Medium | Missing conjunctions/punctuation |
| Word Order | 4% | Parsing | Medium | Adjective order |
| Real Word Spelling | 2% | Context-aware | High | "there" vs "their" |

---

## 12. Exam Strategy Guide

### 12.1 Key Formulas to Memorize

**Must Know (100% certainty):**

```
1. Precision = TP / (TP + FP)

2. Recall = TP / (TP + FN)

3. F-score = 2 × P × R / (P + R)

4. Accuracy = (TP + TN) / Total

5. Bigram Probability:
   P(wᵢ | wᵢ₋₁) = Count(wᵢ₋₁, wᵢ) / Count(wᵢ₋₁)

6. Trigram Probability:
   P(wᵢ | wᵢ₋₂, wᵢ₋₁) = Count(wᵢ₋₂, wᵢ₋₁, wᵢ) / Count(wᵢ₋₂, wᵢ₋₁)

7. Add-One Smoothing:
   P_add1(wᵢ | wᵢ₋₁) = (Count(wᵢ₋₁, wᵢ) + 1) / (Count(wᵢ₋₁) + V)

8. SMT Score:
   Score = log P_TM(target|source) + λ × log P_LM(target)
```

### 12.2 Common Question Types

**Type 1: Calculate Evaluation Metrics**

Given: TP, FP, TN, FN
Calculate: P, R, F₁, Accuracy

**Strategy:**
1. Draw confusion matrix
2. Verify totals
3. Apply formulas systematically
4. Check answers make sense (0 ≤ value ≤ 1)

**Type 2: Language Model Probability**

Given: N-gram counts or probabilities
Calculate: Sentence probability

**Strategy:**
1. Add sentence boundaries (<s>, </s>)
2. Break into n-grams
3. Apply chain rule
4. Use log probabilities for numerical stability
5. Convert back to probability if needed

**Type 3: Error Classification**

Given: Erroneous sentence
Task: Identify error type and correction

**Strategy:**
1. Parse sentence (mentally or written)
2. Check each error type systematically:
   - Subject-verb agreement
   - Verb tense
   - Articles
   - Prepositions
3. Justify correction with rule or corpus evidence

**Type 4: Compare Approaches**

Given: Multiple GEC methods
Task: Compare strengths/weaknesses

**Strategy:**
- Use comparison tables
- Focus on: data requirements, performance, complexity
- Give concrete examples

### 12.3 Calculation Checklist

**Before Starting:**
- [ ] Read question twice
- [ ] Identify what's given
- [ ] Identify what's asked
- [ ] Choose appropriate formula

**During Calculation:**
- [ ] Write out formula first
- [ ] Substitute values clearly
- [ ] Show intermediate steps
- [ ] Use parentheses correctly
- [ ] Keep track of units/percentages

**After Calculation:**
- [ ] Verify answer is in correct range
- [ ] Check if answer makes intuitive sense
- [ ] Verify units (probability, percentage, log, etc.)
- [ ] Round appropriately (usually 2-4 decimal places)

**Common Pitfalls to Avoid:**

```
❌ Confusing Precision and Recall
❌ Forgetting to add sentence boundaries for LM
❌ Using regular probabilities instead of log probabilities
❌ Mixing up rows and columns in confusion matrix
❌ Forgetting to verify total counts
❌ Incorrect parentheses in formulas
❌ Not normalizing probabilities
❌ Confusing P(A|B) with P(B|A)
```

---

**END OF STUDY GUIDE**

This comprehensive markdown file contains all topics, formulas, step-by-step calculations, tables, and practice problems needed for exam preparation in Grammar Checking and Spell Correction in NLP.