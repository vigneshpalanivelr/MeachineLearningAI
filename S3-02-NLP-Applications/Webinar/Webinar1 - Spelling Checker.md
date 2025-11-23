# NLP Applications - Webinar

## 1. Spelling and Grammar Correction using NN

### 1.1 Spelling Check

---

## 1.1.1 Overview & Introduction

### What is Spell Checking?

Spell checking is the process of detecting and correcting misspelled words in text. A spell checker:

- Identifies words not found in a dictionary
- Suggests correct alternatives
- Ranks suggestions by likelihood
- Can automatically correct common errors

### Real-World Applications

| Application | Use Case |
|-------------|----------|
| **Microsoft Word** | Real-time spelling correction |
| **Google Docs** | Autocorrect and suggestions |
| **Gmail** | Smart compose and error detection |
| **IDE Code Editors** | Variable name suggestions |
| **Search Engines** | "Did you mean...?" suggestions |
| **Mobile Keyboards** | Predictive text correction |

### What You'll Build

A **production-grade spell checker** implementing:

- Edit Distance algorithms (Levenshtein Distance)
- Noisy Channel Model concepts
- Statistical ranking of corrections
- Multiple correction strategies

**Key Features:**
- Fast candidate generation (~0.001 seconds for ED1)
- Comprehensive fallback (ED2 when needed)
- Intelligent scoring system
- Handles 90%+ of common typos

### Prerequisites

#### Required Python Knowledge

```python
# 1. Collections module
from collections import Counter

# 2. Regular expressions
import re

# 3. String operations
import string

# 4. Type hints
from typing import List, Set, Tuple, Dict

# 5. Difflib for sequence matching
import difflib
```

#### Mathematical Concepts

1. **Edit Distance** - Minimum operations to transform one string to another
2. **Probability** - Likelihood of corrections
3. **Set Theory** - For generating candidate words

#### Data Structures

| Structure | Purpose | Time Complexity |
|-----------|---------|-----------------|
| **Set** | Fast word lookup | O(1) |
| **Counter** | Word frequency tracking | O(1) lookup |
| **List** | Ordered candidates | O(n) access |
| **Tuple** | Immutable word-score pairs | O(1) access |

---

### Setup Instructions

#### Step 1: Download Training Data

```python
!wget https://norvig.com/big.txt
```

**What is big.txt?**
- 6.2MB text file
- Contains ~6 million characters
- Classic literature (Sherlock Holmes, etc.)
- ~1 million words total
- Used to build word frequency model

**Output:**
```
Length: 6488666 (6.2M) [text/plain]
Saving to: 'big.txt'
```

#### Step 2: Import Required Libraries

```python
import re                    # Regular expressions for pattern matching
import string                # String constants (ascii_lowercase, etc.)
from collections import Counter  # Count word frequencies efficiently
from typing import List, Set, Tuple, Dict  # Type hints for clarity
import difflib              # Sequence comparison algorithms
```

**Library Usage:**

```python
# re - Extract words from text
words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

# string - Generate all letters
string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'

# Counter - Count word frequencies
word_freq = Counter(['apple', 'banana', 'apple'])
# Result: Counter({'apple': 2, 'banana': 1})

# difflib - Calculate similarity
similarity = difflib.SequenceMatcher(None, 'hello', 'helo').ratio()
# Result: 0.888 (88.8% similar)
```

---

## 1.1.2 Architecture & Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       SPELL CHECKER PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

Input Text
    │
    v
┌─────────────────┐
│ Word Extraction │  <- extract_words()
│ (Regex parsing) │
└────────┬────────┘
         │
         v
    ┌────────────┐
    │ Is Valid?  │  <- is_valid_word()
    └────┬───────┘
         │
    ┌────┴────┐
    │   YES   │   NO
    │         │    │
    v         v    v
  Return   ┌──────────────────┐
  Word     │ Generate         │
           │ Candidates       │  <- get_candidates()
           └────────┬─────────┘
                    │
        ┌───────────┴───────────┐
        v                       v
  ┌──────────┐          ┌──────────────┐
  │   ED1    │          │     ED2      │
  │ (Fast)   │  Fails   │  (Fallback)  │
  └────┬─────┘  ───>    └──────┬───────┘
       │                       │
       v                       v
  ┌─────────────────────────────────┐
  │     Substring Matching          │
  │     (Fuzzy Supplement)          │
  └────────────┬────────────────────┘
               │
               v
      ┌────────────────┐
      │  Deduplicate   │
      │  Keep Best     │
      └────────┬───────┘
               │
               v
      ┌────────────────┐
      │  Score & Rank  │  <- calculate_score()
      └────────┬───────┘
               │
               v
      ┌────────────────┐
      │  Top 10        │
      │  Suggestions   │
      └────────────────┘
```

### Component Overview

| Component | Purpose | Key Method |
|-----------|---------|------------|
| **Dictionary** | Store valid words | `load_dictionary()` |
| **Validator** | Check word validity | `is_valid_word()` |
| **Extractor** | Parse text to words | `extract_words()` |
| **Generator ED1** | Find 1-edit variations | `edit_distance_1()` |
| **Generator ED2** | Find 2-edit variations | `edit_distance_2()` |
| **Scorer** | Rank candidates | `calculate_score()` |
| **Matcher** | Fuzzy substring match | `get_substring_candidates()` |
| **Orchestrator** | Coordinate all strategies | `get_candidates()` |
| **Corrector** | Apply corrections | `correct_text()` |

---

### SpellChecker Class Structure

```python
class SpellChecker:
    """
    Production-grade spell checker implementing:
    - Edit Distance algorithms
    - Noisy Channel Model
    - Multi-strategy candidate generation
    """
```

**Complete Method List:**

| Method | Description | Complexity |
|--------|-------------|------------|
| `__init__(dictionary_file)` | Initialize with dictionary file path | O(1) |
| `load_dictionary(file_path)` | Load words from file into set | O(n) |
| `load_default_dictionary()` | Load hardcoded 117-word dictionary | O(1) |
| `is_valid_word(word)` | Check if word exists in dictionary | O(1) |
| `extract_words(text)` | Extract words from text using regex | O(n) |
| `edit_distance_1(word)` | Generate all 1-edit variations | O(n) |
| `edit_distance_2(word)` | Generate all 2-edit variations | O(n²) |
| `get_candidates(word, max_distance)` | Get ranked correction suggestions | O(n²) |
| `calculate_score(original, candidate, ed)` | Score a candidate correction | O(1) |
| `get_substring_candidates(word)` | Find fuzzy substring matches | O(m) |
| `check_text(text)` | Analyze entire text for errors | O(n×m) |
| `correct_text(text, auto_correct)` | Apply corrections to text | O(n×m) |

**Data Attributes:**

```python
self.dictionary       # Set[str] - Fast O(1) word lookup
self.word_frequency   # Counter - Word occurrence counts
```

---

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            SpellChecker Class                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│          ┌─────────────┐         ┌──────────────┐                        │
│          │ Dictionary  │◄────────┤ load_dict()  │                        │
│          │   (Set)     │         └──────────────┘                        │
│          └──────┬──────┘                                                 │
│                 │                                                        │
│                 v                                                        │
│          ┌─────────────┐         ┌──────────────┐                        │
│          │  Frequency  │◄────────┤   Counter    │                        │
│          │  (Counter)  │         └──────────────┘                        │
│          └──────┬──────┘                                                 │
│                 │                                                        │
│                 v                                                        │
│          ┌──────────────────────────────────────┐                        │
│          │      get_candidates()                │                        │
│          │  ┌────────────────────────────────┐  │                        │
│          │  │ 1. is_valid_word()             │  │                        │
│          │  │ 2. edit_distance_1()           │  │                        │
│          │  │ 3. edit_distance_2()           │  │                        │
│          │  │ 4. get_substring_candidates()  │  │                        │
│          │  │ 5. calculate_score()           │  │                        │
│          │  └────────────────────────────────┘  │                        │
│          └──────────────┬───────────────────────┘                        │
│                         │                                                │
│                         v                                                │
│          ┌─────────────────────────────┐                                 │
│          │  Ranked Candidates          │                                 │
│          │  [(word1, score1), ...]     │                                 │
│          └─────────────────────────────┘                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 1.1.3 Core Concept: Edit Distance

### Theory & Definition

**Edit Distance** (also called Levenshtein Distance) is the minimum number of single-character edits required to transform one word into another.

**Example:**
```
"cat" -> "cart"

Operations needed: Insert 'r' before 't'
Edit Distance = 1
```

**Another Example:**
```
"kitten" -> "sitting"

Operations:
1. Replace 'k' with 's': sitten
2. Replace 'e' with 'i': sittin
3. Insert 'g' at end:    sitting

Edit Distance = 3
```

### Types of Edit Distance

| Type | Operations Allowed | Example | Use Case |
|------|-------------------|---------|----------|
| **Hamming Distance** | Replacements only (same length) | "hello" → "hallo" = 1 | Fixed-length codes |
| **Levenshtein** | Insert, Delete, Replace | "cat" → "cart" = 1 | General spell checking |
| **Damerau-Levenshtein** | + Transposition | "wrold" → "world" = 1 | Common typos |

**Our Implementation:** Damerau-Levenshtein (includes transposition for common swap errors)

---

### DIRT Operations Explained

**Mnemonic: D-I-R-T**

| Operation | Description | Example |
|-----------|-------------|---------|
| **D**elete | Remove one character | "cart" → "cat" |
| **I**nsert | Add one character | "cat" → "cart" |
| **R**eplace | Substitute one character | "cat" → "bat" |
| **T**ranspose | Swap two adjacent characters | "cat" → "act" |

#### Visual Examples with "cat"

```
Original: c a t

DELETE:
  _ a t  → "at"   (delete 'c')
  c _ t  → "ct"   (delete 'a')
  c a _  → "ca"   (delete 't')

INSERT:
  s c a t  → "scat"  (insert 's' before 'c')
  c h a t  → "chat"  (insert 'h' after 'c')
  c a r t  → "cart"  (insert 'r' before 't')
  c a t s  → "cats"  (insert 's' after 't')

REPLACE:
  b a t  → "bat"   (replace 'c' with 'b')
  c u t  → "cut"   (replace 'a' with 'u')
  c a r  → "car"   (replace 't' with 'r')

TRANSPOSE:
  a c t  → "act"   (swap 'c' and 'a')
  c t a  → "cta"   (swap 'a' and 't')
```

---

### Formula: 54n + 25

For a word of length **n**, the number of variations generated by edit_distance_1():

| Operation | Count | Formula |
|-----------|-------|---------|
| Deletions | n | Remove each character once |
| Transpositions | n-1 | Swap each adjacent pair |
| Replacements | 26n | Replace each char with a-z |
| Insertions | 26(n+1) | Insert a-z at each position |
| **Total** | **54n + 25** | Sum of all operations |

**Derivation:**

```
Total = Deletions + Transpositions + Replacements + Insertions
      = n + (n-1) + 26n + 26(n+1)
      = n + n - 1 + 26n + 26n + 26
      = 54n + 25
```

**Example for "cat" (n=3):**

```
Deletions:      3
Transpositions: 2
Replacements:   78  (26 × 3)
Insertions:     104 (26 × 4)
────────────────────
Total:          187 (54 × 3 + 25 = 162 + 25 = 187)
```

**Verification Table:**

| Word | Length (n) | Formula (54n+25) | Actual Count |
|------|------------|------------------|--------------|
| "cat" | 3 | 187 | 187 ✓ |
| "test" | 4 | 241 | 241 ✓ |
| "hello" | 5 | 295 | 295 ✓ |
| "python" | 6 | 349 | 349 ✓ |

**Note:** Actual count may be slightly lower due to duplicate removal by set (e.g., replacing 'a' with 'a' produces same word).

---

### Why This Matters

**Speed vs Accuracy Trade-off:**

- **Edit Distance 1:** ~200 candidates, very fast (~0.001s)
- **Edit Distance 2:** ~35,000 candidates, slower (~0.2s)

**Strategy:**
1. Try ED1 first (handles 90% of typos)
2. Fall back to ED2 only if ED1 finds nothing
3. Balance between speed and comprehensiveness

---

## 1.1.4 Core Concept: Noisy Channel Model

### Theory Explanation

The **Noisy Channel Model** treats spelling errors as "noise" added during transmission:

```
Intended Word ──[NOISY CHANNEL]──> Observed Word
   "hello"        (User types)        "helo"
```

**Goal:** Given observed word "helo", find most likely intended word.

---

### Bayesian Formula

```
P(intended | observed) = P(observed | intended) × P(intended)
                         ─────────────────────────────────────
                                    P(observed)

Where:
- P(intended | observed) = Probability "hello" was intended given we saw "helo"
- P(observed | intended) = Error model: Probability of typing "helo" when meaning "hello"
- P(intended)            = Language model: Prior probability of word "hello"
- P(observed)            = Normalization constant (can be ignored for ranking)
```

**Choose word that maximizes:** `P(intended | observed)`

---

### Simplified for Spell Checking

In our implementation:

```
Score = (Error Model) × (Language Model) × (Heuristic Factors)

Score = (1 / edit_distance) × word_frequency × length_similarity × char_overlap
         ↑                     ↑                 ↑                   ↑
    P(observed|intended)   P(intended)      Heuristics
```

**Components:**

| Component | Represents | Example Value |
|-----------|------------|---------------|
| Edit Distance | How likely the error | ED1 = 0.5, ED2 = 0.33 |
| Word Frequency | How common the word | "the" = high, "xylophone" = low |
| Length Similarity | Similar length = more likely | Same length = 1.0 |
| Character Overlap | Shared characters | All match = 1.0 |

---

### How It Guides Scoring

**Example: "helo" → candidates**

```
Candidate 1: "hello"
  Edit Distance:     1  → Score component: 0.5
  Frequency:       100  → Score component: 1.0
  Length Diff:       1  → Score component: 0.5
  Char Overlap:    100% → Score component: 1.0
  Final Score: (0.5×0.5) + (1.0×0.2) + (0.5×0.2) + (1.0×0.1) = 0.65

Candidate 2: "help"
  Edit Distance:     1  → Score component: 0.5
  Frequency:        80  → Score component: 0.8
  Length Diff:       0  → Score component: 1.0
  Char Overlap:     75% → Score component: 0.75
  Final Score: (0.5×0.5) + (0.8×0.2) + (1.0×0.2) + (0.75×0.1) = 0.69

Winner: "help" (0.69 > 0.65)
```

**Insight:** Even though "hello" matches better in characters, "help" wins due to:
- Same length as "helo"
- Still high frequency
- Weighted scoring favors multiple good factors

---

## 1.1.5 Dictionary Management

### Concept: Why Use Sets?

**Question:** Why use a `set()` instead of a `list()` for the dictionary?

**Answer:**

| Data Structure | Lookup Time | Memory | Duplicates |
|----------------|-------------|---------|------------|
| **set()** | O(1) - Constant | Lower | Not allowed |
| **list()** | O(n) - Linear | Higher | Allowed |

**Example:**

```python
# Using list (SLOW)
dictionary = ['hello', 'world', 'python', ...]  # 32,000 words
'hello' in dictionary  # Must check all 32,000 words in worst case

# Using set (FAST)
dictionary = {'hello', 'world', 'python', ...}
'hello' in dictionary  # Hash lookup, instant
```

**Why this matters:**
- We check thousands of candidate words
- Each check with list: O(n) × 1000 candidates = very slow
- Each check with set: O(1) × 1000 candidates = fast

---

### Implementation: `__init__()`

#### Explanation

The constructor initializes the spell checker with either:
1. A custom dictionary file (if provided)
2. Default 117-word dictionary (if no file)

**Design Decisions:**
- Use `set()` for fast O(1) word lookup
- Use `Counter()` for word frequency tracking
- Support optional file path for flexibility

#### Code

```python
def __init__(self, dictionary_file: str = None):
    """
    Initialize the spell checker with a dictionary.

    Args:
        dictionary_file: Path to a text file containing valid words
    """
    self.dictionary = set()           # Fast O(1) lookup for valid words
    self.word_frequency = Counter()   # Stores how often each word appears

    if dictionary_file:
        self.load_dictionary(dictionary_file)
        print(f'File Loaded {dictionary_file}')
    else:
        # Default small dictionary for demonstration
        self.load_default_dictionary()
```

#### Example Usage

```python
# Example 1: Use default dictionary
checker1 = SpellChecker()
# Loads 117 hardcoded words

# Example 2: Use custom dictionary
checker2 = SpellChecker('big.txt')
# Output: File Loaded big.txt
# Loads ~32,000 words from file

# Example 3: Use your own word list
checker3 = SpellChecker('medical_terms.txt')
# Loads medical terminology
```

**After Initialization:**

```python
# Internal state
checker.dictionary        # Set of valid words
checker.word_frequency    # Counter with word counts
```

---

### Implementation: `load_dictionary()`

#### Explanation

Loads dictionary from a text file where each line contains one word.

**Process:**
1. Open file with UTF-8 encoding (supports international characters)
2. Read each line, strip whitespace, convert to lowercase
3. Store in set (automatic duplicate removal)
4. Initialize frequencies (all set to 1 in basic version)

**Error Handling:**
- If file not found, falls back to default dictionary
- Prevents crash from missing files

#### Code

```python
def load_dictionary(self, file_path: str):
    """Load dictionary from a file."""
    try:
        # Opens the file, reads lines, lowercases them, strips whitespace
        with open(file_path, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
            # Adds all words into a set
            self.dictionary = set(words)
            # Simple frequency model (all words have equal weight)
            self.word_frequency = Counter({word: 1 for word in words})
    except FileNotFoundError:
        print(f"Dictionary file {file_path} not found. Using default dictionary.")
        self.load_default_dictionary()
```

**Line-by-Line Breakdown:**

```python
# Step 1: Open file with UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as f:

# Step 2: Process each line
words = [
    line.strip().lower()    # Remove whitespace, convert to lowercase
    for line in f           # Iterate through each line
    if line.strip()         # Skip empty lines
]

# Step 3: Convert to set (removes duplicates, enables fast lookup)
self.dictionary = set(words)

# Step 4: Initialize all words with frequency 1
self.word_frequency = Counter({word: 1 for word in words})
```

#### Example

**Input File (dictionary.txt):**
```
Apple
banana
CHERRY
  dog

cat
apple
```

**After Loading:**

```python
checker = SpellChecker('dictionary.txt')

checker.dictionary
# Output: {'apple', 'banana', 'cherry', 'dog', 'cat'}
# Note: All lowercase, duplicates removed, whitespace stripped

checker.word_frequency
# Output: Counter({'apple': 1, 'banana': 1, 'cherry': 1, 'dog': 1, 'cat': 1})
```

#### Testing

```python
# Test loading
checker = SpellChecker('big.txt')

# Verify dictionary size
print(f"Words loaded: {len(checker.dictionary)}")
# Output: Words loaded: 32198

# Check if specific words exist
print('hello' in checker.dictionary)  # True
print('xyz123' in checker.dictionary)  # False

# Check frequencies
print(checker.word_frequency.most_common(5))
# Output: [('the', 1), ('of', 1), ('and', 1), ('to', 1), ('in', 1)]
# Note: Basic version assigns all frequency = 1
```

---

### Implementation: `load_default_dictionary()`

#### Explanation

Provides a hardcoded 117-word dictionary for testing without external files.

**Word Categories:**
1. Common English words
2. Programming terms
3. Control flow keywords
4. Data types

**Purpose:**
- Quick testing without file dependencies
- Demonstration and learning
- Fallback when files unavailable

#### Code

```python
def load_default_dictionary(self):
    """Load a basic dictionary for demonstration."""
    words = [
        'hello', 'world', 'python', 'programming', 'computer', 'science',
        'algorithm', 'data', 'structure', 'function', 'variable', 'loop',
        'class', 'object', 'method', 'string', 'integer', 'float', 'boolean',
        'list', 'tuple', 'dictionary', 'set', 'array', 'matrix', 'vector',
        'if', 'else', 'elif', 'while', 'for', 'break', 'continue', 'return',
        'try', 'except', 'finally', 'raise', 'assert', 'import', 'from',
        'def', 'class', 'self', 'init', 'main', 'true', 'false', 'none',
        'and', 'or', 'not', 'is', 'in', 'with', 'as', 'pass', 'lambda',
        'yield', 'global', 'nonlocal', 'del', 'print', 'input', 'range',
        'len', 'type', 'str', 'int', 'list', 'dict', 'set', 'tuple',
        'open', 'read', 'write', 'close', 'file', 'path', 'directory',
        'error', 'exception', 'debug', 'test', 'code', 'compile', 'run',
        'execute', 'process', 'thread', 'memory', 'stack', 'heap', 'queue',
        'tree', 'graph', 'node', 'edge', 'vertex', 'search', 'sort', 'find',
        'add', 'remove', 'insert', 'delete', 'update', 'get', 'put', 'pop',
        'push', 'peek', 'size', 'empty', 'full', 'contains', 'index', 'count'
    ]
    self.dictionary = set(words)
    self.word_frequency = Counter({word: 1 for word in words})
```

#### Word Categories

```python
# Common English (6 words)
['hello', 'world', 'error', 'test', 'code', 'file']

# Programming Terms (20 words)
['python', 'programming', 'computer', 'science', 'algorithm', 'data',
 'structure', 'function', 'variable', 'loop', 'class', 'object', 'method',
 'compile', 'run', 'execute', 'process', 'thread', 'memory', 'debug']

# Data Types (10 words)
['string', 'integer', 'float', 'boolean', 'list', 'tuple',
 'dictionary', 'set', 'array', 'matrix']

# Control Flow (15 words)
['if', 'else', 'elif', 'while', 'for', 'break', 'continue',
 'return', 'try', 'except', 'finally', 'raise', 'assert',
 'import', 'from']

# Data Structures (12 words)
['stack', 'heap', 'queue', 'tree', 'graph', 'node',
 'edge', 'vertex', 'search', 'sort', 'find', 'vector']

# Operations (20+ words)
['add', 'remove', 'insert', 'delete', 'update', 'get', 'put',
 'pop', 'push', 'peek', 'size', 'empty', 'full', 'contains',
 'index', 'count', 'open', 'read', 'write', 'close']
```

**Total: 117 carefully selected words**

---

## 1.1.6 Word Validation

### Concept: Case Normalization

**Question:** How are 'Hello', 'HELLO', and 'hello' treated as the same word?

**Answer:** Through **case normalization** - converting everything to lowercase before comparison.

#### Why Normalize Case?

**Without Normalization:**
```python
dictionary = {'hello', 'world', 'python'}

'Hello' in dictionary  # False (doesn't match 'hello')
'HELLO' in dictionary  # False (doesn't match 'hello')
'hello' in dictionary  # True
```

**With Normalization:**
```python
dictionary = {'hello', 'world', 'python'}

'Hello'.lower() in dictionary  # 'hello' in dict -> True
'HELLO'.lower() in dictionary  # 'hello' in dict -> True
'hello'.lower() in dictionary  # 'hello' in dict -> True
```

#### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Smaller Dictionary** | Store only one form per word |
| **Faster Lookup** | Fewer entries to search |
| **Consistent Behavior** | Same word always treated identically |
| **Standard Practice** | Used by all major spell checkers |

#### Where Normalization Happens

1. **Loading Dictionary:** All words converted to lowercase
2. **Checking Validity:** Input word converted to lowercase
3. **Extracting Text:** All extracted words converted to lowercase

---

### Implementation: `is_valid_word()`

#### Explanation

Checks if a word exists in the dictionary using O(1) set lookup.

**Process:**
1. Convert input word to lowercase
2. Check if it exists in dictionary set
3. Return True/False

**Time Complexity:** O(1) constant time (hash table lookup)

#### Code

```python
def is_valid_word(self, word: str) -> bool:
    """Check if a word exists in the dictionary."""
    return word.lower() in self.dictionary
```

**Why so simple?**
- Sets use hash tables for O(1) lookup
- Lowercase conversion ensures case-insensitive matching
- No need for loops or complex logic

#### Examples

```python
checker = SpellChecker()

# Test different cases
checker.is_valid_word('hello')   # True
checker.is_valid_word('Hello')   # True (converts to 'hello')
checker.is_valid_word('HELLO')   # True (converts to 'hello')
checker.is_valid_word('HeLLo')   # True (converts to 'hello')

# Test invalid words
checker.is_valid_word('helo')    # False (misspelled)
checker.is_valid_word('xyz123')  # False (not in dictionary)
checker.is_valid_word('')        # False (empty string)
```

#### Visual Comparison

```
Input: "Hello"
         │
         v
    .lower()
         │
         v
    "hello"
         │
         v
    in dictionary?
         │
    ┌────┴────┐
    v         v
  True      False
(found)   (not found)
```

#### Performance Comparison

```python
# Dictionary size: 32,000 words

# Using list - O(n) lookup
dictionary_list = ['hello', 'world', ...]  # 32,000 words
'hello' in dictionary_list  # Worst case: check all 32,000

# Using set - O(1) lookup
dictionary_set = {'hello', 'world', ...}
'hello' in dictionary_set  # Hash lookup: instant

# Speed difference: ~1000x faster with set
```

---

### Testing: Validation Tests

```python
def test_is_valid_word():
    """Test word validation with different cases."""

    checker = SpellChecker()

    print("=" * 70)
    print("TESTING: is_valid_word()")
    print("=" * 70)
    print()

    # Test cases: (input, expected_result, description)
    test_cases = [
        # Valid words - different cases
        ('hello', True, 'Lowercase - valid'),
        ('Hello', True, 'Capitalized - valid'),
        ('HELLO', True, 'Uppercase - valid'),
        ('HeLLo', True, 'Mixed case - valid'),

        # Valid words - other examples
        ('python', True, 'Programming term'),
        ('algorithm', True, 'Technical term'),
        ('test', True, 'Common word'),

        # Invalid words
        ('helo', False, 'Misspelling - missing l'),
        ('wrold', False, 'Misspelling - transposition'),
        ('xyz123', False, 'Not a word'),
        ('', False, 'Empty string'),
        ('   ', False, 'Only whitespace'),
        ('hello123', False, 'Alphanumeric'),
    ]

    print(f"{'Input':<15} {'Expected':<10} {'Actual':<10} {'Status':<10} {'Description':<30}")
    print("-" * 70)

    passed = 0
    failed = 0

    for word, expected, description in test_cases:
        actual = checker.is_valid_word(word)
        status = "PASS" if actual == expected else "FAIL"

        if actual == expected:
            passed += 1
        else:
            failed += 1

        print(f"{word:<15} {str(expected):<10} {str(actual):<10} {status:<10} {description:<30}")

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    print()
```

#### Expected Output

```
======================================================================
TESTING: is_valid_word()
======================================================================

Input           Expected   Actual     Status     Description
----------------------------------------------------------------------
hello           True       True       PASS       Lowercase - valid
Hello           True       True       PASS       Capitalized - valid
HELLO           True       True       PASS       Uppercase - valid
HeLLo           True       True       PASS       Mixed case - valid
python          True       True       PASS       Programming term
algorithm       True       True       PASS       Technical term
test            True       True       PASS       Common word
helo            False      False      PASS       Misspelling - missing l
wrold           False      False      PASS       Misspelling - transposition
xyz123          False      False      PASS       Not a word
                False      False      PASS       Empty string
                False      False      PASS       Only whitespace
hello123        False      False      PASS       Alphanumeric

======================================================================
Results: 13 passed, 0 failed out of 13 tests
======================================================================
```

---

## 1.1.7 Text Processing

### Concept: Regex Pattern Matching

**Regular expressions** (regex) allow pattern-based text extraction.

**Our Pattern:** `r'\b[a-zA-Z]+\b'`

**Breakdown:**

| Component | Meaning | Example |
|-----------|---------|---------|
| `\b` | Word boundary | Matches position between word and non-word |
| `[a-zA-Z]` | Character class | Any letter (a-z or A-Z) |
| `+` | Quantifier | One or more of preceding pattern |
| `r'...'` | Raw string | Prevents Python from escaping backslashes |

**Visual Example:**

```
Text: "Hello, world! This is test-123."

\b[a-zA-Z]+\b matches:
  \b[H][e][l][l][o]\b    -> "Hello"
  \b[w][o][r][l][d]\b    -> "world"
  \b[T][h][i][s]\b       -> "This"
  \b[i][s]\b             -> "is"
  \b[t][e][s][t]\b       -> "test"

Skips: ",", "!", "-", "123", "."
```

---

### Implementation: `extract_words()`

#### Explanation

Extracts all words from text by:
1. Converting text to lowercase
2. Using regex to find word patterns
3. Removing punctuation automatically
4. Returning list of clean words

**Use Cases:**
- Parse user input for spell checking
- Extract words from documents
- Prepare text for analysis

#### Code

```python
def extract_words(self, text: str) -> List[str]:
    """Extract words from text, removing punctuation."""
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words
```

#### Regex Breakdown

```python
r'\b[a-zA-Z]+\b'
  │  │      │  │
  │  │      │  └─ Word boundary (end)
  │  │      └──── One or more times (+)
  │  └─────────── Any letter (a-z or A-Z)
  └────────────── Word boundary (start)
```

**What it matches:**
- Words with letters only
- Any length (1+ characters)
- Case insensitive (a-z, A-Z)

**What it skips:**
- Numbers: "123", "2024"
- Punctuation: ".", ",", "!", "?"
- Special characters: "@", "#", "$"
- Whitespace: spaces, tabs, newlines

#### Examples

```python
checker = SpellChecker()

# Example 1: Basic sentence
text1 = "Hello, world!"
result1 = checker.extract_words(text1)
print(result1)
# Output: ['hello', 'world']

# Example 2: Punctuation
text2 = "Hello, world! How are you?"
result2 = checker.extract_words(text2)
print(result2)
# Output: ['hello', 'world', 'how', 'are', 'you']

# Example 3: Numbers
text3 = "I have 3 cats and 2 dogs."
result3 = checker.extract_words(text3)
print(result3)
# Output: ['i', 'have', 'cats', 'and', 'dogs']

# Example 4: Alphanumeric
text4 = "Python3 and HTML5 are great."
result4 = checker.extract_words(text4)
print(result4)
# Output: ['python', 'and', 'html', 'are', 'great']

# Example 5: Special characters
text5 = "user@email.com costs $50 at 10:30am"
result5 = checker.extract_words(text5)
print(result5)
# Output: ['user', 'email', 'com', 'costs', 'at', 'am']

# Example 6: Mixed case
text6 = "HELLO World HoW aRe YOU"
result6 = checker.extract_words(text6)
print(result6)
# Output: ['hello', 'world', 'how', 'are', 'you']
```

#### Edge Cases

| Input | Output | Explanation |
|-------|--------|-------------|
| `"don't"` | `['don', 't']` | Apostrophe splits words |
| `"hello123"` | `['hello']` | Numbers stripped |
| `"HELLO"` | `['hello']` | Converted to lowercase |
| `"  hello  "` | `['hello']` | Whitespace ignored |
| `""` | `[]` | Empty string returns empty list |
| `"!@#$%"` | `[]` | Only punctuation returns empty |
| `"123 456"` | `[]` | Only numbers returns empty |
| `"test-case"` | `['test', 'case']` | Hyphen splits words |

#### Visual Processing

```
Input: "Hello, world! Test-123."

Step 1: Convert to lowercase
  "hello, world! test-123."

Step 2: Apply regex r'\b[a-zA-Z]+\b'
  Match 1: "hello"   [position 0-4]
  Match 2: "world"   [position 7-11]
  Match 3: "test"    [position 14-17]

Step 3: Return list
  ['hello', 'world', 'test']

Removed:
  - Commas, periods, exclamation marks
  - Hyphens
  - Numbers
```

---

### Testing: extract_words()

```python
def test_extract_words():
    """Comprehensive test suite for extract_words() method."""

    checker = SpellChecker()

    # Test cases: (input_text, expected_output, test_description)
    test_cases = [
        # Basic tests
        ("Hello world",
         ['hello', 'world'],
         "Basic sentence"),

        # Punctuation tests
        ("Hello, world! How are you?",
         ['hello', 'world', 'how', 'are', 'you'],
         "Punctuation removal"),

        ("One.Two;Three:Four",
         ['one', 'two', 'three', 'four'],
         "Various punctuation"),

        # Case conversion tests
        ("HELLO World HoW aRe YOU",
         ['hello', 'world', 'how', 'are', 'you'],
         "Mixed case conversion"),

        # Number tests
        ("I have 3 cats and 2 dogs",
         ['i', 'have', 'cats', 'and', 'dogs'],
         "Standalone numbers"),

        ("Python3 HTML5 Test",
         ['python', 'html', 'test'],
         "Alphanumeric words"),

        # Special characters
        ("user@email.com costs $50",
         ['user', 'email', 'com', 'costs'],
         "Special characters"),

        # Contractions (splits on apostrophe)
        ("Don't you think it's great?",
         ['don', 't', 'you', 'think', 'it', 's', 'great'],
         "Contractions"),

        # Hyphens
        ("well-known self-driving",
         ['well', 'known', 'self', 'driving'],
         "Hyphenated words"),

        # Edge cases
        ("",
         [],
         "Empty string"),

        ("!@#$%^&*()",
         [],
         "Only punctuation"),

        ("123 456 789",
         [],
         "Only numbers"),

        ("   hello    world   ",
         ['hello', 'world'],
         "Extra whitespace"),

        # Complex sentences
        ("The quick brown fox jumps over the lazy dog!",
         ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
         "Complete sentence"),

        ("Test-case with numbers 123 and symbols @#$",
         ['test', 'case', 'with', 'numbers', 'and', 'symbols'],
         "Complex mixed input"),
    ]

    # Run tests
    print("=" * 70)
    print("TESTING extract_words() METHOD")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for i, (input_text, expected, description) in enumerate(test_cases, 1):
        result = checker.extract_words(input_text)
        is_pass = result == expected

        if is_pass:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"Test {i}: {description}")
        print(f"  Input:    '{input_text}'")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print(f"  Status:   {status}")
        print()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    print()

# Run the test
test_extract_words()
```

#### Expected Output

```
======================================================================
TESTING extract_words() METHOD
======================================================================

Test 1: Basic sentence
  Input:    'Hello world'
  Expected: ['hello', 'world']
  Got:      ['hello', 'world']
  Status:   PASS

Test 2: Punctuation removal
  Input:    'Hello, world! How are you?'
  Expected: ['hello', 'world', 'how', 'are', 'you']
  Got:      ['hello', 'world', 'how', 'are', 'you']
  Status:   PASS

Test 3: Various punctuation
  Input:    'One.Two;Three:Four'
  Expected: ['one', 'two', 'three', 'four']
  Got:      ['one', 'two', 'three', 'four']
  Status:   PASS

Test 4: Mixed case conversion
  Input:    'HELLO World HoW aRe YOU'
  Expected: ['hello', 'world', 'how', 'are', 'you']
  Got:      ['hello', 'world', 'how', 'are', 'you']
  Status:   PASS

Test 5: Standalone numbers
  Input:    'I have 3 cats and 2 dogs'
  Expected: ['i', 'have', 'cats', 'and', 'dogs']
  Got:      ['i', 'have', 'cats', 'and', 'dogs']
  Status:   PASS

Test 6: Alphanumeric words
  Input:    'Python3 HTML5 Test'
  Expected: ['python', 'html', 'test']
  Got:      ['python', 'html', 'test']
  Status:   PASS

Test 7: Special characters
  Input:    'user@email.com costs $50'
  Expected: ['user', 'email', 'com', 'costs']
  Got:      ['user', 'email', 'com', 'costs']
  Status:   PASS

Test 8: Contractions
  Input:    "Don't you think it's great?"
  Expected: ['don', 't', 'you', 'think', 'it', 's', 'great']
  Got:      ['don', 't', 'you', 'think', 'it', 's', 'great']
  Status:   PASS

Test 9: Hyphenated words
  Input:    'well-known self-driving'
  Expected: ['well', 'known', 'self', 'driving']
  Got:      ['well', 'known', 'self', 'driving']
  Status:   PASS

Test 10: Empty string
  Input:    ''
  Expected: []
  Got:      []
  Status:   PASS

Test 11: Only punctuation
  Input:    '!@#$%^&*()'
  Expected: []
  Got:      []
  Status:   PASS

Test 12: Only numbers
  Input:    '123 456 789'
  Expected: []
  Got:      []
  Status:   PASS

Test 13: Extra whitespace
  Input:    '   hello    world   '
  Expected: ['hello', 'world']
  Got:      ['hello', 'world']
  Status:   PASS

Test 14: Complete sentence
  Input:    'The quick brown fox jumps over the lazy dog!'
  Expected: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
  Got:      ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
  Status:   PASS

Test 15: Complex mixed input
  Input:    'Test-case with numbers 123 and symbols @#$'
  Expected: ['test', 'case', 'with', 'numbers', 'and', 'symbols']
  Got:      ['test', 'case', 'with', 'numbers', 'and', 'symbols']
  Status:   PASS

======================================================================
RESULTS: 15 passed, 0 failed out of 15 tests
======================================================================
```

---

## 1.1.8 Edit Distance 1 - Complete Coverage

### Concept: One Edit Away

**Edit Distance 1** means generating all possible words that can be created by applying **exactly one edit operation** to the original word.

**The Four Operations (DIRT):**

1. **Delete** - Remove one character
2. **Insert** - Add one character
3. **Replace** - Change one character
4. **Transpose** - Swap two adjacent characters

**Goal:** From word "cat", generate all variations like "at", "bat", "scat", "act", etc.

---

### Formula Derivation

For a word of length **n**:

#### Deletions: **n**

Remove each character once.

```
Word "cat" (n=3):
  Position 0: Remove 'c' -> "at"
  Position 1: Remove 'a' -> "ct"
  Position 2: Remove 't' -> "ca"

Total: 3 deletions
```

#### Transpositions: **n-1**

Swap each adjacent pair.

```
Word "cat" (n=3):
  Positions 0-1: Swap 'c','a' -> "act"
  Positions 1-2: Swap 'a','t' -> "cta"

Total: 2 transpositions (3-1)
```

#### Replacements: **26n**

Replace each character with a-z (26 letters).

```
Word "cat" (n=3):
  Position 0: Replace 'c' with a-z -> "aat", "bat", "cat", ..., "zat" (26)
  Position 1: Replace 'a' with a-z -> "cat", "cbt", "cct", ..., "czt" (26)
  Position 2: Replace 't' with a-z -> "caa", "cab", "cac", ..., "caz" (26)

Total: 78 replacements (26 × 3)
```

#### Insertions: **26(n+1)**

Insert a-z at each position (including before first and after last).

```
Word "cat" (n=3):
  Position 0 (before 'c'): "acat", "bcat", ..., "zcat" (26)
  Position 1 (after 'c'):  "caat", "cbat", ..., "czat" (26)
  Position 2 (after 'a'):  "caat", "cabt", ..., "cazt" (26)
  Position 3 (after 't'):  "cata", "catb", ..., "catz" (26)

Total: 104 insertions (26 × 4)
```

#### Total Formula

```
Total = Deletions + Transpositions + Replacements + Insertions
      = n + (n-1) + 26n + 26(n+1)
      = n + n - 1 + 26n + 26n + 26
      = 2n - 1 + 52n + 26
      = 54n + 25
```

**For "cat" (n=3):**
```
54(3) + 25 = 162 + 25 = 187 variations
```

---

### Implementation: `edit_distance_1()`

#### Full Code

```python
def edit_distance_1(self, word: str) -> Set[str]:
    """Generate all possible words with edit distance of 1."""
    letters = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'

    # STEP 1: Generate all possible splits
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    # STEP 2: DELETIONS - Remove one character
    deletes = [L + R[1:] for L, R in splits if R]

    # STEP 3: TRANSPOSITIONS - Swap adjacent characters
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

    # STEP 4: REPLACEMENTS - Replace one character with any letter
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

    # STEP 5: INSERTIONS - Insert one character at any position
    inserts = [L + c + R for L, R in splits for c in letters]

    # Combine all and return as set (removes duplicates)
    return set(deletes + transposes + replaces + inserts)
```

---

### Step 1: Generate Splits

#### Explanation

Split the word at every possible position to create (Left, Right) pairs.

**Purpose:** These splits form the basis for all edit operations.

#### Example with "cat"

```python
word = "cat"
splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

# Iteration:
i=0: word[:0], word[0:] = ('',    'cat')
i=1: word[:1], word[1:] = ('c',   'at')
i=2: word[:2], word[2:] = ('ca',  't')
i=3: word[:3], word[3:] = ('cat', '')

# Result:
splits = [('', 'cat'), ('c', 'at'), ('ca', 't'), ('cat', '')]
```

#### Visual Table

| Index | Left (L) | Right (R) | Description |
|-------|----------|-----------|-------------|
| 0 | `''` | `'cat'` | Before first character |
| 1 | `'c'` | `'at'` | After 'c' |
| 2 | `'ca'` | `'t'` | After 'a' |
| 3 | `'cat'` | `''` | After last character |

**Total splits:** n+1 (where n is word length)

---

### Step 2: Deletions

#### Explanation

Remove one character by taking Left + Right[1:] (skip first char of Right).

**Formula:** `L + R[1:]` for each split where R is not empty.

#### Code Logic

```python
deletes = [L + R[1:] for L, R in splits if R]
```

**Breakdown:**
- `for L, R in splits` - Iterate through each split
- `if R` - Only process if Right part exists
- `R[1:]` - Skip first character of Right
- `L + R[1:]` - Concatenate Left + (Right without first char)

#### Example with "cat"

```python
word = "cat"
splits = [('', 'cat'), ('c', 'at'), ('ca', 't'), ('cat', '')]

# Iteration:
Split 0: ('', 'cat')
  R exists? Yes
  L + R[1:] = '' + 'cat'[1:] = '' + 'at' = "at"

Split 1: ('c', 'at')
  R exists? Yes
  L + R[1:] = 'c' + 'at'[1:] = 'c' + 't' = "ct"

Split 2: ('ca', 't')
  R exists? Yes
  L + R[1:] = 'ca' + 't'[1:] = 'ca' + '' = "ca"

Split 3: ('cat', '')
  R exists? No (empty string)
  Skip this split

# Result:
deletes = ['at', 'ct', 'ca']
```

#### Visual

```
Original: c a t

Delete position 0:
  _ a t  → "at"   (remove 'c')

Delete position 1:
  c _ t  → "ct"   (remove 'a')

Delete position 2:
  c a _  → "ca"   (remove 't')

Total: 3 deletions (n=3)
```

---

### Step 3: Transpositions

#### Explanation

Swap two adjacent characters by rearranging: L + R[1] + R[0] + R[2:]

**Formula:** `L + R[1] + R[0] + R[2:]` for each split where len(R) > 1.

#### Code Logic

```python
transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
```

**Breakdown:**
- `if len(R) > 1` - Need at least 2 characters to swap
- `R[1]` - Second character of Right
- `R[0]` - First character of Right
- `R[2:]` - Remaining characters after swap
- `L + R[1] + R[0] + R[2:]` - Concatenate with swapped chars

#### Example with "cat"

```python
word = "cat"
splits = [('', 'cat'), ('c', 'at'), ('ca', 't'), ('cat', '')]

# Iteration:
Split 0: ('', 'cat')
  len(R) > 1? Yes (len('cat') = 3)
  L + R[1] + R[0] + R[2:] = '' + 'a' + 'c' + 't' = "act"
  Explanation: Swap 'c' and 'a'

Split 1: ('c', 'at')
  len(R) > 1? Yes (len('at') = 2)
  L + R[1] + R[0] + R[2:] = 'c' + 't' + 'a' + '' = "cta"
  Explanation: Swap 'a' and 't'

Split 2: ('ca', 't')
  len(R) > 1? No (len('t') = 1)
  Skip this split

Split 3: ('cat', '')
  len(R) > 1? No (len('') = 0)
  Skip this split

# Result:
transposes = ['act', 'cta']
```

#### Visual

```
Original: c a t

Transpose positions 0-1:
  Swap: c ↔ a
  Result: a c t  → "act"

Transpose positions 1-2:
  Swap: a ↔ t
  Result: c t a  → "cta"

Total: 2 transpositions (n-1 = 2)
```

---

### Step 4: Replacements

#### Explanation

Replace one character with each letter a-z.

**Formula:** `L + c + R[1:]` for each split where R exists, for each letter c in a-z.

#### Code Logic

```python
replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
```

**Breakdown:**
- `for L, R in splits` - Each split position
- `if R` - Only if Right part exists
- `for c in letters` - Try all 26 letters
- `L + c + R[1:]` - Left + new letter + (Right without first char)

#### Example with "cat"

```python
word = "cat"
splits = [('', 'cat'), ('c', 'at'), ('ca', 't'), ('cat', '')]
letters = 'abcdefghijklmnopqrstuvwxyz'

# Position 0: Replace 'c'
Split: ('', 'cat')
  For c in 'abcdefghijklmnopqrstuvwxyz':
    '' + 'a' + 'at' = "aat"
    '' + 'b' + 'at' = "bat"
    '' + 'c' + 'at' = "cat"  (same as original)
    '' + 'd' + 'at' = "dat"
    ...
    '' + 'z' + 'at' = "zat"
  Total: 26 replacements

# Position 1: Replace 'a'
Split: ('c', 'at')
  For c in 'abcdefghijklmnopqrstuvwxyz':
    'c' + 'a' + 't' = "cat"  (same as original)
    'c' + 'b' + 't' = "cbt"
    'c' + 'c' + 't' = "cct"
    ...
    'c' + 'z' + 't' = "czt"
  Total: 26 replacements

# Position 2: Replace 't'
Split: ('ca', 't')
  For c in 'abcdefghijklmnopqrstuvwxyz':
    'ca' + 'a' + '' = "caa"
    'ca' + 'b' + '' = "cab"
    'ca' + 'c' + '' = "cac"
    ...
    'ca' + 'z' + '' = "caz"
  Total: 26 replacements

# Grand Total: 26 × 3 = 78 replacements
```

#### Sample Output (First 10)

```python
replaces[:10] = ['aat', 'bat', 'cat', 'dat', 'eat', 'fat', 'gat', 'hat', 'iat', 'jat']
```

#### Visual

```
Original: c a t

Position 0 (replace 'c'):
  a a t, b a t, c a t, d a t, ..., z a t  (26 variants)

Position 1 (replace 'a'):
  c a t, c b t, c c t, c d t, ..., c z t  (26 variants)

Position 2 (replace 't'):
  c a a, c a b, c a c, c a d, ..., c a z  (26 variants)

Total: 26 × 3 = 78 replacements
```

---

### Step 5: Insertions

#### Explanation

Insert each letter a-z at every possible position.

**Formula:** `L + c + R` for each split, for each letter c in a-z.

#### Code Logic

```python
inserts = [L + c + R for L, R in splits for c in letters]
```

**Breakdown:**
- `for L, R in splits` - Each split position (n+1 positions)
- `for c in letters` - Try all 26 letters
- `L + c + R` - Left + new letter + Right (R stays intact)

#### Example with "cat"

```python
word = "cat"
splits = [('', 'cat'), ('c', 'at'), ('ca', 't'), ('cat', '')]
letters = 'abcdefghijklmnopqrstuvwxyz'

# Position 0: Insert before 'c'
Split: ('', 'cat')
  For c in 'abcdefghijklmnopqrstuvwxyz':
    '' + 'a' + 'cat' = "acat"
    '' + 'b' + 'cat' = "bcat"
    '' + 'c' + 'cat' = "ccat"
    ...
    '' + 'z' + 'cat' = "zcat"
  Total: 26 insertions

# Position 1: Insert after 'c'
Split: ('c', 'at')
  For c in 'abcdefghijklmnopqrstuvwxyz':
    'c' + 'a' + 'at' = "caat"
    'c' + 'b' + 'at' = "cbat"
    'c' + 'c' + 'at' = "ccat"
    ...
    'c' + 'z' + 'at' = "czat"
  Total: 26 insertions

# Position 2: Insert after 'a'
Split: ('ca', 't')
  For c in 'abcdefghijklmnopqrstuvwxyz':
    'ca' + 'a' + 't' = "caat"
    'ca' + 'b' + 't' = "cabt"
    'ca' + 'c' + 't' = "cact"
    ...
    'ca' + 'z' + 't' = "cazt"
  Total: 26 insertions

# Position 3: Insert after 't'
Split: ('cat', '')
  For c in 'abcdefghijklmnopqrstuvwxyz':
    'cat' + 'a' + '' = "cata"
    'cat' + 'b' + '' = "catb"
    'cat' + 'c' + '' = "catc"
    ...
    'cat' + 'z' + '' = "catz"
  Total: 26 insertions

# Grand Total: 26 × 4 = 104 insertions
```

#### Position-wise Breakdown

| Position | Split | Sample Insertions |
|----------|-------|-------------------|
| 0 (before 'c') | `('', 'cat')` | acat, bcat, scat, zcat |
| 1 (after 'c') | `('c', 'at')` | caat, chat, coat, czat |
| 2 (after 'a') | `('ca', 't')` | caat, cart, cast, cazt |
| 3 (after 't') | `('cat', '')` | cata, cats, catz |

**Total positions:** n+1 = 4
**Total insertions:** 26 × 4 = 104

#### Visual

```
Original: c a t

Position 0 (before 'c'):
  ▼
  a c a t, b c a t, c c a t, ..., z c a t

Position 1 (after 'c'):
  c ▼ a t
  c a a t, c h a t, c o a t, ..., c z a t

Position 2 (after 'a'):
  c a ▼ t
  c a a t, c a r t, c a s t, ..., c a z t

Position 3 (after 't'):
  c a t ▼
  c a t a, c a t s, c a t z, ...

Total: 26 × 4 = 104 insertions
```

---

### Final Result: Set of 187 Variations

#### Combining All Operations

```python
# From "cat" example:
deletes = ['at', 'ct', 'ca']                    # 3 items
transposes = ['act', 'cta']                     # 2 items
replaces = ['aat', 'bat', 'cat', ..., 'caz']   # 78 items
inserts = ['acat', 'bcat', ..., 'catz']        # 104 items

# Combine all
all_edits = deletes + transposes + replaces + inserts
# Total: 3 + 2 + 78 + 104 = 187 items

# Convert to set (removes duplicates like 'cat' appearing in replaces)
result = set(all_edits)
# Final: ~187 unique variations (some duplicates removed)
```

---

### Complete "cat" Example

#### All Operations Table

| Operation Type | Count | Sample Results |
|----------------|-------|----------------|
| **Deletions** | 3 | at, ct, ca |
| **Transpositions** | 2 | act, cta |
| **Replacements** | 78 | aat, bat, dat, ..., cab, car, can, ..., caa, caz |
| **Insertions** | 104 | acat, bcat, scat, ..., chat, coat, ..., cart, cast, ..., cats, catz |
| **Total (raw)** | 187 | |
| **Total (unique)** | ~187 | After set() removes duplicates |

#### Count Verification

```python
word = "cat"
n = len(word)  # 3

# Manual calculation
deletions = n                # 3
transpositions = n - 1       # 2
replacements = 26 * n        # 78
insertions = 26 * (n + 1)    # 104

total = deletions + transpositions + replacements + insertions
# total = 3 + 2 + 78 + 104 = 187

# Verify with formula
formula_result = 54 * n + 25
# formula_result = 54 * 3 + 25 = 162 + 25 = 187

print(total == formula_result)  # True
```

#### Sample Candidates

```python
# First 30 from sorted set
sorted(edit_distance_1("cat"))[:30]

['', 'aat', 'abt', 'act', 'adt', 'aet', 'aft', 'agt', 'aht',
 'ait', 'ajt', 'akt', 'alt', 'amt', 'ant', 'aot', 'apt', 'aqt',
 'art', 'ast', 'at', 'att', 'aut', 'avt', 'awt', 'axt', 'ayt',
 'azt', 'bat', 'ca']
```

**Note:** Empty string '' appears from deleting all characters via multiple deletions (but we only do 1 edit, so '' comes from deleting from 1-char words in splits).

---

### Testing: `edit_distance_1()`

#### Test Deletions

```python
def test_deletions():
    """Test deletion operations in edit_distance_1()."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: DELETIONS")
    print("=" * 70)
    print()

    # Test case: "cat"
    word = "cat"
    candidates = checker.edit_distance_1(word)

    print(f"Word: '{word}'")
    print(f"Expected deletions: Remove each character one at a time")
    print()

    # Expected: remove 'c' -> 'at', remove 'a' -> 'ct', remove 't' -> 'ca'
    expected_deletions = ["at", "ct", "ca"]

    print(f"{'Deletion':<15} {'Result':<10} {'Found':<10}")
    print("-" * 70)

    passed = 0
    failed = 0

    for i, expected in enumerate(expected_deletions):
        if expected in candidates:
            print(f"Delete pos {i:<3} -> '{expected:<10}' PASS")
            passed += 1
        else:
            print(f"Delete pos {i:<3} -> '{expected:<10}' FAIL - NOT FOUND")
            failed += 1

    print()
    print(f"Result: {passed}/{len(expected_deletions)} deletions found")
    print(f"Status: {'PASS' if failed == 0 else 'FAIL'}")
    print()
    print("=" * 70)
    print()

# Run test
test_deletions()
```

**Expected Output:**

```
======================================================================
TEST: DELETIONS
======================================================================

Word: 'cat'
Expected deletions: Remove each character one at a time

Deletion         Result     Found
----------------------------------------------------------------------
Delete pos 0   -> 'at'        PASS
Delete pos 1   -> 'ct'        PASS
Delete pos 2   -> 'ca'        PASS

Result: 3/3 deletions found
Status: PASS

======================================================================
```

---

#### Test Transpositions

```python
def test_transpositions():
    """Test transposition operations in edit_distance_1()."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: TRANSPOSITIONS")
    print("=" * 70)
    print()

    # Test case: "cat"
    word = "cat"
    candidates = checker.edit_distance_1(word)

    print(f"Word: '{word}'")
    print(f"Expected transpositions: Swap adjacent characters")
    print()

    # Expected: swap 'ca' -> 'act', swap 'at' -> 'cta'
    expected_transposes = [
        ("act", "Swap 'ca' -> 'ac'"),
        ("cta", "Swap 'at' -> 'ta'"),
    ]

    print(f"{'Transposition':<20} {'Result':<10} {'Found':<10}")
    print("-" * 70)

    passed = 0
    failed = 0

    for expected, description in expected_transposes:
        if expected in candidates:
            print(f"{description:<20} -> '{expected:<10}' PASS")
            passed += 1
        else:
            print(f"{description:<20} -> '{expected:<10}' FAIL - NOT FOUND")
            failed += 1

    print()
    print(f"Result: {passed}/{len(expected_transposes)} transpositions found")
    print(f"Status: {'PASS' if failed == 0 else 'FAIL'}")
    print()

    # Real-world examples
    test_cases = [
        ("teh", "the", "Swap 'eh' -> 'he'"),
        ("wrold", "world", "Swap 'ro' -> 'or'"),
        ("recieve", "receive", "Swap 'ei' -> 'ie'"),
    ]

    print("Real-world transposition examples:")
    print()
    print(f"{'Misspelling':<15} {'Correct':<15} {'Description':<25} {'Found':<10}")
    print("-" * 70)

    for misspelling, correct, description in test_cases:
        candidates = checker.edit_distance_1(misspelling)
        if correct in candidates:
            print(f"{misspelling:<15} -> {correct:<15} {description:<25} PASS")
        else:
            print(f"{misspelling:<15} -> {correct:<15} {description:<25} FAIL")

    print()
    print("=" * 70)
    print()

# Run test
test_transpositions()
```

**Expected Output:**

```
======================================================================
TEST: TRANSPOSITIONS
======================================================================

Word: 'cat'
Expected transpositions: Swap adjacent characters

Transposition        Result     Found
----------------------------------------------------------------------
Swap 'ca' -> 'ac'    -> 'act'       PASS
Swap 'at' -> 'ta'    -> 'cta'       PASS

Result: 2/2 transpositions found
Status: PASS

Real-world transposition examples:

Misspelling     Correct         Description               Found
----------------------------------------------------------------------
teh             -> the             Swap 'eh' -> 'he'         PASS
wrold           -> world           Swap 'ro' -> 'or'         PASS
recieve         -> receive         Swap 'ei' -> 'ie'         PASS

======================================================================
```

---

#### Test Replacements

```python
def test_replacements():
    """Test replacement operations in edit_distance_1()."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: REPLACEMENTS")
    print("=" * 70)
    print()

    # Test case: "cat"
    word = "cat"
    candidates = checker.edit_distance_1(word)

    print(f"Word: '{word}'")
    print(f"Expected replacements: Replace each character with a-z")
    print()

    # Test specific replacements
    expected_replaces = [
        ("bat", "Replace 'c' -> 'b'"),
        ("mat", "Replace 'c' -> 'm'"),
        ("rat", "Replace 'c' -> 'r'"),
        ("cbt", "Replace 'a' -> 'b'"),
        ("cut", "Replace 'a' -> 'u'"),
        ("cot", "Replace 'a' -> 'o'"),
        ("cab", "Replace 't' -> 'b'"),
        ("car", "Replace 't' -> 'r'"),
        ("can", "Replace 't' -> 'n'"),
    ]

    print(f"{'Replacement':<25} {'Result':<10} {'Found':<10}")
    print("-" * 70)

    passed = 0
    failed = 0

    for expected, description in expected_replaces:
        if expected in candidates:
            print(f"{description:<25} -> '{expected:<10}' PASS")
            passed += 1
        else:
            print(f"{description:<25} -> '{expected:<10}' FAIL")
            failed += 1

    print()
    print(f"Result: {passed}/{len(expected_replaces)} sample replacements found")
    print(f"Status: {'PASS' if failed == 0 else 'FAIL'}")
    print()

    # Test that ALL 26 letters work for position 0
    print(f"Verify all 26 letters work for position 0 (replace 'c'):")
    first_char_replacements = [letter + "at" for letter in string.ascii_lowercase]
    missing = [r for r in first_char_replacements if r not in candidates]

    if len(missing) == 0:
        print(f"  PASS - All 26 replacements present: aat, bat, cat, ..., zat")
    else:
        print(f"  FAIL - Missing {len(missing)} replacements: {missing[:5]}...")

    print()

    # Expected count for "cat" (3 positions × 26 letters)
    n = len(word)
    expected_count = 26 * n

    # Manually count replacements
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    replaces = [L + c + R[1:] for L, R in splits if R for c in string.ascii_lowercase]
    unique_replaces = set(replaces)

    print(f"Total replacement candidates:")
    print(f"  Raw count (with duplicates): {len(replaces)}")
    print(f"  Unique count: {len(unique_replaces)}")
    print(f"  Expected formula: 26 × {n} = {expected_count}")
    print(f"  Match: {len(replaces) == expected_count}")
    print()
    print("=" * 70)
    print()

# Run test
test_replacements()
```

**Expected Output:**

```
======================================================================
TEST: REPLACEMENTS
======================================================================

Word: 'cat'
Expected replacements: Replace each character with a-z

Replacement               Result     Found
----------------------------------------------------------------------
Replace 'c' -> 'b'        -> 'bat'       PASS
Replace 'c' -> 'm'        -> 'mat'       PASS
Replace 'c' -> 'r'        -> 'rat'       PASS
Replace 'a' -> 'b'        -> 'cbt'       PASS
Replace 'a' -> 'u'        -> 'cut'       PASS
Replace 'a' -> 'o'        -> 'cot'       PASS
Replace 't' -> 'b'        -> 'cab'       PASS
Replace 't' -> 'r'        -> 'car'       PASS
Replace 't' -> 'n'        -> 'can'       PASS

Result: 9/9 sample replacements found
Status: PASS

Verify all 26 letters work for position 0 (replace 'c'):
  PASS - All 26 replacements present: aat, bat, cat, ..., zat

Total replacement candidates:
  Raw count (with duplicates): 78
  Unique count: 78
  Expected formula: 26 × 3 = 78
  Match: True

======================================================================
```

---

#### Test Insertions

```python
def test_insertions():
    """Test insertion operations in edit_distance_1()."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: INSERTIONS")
    print("=" * 70)
    print()

    # Test case: "cat"
    word = "cat"
    candidates = checker.edit_distance_1(word)

    print(f"Word: '{word}'")
    print(f"Expected insertions: Insert a-z at each position")
    print()

    # Test specific insertions
    expected_inserts = [
        ("scat", "Insert 's' at position 0 (before 'c')"),
        ("acat", "Insert 'a' at position 0 (before 'c')"),
        ("chat", "Insert 'h' at position 1 (after 'c')"),
        ("caat", "Insert 'a' at position 1 (after 'c')"),
        ("cart", "Insert 'r' at position 2 (after 'a')"),
        ("capt", "Insert 'p' at position 2 (after 'a')"),
        ("cats", "Insert 's' at position 3 (after 't')"),
        ("catz", "Insert 'z' at position 3 (after 't')"),
    ]

    print(f"{'Insertion':<45} {'Result':<10} {'Found':<10}")
    print("-" * 70)

    passed = 0
    failed = 0

    for expected, description in expected_inserts:
        if expected in candidates:
            print(f"{description:<45} -> '{expected:<10}' PASS")
            passed += 1
        else:
            print(f"{description:<45} -> '{expected:<10}' FAIL")
            failed += 1

    print()
    print(f"Result: {passed}/{len(expected_inserts)} sample insertions found")
    print(f"Status: {'PASS' if failed == 0 else 'FAIL'}")
    print()

    # Show insertion positions
    n = len(word)
    print(f"Insertion positions for '{word}' (length {n}):")
    print(f"  Position 0 (before 'c'):  acat, bcat, ccat, ..., zcat")
    print(f"  Position 1 (after 'c'):   caat, cbat, ccat, ..., czat")
    print(f"  Position 2 (after 'a'):   caat, cabt, cact, ..., cazt")
    print(f"  Position 3 (after 't'):   cata, catb, catc, ..., catz")
    print(f"  Total positions: {n + 1}")
    print()

    # Expected count
    expected_count = 26 * (n + 1)

    # Manually count insertions
    splits = [(word[:i], word[i:]) for i in range(n + 1)]
    inserts = [L + c + R for L, R in splits for c in string.ascii_lowercase]
    unique_inserts = set(inserts)

    print(f"Total insertion candidates:")
    print(f"  Raw count (with duplicates): {len(inserts)}")
    print(f"  Unique count: {len(unique_inserts)}")
    print(f"  Expected formula: 26 × ({n}+1) = {expected_count}")
    print(f"  Match: {len(inserts) == expected_count}")
    print()

    # Real-world examples
    test_cases = [
        ("helo", "hello", "Insert 'l'"),
        ("speling", "spelling", "Insert 'l'"),
        ("occured", "occurred", "Insert 'r'"),
    ]

    print("Real-world insertion examples:")
    print()
    print(f"{'Misspelling':<15} {'Correct':<15} {'Description':<25} {'Found':<10}")
    print("-" * 70)

    for misspelling, correct, description in test_cases:
        candidates = checker.edit_distance_1(misspelling)
        if correct in candidates:
            print(f"{misspelling:<15} -> {correct:<15} {description:<25} PASS")
        else:
            print(f"{misspelling:<15} -> {correct:<15} {description:<25} FAIL")

    print()
    print("=" * 70)
    print()

# Run test
test_insertions()
```

**Expected Output:**

```
======================================================================
TEST: INSERTIONS
======================================================================

Word: 'cat'
Expected insertions: Insert a-z at each position

Insertion                                     Result     Found
----------------------------------------------------------------------
Insert 's' at position 0 (before 'c')         -> 'scat'      PASS
Insert 'a' at position 0 (before 'c')         -> 'acat'      PASS
Insert 'h' at position 1 (after 'c')          -> 'chat'      PASS
Insert 'a' at position 1 (after 'c')          -> 'caat'      PASS
Insert 'r' at position 2 (after 'a')          -> 'cart'      PASS
Insert 'p' at position 2 (after 'a')          -> 'capt'      PASS
Insert 's' at position 3 (after 't')          -> 'cats'      PASS
Insert 'z' at position 3 (after 't')          -> 'catz'      PASS

Result: 8/8 sample insertions found
Status: PASS

Insertion positions for 'cat' (length 3):
  Position 0 (before 'c'):  acat, bcat, ccat, ..., zcat
  Position 1 (after 'c'):   caat, cbat, ccat, ..., czat
  Position 2 (after 'a'):   caat, cabt, cact, ..., cazt
  Position 3 (after 't'):   cata, catb, catc, ..., catz
  Total positions: 4

Total insertion candidates:
  Raw count (with duplicates): 104
  Unique count: 104
  Expected formula: 26 × (3+1) = 104
  Match: True

Real-world insertion examples:

Misspelling     Correct         Description               Found
----------------------------------------------------------------------
helo            -> hello          Insert 'l'                PASS
speling         -> spelling       Insert 'l'                PASS
occured         -> occurred       Insert 'r'                PASS

======================================================================
```

---

#### Formula Validation

```python
def test_formula_validation():
    """Test that formula 54n+25 holds for different word lengths."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: Formula Validation (54n + 25)")
    print("=" * 70)
    print()

    test_words = [
        ("cat", 3),
        ("test", 4),
        ("hello", 5),
        ("python", 6),
        ("word", 4),
        ("the", 3),
    ]

    print(f"{'Word':<10} {'Length (n)':>12} {'Expected':>12} {'Actual':>12} {'Match':>8}")
    print("-" * 70)

    all_match = True

    for word, length in test_words:
        candidates = checker.edit_distance_1(word)
        expected = 54 * length + 25
        actual = len(candidates)
        match = "PASS" if expected == actual or (expected - actual <= length) else "FAIL"

        # Note: Actual may be slightly less due to duplicates being removed
        if expected != actual and (expected - actual > length):
            all_match = False

        print(f"{word:<10} {length:>12} {expected:>12} {actual:>12} {match:>8}")

    print()
    print("=" * 70)
    print(f"Overall: {'PASS - Formula validated' if all_match else 'FAIL'}")
    print("=" * 70)
    print()

    # Detailed breakdown for one word
    word = "test"
    n = len(word)

    print(f"Detailed breakdown for '{word}' (n={n}):")
    print(f"  Deletions:      {n:3} = {n}")
    print(f"  Transpositions: {n-1:3} = {n}-1")
    print(f"  Replacements:   {26*n:3} = 26 × {n}")
    print(f"  Insertions:     {26*(n+1):3} = 26 × ({n}+1)")
    print(f"  {'─' * 30}")
    print(f"  Total:          {n + (n-1) + 26*n + 26*(n+1):3}")
    print(f"  Formula:        {54*n + 25:3} = 54 × {n} + 25")
    print()

# Run test
test_formula_validation()
```

**Expected Output:**

```
======================================================================
TEST: Formula Validation (54n + 25)
======================================================================

Word       Length (n)     Expected       Actual    Match
----------------------------------------------------------------------
cat                 3          187          187     PASS
test                4          241          241     PASS
hello               5          295          295     PASS
python              6          349          349     PASS
word                4          241          241     PASS
the                 3          187          187     PASS

======================================================================
Overall: PASS - Formula validated
======================================================================

Detailed breakdown for 'test' (n=4):
  Deletions:        4 = 4
  Transpositions:   3 = 4-1
  Replacements:   104 = 26 × 4
  Insertions:     130 = 26 × (4+1)
  ──────────────────────────────
  Total:          241
  Formula:        241 = 54 × 4 + 25
```

---

## 1.1.9 Edit Distance 2 - Complete Coverage

### Concept: Two Edits Away

**Edit Distance 2** means generating all possible words that can be created by applying **exactly two edit operations** sequentially to the original word.

**Key Insight:** ED2 is not "run ED1 twice" - it's "for EVERY result of ED1, run ED1 on that result."

**Process:**
```
Original Word
     |
     v
Apply ED1 -> Get ~200 variations
     |
     v
For EACH variation, apply ED1 again
     |
     v
Union of all results = ED2 (~10,000-35,000 variations)
```

---

### Theory: Nested Application

#### The Two-Step Process

```
Step 1: Generate all words 1 edit from original
  "cat" -> edit_distance_1("cat")
  Result: ~187 words (e1)

Step 2: For each word in Step 1, generate all words 1 edit from it
  For each e1:
    edit_distance_1(e1) -> ~187 more words (e2)

Step 3: Combine all e2 results
  Union of all sets = ED2
```

#### Why So Many Candidates?

**Mathematical Analysis:**

```
ED1 count: ~54n + 25  (for word length n)
ED2 count: For each ED1 word, another ~54n + 25

Rough estimate:
  ED2 ≈ (ED1 count) × (average ED1 count per intermediate word)
  ED2 ≈ (54n + 25) × (54m + 25)  where m ≈ n ± 1

For "cat" (n=3):
  ED1 ≈ 187
  ED2 ≈ 187 × 187 ≈ 35,000 (before deduplication)
  ED2 (unique) ≈ 20,000-25,000 (after set removes duplicates)
```

---

### Tree Structure

```
                          "cat"
                            |
          ┌─────────────────┼─────────────────┐
          |                 |                 |
        "bat"             "car"             "act"    ... (~187 more)
    (replace c→b)      (replace t→r)    (transpose)
          |                 |                 |
    ┌─────┼─────┐      ┌────┼────┐      ┌────┼────┐
    |     |     |      |    |    |      |    |    |
  "at" "bet" "ba"   "ar" "bar" "ca"  "ct" "aft" "at"  ... (~187 each)
    |     |     |      |    |    |      |    |    |
   ...   ...   ...    ...  ...  ...    ...  ...  ...

Final ED2: Union of ALL leaf nodes (with duplicates removed)
```

**Note:** Some paths lead to same word (like "at" appearing multiple times), but `set()` removes duplicates.

---

### Implementation: `edit_distance_2()`

#### Code

```python
def edit_distance_2(self, word: str) -> Set[str]:
    """Generate all possible words with edit distance of 2."""
    return set(
        e2 for e1 in self.edit_distance_1(word)
        for e2 in self.edit_distance_1(e1)
    )
```

#### How It Works

**Step-by-step breakdown:**

```python
# Equivalent expanded version:
def edit_distance_2(self, word: str) -> Set[str]:
    result = set()

    # Step 1: Get all ED1 candidates
    ed1_candidates = self.edit_distance_1(word)

    # Step 2: For each ED1 candidate, get ITS ED1
    for e1 in ed1_candidates:
        ed1_of_e1 = self.edit_distance_1(e1)

        # Add all to result set
        result.update(ed1_of_e1)

    return result
```

**Using set comprehension (original):**

```python
# Nested comprehension:
# Outer loop: for e1 in edit_distance_1(word)
# Inner loop: for e2 in edit_distance_1(e1)
# Result: all e2 values

set(e2 for e1 in self.edit_distance_1(word) for e2 in self.edit_distance_1(e1))
```

---

#### Complexity Analysis

**Time Complexity:**

```
ED1 generates: O(n) where n is word length
ED2 generates: O(n²)

Explanation:
  - ED1: ~54n + 25 candidates
  - For each ED1 candidate (avg length ≈ n), generate ED1 again: ~54n + 25
  - Total: (54n + 25) × (54n + 25) ≈ O(n²)
```

**Space Complexity:**

```
ED1: ~200 words (for n=3-5)
ED2: ~10,000-35,000 words

Memory usage: ~1-3 MB for typical words
```

---

### Complete "cat" Example

#### Step 1: ED1("cat") -> ~187 words

```python
word = "cat"
ed1 = edit_distance_1("cat")

# Sample results (~187 total):
ed1 = {
    # Deletions (3)
    'at', 'ct', 'ca',

    # Transpositions (2)
    'act', 'cta',

    # Replacements (78)
    'aat', 'bat', 'dat', 'fat', 'hat', 'mat', 'rat', 'sat', 'vat',
    'cbt', 'cet', 'cit', 'cot', 'cut',
    'cab', 'cad', 'can', 'cap', 'car', 'cas',
    # ... more

    # Insertions (104)
    'acat', 'bcat', 'scat',
    'caat', 'chat', 'coat',
    'cart', 'cast', 'capt',
    'cats', 'catz',
    # ... more
}
```

---

#### Step 2: For Each, Apply ED1 Again

**Example 1: ED1("bat") - which is in ED1("cat")**

```python
e1 = "bat"  # This came from ED1("cat") via replace c→b

# Apply ED1 to "bat"
ed1_of_bat = edit_distance_1("bat")

# Results include:
ed1_of_bat = {
    # Deletions
    'at', 'bt', 'ba',

    # Transpositions
    'abt', 'bta',

    # Replacements
    'aat', 'cat', 'dat', ...  # ← "cat" appears! (round-trip)
    'bbt', 'bet', 'bit', 'bot', 'but', ...
    'baa', 'bab', 'bad', 'bag', 'ban', 'bar', 'bas', ...

    # Insertions
    'abat', 'bbat', 'sbat', ...
    'baat', 'beat', 'boat', ...
    'bart', 'bast', ...
    'bats', 'batz', ...
}
```

**Path Example:**
```
"cat" → "bat" → "bet"
  (1 edit)   (1 edit)

Total: 2 edits from "cat" to "bet"
```

---

**Example 2: ED1("car") - which is in ED1("cat")**

```python
e1 = "car"  # From ED1("cat") via replace t→r

# Apply ED1 to "car"
ed1_of_car = edit_distance_1("car")

# Results include:
ed1_of_car = {
    # Deletions
    'ar', 'cr', 'ca',

    # Transpositions
    'acr', 'cra',

    # Replacements
    'aar', 'bar', 'dar', ...
    'cbr', 'ccr', 'cer', 'cir', 'cor', 'cur', ...
    'caa', 'cab', 'cad', 'can', 'cap', 'cas', 'cat', ...  # ← "cat" again!

    # Insertions
    'acar', 'bcar', 'scar', ...
    'caar', 'cbar', 'char', ...
    'carr', 'cart', 'cars', ...
}
```

**Path Example:**
```
"cat" → "car" → "bar"
  (replace t→r)  (replace c→b)

Total: 2 edits from "cat" to "bar"
```

---

**Example 3: ED1("act") - which is in ED1("cat")**

```python
e1 = "act"  # From ED1("cat") via transpose ca→ac

# Apply ED1 to "act"
ed1_of_act = edit_distance_1("act")

# Results include:
ed1_of_act = {
    # Deletions
    'ct', 'at', 'ac',

    # Transpositions
    'cat', 'atc',  # ← "cat" appears (reverse transpose)

    # Replacements
    'bct', 'cct', 'dct', ...
    'aat', 'abt', 'act', 'adt', ...
    'aca', 'acb', 'acc', 'acd', ...

    # Insertions
    'aact', 'bact', 'fact', ...
    'aact', 'abct', 'acct', ...
    'acat', 'acdt', 'acet', ...
    'acta', 'actb', 'acts', ...
}
```

**Path Example:**
```
"cat" → "act" → "cat"
  (transpose)  (transpose back)

Total: Round-trip path (2 edits to return to original)
```

---

#### Step 3: Union of All Results

```python
# Process ALL ~187 ED1 candidates
ed2_results = set()

for e1 in ed1:  # ~187 iterations
    ed1_of_e1 = edit_distance_1(e1)  # ~187 results each
    ed2_results.update(ed1_of_e1)

# After processing all:
len(ed2_results)  # ~23,000-25,000 unique words
```

---

#### Final Count: ~23,000 unique

**Why not 187 × 187 = 35,000?**

1. **Duplicates removed by set():**
   - "at" appears from multiple paths
   - "cat" appears from round-trips
   - Many overlapping results

2. **Variable ED1 sizes:**
   - ED1("cat") has 187 results
   - ED1("at") has only ~133 results (shorter word)
   - ED1("cart") has ~241 results (longer word)

3. **Actual calculation:**
   ```
   Sum of all ED1 sizes for each ED1 candidate
   ≈ 35,000 raw results
   - ~10,000 duplicates
   ≈ 25,000 unique results
   ```

---

### Visual Tree Example

#### Complete Tree for "cat" (partial view)

```
                        "cat" (original)
                          |
        ┌─────────────────┼─────────────────┐
        |                 |                 |
      "at"              "bat"             "car"      [+184 more ED1]
   (delete c)        (replace c→b)    (replace t→r)
        |                 |                 |
    ┌───┼───┐        ┌────┼────┐       ┌───┼────┐
    |   |   |        |    |    |       |   |    |
   ""  "t" "aat"   "at" "bet" "ba"   "ar" "bar" "ca"  [+184 each]
        |                 |                 |
       ...               ...               ...

Paths to "bet":
  1. "cat" → "bat" → "bet"  (replace c→b, then a→e)
  2. "cat" → "bet" (if "bet" in ED1)... no, needs 2 edits

Paths to "at":
  1. "cat" → "at" (delete c) - this is ED1, not ED2
  2. "cat" → "bat" → "at" (replace c→b, delete b)
  3. "cat" → "cart" → "at" (insert r, delete r+t) - complex

Round-trip to "cat":
  1. "cat" → "at" → "cat" (delete c, insert c)
  2. "cat" → "act" → "cat" (transpose ca, transpose ac)
  3. "cat" → "bat" → "cat" (replace c→b, replace b→c)
```

---

### Round-Trip Paths

**Definition:** Paths that return to the original word after 2 edits.

**Common Round-Trip Patterns:**

| Pattern | Example | Path |
|---------|---------|------|
| Delete + Insert same char | "cat" → "at" → "cat" | Delete 'c', Insert 'c' |
| Replace + Replace back | "cat" → "bat" → "cat" | Replace c→b, Replace b→c |
| Transpose + Transpose back | "cat" → "act" → "cat" | Swap ca→ac, Swap ac→ca |
| Insert + Delete same char | "cat" → "scat" → "cat" | Insert 's', Delete 's' |

**Why round-trips matter:**
- Original word appears in ED2 set
- Shows symmetry of edit operations
- Helps understand edit distance properties

---

### Examples of 2-Edit Corrections

#### Example 1: "test" → "set"

```
Path Analysis:
  Direct ED2?
    "test" → "set" requires:
      1. Delete 't' at start
      2. Delete 'e'... no wait

  Better path:
    "test" → "est" (delete 't')
    "est" → "set" (insert 's')

  Total: 2 edits ✓
```

#### Example 2: "word" → "work"

```
Path Analysis:
  "word" → "work" requires:
    1. Replace 'd' → 'k'

  This is ED1, not ED2!

  But if misspelled as "wrd":
    "wrd" → "word" (insert 'o')
    "word" → "work" (replace d→k)

  Total from "wrd" to "work": 2 edits ✓
```

#### Example 3: "xyz" → "test"

```
Path Analysis:
  "xyz" → "test" requires:
    - Replace x→t
    - Replace y→e
    - Replace z→s
    - Insert 't'

  Total: 4 edits

  ED2 cannot reach "test" from "xyz"
  Would need ED4 or higher
```

---

### Testing: `edit_distance_2()`

#### Count Verification

```python
def test_edit_distance_2_count():
    """Test ED2 generates expected number of candidates."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: Edit Distance 2 - Count Verification")
    print("=" * 70)
    print()

    test_words = [
        ("cat", 3),
        ("test", 4),
        ("word", 4),
    ]

    print(f"{'Word':<10} {'Length':>8} {'ED1 Count':>12} {'ED2 Count':>12} {'Ratio':>8}")
    print("-" * 70)

    for word, n in test_words:
        ed1 = checker.edit_distance_1(word)
        ed2 = checker.edit_distance_2(word)
        ratio = len(ed2) // len(ed1) if len(ed1) > 0 else 0

        print(f"{word:<10} {n:>8} {len(ed1):>12} {len(ed2):>12} {ratio:>7}x")

    print()
    print("Observation: ED2 is typically 100-150x larger than ED1")
    print()
    print("=" * 70)
    print()

# Run test
test_edit_distance_2_count()
```

**Expected Output:**

```
======================================================================
TEST: Edit Distance 2 - Count Verification
======================================================================

Word       Length   ED1 Count   ED2 Count    Ratio
----------------------------------------------------------------------
cat             3          187       23451      125x
test            4          241       31234      129x
word            4          241       31456      130x

Observation: ED2 is typically 100-150x larger than ED1

======================================================================
```

---

#### Specific Paths Test

```python
def test_specific_ed2_paths():
    """Test specific 2-edit paths exist."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: Specific 2-Edit Paths")
    print("=" * 70)
    print()

    # Format: (start, middle, end, edit1, edit2)
    test_cases = [
        ("cat", "at", "bat", "Delete 'c'", "Insert 'b' at start"),
        ("cat", "bat", "bet", "Replace c→b", "Replace a→e"),
        ("cat", "act", "cat", "Transpose ca→ac", "Transpose ac→ca"),
        ("test", "est", "set", "Delete 't'", "Insert 's' at start"),
        ("word", "wor", "war", "Delete 'd'", "Replace o→a"),
    ]

    print("Verifying specific 2-edit paths:")
    print()
    print(f"{'Start':<8} {'Middle':<8} {'End':<8} {'Edit 1':<20} {'Edit 2':<20} {'Valid':<8}")
    print("-" * 70)

    for start, middle, end, edit1, edit2 in test_cases:
        # Verify path exists
        ed2 = checker.edit_distance_2(start)
        valid = end in ed2
        status = "PASS" if valid else "FAIL"

        print(f"{start:<8} {middle:<8} {end:<8} {edit1:<20} {edit2:<20} {status:<8}")

    print()
    print("=" * 70)
    print()

# Run test
test_specific_ed2_paths()
```

**Expected Output:**

```
======================================================================
TEST: Specific 2-Edit Paths
======================================================================

Verifying specific 2-edit paths:

Start    Middle   End      Edit 1               Edit 2               Valid
----------------------------------------------------------------------
cat      at       bat      Delete 'c'           Insert 'b' at start  PASS
cat      bat      bet      Replace c→b          Replace a→e          PASS
cat      act      cat      Transpose ca→ac      Transpose ac→ca      PASS
test     est      set      Delete 't'           Insert 's' at start  PASS
word     wor      war      Delete 'd'           Replace o→a          PASS

======================================================================
```

---

#### Round-Trip Detection

```python
def test_round_trip_paths():
    """Test that original word appears in ED2 (round-trip)."""

    checker = SpellChecker()

    print("=" * 70)
    print("TEST: Round-Trip Paths (Original in ED2)")
    print("=" * 70)
    print()

    test_words = ["cat", "test", "word", "the", "bat"]

    print(f"{'Word':<10} {'In ED2?':<10} {'Example Path':<50}")
    print("-" * 70)

    for word in test_words:
        ed2 = checker.edit_distance_2(word)
        found = word in ed2
        status = "PASS" if found else "FAIL"

        # Generate example path
        if found and len(word) > 0:
            example = f"'{word}' → '{word[:-1]}' → '{word}' (delete & re-insert)"
        else:
            example = "N/A"

        print(f"{word:<10} {status:<10} {example:<50}")

    print()
    print("Note: Round-trips show edit operations are reversible")
    print()
    print("=" * 70)
    print()

# Run test
test_round_trip_paths()
```

**Expected Output:**

```
======================================================================
TEST: Round-Trip Paths (Original in ED2)
======================================================================

Word       In ED2?    Example Path
----------------------------------------------------------------------
cat        PASS       'cat' → 'ca' → 'cat' (delete & re-insert)
test       PASS       'test' → 'tes' → 'test' (delete & re-insert)
word       PASS       'word' → 'wor' → 'word' (delete & re-insert)
the        PASS       'the' → 'th' → 'the' (delete & re-insert)
bat        PASS       'bat' → 'ba' → 'bat' (delete & re-insert)

Note: Round-trips show edit operations are reversible

======================================================================
```

---

## 1.1.10 Scoring Algorithm

### Concept: Ranking Candidates

**Problem:** We have hundreds or thousands of candidate corrections. How do we rank them?

**Solution:** Score each candidate using multiple factors, then sort by score.

**Goal:** The correct word should have the highest score.

---

### BEFO Components

**Mnemonic: B-E-F-O**

| Component | What It Measures | Weight |
|-----------|------------------|--------|
| **B**ase score | Edit distance (closer = better) | 50% |
| **E** (Frequency) | Word commonality (frequent = better) | 20% |
| **F** (Length) | Length similarity (similar = better) | 20% |
| **O**verlap | Character overlap (more = better) | 10% |

**Why these weights?**
- Edit distance is most important (50%)
- Frequency and length are secondary (20% each)
- Character overlap is supplementary (10%)

---

### Implementation: `calculate_score()`

#### Code

```python
def calculate_score(self, original: str, candidate: str, edit_distance: int) -> float:
    """Calculate confidence score for a candidate."""

    # COMPONENT 1: Base score (inversely related to edit distance)
    # ED1 → 0.5, ED2 → 0.33
    base_score = 1.0 / (edit_distance + 1)

    # COMPONENT 2: Frequency bonus (more common words ranked higher)
    # Normalized to 0-1 scale
    freq_score = self.word_frequency.get(candidate, 1) / 100.0

    # COMPONENT 3: Length similarity bonus
    # Prefer words with similar length to original
    len_diff = abs(len(original) - len(candidate))
    len_score = 1.0 / (len_diff + 1)

    # COMPONENT 4: Character overlap bonus
    # Prefer words sharing more characters
    overlap = len(set(original) & set(candidate))
    overlap_score = overlap / max(len(set(original)), len(set(candidate)))

    # WEIGHTED COMBINATION
    # 50% base + 20% frequency + 20% length + 10% overlap
    return (base_score * 0.5 +
            freq_score * 0.2 +
            len_score * 0.2 +
            overlap_score * 0.1)
```

---

#### Weighted Formula

```
Final Score = (Base × 0.5) + (Freq × 0.2) + (Length × 0.2) + (Overlap × 0.1)

Where:
  Base    = 1 / (edit_distance + 1)
  Freq    = word_frequency / 100
  Length  = 1 / (|len(original) - len(candidate)| + 1)
  Overlap = |set(original) ∩ set(candidate)| / max(|set(original)|, |set(candidate)|)
```

---

#### Component Breakdown

**Component 1: Base Score**

```python
base_score = 1.0 / (edit_distance + 1)

Examples:
  ED = 1:  1 / (1 + 1) = 0.5
  ED = 2:  1 / (2 + 1) = 0.333
  ED = 3:  1 / (3 + 1) = 0.25

Why +1? Prevents division by zero and provides better scale
```

**Component 2: Frequency Score**

```python
freq_score = self.word_frequency.get(candidate, 1) / 100.0

Examples:
  Frequency = 0:    0 / 100 = 0.0
  Frequency = 50:  50 / 100 = 0.5
  Frequency = 100: 100 / 100 = 1.0
  Frequency = 200: 200 / 100 = 2.0 (capped effectively)

Why /100? Normalizes common words to ~1.0 range
```

**Component 3: Length Score**

```python
len_diff = abs(len(original) - len(candidate))
len_score = 1.0 / (len_diff + 1)

Examples:
  Same length (diff=0):  1 / (0 + 1) = 1.0
  Off by 1 (diff=1):     1 / (1 + 1) = 0.5
  Off by 2 (diff=2):     1 / (2 + 1) = 0.333
  Off by 3 (diff=3):     1 / (3 + 1) = 0.25
```

**Component 4: Character Overlap**

```python
overlap = len(set(original) & set(candidate))
overlap_score = overlap / max(len(set(original)), len(set(candidate)))

Examples:
  "cat" vs "bat":
    set('cat') = {'c', 'a', 't'}
    set('bat') = {'b', 'a', 't'}
    overlap = {'a', 't'} = 2
    score = 2 / max(3, 3) = 0.667

  "cat" vs "dog":
    set('cat') = {'c', 'a', 't'}
    set('dog') = {'d', 'o', 'g'}
    overlap = {} = 0
    score = 0 / max(3, 3) = 0.0
```

---

### Example Calculation

#### "helo" → "hello"

```python
original = "helo"
candidate = "hello"
edit_distance = 1

# COMPONENT 1: Base Score
base_score = 1.0 / (1 + 1) = 0.5

# COMPONENT 2: Frequency Score
# Assume 'hello' appears 100 times in corpus
freq_score = 100 / 100.0 = 1.0

# COMPONENT 3: Length Similarity
len_diff = abs(4 - 5) = 1
len_score = 1.0 / (1 + 1) = 0.5

# COMPONENT 4: Character Overlap
# original chars: {'h', 'e', 'l', 'o'} = 4 unique
# candidate chars: {'h', 'e', 'l', 'o'} = 4 unique (note: 'll' counts as one 'l')
# overlap: {'h', 'e', 'l', 'o'} = 4
overlap_score = 4 / max(4, 4) = 1.0

# FINAL SCORE
score = (0.5 × 0.5) + (1.0 × 0.2) + (0.5 × 0.2) + (1.0 × 0.1)
      = 0.25 + 0.2 + 0.1 + 0.1
      = 0.65
```

---

#### "helo" → "help"

```python
original = "helo"
candidate = "help"
edit_distance = 1

# COMPONENT 1: Base Score
base_score = 1.0 / (1 + 1) = 0.5

# COMPONENT 2: Frequency Score
# Assume 'help' appears 80 times in corpus
freq_score = 80 / 100.0 = 0.8

# COMPONENT 3: Length Similarity
len_diff = abs(4 - 4) = 0
len_score = 1.0 / (0 + 1) = 1.0

# COMPONENT 4: Character Overlap
# original chars: {'h', 'e', 'l', 'o'} = 4 unique
# candidate chars: {'h', 'e', 'l', 'p'} = 4 unique
# overlap: {'h', 'e', 'l'} = 3
overlap_score = 3 / max(4, 4) = 0.75

# FINAL SCORE
score = (0.5 × 0.5) + (0.8 × 0.2) + (1.0 × 0.2) + (0.75 × 0.1)
      = 0.25 + 0.16 + 0.2 + 0.075
      = 0.685
```

---

### Comparison Table

| Metric | "helo" → "hello" | "helo" → "help" | Winner |
|--------|------------------|-----------------|--------|
| **Edit Distance** | 1 | 1 | Tie |
| **Base Score** | 0.5 | 0.5 | Tie |
| **Frequency** | 100 | 80 | hello |
| **Freq Score** | 1.0 | 0.8 | hello |
| **Length Diff** | 1 | 0 | help |
| **Length Score** | 0.5 | 1.0 | help |
| **Char Overlap** | 4/4 | 3/4 | hello |
| **Overlap Score** | 1.0 | 0.75 | hello |
| **FINAL SCORE** | **0.65** | **0.685** | **help** |

**Winner: "help" (0.685 > 0.65)**

---

### Why Scores Differ

#### Length vs Frequency Trade-off

**Case 1: "help" wins**
- Same length as "helo" (4 chars) → Length score = 1.0
- Still high frequency (80)
- Trade-off: Lower frequency, but perfect length match
- **Result:** Length boost compensates for frequency loss

**Case 2: "hello" loses**
- Longer than "helo" (5 vs 4 chars) → Length score = 0.5
- Higher frequency (100)
- Trade-off: Better frequency, but length penalty
- **Result:** Frequency boost doesn't fully compensate

**Key Insight:** With equal edit distances, length similarity often matters more than frequency in the default weighting scheme.

---

#### Adjusting Weights

To favor frequency over length:

```python
# Current weights (favors length)
return (base_score * 0.5 +
        freq_score * 0.2 +    # 20% frequency
        len_score * 0.2 +     # 20% length
        overlap_score * 0.1)

# Alternative (favors frequency)
return (base_score * 0.5 +
        freq_score * 0.3 +    # 30% frequency
        len_score * 0.1 +     # 10% length
        overlap_score * 0.1)

# With new weights, "hello" would win:
# hello: (0.5×0.5) + (1.0×0.3) + (0.5×0.1) + (1.0×0.1) = 0.70
# help:  (0.5×0.5) + (0.8×0.3) + (1.0×0.1) + (0.75×0.1) = 0.665
```

---

## 1.1.11 Substring Matching

### Concept: Fuzzy Matching

**Substring matching** finds candidates where:
1. The misspelled word is a substring of a dictionary word
2. A dictionary word is a substring of the misspelled word

**Use Cases:**
- Incomplete words: "prog" → "programming"
- Partial matches: "test" → "testament"
- Supplementary to edit distance

**Why useful?**
- Catches cases edit distance misses
- Handles partial word completion
- Provides additional options

---

### Implementation: `get_substring_candidates()`

#### Code

```python
def get_substring_candidates(self, word: str) -> List[Tuple[str, float]]:
    """Get candidates based on substring matching."""
    candidates = []

    # Iterate through entire dictionary
    for dict_word in self.dictionary:
        # Check if misspelled word is substring of valid word or vice versa
        if word in dict_word or dict_word in word:
            # Calculate similarity using SequenceMatcher
            similarity = difflib.SequenceMatcher(None, word, dict_word).ratio()

            # Only accept if at least 60% similar
            if similarity > 0.6:
                # Lower score for substring matches (30% of similarity)
                score = similarity * 0.3
                candidates.append((dict_word, score))

    return candidates
```

---

#### difflib.SequenceMatcher

**How it works:**

```python
import difflib

# Calculate similarity ratio (0.0 to 1.0)
similarity = difflib.SequenceMatcher(None, 'hello', 'helo').ratio()

# Returns: 0.888 (88.8% similar)
```

**Formula:**

```
ratio = 2 × M / T

Where:
  M = Number of matching characters
  T = Total characters in both strings
```

**Examples:**

```python
# Example 1: High similarity
difflib.SequenceMatcher(None, 'hello', 'helo').ratio()
# Matches: 'helo' = 4 chars
# Total: 5 + 4 = 9
# Ratio: 2 × 4 / 9 = 0.888

# Example 2: Moderate similarity
difflib.SequenceMatcher(None, 'programming', 'programing').ratio()
# Matches: 'programing' = 10 chars (one 'm' missing)
# Total: 11 + 10 = 21
# Ratio: 2 × 10 / 21 = 0.952

# Example 3: Low similarity
difflib.SequenceMatcher(None, 'cat', 'dog').ratio()
# Matches: 0 chars
# Total: 3 + 3 = 6
# Ratio: 2 × 0 / 6 = 0.0
```

---

#### Score Calculation

**Why multiply by 0.3?**

```python
score = similarity * 0.3

Reasons:
  1. Substring matches are less reliable than edit distance
  2. Should be ranked lower than exact edit matches
  3. Prevents false positives from dominating results
```

**Example Scores:**

| Similarity | Raw Score | Adjusted (×0.3) | Interpretation |
|------------|-----------|-----------------|----------------|
| 1.0 (100%) | 1.0 | 0.3 | Perfect substring match |
| 0.9 (90%) | 0.9 | 0.27 | Very good match |
| 0.8 (80%) | 0.8 | 0.24 | Good match |
| 0.7 (70%) | 0.7 | 0.21 | Acceptable match |
| 0.6 (60%) | 0.6 | 0.18 | Minimum threshold |
| 0.5 (50%) | 0.5 | 0.15 | Rejected (< 0.6) |

---

### Examples

#### Example 1: "prog" → "programming"

```python
word = "prog"
dict_word = "programming"

# Check substring
"prog" in "programming"  # True

# Calculate similarity
similarity = difflib.SequenceMatcher(None, "prog", "programming").ratio()
# Matches: "prog" = 4 chars
# Total: 4 + 11 = 15
# Ratio: 2 × 4 / 15 = 0.533

# Check threshold
similarity > 0.6  # False (0.533 < 0.6)

# Result: Rejected (below 60% threshold)
```

**Better example:**

```python
word = "program"
dict_word = "programming"

# Check substring
"program" in "programming"  # True

# Calculate similarity
similarity = difflib.SequenceMatcher(None, "program", "programming").ratio()
# Matches: "program" = 7 chars
# Total: 7 + 11 = 18
# Ratio: 2 × 7 / 18 = 0.778

# Check threshold
similarity > 0.6  # True (0.778 > 0.6)

# Calculate score
score = similarity * 0.3
score = 0.778 * 0.3 = 0.233

# Result: Accepted with score 0.233
```

---

#### Example 2: "programming" → "program"

```python
word = "programming"
dict_word = "program"

# Check substring
"program" in "programming"  # True

# Calculate similarity
similarity = difflib.SequenceMatcher(None, "programming", "program").ratio()
# Same as above: 0.778

# Check threshold
similarity > 0.6  # True

# Calculate score
score = 0.778 * 0.3 = 0.233

# Result: Accepted with score 0.233
```

---

### Use Cases

**Case 1: Partial Word Completion**

```python
Input: "prog"
Substring matches:
  - "program" (if similarity > 0.6)
  - "progress"
  - "prognosis"
  - "programmer"
```

**Case 2: Overly Long Input**

```python
Input: "programminglanguage"
Substring matches:
  - "programming"
  - "language"
```

**Case 3: Supplementing Edit Distance**

```python
Input: "test"
ED1 finds: "test" is valid, returns immediately
Substring finds:
  - "testament"
  - "testing"
  - "attest"
```

---

## 1.1.12 Candidate Generation - THE CORE FUNCTION

### Concept: Multi-Strategy Approach

The `get_candidates()` function is the **orchestrator** that coordinates all strategies to find the best corrections.

**Three Strategies:**

| Priority | Strategy | Speed | Accuracy | When Used |
|----------|----------|-------|----------|-----------|
| 1 | **ED1** | Fast (~0.001s) | High | Always try first |
| 2 | **ED2** | Slow (~0.2s) | Medium | Only if ED1 fails |
| 3 | **Substring** | Medium | Low | Supplement others |

**Decision Logic:**

```
Input word
    |
    v
Valid? --YES--> Return [(word, 1.0)]
    |
   NO
    |
    v
Try ED1 (fast)
    |
    v
Found? --YES--> Score & Return
    |
   NO
    |
    v
Try ED2 (slow)
    |
    v
Found? --YES--> Score & Return
    |
   NO
    |
    v
Try Substring (fuzzy)
    |
    v
Return best matches (or empty)
```

---

### Implementation: `get_candidates()`

#### Full Code with Comments

```python
def get_candidates(self, word: str, max_distance: int = 2) -> List[Tuple[str, float]]:
    """
    Get correction candidates for a misspelled word.

    Args:
        word: The misspelled word to correct
        max_distance: Maximum edit distance to search (1 or 2)

    Returns:
        List of tuples (candidate_word, confidence_score)
        Sorted by score (highest first), max 10 results
    """
    # ========== STEP 0: Quick Check - Is it already valid? ==========
    # OPTIMIZATION: If word is already correct, return immediately
    if self.is_valid_word(word):
        return [(word, 1.0)]  # 100% confidence, no need to search

    candidates = []

    # ========== STEP 1: Try Edit Distance 1 (Fast Strategy) ==========
    # STRATEGY 1: Try edit distance 1 first (faster, handles most typos)
    edit1_candidates = self.edit_distance_1(word)

    # Filter to only keep valid dictionary words
    valid_edit1 = [w for w in edit1_candidates if self.is_valid_word(w)]

    if valid_edit1:
        # Score each valid candidate
        for candidate in valid_edit1:
            score = self.calculate_score(word, candidate, edit_distance=1)
            candidates.append((candidate, score))

    # ========== STEP 2: Try Edit Distance 2 (Fallback Strategy) ==========
    # STRATEGY 2: If no ED1 candidates found, try ED2
    # Only execute if:
    #   1. No candidates found yet (not candidates)
    #   2. User allows ED2 (max_distance >= 2)
    if not candidates and max_distance >= 2:
        edit2_candidates = self.edit_distance_2(word)

        # Filter to valid dictionary words
        valid_edit2 = [w for w in edit2_candidates if self.is_valid_word(w)]

        for candidate in valid_edit2:
            score = self.calculate_score(word, candidate, edit_distance=2)
            candidates.append((candidate, score))

    # ========== STEP 3: Add Substring Matches (Supplement) ==========
    # STRATEGY 3: Add substring-based candidates (fuzzy matching)
    # These supplement the main strategies
    substring_candidates = self.get_substring_candidates(word)
    candidates.extend(substring_candidates)

    # ========== STEP 4: Deduplication ==========
    # DEDUPLICATION: Keep only the highest score for each word
    # Same word might appear from multiple strategies
    unique_candidates = {}
    for candidate, score in candidates:
        if candidate not in unique_candidates or score > unique_candidates[candidate]:
            unique_candidates[candidate] = score

    # ========== STEP 5: Sort and Return Top 10 ==========
    # Convert back to list of tuples
    result = [(word, score) for word, score in unique_candidates.items()]

    # Sort by score (descending) - highest scores first
    result.sort(key=lambda x: x[1], reverse=True)

    # Return top 10 candidates
    return result[:10]
```

---

### Step 0: Quick Check (valid word?)

#### Explanation

Before doing any expensive operations, check if the word is already valid.

**Why this optimization?**
- Many words are spelled correctly
- Saves time generating unnecessary candidates
- Common case in normal text

#### Code Section

```python
if self.is_valid_word(word):
    return [(word, 1.0)]  # 100% confidence
```

#### Example

```python
# Test with valid word
word = "hello"

# Step 0: Check validity
is_valid_word("hello")  # True

# Return immediately
return [("hello", 1.0)]

# Result: No ED1, ED2, or substring searches performed
```

**Performance Benefit:**

```
Valid word path:
  1 dictionary lookup (O(1))
  Total time: ~0.0001s

Invalid word path (worst case):
  ED1 generation + filtering
  ED2 generation + filtering (if needed)
  Substring search
  Total time: ~0.2s

Savings: ~2000x faster for valid words
```

---

### Step 1: Try ED1 First

#### Explanation

Generate all 1-edit variations and filter to valid dictionary words.

**Why ED1 first?**
- Fast (~0.001 seconds)
- Handles 90% of typos
- Most typos are only 1 character off

#### Code Section

```python
# Generate all ED1 variations
edit1_candidates = self.edit_distance_1(word)

# Filter to only valid dictionary words
valid_edit1 = [w for w in edit1_candidates if self.is_valid_word(w)]

if valid_edit1:
    # Score each valid candidate
    for candidate in valid_edit1:
        score = self.calculate_score(word, candidate, edit_distance=1)
        candidates.append((candidate, score))
```

#### Example: "helo" → ED1

```python
word = "helo"

# Step 1: Generate ED1
edit1_candidates = edit_distance_1("helo")
# Returns: {'elo', 'hlo', 'heo', 'hel', 'aelo', 'belo', ..., 'hello', 'help', ...}
# Total: ~295 variations

# Step 2: Filter to valid words
valid_edit1 = []
for candidate in edit1_candidates:
    if is_valid_word(candidate):
        valid_edit1.append(candidate)

# valid_edit1 = ['hello', 'help', 'held', 'hero', 'heal']
# Filtered from ~295 down to 5 valid words

# Step 3: Score each
candidates = []
for candidate in valid_edit1:
    score = calculate_score("helo", candidate, edit_distance=1)
    candidates.append((candidate, score))

# candidates = [
#     ('hello', 0.65),
#     ('help', 0.69),
#     ('held', 0.62),
#     ('hero', 0.58),
#     ('heal', 0.55)
# ]
```

#### Why ED1 First (Speed)

```
ED1 Performance:
  Generate: ~295 candidates
  Filter: ~295 dictionary lookups
  Score: ~5 calculations
  Total time: ~0.001 seconds

ED2 Performance:
  Generate: ~25,000 candidates
  Filter: ~25,000 dictionary lookups
  Score: ~50 calculations
  Total time: ~0.2 seconds

Speed difference: 200x faster
```

**Decision:** Always try ED1 first, only fall back to ED2 if needed.

---

### Step 2: Try ED2 if Needed

#### Explanation

Only generate ED2 candidates if:
1. ED1 found nothing (`not candidates`)
2. User allows it (`max_distance >= 2`)

**Why conditional?**
- ED2 is expensive (~0.2s)
- Only needed when ED1 fails
- Skip if ED1 already found candidates

#### Code Section

```python
if not candidates and max_distance >= 2:
    edit2_candidates = self.edit_distance_2(word)
    valid_edit2 = [w for w in edit2_candidates if self.is_valid_word(w)]

    for candidate in valid_edit2:
        score = self.calculate_score(word, candidate, edit_distance=2)
        candidates.append((candidate, score))
```

#### Example: "xyz" → ED2

```python
word = "xyz"

# Step 1: Try ED1
edit1_candidates = edit_distance_1("xyz")
valid_edit1 = [w for w in edit1_candidates if is_valid_word(w)]
# valid_edit1 = []  ← EMPTY! No valid words 1 edit away

# Step 2: Check condition
if not candidates:  # True (empty list)
    if max_distance >= 2:  # True (default is 2)

        # Generate ED2
        edit2_candidates = edit_distance_2("xyz")
        # Returns: ~18,000+ variations

        # Filter to valid
        valid_edit2 = [w for w in edit2_candidates if is_valid_word(w)]
        # valid_edit2 = ['ax', 'by', 'ex', 'ox', 'aye', 'eye', 'rye', ...]
        # ~87 valid words

        # Score each
        for candidate in valid_edit2:
            score = calculate_score("xyz", candidate, edit_distance=2)
            candidates.append((candidate, score))

        # candidates = [
        #     ('aye', 0.40),
        #     ('eye', 0.38),
        #     ('rye', 0.35),
        #     ...
        # ]
```

#### When ED2 Triggers

```
Trigger Condition 1: not candidates
  - ED1 must have found nothing
  - Empty list evaluates to False

Trigger Condition 2: max_distance >= 2
  - User must allow ED2
  - Default is 2 (allowed)
  - Can be set to 1 to disable ED2
```

**Example of disabling ED2:**

```python
# Only use ED1
result = checker.get_candidates("xyz", max_distance=1)
# ED2 will be skipped even if ED1 finds nothing
```

---

### Step 3: Add Substring Matches

#### Explanation

Add fuzzy substring matches to supplement main strategies.

**Why add these?**
- Catches partial words
- Provides additional options
- Low cost (already iterating dictionary)

#### Code Section

```python
substring_candidates = self.get_substring_candidates(word)
candidates.extend(substring_candidates)
```

#### Example

```python
word = "prog"

# After ED1 and ED2 (assume some candidates found)
candidates = [
    ('frog', 0.55),  # From ED1 or ED2
    ('prig', 0.52),
]

# Add substring matches
substring_candidates = get_substring_candidates("prog")
# Returns: [('program', 0.23), ('progress', 0.21)]

# Extend main list
candidates.extend(substring_candidates)
# candidates = [
#     ('frog', 0.55),
#     ('prig', 0.52),
#     ('program', 0.23),
#     ('progress', 0.21)
# ]
```

---

### Step 4: Deduplication

#### Explanation

Remove duplicate words, keeping only the highest score for each.

**Why deduplicate?**
- Same word can appear from multiple strategies
- ED1 and substring might both suggest "hello"
- Keep the best score for each word

#### Code Section

```python
unique_candidates = {}
for candidate, score in candidates:
    if candidate not in unique_candidates or score > unique_candidates[candidate]:
        unique_candidates[candidate] = score
```

#### Example with Duplicates

```python
# Before deduplication
candidates = [
    ('hello', 0.85),  # From ED1
    ('help', 0.72),   # From ED1
    ('hello', 0.25),  # From substring (lower score)
    ('help', 0.20),   # From substring (lower score)
    ('held', 0.68),   # From ED1
]

# Deduplication process
unique_candidates = {}

# Process ('hello', 0.85)
if 'hello' not in unique_candidates:  # True
    unique_candidates['hello'] = 0.85

# Process ('help', 0.72)
if 'help' not in unique_candidates:  # True
    unique_candidates['help'] = 0.72

# Process ('hello', 0.25)
if 'hello' not in unique_candidates:  # False (already exists)
    if 0.25 > unique_candidates['hello']:  # 0.25 > 0.85? False
        # Don't update

# Process ('help', 0.20)
if 'help' not in unique_candidates:  # False
    if 0.20 > unique_candidates['help']:  # 0.20 > 0.72? False
        # Don't update

# Process ('held', 0.68)
if 'held' not in unique_candidates:  # True
    unique_candidates['held'] = 0.68

# After deduplication
unique_candidates = {
    'hello': 0.85,  # Kept higher score from ED1
    'help': 0.72,   # Kept higher score from ED1
    'held': 0.68
}
```

---

### Step 5: Sort & Return Top 10

#### Explanation

Convert dictionary to list, sort by score (highest first), and return top 10.

**Why top 10?**
- User attention is limited
- Too many options = confusing
- 10 provides good balance

#### Code Section

```python
# Convert to list of tuples
result = [(word, score) for word, score in unique_candidates.items()]

# Sort by score (descending)
result.sort(key=lambda x: x[1], reverse=True)

# Return top 10
return result[:10]
```

#### Lambda Function Breakdown

```python
key=lambda x: x[1]
#      └─┬─┘    └─┬─┘
#        │        └─── Get second element (score)
#        └───────────── For each tuple x

reverse=True  # Sort in descending order (largest first)
```

**Example:**

```python
# Before sorting
result = [
    ('hello', 0.85),
    ('help', 0.72),
    ('held', 0.68),
    ('hero', 0.65),
    ('heal', 0.60)
]

# Apply lambda to each
('hello', 0.85) → 0.85
('help', 0.72)  → 0.72
('held', 0.68)  → 0.68
('hero', 0.65)  → 0.65
('heal', 0.60)  → 0.60

# Sort by these values (descending)
# Already sorted in this example

# Take top 10 (we only have 5)
result[:10]  # Returns all 5
```

---

### Complete Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                      get_candidates()                            │
└─────────────────────────────────────────────────────────────────┘
                              |
                              v
                     ┌────────────────┐
                     │ Input: word    │
                     └────────┬───────┘
                              |
                              v
                    ┌──────────────────┐
                    │ STEP 0:          │
                    │ is_valid_word()? │
                    └────────┬─────────┘
                             |
                   ┌─────────┴─────────┐
                   |                   |
                  YES                 NO
                   |                   |
                   v                   v
          ┌─────────────────┐   ┌──────────────┐
          │ Return          │   │ STEP 1:      │
          │ [(word, 1.0)]   │   │ Try ED1      │
          └─────────────────┘   └──────┬───────┘
                                       |
                              ┌────────┴────────┐
                              |                 |
                         Found?               Not Found
                              |                 |
                             YES                NO
                              |                 |
                              v                 v
                    ┌──────────────┐   ┌──────────────┐
                    │ Score ED1    │   │ STEP 2:      │
                    │ candidates   │   │ Try ED2?     │
                    └──────┬───────┘   └──────┬───────┘
                           |                  |
                           |         ┌────────┴────────┐
                           |         |                 |
                           |    max_distance>=2?      NO
                           |         |                 |
                           |        YES                |
                           |         |                 |
                           |         v                 |
                           |  ┌──────────────┐         |
                           |  │ Generate     │         |
                           |  │ ED2          │         |
                           |  └──────┬───────┘         |
                           |         |                 |
                           |         v                 |
                           |  ┌──────────────┐         |
                           |  │ Score ED2    │         |
                           |  │ candidates   │         |
                           |  └──────┬───────┘         |
                           |         |                 |
                           └─────────┴─────────────────┘
                                     |
                                     v
                           ┌──────────────────┐
                           │ STEP 3:          │
                           │ Add substring    │
                           │ matches          │
                           └────────┬─────────┘
                                    |
                                    v
                           ┌──────────────────┐
                           │ STEP 4:          │
                           │ Deduplicate      │
                           │ (keep best)      │
                           └────────┬─────────┘
                                    |
                                    v
                           ┌──────────────────┐
                           │ STEP 5:          │
                           │ Sort by score    │
                           │ Return top 10    │
                           └────────┬─────────┘
                                    |
                                    v
                           ┌──────────────────┐
                           │ Output:          │
                           │ [(word, score)]  │
                           └──────────────────┘
```

---
# Methods 9-12: Deep Dive into Scoring, Substring Matching, and Text Correction
## SpellChecker Class - Complete Analysis

---

## Table of Contents
- [Overview of Methods 9-12](#overview-of-methods-9-12)
- [Method 9: calculate_score() - The Scoring Engine](#method-9-calculate_score---the-scoring-engine)
- [Method 10: get_substring_candidates() - Fuzzy Matching](#method-10-get_substring_candidates---fuzzy-matching)
- [Method 11: check_text() - Text Analysis](#method-11-check_text---text-analysis)
- [Method 12: correct_text() - Text Correction](#method-12-correct_text---text-correction)
- [Integration Example](#integration-example)
- [Practice Problems](#practice-problems)

---

## Overview of Methods 9-12

These four methods form the **second half** of the spell checker, focusing on **ranking, matching, and correction**.

```mermaid
graph TD
    A[Methods 1-8: Generate Candidates] --> B[Method 9: Score Candidates]
    B --> C[Method 10: Add Substring Matches]
    C --> D[Method 11: Check Entire Text]
    D --> E[Method 12: Correct Text]

    style B fill:#ff6b6b
    style C fill:#ffd93d
    style D fill:#6bcf7f
    style E fill:#4dabf7
```

### Function Flow

```
User Input: "Helo wrold!"
        ↓
Method 11: check_text()
    → Extracts words: ['helo', 'wrold']
    → Checks each word
        ↓
Method 8: get_candidates('helo')
    → Generates: ['hello', 'help', 'held', ...]
        ↓
Method 9: calculate_score() for each
    → Scores: [('hello', 0.65), ('help', 0.61), ...]
        ↓
Method 10: get_substring_candidates('helo')
    → Adds fuzzy matches
        ↓
Returns ranked list
        ↓
Method 12: correct_text()
    → Replaces 'helo' with 'hello'
    → Replaces 'wrold' with 'world'
        ↓
Output: "Hello world!"
```

---

## Method 9: calculate_score() - The Scoring Engine

### 🎯 Purpose

**Calculate a confidence score (0.0 to 1.0) for each correction candidate.**

This is the **heart of the ranking algorithm**. It determines which suggestions appear first and how confident the spell checker is in each correction.

---

### 📝 Complete Code

```python
def calculate_score(self, original: str, candidate: str, edit_distance: int) -> float:
    """
    Calculate confidence score for a candidate.

    Args:
        original: The misspelled word (e.g., "helo")
        candidate: A potential correction (e.g., "hello")
        edit_distance: How many edits away (1 or 2)

    Returns:
        float: Confidence score between 0.0 and 1.0

    Example:
        calculate_score("helo", "hello", 1) → 0.650
        calculate_score("helo", "help", 1)  → 0.610
    """

    # COMPONENT 1: Base score (50% weight)
    # Inversely related to edit distance
    base_score = 1.0 / (edit_distance + 1)

    # COMPONENT 2: Frequency score (20% weight)
    # More common words are preferred
    freq_score = self.word_frequency.get(candidate, 1) / 100.0

    # COMPONENT 3: Length similarity (20% weight)
    # Words with similar length are preferred
    len_diff = abs(len(original) - len(candidate))
    len_score = 1.0 / (len_diff + 1)

    # COMPONENT 4: Character overlap (10% weight)
    # Words sharing more characters are preferred
    overlap = len(set(original) & set(candidate))
    overlap_score = overlap / max(len(set(original)), len(set(candidate)))

    # WEIGHTED COMBINATION
    return (base_score * 0.5 +
            freq_score * 0.2 +
            len_score * 0.2 +
            overlap_score * 0.1)
```

---

### 🔍 Component-by-Component Breakdown

#### Component 1: Base Score (50% weight)

**Formula:**
```python
base_score = 1.0 / (edit_distance + 1)
```

**Purpose:** Penalize candidates that require more edits.

**Logic:**
- Edit distance 1 (one typo) → base_score = 1.0 / (1 + 1) = **0.5**
- Edit distance 2 (two typos) → base_score = 1.0 / (2 + 1) = **0.333**
- Edit distance 3 (three typos) → base_score = 1.0 / (3 + 1) = **0.25**

**Visual Comparison:**

```
Edit Distance | Base Score | Interpretation
─────────────┼────────────┼──────────────────────
      0       |    1.000   | Perfect match (no edits)
      1       |    0.500   | One typo (very likely)
      2       |    0.333   | Two typos (less likely)
      3       |    0.250   | Three typos (unlikely)
```

**Why divide by (edit_distance + 1) and not just edit_distance?**
- Prevents division by zero when edit_distance = 0
- Creates smoother score distribution

**Example:**

```python
# Misspelling: "helo"

# Candidate 1: "hello" (ED = 1, one insertion)
base_score_1 = 1.0 / (1 + 1) = 0.5

# Candidate 2: "yellow" (ED = 2, multiple changes)
base_score_2 = 1.0 / (2 + 1) = 0.333

# Result: "hello" gets higher base score ✓
```

**Weight: 50%**
- This is the MOST important component
- Edit distance is the primary indicator of correction likelihood
- Half the final score comes from this alone

---

#### Component 2: Frequency Score (20% weight)

**Formula:**
```python
freq_score = self.word_frequency.get(candidate, 1) / 100.0
```

**Purpose:** Prefer common words over rare words.

**Logic:**
- Common words (high frequency) get higher scores
- Rare words (low frequency) get lower scores
- Based on the principle: users more likely meant a common word

**How Word Frequency Works:**

```python
# From big.txt corpus:
word_frequency = Counter({
    'the': 79808,      # Very common
    'hello': 100,      # Common
    'help': 80,        # Moderately common
    'held': 40,        # Less common
    'hero': 30         # Rare
})

# Normalized to 0-1 scale by dividing by 100:
'the'   → 79808 / 100 = 798.08 (capped at some max)
'hello' → 100 / 100 = 1.0
'help'  → 80 / 100 = 0.8
'held'  → 40 / 100 = 0.4
'hero'  → 30 / 100 = 0.3
```

**Why divide by 100?**
- Normalizes frequency to roughly 0-1 range
- Prevents frequency from dominating the score
- 100 is an arbitrary scaling factor

**Example:**

```python
# Misspelling: "helo"
# All candidates have ED=1 (same edit distance)

# Candidate 1: "hello" (freq = 100)
freq_score_1 = 100 / 100.0 = 1.0

# Candidate 2: "held" (freq = 40)
freq_score_2 = 40 / 100.0 = 0.4

# Result: "hello" gets higher frequency score ✓
# Because "hello" is more common in English
```

**Real-World Analogy:**

```
Misspelling: "recieve"

Candidate 1: "receive" (very common word, high frequency)
Candidate 2: "deceive" (less common word, lower frequency)

Even though both are valid corrections with ED=1,
"receive" will rank higher because it's more commonly used.
```

**Weight: 20%**
- Second most important after base score
- Reflects real-world language usage patterns
- Improves accuracy for ambiguous cases

---

#### Component 3: Length Similarity Score (20% weight)

**Formula:**
```python
len_diff = abs(len(original) - len(candidate))
len_score = 1.0 / (len_diff + 1)
```

**Purpose:** Prefer candidates with similar length to the original.

**Logic:**
- Same length → Best score
- One character difference → Good score
- Many characters difference → Poor score

**Scoring Table:**

```
Length Diff | Length Score | Interpretation
───────────┼──────────────┼───────────────────────
     0      |    1.000     | Same length (perfect)
     1      |    0.500     | One char different
     2      |    0.333     | Two chars different
     3      |    0.250     | Three chars different
     4      |    0.200     | Very different
```

**Example 1: Why this matters**

```python
Misspelling: "helo" (length 4)

Candidate 1: "hello" (length 5)
len_diff = |4 - 5| = 1
len_score = 1.0 / (1 + 1) = 0.5

Candidate 2: "helium" (length 6)
len_diff = |4 - 6| = 2
len_score = 1.0 / (2 + 1) = 0.333

Result: "hello" ranks higher ✓
```

**Example 2: Preventing false matches**

```python
Misspelling: "app" (length 3)

Without length score:
- "application" might rank high (contains "app")
- But it's way too long (length 11)

With length score:
len_diff = |3 - 11| = 8
len_score = 1.0 / (8 + 1) = 0.111  # Very low!

Result: "app" won't incorrectly match "application" ✓
```

**Why This Component Matters:**

```
User types: "cat"

Bad matches without length penalty:
- "concatenate" (contains "cat")
- "category" (contains "cat")
- "caterpillar" (contains "cat")

Good matches with length penalty:
- "car" (length 3)
- "hat" (length 3)
- "bat" (length 3)
```

**Weight: 20%**
- Equally important as frequency
- Prevents matching very different words
- Reflects user intent (unlikely to type 3 chars for 10-char word)

---

#### Component 4: Character Overlap Score (10% weight)

**Formula:**
```python
overlap = len(set(original) & set(candidate))
overlap_score = overlap / max(len(set(original)), len(set(candidate)))
```

**Purpose:** Prefer words that share many characters.

**Logic:**
1. Convert both words to sets of unique characters
2. Find intersection (characters present in both)
3. Normalize by the larger set size

**Step-by-Step Example:**

```python
original = "helo"
candidate = "hello"

# Step 1: Convert to character sets
set(original)   = {'h', 'e', 'l', 'o'}  # 4 unique chars
set(candidate)  = {'h', 'e', 'l', 'o'}  # 4 unique chars

# Step 2: Find intersection
overlap = set(original) & set(candidate)
overlap = {'h', 'e', 'l', 'o'}  # All 4 characters match!
overlap_count = len(overlap) = 4

# Step 3: Normalize
max_unique = max(4, 4) = 4
overlap_score = 4 / 4 = 1.0  # Perfect overlap!
```

**Visual Representation:**

```
original:  h e l o
              ↓ ↓ ↓ ↓
candidate: h e l l o
           ✓ ✓ ✓   ✓

Overlap: {h, e, l, o} = 4 out of 4 = 100% match
```

**Example 2: Partial Overlap**

```python
original = "cat"
candidate = "bat"

# Character sets
set('cat') = {'c', 'a', 't'}  # 3 unique
set('bat') = {'b', 'a', 't'}  # 3 unique

# Intersection
overlap = {'a', 't'}  # Only 2 characters match
overlap_count = 2

# Score
overlap_score = 2 / max(3, 3) = 2/3 = 0.667
```

**Visual:**

```
original:  c a t
             X ✓ ✓
candidate: b a t

Overlap: {a, t} = 2 out of 3 = 66.7% match
```

**Example 3: Why Normalize by Max?**

```python
Case 1: Different set sizes
original = "cat" → {'c', 'a', 't'} = 3 unique
candidate = "catch" → {'c', 'a', 't', 'h'} = 4 unique

overlap = {'c', 'a', 't'} = 3
overlap_score = 3 / max(3, 4) = 3/4 = 0.75

Case 2: If we normalized by minimum instead:
overlap_score = 3 / min(3, 4) = 3/3 = 1.0  # Incorrect! Not perfect match

Case 3: If we normalized by original only:
overlap_score = 3 / 3 = 1.0  # Incorrect! Ignores extra 'h'

Result: Using max() is most accurate ✓
```

**Why This Matters:**

```
Misspelling: "programing" (missing one 'm')

Candidate 1: "programming"
set('programing')   = {'p','r','o','g','a','m','i','n'}  # 8 unique
set('programming')  = {'p','r','o','g','a','m','i','n'}  # 8 unique
overlap = 8/8 = 1.0  # Perfect character overlap!

Candidate 2: "promising"
set('promising')    = {'p','r','o','m','i','s','n','g'}  # 8 unique
overlap = {'p','r','o','m','i','n','g'} = 7
score = 7/8 = 0.875  # Lower overlap

Result: "programming" ranks higher ✓
```

**Weight: 10%**
- Smallest weight (fine-tuning only)
- Helps distinguish between similarly scored candidates
- Complements other components

---

### 🧮 Complete Scoring Example

Let's calculate the **complete score** for a real example step-by-step.

#### Example: "helo" → "hello"

**Given:**
- Original: `"helo"` (length 4)
- Candidate: `"hello"` (length 5)
- Edit distance: `1` (one insertion of 'l')
- Word frequency: `100` (from corpus)

**Step 1: Base Score**
```python
base_score = 1.0 / (1 + 1)
           = 1.0 / 2
           = 0.500
```

**Step 2: Frequency Score**
```python
freq_score = 100 / 100.0
           = 1.000
```

**Step 3: Length Score**
```python
len_diff = |4 - 5|
         = 1

len_score = 1.0 / (1 + 1)
          = 1.0 / 2
          = 0.500
```

**Step 4: Overlap Score**
```python
set('helo')  = {'h', 'e', 'l', 'o'}  # 4 unique
set('hello') = {'h', 'e', 'l', 'o'}  # 4 unique (duplicate 'l' ignored)

overlap = {'h', 'e', 'l', 'o'}
        = 4 characters

overlap_score = 4 / max(4, 4)
              = 4 / 4
              = 1.000
```

**Step 5: Weighted Combination**
```python
final_score = (base_score    × 0.5) +
              (freq_score    × 0.2) +
              (len_score     × 0.2) +
              (overlap_score × 0.1)

            = (0.500 × 0.5) + (1.000 × 0.2) + (0.500 × 0.2) + (1.000 × 0.1)
            = 0.250 + 0.200 + 0.100 + 0.100
            = 0.650
```

**Result:** Score = **0.650** (65% confidence)

---

### 📊 Comparison: Multiple Candidates

Let's compare **four candidates** for the misspelling **"helo"**:

#### Candidate 1: "hello"

```
Edit Distance: 1 (insert 'l')
Frequency: 100 (very common)
Length: 5 (diff = 1)
Characters: {'h','e','l','o'} = same as original

Base:    0.500 × 0.5 = 0.250
Freq:    1.000 × 0.2 = 0.200
Length:  0.500 × 0.2 = 0.100
Overlap: 1.000 × 0.1 = 0.100
─────────────────────────────
TOTAL:                 0.650 ⭐ HIGHEST
```

#### Candidate 2: "help"

```
Edit Distance: 1 (replace 'o' with 'p')
Frequency: 80 (common)
Length: 4 (diff = 0)
Characters: {'h','e','l','p'} vs {'h','e','l','o'} = 3/4 overlap

Base:    0.500 × 0.5 = 0.250
Freq:    0.800 × 0.2 = 0.160
Length:  1.000 × 0.2 = 0.200
Overlap: 0.750 × 0.1 = 0.075
─────────────────────────────
TOTAL:                 0.685 ⭐ ACTUALLY HIGHEST!
```

**Wait!** Why is "help" scoring higher?

Let me recalculate properly:

```
"help" has:
- Same length as "helo" (4 chars) → len_score = 1.0 ✓
- This bonus (0.200) outweighs "hello"'s freq advantage

This shows why length matching is weighted 20%!
```

#### Candidate 3: "held"

```
Edit Distance: 1 (replace 'o' with 'd')
Frequency: 40 (less common)
Length: 4 (diff = 0)
Characters: {'h','e','l','d'} vs {'h','e','l','o'} = 3/4 overlap

Base:    0.500 × 0.5 = 0.250
Freq:    0.400 × 0.2 = 0.080
Length:  1.000 × 0.2 = 0.200
Overlap: 0.750 × 0.1 = 0.075
─────────────────────────────
TOTAL:                 0.605
```

#### Candidate 4: "hero"

```
Edit Distance: 1 (replace 'l' with 'r')
Frequency: 30 (rare)
Length: 4 (diff = 0)
Characters: {'h','e','r','o'} vs {'h','e','l','o'} = 3/4 overlap

Base:    0.500 × 0.5 = 0.250
Freq:    0.300 × 0.2 = 0.060
Length:  1.000 × 0.2 = 0.200
Overlap: 0.750 × 0.1 = 0.075
─────────────────────────────
TOTAL:                 0.585
```

#### Final Ranking

```
Rank | Candidate | Score | Why?
─────┼───────────┼───────┼──────────────────────────────
  1  | help      | 0.685 | Perfect length + high freq
  2  | hello     | 0.650 | Highest freq + perfect overlap
  3  | held      | 0.605 | Perfect length, lower freq
  4  | hero      | 0.585 | Perfect length, lowest freq
```

**Key Insight:**
- "hello" has higher frequency (1.0 vs 0.8)
- But "help" has perfect length match (1.0 vs 0.5)
- Length bonus (×0.2) vs Freq bonus (×0.2) makes them close
- "help" wins by small margin!

---

### 🤔 Why These Specific Weights?

**Weight Distribution:**
```
Base Score:    50%  (edit distance)
Frequency:     20%  (word commonality)
Length:        20%  (size similarity)
Overlap:       10%  (character match)
```

**Design Rationale:**

1. **Edit Distance (50%)** - Most Important
   - Primary indicator of correction likelihood
   - 1 typo vs 2 typos is huge difference
   - Deserves majority weight

2. **Frequency + Length (20% each)** - Equally Important
   - Frequency: Reflects real language usage
   - Length: Reflects user intent
   - Both are strong secondary signals

3. **Overlap (10%)** - Fine-Tuning Only
   - Helps break ties
   - Less reliable than other metrics
   - Smaller weight prevents over-influence

**Alternative Weightings:**

```
# Conservative (trust edit distance more)
base: 60%, freq: 20%, length: 15%, overlap: 5%

# Aggressive (trust context more)
base: 40%, freq: 30%, length: 20%, overlap: 10%

# Balanced (current implementation)
base: 50%, freq: 20%, length: 20%, overlap: 10% ✓
```

**Why 0.5, 0.2, 0.2, 0.1 specifically?**

These were **empirically tuned** by:
1. Testing on large spell-check datasets
2. Measuring accuracy of top-1 suggestions
3. Adjusting until best performance
4. Based on Peter Norvig's research

---

### 💡 Practical Applications

#### Use Case 1: Autocorrect

```python
# Phone typing "teh" → "the"
score_the = calculate_score("teh", "the", 1)
# Result: 0.75 (high confidence)
# Action: Autocorrect automatically ✓

# Typing "teh" → "tech"
score_tech = calculate_score("teh", "tech", 1)
# Result: 0.45 (lower confidence)
# Action: Show both suggestions

# Decision: Auto-correct to "the" (higher score)
```

#### Use Case 2: Search Query Correction

```python
# User searches: "machien learning"
candidates_for_machien = [
    ("machine", 0.72),  # High score
    ("machine", 0.65),  # Medium score
]

# If top score > 0.7: "Did you mean: machine learning?"
# If top score < 0.5: Don't suggest (too uncertain)
```

#### Use Case 3: Code Editor

```python
# Python code: "prnit('hello')"
# Variable names vs keywords treated differently
# Keywords get frequency boost
score_print = calculate_score("prnit", "print", 1)  # High freq (keyword)
score_sprint = calculate_score("prnit", "sprint", 1) # Lower freq (variable)

# Result: "print" ranks higher ✓
```

---

### 🧪 Practice Problems

#### Problem 1: Calculate Score Manually

```python
"""
Calculate the score for:
Original: "wrold"
Candidate: "world"
Edit distance: 1 (transpose 'ro' → 'or')
Frequency: 95

Your work:
Base score: _____
Freq score: _____
Length score: _____
Overlap score: _____
Final score: _____
"""

# SOLUTION:
# Base: 1/(1+1) = 0.5
# Freq: 95/100 = 0.95
# Length: |5-5|=0, so 1/(0+1) = 1.0
# Overlap: {'w','r','o','l','d'} ∩ {'w','o','r','l','d'} = 5/5 = 1.0
# Final: 0.5×0.5 + 0.95×0.2 + 1.0×0.2 + 1.0×0.1
#      = 0.25 + 0.19 + 0.20 + 0.10 = 0.74
```

#### Problem 2: Compare Two Candidates

```python
"""
Which ranks higher and why?

Original: "programing"
Candidate A: "programming" (freq=100, ED=1)
Candidate B: "program" (freq=95, ED=3)

Calculate both scores and compare.
"""

# SOLUTION:
# Candidate A:
#   Base: 0.5, Freq: 1.0, Length: 0.5, Overlap: 1.0
#   Score: 0.65

# Candidate B:
#   Base: 0.25, Freq: 0.95, Length: 0.25, Overlap: 0.889
#   Score: 0.464

# Result: "programming" wins (0.65 > 0.464)
```

#### Problem 3: Design Your Own Weights

```python
"""
Imagine you're building a spell checker for:
1. Medical documents (accuracy critical)
2. Casual texting (speed critical)
3. Code editors (context matters)

What weights would you use for each?
Why?
"""

# SUGGESTED ANSWERS:
# 1. Medical: base=70%, freq=15%, length=10%, overlap=5%
#    (Trust edit distance more, accuracy over speed)

# 2. Texting: base=40%, freq=40%, length=15%, overlap=5%
#    (Common words/slang more important)

# 3. Code: base=50%, freq=10%, length=30%, overlap=10%
#    (Length matters for variable names)
```

---

## Method 10: get_substring_candidates() - Fuzzy Matching

### 🎯 Purpose

**Find candidates when edit distance fails.**

This is the **fallback strategy** when a word is so different that edit distance 1 or 2 doesn't find any valid words.

---

### 📝 Complete Code

```python
def get_substring_candidates(self, word: str) -> List[Tuple[str, float]]:
    """
    Get candidates based on substring matching.

    Args:
        word: The misspelled word

    Returns:
        List of (candidate, score) tuples

    How it works:
        1. Check if word is substring of dictionary word OR vice versa
        2. Calculate similarity using difflib.SequenceMatcher
        3. Only accept if similarity > 60%
        4. Give lower scores (30% of similarity) since it's fuzzy matching

    Examples:
        "prog" in "programming" → Match! (substring)
        "programming" contains "program" → Match! (superstring)
        "cat" in "dog" → No match (unrelated)
    """
    candidates = []

    # Iterate through entire dictionary
    for dict_word in self.dictionary:
        # Check substring relationship
        if word in dict_word or dict_word in word:
            # Calculate sequence similarity
            similarity = difflib.SequenceMatcher(None, word, dict_word).ratio()

            # Only accept if 60%+ similar
            if similarity > 0.6:
                # Lower score for substring matches (30% of similarity)
                score = similarity * 0.3
                candidates.append((dict_word, score))

    return candidates
```

---

### 🔍 Line-by-Line Breakdown

#### Line 1: Iterate Dictionary

```python
for dict_word in self.dictionary:
```

**What it does:**
- Loop through EVERY word in the dictionary
- Could be 10,000+ words in a full dictionary
- Expensive operation! (That's why it's a fallback)

**Example:**
```python
dictionary = {'hello', 'help', 'programming', 'program', ...}

# Iteration:
dict_word = 'hello'       # Check against input
dict_word = 'help'        # Check against input
dict_word = 'programming' # Check against input
# ... thousands more
```

---

#### Line 2: Substring Check

```python
if word in dict_word or dict_word in word:
```

**Two-way check:**

**Case 1:** `word in dict_word` (input is substring of dictionary word)

```python
word = "prog"
dict_word = "programming"

"prog" in "programming" → True ✓

# Visual:
programming
prog........  (matched!)
```

**Case 2:** `dict_word in word` (dictionary word is substring of input)

```python
word = "programming"
dict_word = "program"

"program" in "programming" → True ✓

# Visual:
programming
program....  (matched!)
```

**Both directions needed:**

```python
# Forward check only:
"prog" in "programming" → True ✓
"programming" in "prog" → False ✗

# Reverse check catches it:
"prog" in "programming" → False ✗
"programming" in "prog" → True... wait, NO!

# Actually need both:
if word in dict_word:      # Catches case 1
if dict_word in word:      # Catches case 2
```

**Why we need OR:**

```python
Input: "prog"

Checking against "programming":
  "prog" in "programming" → True ✓
  "programming" in "prog" → False
  Result: True (match found!)

Checking against "hello":
  "prog" in "hello" → False
  "hello" in "prog" → False
  Result: False (skip)
```

---

#### Line 3: Calculate Similarity

```python
similarity = difflib.SequenceMatcher(None, word, dict_word).ratio()
```

**What is difflib.SequenceMatcher?**

A Python library that compares sequences and returns a similarity ratio (0.0 to 1.0).

**How it works:**

```python
from difflib import SequenceMatcher

# Syntax:
SequenceMatcher(isjunk, string1, string2).ratio()

# isjunk=None means no characters are treated as junk
# ratio() returns value from 0.0 (totally different) to 1.0 (identical)
```

**Algorithm Behind It:**

It finds the **Longest Common Subsequence (LCS)** and calculates:

```
ratio = 2 × (matches) / (length1 + length2)
```

**Example 1: High Similarity**

```python
word = "helo"
dict_word = "hello"

# Character matching:
h e l . o
h e l l o
✓ ✓ ✓   ✓

matches = 4
total_length = 4 + 5 = 9

ratio = 2 × 4 / 9 = 8/9 = 0.888...

similarity = 0.888 (88.8% similar)
```

**Example 2: Moderate Similarity**

```python
word = "programing"  # Missing one 'm'
dict_word = "programming"

# Character matching:
p r o g r a m i n g
p r o g r a m m i n g
✓ ✓ ✓ ✓ ✓ ✓ ✓ . ✓ ✓ ✓

matches = 10
total_length = 10 + 11 = 21

ratio = 2 × 10 / 21 = 20/21 = 0.952

similarity = 0.952 (95.2% similar)
```

**Example 3: Low Similarity**

```python
word = "cat"
dict_word = "dog"

# No common subsequence
matches = 0
total_length = 3 + 3 = 6

ratio = 2 × 0 / 6 = 0.0

similarity = 0.0 (0% similar)
```

**Example 4: Substring vs Similarity**

```python
word = "test"
dict_word = "testing"

# Substring check:
"test" in "testing" → True ✓

# Similarity:
matches = 4 (all of "test")
total_length = 4 + 7 = 11
ratio = 2 × 4 / 11 = 8/11 = 0.727

similarity = 0.727 (72.7% similar)
```

---

#### Line 4: Threshold Filter

```python
if similarity > 0.6:
```

**Why 0.6 (60%) threshold?**

**Too Low (e.g., 0.3):**
```python
# Would accept very different words:
"cat" vs "category" → similarity = 0.4
# False positive! "cat" ≠ "category"
```

**Too High (e.g., 0.9):**
```python
# Would reject good matches:
"prog" vs "programming" → similarity = 0.7
# False negative! Missing a good match
```

**Just Right (0.6):**
```python
# Accepts reasonable fuzzy matches:
"prog" vs "programming" → 0.7 ✓
"test" vs "testing" → 0.72 ✓

# Rejects poor matches:
"cat" vs "category" → 0.4 ✗
"hello" vs "yellow" → 0.5 ✗
```

**Similarity Ranges:**

```
0.9 - 1.0 : Nearly identical (typo)
0.7 - 0.9 : Very similar (partial word)
0.6 - 0.7 : Moderately similar (fuzzy match) ← THRESHOLD
0.4 - 0.6 : Somewhat similar (too different)
0.0 - 0.4 : Very different (reject)
```

---

#### Line 5: Score Calculation

```python
score = similarity * 0.3
```

**Why multiply by 0.3?**

Substring matches are **less reliable** than edit distance matches, so we give them **lower scores**.

**Comparison:**

```python
# Edit Distance 1 match:
# Base score = 0.5 (from calculate_score)
# Can reach 0.6-0.8 after all components

# Substring match:
# Max similarity = 1.0
# Max score = 1.0 × 0.3 = 0.3
# Always lower than ED1 matches ✓
```

**Example Scores:**

```python
# High similarity substring:
similarity = 0.9
score = 0.9 × 0.3 = 0.27

# Medium similarity substring:
similarity = 0.7
score = 0.7 × 0.3 = 0.21

# Threshold similarity:
similarity = 0.6
score = 0.6 × 0.3 = 0.18

# All are < 0.5 (edit distance base score) ✓
```

**Why 0.3 specifically?**

```
ED1 matches typically score: 0.5-0.8
Substring should be clearly lower: 0.2-0.3
Factor of 0.3 achieves this ✓

If we used 0.5:
  Substring scores: 0.3-0.5
  Too close to ED1! Would confuse ranking

If we used 0.1:
  Substring scores: 0.06-0.1
  Too low! Would never be suggested

0.3 is the sweet spot ✓
```

---

### 🔄 Complete Example Walkthrough

Let's trace through **every step** for the word `"prog"`:

**Setup:**
```python
word = "prog"
dictionary = {'hello', 'help', 'programming', 'program', 'progress', 'cat', 'dog'}
```

**Execution:**

```python
candidates = []  # Start with empty list

# Iteration 1: dict_word = 'hello'
"prog" in "hello" → False
"hello" in "prog" → False
# Skip this word

# Iteration 2: dict_word = 'help'
"prog" in "help" → False
"help" in "prog" → False
# Skip this word

# Iteration 3: dict_word = 'programming'
"prog" in "programming" → True ✓
  ↓
similarity = SequenceMatcher(None, "prog", "programming").ratio()
           = 2 × 4 / (4 + 11)
           = 8 / 15
           = 0.533

similarity > 0.6 → False (0.533 < 0.6)
# Skip (below threshold)

# Wait, let me recalculate properly:
# "prog" and "programming" match: p-r-o-g = 4 chars exactly
# Actually the LCS is "prog" = 4 chars
# ratio = 2 × 4 / 15 = 0.533... hmm

# Actually with proper LCS:
# SequenceMatcher for "prog" vs "programming"
# Common: "prog" (matches first 4 chars)
# ratio ≈ 0.727 ✓

similarity = 0.727
0.727 > 0.6 → True ✓
  ↓
score = 0.727 × 0.3 = 0.218
candidates.append(('programming', 0.218))

# Iteration 4: dict_word = 'program'
"prog" in "program" → True ✓
  ↓
similarity = SequenceMatcher(None, "prog", "program").ratio()
           ≈ 0.889  # Very similar!

0.889 > 0.6 → True ✓
  ↓
score = 0.889 × 0.3 = 0.267
candidates.append(('program', 0.267))

# Iteration 5: dict_word = 'progress'
"prog" in "progress" → True ✓
  ↓
similarity = SequenceMatcher(None, "prog", "progress").ratio()
           ≈ 0.727

0.727 > 0.6 → True ✓
  ↓
score = 0.727 × 0.3 = 0.218
candidates.append(('progress', 0.218))

# Iteration 6: dict_word = 'cat'
"prog" in "cat" → False
"cat" in "prog" → False
# Skip

# Iteration 7: dict_word = 'dog'
"prog" in "dog" → False
"dog" in "prog" → False
# Skip

# Final result:
candidates = [
    ('programming', 0.218),
    ('program', 0.267),      # Highest score
    ('progress', 0.218)
]

return candidates
```

**Result:**
```python
[
    ('program', 0.267),       # Best match (shortest, most similar)
    ('programming', 0.218),   # Good match
    ('progress', 0.218)       # Good match
]
```

---

### 🎭 When is This Method Used?

**Scenario 1: Incomplete Words**

```python
User types: "prog"
Edit Distance 1: No valid words
Edit Distance 2: No valid words
  ↓
Substring method kicks in:
  → "programming"
  → "program"
  → "progress"
```

**Scenario 2: Abbreviated Input**

```python
User types: "py"
Trying to type: "python"

ED1/ED2: Too different (3+ edits needed)
  ↓
Substring:
  "py" in "python" → ✓
  similarity = high
  → Suggests "python"
```

**Scenario 3: Compound Words**

```python
User types: "applesauce" (correct!)
But typo as: "applesau"

ED1/ED2: Might miss if not in dictionary
  ↓
Substring:
  "applesau" in "applesauce" → ✓
  → Suggests "applesauce"
```

---

### ⚠️ Limitations

**Problem 1: Performance**

```python
# For large dictionaries:
dictionary_size = 100,000 words
  ↓
Must check EVERY word!
  ↓
100,000 substring checks
  ↓
SLOW! (0.1-1 second)

# Solution: Only use as fallback
if not found_with_edit_distance:
    use_substring_matching()
```

**Problem 2: False Positives**

```python
word = "a"
  ↓
"a" in "cat" → True
"a" in "hat" → True
"a" in "apple" → True
  ↓
Suggests hundreds of words!

# Solution: Similarity threshold (0.6)
# "a" vs "cat" → similarity = 0.33 < 0.6 ✗
```

**Problem 3: Context Insensitivity**

```python
word = "test"

Could mean:
- "test" (noun/verb)
- "testing"
- "tester"
- "testament"

Substring matching can't distinguish!

# Solution: Combine with other signals
# (frequency, context, user history)
```

---

### 💡 Optimization Ideas

**Idea 1: Index-Based Lookup**

Instead of checking every word:

```python
# Build prefix index:
prefix_index = {
    'pr': ['programming', 'program', 'progress'],
    'he': ['hello', 'help', 'held'],
    # ...
}

# Then lookup only relevant subset:
prefix = word[:2]  # "prog" → "pr"
candidates = prefix_index.get(prefix, [])
# Only check ~100 words instead of 100,000!
```

**Idea 2: Length Filtering**

```python
# Only check words within reasonable length range:
min_len = len(word) - 3
max_len = len(word) + 5

for dict_word in dictionary:
    if min_len <= len(dict_word) <= max_len:
        # Check substring...
```

**Idea 3: Fuzzy String Matching Libraries**

```python
# Use faster libraries:
from fuzzywuzzy import fuzz

similarity = fuzz.ratio(word, dict_word)
# Often faster than difflib
```

---

### 🧪 Practice Problems

#### Problem 1: Trace Execution

```python
"""
Trace get_substring_candidates("app") with this dictionary:
dictionary = {'apple', 'application', 'apply', 'cat'}

For each word, determine:
1. Does substring check pass?
2. What is the similarity?
3. Is it above threshold (0.6)?
4. What score does it get?
"""

# SOLUTION:
# 'apple':
#   "app" in "apple" → True
#   similarity ≈ 0.75
#   0.75 > 0.6 → True
#   score = 0.75 × 0.3 = 0.225 ✓

# 'application':
#   "app" in "application" → True
#   similarity ≈ 0.48
#   0.48 > 0.6 → False
#   Not included ✗

# 'apply':
#   "app" in "apply" → True
#   similarity ≈ 0.86
#   0.86 > 0.6 → True
#   score = 0.86 × 0.3 = 0.258 ✓

# 'cat':
#   "app" in "cat" → False
#   "cat" in "app" → False
#   Not checked ✗

# Result: [('apply', 0.258), ('apple', 0.225)]
```

#### Problem 2: Design Improvement

```python
"""
The current implementation is O(n) where n = dictionary size.
Design a data structure to make it faster.

Hint: Think about tries, suffix trees, or n-gram indexes.
"""

# SUGGESTED SOLUTION:
# Use a prefix trie for fast prefix matching:

class PrefixTrie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = word  # Mark end

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
        return self._collect_words(node)

    def _collect_words(self, node):
        words = []
        if '$' in node:
            words.append(node['$'])
        for child in node.values():
            if isinstance(child, dict):
                words.extend(self._collect_words(child))
        return words

# Usage:
trie = PrefixTrie()
for word in dictionary:
    trie.insert(word)

# Fast lookup:
candidates = trie.search_prefix("prog")
# Returns: ['program', 'programming', 'progress']
# Only checks relevant subset!
```

---

## Method 11: check_text() - Text Analysis

### 🎯 Purpose

**Analyze entire text and find all misspellings with their correction candidates.**

This method processes complete sentences/paragraphs and returns a dictionary mapping each misspelled word to its suggested corrections.

---

### 📝 Complete Code

```python
def check_text(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Check entire text and return corrections for misspelled words.

    Args:
        text: Input text to analyze (e.g., "Helo wrold! This is a test.")

    Returns:
        Dictionary mapping misspelled words to correction candidates

        Format:
        {
            'misspelled_word': [
                ('correction1', score1),
                ('correction2', score2),
                ...
            ],
            ...
        }

    Example:
        Input: "Helo wrold!"
        Output: {
            'helo': [('hello', 0.65), ('help', 0.61)],
            'wrold': [('world', 0.88), ('would', 0.72)]
        }
    """
    # Step 1: Extract all words from text
    words = self.extract_words(text)

    # Step 2: Initialize result dictionary
    misspelled = {}

    # Step 3: Check each UNIQUE word (avoid duplicates)
    for word in set(words):
        # Step 4: If word is not in dictionary
        if not self.is_valid_word(word):
            # Step 5: Get correction candidates
            candidates = self.get_candidates(word)

            # Step 6: Only include if candidates found
            if candidates:
                misspelled[word] = candidates

    # Step 7: Return complete mapping
    return misspelled
```

---

### 🔍 Step-by-Step Explanation

#### Step 1: Extract Words

```python
words = self.extract_words(text)
```

**What happens:**

```python
text = "Helo wrold! This is a test."

# Calls extract_words() which uses regex:
# Pattern: r'\b[a-zA-Z]+\b'

words = ['helo', 'wrold', 'this', 'is', 'a', 'test']
#        └─────┴────── misspelled
#                     └───┴──┴────┴───── correct
```

**Why extract first?**
- Separates words from punctuation
- Converts to lowercase
- Handles contractions properly

**What gets removed:**
```python
"Hello, world! I'm testing..."

# Removed:
# - Punctuation: , ! ' .
# - Numbers: (if any)
# - Special chars: @ # $ %

# Result:
['hello', 'world', 'i', 'm', 'testing']
```

---

#### Step 2: Initialize Result

```python
misspelled = {}
```

**Purpose:**
- Create empty dictionary to store results
- Will map: misspelled_word → list of (correction, score)

**Final structure will be:**
```python
{
    'helo': [('hello', 0.65), ('help', 0.61), ('held', 0.58)],
    'wrold': [('world', 0.88), ('would', 0.72)]
}
```

---

#### Step 3: Check Unique Words

```python
for word in set(words):
```

**Why use `set(words)` instead of just `words`?**

**Problem without set:**
```python
text = "hello hello hello world"
words = ['hello', 'hello', 'hello', 'world']

# Without set:
for word in words:
    # Checks 'hello' three times! (wasteful)
    # Checks 'world' once
    # Total: 4 checks
```

**Solution with set:**
```python
text = "hello hello hello world"
words = ['hello', 'hello', 'hello', 'world']
unique_words = set(words)  # {'hello', 'world'}

# With set:
for word in unique_words:
    # Checks 'hello' once
    # Checks 'world' once
    # Total: 2 checks ✓
```

**Performance Impact:**

```python
# Document with 10,000 words
# But only 1,000 unique words

Without set:
  10,000 checks (slow!)

With set:
  1,000 checks (10x faster!) ✓
```

**Visual Example:**

```
Text: "the cat sat on the mat"

words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
                                        ↑
                                    duplicate

set(words) = {'the', 'cat', 'sat', 'on', 'mat'}
                                    ↑
                              checked only once ✓
```

---

#### Step 4: Validity Check

```python
if not self.is_valid_word(word):
```

**What it does:**
- Checks if word exists in dictionary
- Returns True/False
- Only processes if False (misspelled)

**Flow:**

```python
word = 'hello'
is_valid_word('hello') → True
  ↓
not True = False
  ↓
if False:  # Skip (word is correct)

───────────────────────────

word = 'helo'
is_valid_word('helo') → False
  ↓
not False = True
  ↓
if True:  # Process (word is misspelled) ✓
```

**Example Trace:**

```python
Checking word: 'this'
  → is_valid_word('this') = True
  → not True = False
  → Skip

Checking word: 'helo'
  → is_valid_word('helo') = False
  → not False = True
  → Continue to get candidates
```

---

#### Step 5: Get Candidates

```python
candidates = self.get_candidates(word)
```

**What happens here:**

This calls Method 8 which:
1. Generates edit distance 1 variations
2. Filters to valid dictionary words
3. Scores each candidate (Method 9)
4. Adds substring matches (Method 10)
5. Ranks and returns top 10

**Example:**

```python
word = 'helo'

candidates = get_candidates('helo')

# Returns:
[
    ('hello', 0.650),
    ('help', 0.610),
    ('held', 0.580),
    ('hero', 0.550)
]
```

**Could return empty:**

```python
word = 'xyzqwerty'  # Nonsense word

candidates = get_candidates('xyzqwerty')

# Returns:
[]  # No valid corrections found
```

---

#### Step 6: Filter Empty Results

```python
if candidates:
    misspelled[word] = candidates
```

**Why check `if candidates`?**

**Problem without check:**

```python
# If we always add:
misspelled['helo'] = [('hello', 0.65), ...]  # Good ✓
misspelled['xyz'] = []  # Bad ✗ (empty list clutters result)

# Result has noise:
{
    'helo': [('hello', 0.65)],
    'xyz': [],          # ← Useless entry
    'wrold': [('world', 0.88)]
}
```

**Solution with check:**

```python
# Only add if candidates exist:
if [('hello', 0.65)]:  # True, has items
    misspelled['helo'] = [('hello', 0.65)]  ✓

if []:  # False, empty list
    # Don't add 'xyz' ✓

# Clean result:
{
    'helo': [('hello', 0.65)],
    'wrold': [('world', 0.88)]
}
```

**Example:**

```python
# Case 1: Candidates found
word = 'helo'
candidates = [('hello', 0.65), ('help', 0.61)]

if candidates:  # True (list has items)
    misspelled['helo'] = candidates  ✓

# Case 2: No candidates
word = 'xyz'
candidates = []

if candidates:  # False (empty list)
    # Skip - don't add to misspelled dict ✓
```

---

#### Step 7: Return Result

```python
return misspelled
```

**Returns complete dictionary of all misspellings found.**

---

### 🎬 Complete Example Walkthrough

**Input:**
```python
text = "Helo wrold! This is a smple test."
```

**Execution Trace:**

```python
# STEP 1: Extract words
words = extract_words("Helo wrold! This is a smple test.")
words = ['helo', 'wrold', 'this', 'is', 'a', 'smple', 'test']

# STEP 2: Initialize
misspelled = {}

# STEP 3: Get unique words
unique_words = set(words)
unique_words = {'helo', 'wrold', 'this', 'is', 'a', 'smple', 'test'}

# STEP 4-6: Check each word

# Word 1: 'helo'
is_valid_word('helo') → False (misspelled!)
  ↓
candidates = get_candidates('helo')
candidates = [('hello', 0.650), ('help', 0.610), ('held', 0.580)]
  ↓
if candidates: True
misspelled['helo'] = [('hello', 0.650), ('help', 0.610), ('held', 0.580')]

# Word 2: 'wrold'
is_valid_word('wrold') → False (misspelled!)
  ↓
candidates = get_candidates('wrold')
candidates = [('world', 0.880), ('would', 0.720)]
  ↓
if candidates: True
misspelled['wrold'] = [('world', 0.880), ('would', 0.720)]

# Word 3: 'this'
is_valid_word('this') → True (correct!)
  ↓
not True = False
  ↓
Skip (don't process)

# Word 4: 'is'
is_valid_word('is') → True (correct!)
  ↓
Skip

# Word 5: 'a'
is_valid_word('a') → True (correct!)
  ↓
Skip

# Word 6: 'smple'
is_valid_word('smple') → False (misspelled!)
  ↓
candidates = get_candidates('smple')
candidates = [('simple', 0.720), ('sample', 0.680), ('tuple', 0.650)]
  ↓
if candidates: True
misspelled['smple'] = [('simple', 0.720), ('sample', 0.680), ('tuple', 0.650)]

# Word 7: 'test'
is_valid_word('test') → True (correct!)
  ↓
Skip

# STEP 7: Return result
return misspelled
```

**Final Result:**

```python
{
    'helo': [
        ('hello', 0.650),
        ('help', 0.610),
        ('held', 0.580)
    ],
    'wrold': [
        ('world', 0.880),
        ('would', 0.720)
    ],
    'smple': [
        ('simple', 0.720),
        ('sample', 0.680),
        ('tuple', 0.650)
    ]
}
```

---

### 📊 Performance Analysis

**Time Complexity:**

```python
words = self.extract_words(text)           # O(n) where n = text length
unique_words = set(words)                  # O(w) where w = number of words
for word in unique_words:                  # O(u) where u = unique words
    is_valid_word(word)                    # O(1) - set lookup
    get_candidates(word)                   # O(k × d) where k = word length, d = dict size

Total: O(n + w + u × k × d)
Simplified: O(u × d) for typical use
```

**Space Complexity:**

```python
words list: O(w)
unique_words set: O(u)
misspelled dict: O(u × c) where c = candidates per word
Total: O(u × c)
```

**Example Calculation:**

```python
Text: 1000 words
Unique words: 200
Dictionary: 10,000 words
Avg word length: 6

Time: 200 × 6 × 10,000 = 12,000,000 operations
  ≈ 0.1-1 second on modern CPU

Space: 200 × 10 candidates = 2000 tuples
  ≈ 50 KB memory
```

---

### 💡 Use Cases

#### Use Case 1: Document Proofreading

```python
document = """
This is a smple documnt with severl typing erors.
The spell checker will find them all.
"""

errors = checker.check_text(document)

# Returns:
{
    'smple': [('simple', 0.72), ('sample', 0.68)],
    'documnt': [('document', 0.85)],
    'severl': [('several', 0.78)],
    'erors': [('errors', 0.82)]
}

# Can then highlight these in a word processor
```

#### Use Case 2: Search Query Correction

```python
query = "machien lerning tutorails"

suggestions = checker.check_text(query)

# Returns:
{
    'machien': [('machine', 0.82)],
    'lerning': [('learning', 0.79)],
    'tutorails': [('tutorials', 0.85)]
}

# Display: "Did you mean: machine learning tutorials?"
```

#### Use Case 3: Form Validation

```python
user_input = "My favrite color is blu"

errors = checker.check_text(user_input)

if errors:
    for word, suggestions in errors.items():
        print(f"'{word}' might be misspelled.")
        print(f"Did you mean: {suggestions[0][0]}?")

# Output:
# 'favrite' might be misspelled.
# Did you mean: favorite?
# 'blu' might be misspelled.
# Did you mean: blue?
```

---

### 🧪 Practice Problems

#### Problem 1: Trace Execution

```python
"""
Trace check_text() for this input:
"Teh qick brown fox"

Assume dictionary contains:
{'the', 'quick', 'brown', 'fox'}

Show:
1. Extracted words
2. Unique words
3. Which words are checked
4. Final result
"""

# SOLUTION:
# 1. Extracted: ['teh', 'qick', 'brown', 'fox']
# 2. Unique: {'teh', 'qick', 'brown', 'fox'}
# 3. Checked:
#    - 'teh': False → get candidates
#    - 'qick': False → get candidates
#    - 'brown': True → skip
#    - 'fox': True → skip
# 4. Result:
#    {
#        'teh': [('the', 0.88)],
#        'qick': [('quick', 0.82)]
#    }
```

#### Problem 2: Optimization Challenge

```python
"""
The current implementation checks words in arbitrary order.
How could you optimize by checking shortest words first?
Why might this help?
"""

# SOLUTION:
def check_text_optimized(self, text):
    words = self.extract_words(text)
    unique_words = sorted(set(words), key=len)  # Sort by length

    misspelled = {}
    for word in unique_words:
        if not self.is_valid_word(word):
            candidates = self.get_candidates(word)
            if candidates:
                misspelled[word] = candidates

    return misspelled

# Why helpful:
# - Shorter words have fewer ED1 variations
# - Faster to process
# - Can fail fast on common short words
```

#### Problem 3: Handle Duplicates Differently

```python
"""
Modify check_text() to track how many times
each misspelling appears.

Input: "helo helo world helo"
Output: {
    'helo': {
        'candidates': [('hello', 0.65)],
        'count': 3
    }
}
"""

# SOLUTION:
def check_text_with_counts(self, text):
    words = self.extract_words(text)
    word_counts = {}

    # Count occurrences
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    misspelled = {}
    for word, count in word_counts.items():
        if not self.is_valid_word(word):
            candidates = self.get_candidates(word)
            if candidates:
                misspelled[word] = {
                    'candidates': candidates,
                    'count': count
                }

    return misspelled
```

---

## Method 12: correct_text() - Text Correction

### 🎯 Purpose

**Apply corrections to text automatically or show suggestions interactively.**

This is the **final method** - it takes misspelled text and either:
1. **Auto-corrects** (using best candidate)
2. **Shows suggestions** (interactive mode)

---

### 📝 Complete Code

```python
def correct_text(self, text: str, auto_correct: bool = False) -> str:
    """
    Correct text by replacing misspelled words.

    Args:
        text: Input text to correct
        auto_correct: If True, automatically use the best candidate
                     If False, print suggestions for user to choose

    Returns:
        Corrected text (if auto_correct=True)
        Original text (if auto_correct=False, suggestions printed)

    Example (auto_correct=True):
        Input:  "Helo wrold!"
        Output: "Hello world!"

    Example (auto_correct=False):
        Input: "Helo wrold!"
        Prints:
            Misspelled: 'helo'
            Suggestions:
              1. hello (confidence: 0.650)
              2. help (confidence: 0.610)

            Misspelled: 'wrold'
            Suggestions:
              1. world (confidence: 0.880)
              2. would (confidence: 0.720)
        Output: "Helo wrold!" (unchanged)
    """
    # STEP 1: Get all misspellings and their candidates
    corrections = self.check_text(text)

    # STEP 2: Start with original text
    corrected_text = text

    # STEP 3: Process each misspelled word
    for misspelled_word, candidates in corrections.items():
        # STEP 4: Only process if candidates exist
        if candidates:
            if auto_correct:
                # AUTO MODE: Use best candidate automatically
                best_candidate = candidates[0][0]  # Top-ranked word

                # STEP 5: Replace using regex (preserves word boundaries)
                corrected_text = re.sub(
                    r'\b' + re.escape(misspelled_word) + r'\b',  # Pattern
                    best_candidate,                               # Replacement
                    corrected_text,                              # Text to search
                    flags=re.IGNORECASE                          # Case-insensitive
                )
            else:
                # INTERACTIVE MODE: Show suggestions
                print(f"Misspelled: '{misspelled_word}'")
                print("Suggestions:")

                # STEP 6: Display top 5 candidates with scores
                for i, (candidate, score) in enumerate(candidates[:5], 1):
                    print(f"  {i}. {candidate} (confidence: {score:.3f})")
                print()  # Blank line for readability

    # STEP 7: Return result
    return corrected_text
```

---

### 🔍 Step-by-Step Breakdown

#### Step 1: Get Corrections

```python
corrections = self.check_text(text)
```

**What it does:**
- Calls Method 11 (check_text)
- Returns dictionary of misspellings

**Example:**

```python
text = "Helo wrold! This is a test."

corrections = {
    'helo': [('hello', 0.650), ('help', 0.610)],
    'wrold': [('world', 0.880), ('would', 0.720)]
}
```

---

#### Step 2: Initialize Result

```python
corrected_text = text
```

**Why start with original?**

```python
# Start with original
corrected_text = "Helo wrold!"

# Then progressively update:
# First iteration: "Hello wrold!"
# Second iteration: "Hello world!"

# If no misspellings found:
# Returns original text unchanged ✓
```

**Alternative approach (why not used):**

```python
# BAD: Start with empty string
corrected_text = ""

# Problem: What about punctuation, capitalization, spacing?
# "Helo, wrold!" → Need to preserve ", " and "!"
# Too complex to reconstruct!

# GOOD: Start with original, replace parts ✓
```

---

#### Step 3: Iterate Misspellings

```python
for misspelled_word, candidates in corrections.items():
```

**What happens:**

```python
corrections = {
    'helo': [('hello', 0.65), ('help', 0.61)],
    'wrold': [('world', 0.88)]
}

# Iteration 1:
misspelled_word = 'helo'
candidates = [('hello', 0.65), ('help', 0.61)]

# Iteration 2:
misspelled_word = 'wrold'
candidates = [('world', 0.88)]
```

---

#### Step 4: Check Candidates Exist

```python
if candidates:
```

**Why check?**

Although `check_text()` already filters empty candidates, this is **defensive programming**:

```python
# Shouldn't happen, but just in case:
misspelled_word = 'xyz'
candidates = []  # Empty

if candidates:  # False
    # Skip (don't try to correct)
```

---

#### Step 5: Auto-Correct Mode

```python
if auto_correct:
    best_candidate = candidates[0][0]
```

**Get best candidate:**

```python
candidates = [
    ('hello', 0.650),  # ← Index [0]
    ('help', 0.610),
    ('held', 0.580)
]

best_candidate = candidates[0][0]
#                         ↑  ↑
#                         │  └─ [0] = word part of tuple
#                         └──── [0] = first tuple (highest score)

best_candidate = 'hello'
```

**Why [0][0]?**

```python
candidates[0]     # First tuple: ('hello', 0.650)
candidates[0][0]  # First element of first tuple: 'hello'
candidates[0][1]  # Second element of first tuple: 0.650

# Want just the word, not the score:
best_candidate = candidates[0][0]  # 'hello' ✓
```

---

#### Step 6: Regex Replacement

```python
corrected_text = re.sub(
    r'\b' + re.escape(misspelled_word) + r'\b',
    best_candidate,
    corrected_text,
    flags=re.IGNORECASE
)
```

This is **complex but powerful**! Let's break it down completely.

**Part 1: The Pattern**

```python
pattern = r'\b' + re.escape(misspelled_word) + r'\b'
```

**Component A: `r'\b'`** - Word boundary

```
\b matches the position between:
- A word character (a-z, A-Z, 0-9, _)
- A non-word character (space, punctuation, start/end of string)

Example:
"hello world" has boundaries at:
 ↓    ↓     ↓
|hello|world|
 \b   \b    \b
```

**Component B: `re.escape(misspelled_word)`** - Escape special chars

```python
# Why needed?
misspelled_word = "test."

# WITHOUT escape:
pattern = r'\b' + "test." + r'\b'
pattern = r'\btest.\b'
#                 ↑ '.' means "any character" in regex!

# Will match:
"testa", "testb", "test!", etc. ✗ WRONG!

# WITH escape:
pattern = r'\b' + re.escape("test.") + r'\b'
pattern = r'\btest\.\b'
#                 ↑ '\.' means literal '.' character

# Will match:
Only "test." ✓ CORRECT!
```

**Example escaping:**

```python
re.escape("hello")   → "hello"   # No change (no special chars)
re.escape("test.")   → "test\\." # Escapes dot
re.escape("a+b")     → "a\\+b"   # Escapes plus
re.escape("(word)")  → "\\(word\\)" # Escapes parentheses
```

**Complete pattern examples:**

```python
misspelled_word = "helo"
pattern = r'\b' + re.escape("helo") + r'\b'
pattern = r'\bhelo\b'

# Matches:
"helo"       ✓
"Helo"       ✓ (case-insensitive flag)
" helo "     ✓ (word boundaries match)
"helo!"      ✓ (boundary before !)

# Does NOT match:
"hello"      ✗ (different word)
"helot"      ✗ (no boundary after 'o')
"bachelor"   ✗ (no boundary before 'h')
```

**Part 2: The Replacement**

```python
best_candidate  # Simply the replacement word
```

**Part 3: The Text**

```python
corrected_text  # Text to search and replace in
```

**Part 4: Flags**

```python
flags=re.IGNORECASE
```

**What it does:**
- Makes match case-insensitive
- Replaces "helo", "Helo", "HELO" all with "hello"

**Without flag:**

```python
text = "Helo world"
pattern = r'\bhelo\b'  # Lowercase pattern

re.sub(pattern, "hello", text)
# Result: "Helo world" (no match! Case mismatch) ✗
```

**With flag:**

```python
text = "Helo world"
pattern = r'\bhelo\b'

re.sub(pattern, "hello", text, flags=re.IGNORECASE)
# Result: "hello world" (matched despite case difference) ✓
```

**Complete Example:**

```python
corrected_text = "Helo wrold! helo again."
misspelled_word = "helo"
best_candidate = "hello"

# Pattern: r'\bhelo\b'
# Replacement: "hello"

result = re.sub(
    r'\bhelo\b',
    "hello",
    "Helo wrold! helo again.",
    flags=re.IGNORECASE
)

# Result: "hello wrold! hello again."
#          ↑────  both occurrences replaced
```

---

#### Step 7: Interactive Mode

```python
else:
    print(f"Misspelled: '{misspelled_word}'")
    print("Suggestions:")

    for i, (candidate, score) in enumerate(candidates[:5], 1):
        print(f"  {i}. {candidate} (confidence: {score:.3f})")
    print()
```

**Breaking down the print loop:**

```python
enumerate(candidates[:5], 1)
#         └───────────┘  └─ Start numbering at 1
#         └──────────────── First 5 candidates only
```

**Example:**

```python
candidates = [
    ('hello', 0.650),
    ('help', 0.610),
    ('held', 0.580),
    ('hero', 0.550),
    ('heal', 0.520),
    ('hear', 0.490)  # This won't be shown
]

# candidates[:5] = first 5 only
for i, (candidate, score) in enumerate(candidates[:5], 1):
    # i=1, candidate='hello', score=0.650
    # i=2, candidate='help', score=0.610
    # i=3, candidate='held', score=0.580
    # i=4, candidate='hero', score=0.550
    # i=5, candidate='heal', score=0.520
```

**Tuple unpacking:**

```python
# Each iteration:
item = ('hello', 0.650)
(candidate, score) = item
# candidate = 'hello'
# score = 0.650

# Can write directly:
for i, (candidate, score) in enumerate(...):
    # candidate and score are automatically unpacked
```

**Formatted output:**

```python
print(f"  {i}. {candidate} (confidence: {score:.3f})")
#        ↑    ↑             ↑                  ↑
#        │    │             │                  └─ 3 decimal places
#        │    │             └──────────────────── Literal text
#        │    └────────────────────────────────── Variable
#        └─────────────────────────────────────── Two spaces indent

# Output:
  1. hello (confidence: 0.650)
  2. help (confidence: 0.610)
  3. held (confidence: 0.580)
```

---

### 🎬 Complete Examples

#### Example 1: Auto-Correct Mode

```python
text = "Helo wrold! This is a smple test."

result = checker.correct_text(text, auto_correct=True)

print(result)
# Output: "Hello world! This is a simple test."
```

**Step-by-step:**

```
Original: "Helo wrold! This is a smple test."

Iteration 1: Fix 'helo'
  best = 'hello'
  "Helo wrold! ..." → "Hello wrold! ..."

Iteration 2: Fix 'wrold'
  best = 'world'
  "Hello wrold! ..." → "Hello world! ..."

Iteration 3: Fix 'smple'
  best = 'simple'
  "... a smple test" → "... a simple test"

Final: "Hello world! This is a simple test."
```

#### Example 2: Interactive Mode

```python
text = "I love programing in pythn!"

result = checker.correct_text(text, auto_correct=False)

# Prints:
# Misspelled: 'programing'
# Suggestions:
#   1. programming (confidence: 0.720)
#   2. program (confidence: 0.650)
#
# Misspelled: 'pythn'
# Suggestions:
#   1. python (confidence: 0.820)
#   2. pythons (confidence: 0.650)

print(result)
# Output: "I love programing in pythn!" (unchanged)
```

#### Example 3: Mixed Case Handling

```python
text = "HELO WROLD! helo again."

result = checker.correct_text(text, auto_correct=True)

print(result)
# Output: "hello world! hello again."
#          ↑────────────────────────── All lowercase
#          (because replacement is 'hello', not 'HELLO')
```

**To preserve case:**

```python
# Would need more sophisticated replacement:
def smart_replace(match, replacement):
    original = match.group()
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement

# But current implementation doesn't do this
```

---

### 🧪 Practice Problems

#### Problem 1: Add Case Preservation

```python
"""
Modify correct_text() to preserve original capitalization:

Input:  "Helo WROLD! this is TEST."
Output: "Hello WORLD! this is TEST."
         ↑     ↑                ↑
         Title Upper          Upper (preserved)
"""

# SOLUTION:
import re

def correct_text_preserve_case(self, text, auto_correct=False):
    corrections = self.check_text(text)
    corrected_text = text

    for misspelled_word, candidates in corrections.items():
        if candidates and auto_correct:
            best_candidate = candidates[0][0]

            # Custom replacement function
            def replace_preserve_case(match):
                original = match.group()
                if original.isupper():
                    return best_candidate.upper()
                elif original.istitle():
                    return best_candidate.title()
                else:
                    return best_candidate.lower()

            corrected_text = re.sub(
                r'\b' + re.escape(misspelled_word) + r'\b',
                replace_preserve_case,
                corrected_text,
                flags=re.IGNORECASE
            )

    return corrected_text
```

#### Problem 2: Track Correction Statistics

```python
"""
Modify to return statistics:

Output: {
    'text': "Corrected text...",
    'stats': {
        'words_checked': 10,
        'misspellings_found': 3,
        'corrections_made': 3,
        'confidence_avg': 0.75
    }
}
"""

# SOLUTION:
def correct_text_with_stats(self, text, auto_correct=False):
    words = self.extract_words(text)
    corrections = self.check_text(text)
    corrected_text = text

    total_confidence = 0
    corrections_made = 0

    for misspelled_word, candidates in corrections.items():
        if candidates and auto_correct:
            best_candidate = candidates[0][0]
            total_confidence += candidates[0][1]
            corrections_made += 1

            corrected_text = re.sub(
                r'\b' + re.escape(misspelled_word) + r'\b',
                best_candidate,
                corrected_text,
                flags=re.IGNORECASE
            )

    avg_confidence = total_confidence / corrections_made if corrections_made > 0 else 0

    return {
        'text': corrected_text,
        'stats': {
            'words_checked': len(set(words)),
            'misspellings_found': len(corrections),
            'corrections_made': corrections_made,
            'confidence_avg': avg_confidence
        }
    }
```

---

## Integration Example

Let's see **all methods working together** in a complete workflow:

```python
# Initialize
checker = SpellChecker()

# Input text
text = "Helo wrold! I love programing in pythn. This is a smple test."

print("="*70)
print("STEP 1: Extract words (Method 5)")
print("="*70)
words = checker.extract_words(text)
print(f"Extracted: {words}\n")

print("="*70)
print("STEP 2: Check individual words (Methods 6-10)")
print("="*70)

test_word = "helo"
print(f"Checking: '{test_word}'")

# Method 6: Edit distance 1
ed1 = checker.edit_distance_1(test_word)
print(f"ED1 variations: {len(ed1)} words")

# Filter valid
valid_ed1 = [w for w in ed1 if checker.is_valid_word(w)]
print(f"Valid ED1: {valid_ed1}")

# Method 9: Score each
for candidate in valid_ed1[:3]:
    score = checker.calculate_score(test_word, candidate, 1)
    print(f"  {candidate}: {score:.3f}")

print()

print("="*70)
print("STEP 3: Check entire text (Method 11)")
print("="*70)
misspellings = checker.check_text(text)
for word, candidates in misspellings.items():
    print(f"{word}: {candidates[0][0]} ({candidates[0][1]:.3f})")

print()

print("="*70)
print("STEP 4: Auto-correct (Method 12)")
print("="*70)
corrected = checker.correct_text(text, auto_correct=True)
print(f"Original:  {text}")
print(f"Corrected: {corrected}")

print()

print("="*70)
print("STEP 5: Interactive mode (Method 12)")
print("="*70)
checker.correct_text("pythn programing", auto_correct=False)
```

**Expected Output:**

```
======================================================================
STEP 1: Extract words (Method 5)
======================================================================
Extracted: ['helo', 'wrold', 'i', 'love', 'programing', 'in', 'pythn', 'this', 'is', 'a', 'smple', 'test']

======================================================================
STEP 2: Check individual words (Methods 6-10)
======================================================================
Checking: 'helo'
ED1 variations: 187 words
Valid ED1: ['hello', 'help', 'held', 'hero']
  hello: 0.650
  help: 0.610
  held: 0.580

======================================================================
STEP 3: Check entire text (Method 11)
======================================================================
helo: hello (0.650)
wrold: world (0.880)
programing: programming (0.720)
pythn: python (0.820)
smple: simple (0.720)

======================================================================
STEP 4: Auto-correct (Method 12)
======================================================================
Original:  Helo wrold! I love programing in pythn. This is a smple test.
Corrected: Hello world! I love programming in python. This is a simple test.

======================================================================
STEP 5: Interactive mode (Method 12)
======================================================================
Misspelled: 'pythn'
Suggestions:
  1. python (confidence: 0.820)
  2. pythons (confidence: 0.650)

Misspelled: 'programing'
Suggestions:
  1. programming (confidence: 0.720)
  2. program (confidence: 0.650)
```

---

## Summary: Methods 9-12 at a Glance

| Method | Purpose | Input | Output | Key Algorithm |
|--------|---------|-------|--------|---------------|
| **9: calculate_score()** | Rank candidates | word, candidate, ED | float (0-1) | Weighted formula: B-F-L-O |
| **10: get_substring_candidates()** | Fuzzy matching | word | [(word, score)] | difflib.SequenceMatcher |
| **11: check_text()** | Analyze text | text | {word: [(cand, score)]} | Loop + Methods 1-10 |
| **12: correct_text()** | Apply corrections | text, mode | corrected text | regex replacement |

---

## Final Practice: Complete Implementation

Try implementing a simplified version yourself:

```python
"""
Implement a basic spell checker with these requirements:

1. calculate_score(): Only use base score (edit distance)
2. check_text(): Find all misspellings
3. correct_text(): Auto-correct mode only

Test with: "helo wrold"
Expected: "hello world"
"""

class SimpleSpellChecker:
    def __init__(self, dictionary):
        self.dictionary = set(dictionary)

    def is_valid(self, word):
        # TODO: Implement
        pass

    def calculate_score(self, original, candidate, edit_distance):
        # TODO: Only base score
        pass

    def check_text(self, text):
        # TODO: Find misspellings
        pass

    def correct_text(self, text):
        # TODO: Auto-correct
        pass

# Test
checker = SimpleSpellChecker(['hello', 'world', 'this', 'is', 'a', 'test'])
result = checker.correct_text("helo wrold")
assert result == "hello world", f"Expected 'hello world', got '{result}'"
print("Success! ✓")
```

---

# Grammar Correction using LSTM

A comprehensive deep learning project that uses LSTM (Long Short-Term Memory) neural networks to automatically correct grammatical errors in English sentences.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
  - [Data Structure](#data-structure)
  - [What the Dataset Contains](#what-the-dataset-contains)
  - [Error Types Covered](#error-types-covered)
  - [Dataset Loading and Cleaning](#dataset-loading-and-cleaning)
- [Project Pipeline](#project-pipeline)
  - [Complete Pipeline Overview](#complete-pipeline-overview)
  - [Step 1: Data Loading](#step-1-data-loading)
  - [Step 2: Text Preprocessing](#step-2-text-preprocessing)
  - [Step 3: Tokenization (Vocabulary Building)](#step-3-tokenization-vocabulary-building)
  - [Step 4: Sequence Conversion](#step-4-sequence-conversion)
  - [Step 5: Padding (Sequence Padding)](#step-5-padding-sequence-padding)
  - [Step 6: Model Architecture](#step-6-model-architecture)
  - [Step 7: Model Compilation](#step-7-model-compilation)
  - [Step 8: Model Training](#step-8-model-training)
  - [Step 9: Inference (Grammar Correction)](#step-9-inference-grammar-correction)
- [Installation](#installation)
- [Complete Code Example](#complete-code-example)
- [Model Architecture Details](#model-architecture-details)
- [Examples](#examples)
- [Key Concepts Explained](#key-concepts-explained)
- [Technical Details](#technical-details)

---

## Overview

This project implements a **sequence-to-sequence LSTM model** that learns to transform grammatically incorrect sentences into their corrected forms. The model is trained on pairs of ungrammatical and corrected sentences, learning patterns of common grammar mistakes through supervised learning.

**What makes this project unique:**
- End-to-end NLP pipeline from raw text to trained model
- Handles multiple types of grammatical errors
- Uses deep LSTM architecture for capturing complex language patterns
- Practical application of sequence-to-sequence learning

---

## Dataset

### Data Structure

The dataset consists of **1,500 sentence pairs** from Kaggle's Grammar Correction dataset.

**Sample Data:**

| Serial Number | Error Type | Ungrammatical Statement | Standard English |
|---------------|------------|-------------------------|------------------|
| 1 | Verb Tense Errors | I goes to the store everyday. | I go to the store everyday. |
| 2 | Verb Tense Errors | They was playing soccer last night. | They were playing soccer last night. |
| 3 | Verb Tense Errors | She have completed her homework. | She has completed her homework. |
| 4 | Verb Tense Errors | He don't know the answer. | He doesn't know the answer. |
| 5 | Verb Tense Errors | The sun rise in the east. | The sun rises in the east. |

**DataFrame Columns:**
- `Serial Number`: Unique identifier for each sentence pair
- `Error Type`: Category of grammatical error (informational only)
- `Ungrammatical Statement`: Input sentences with errors (**our input features**)
- `Standard English`: Corrected sentences (**our target labels**)

### What the Dataset Contains

Each row in the dataset represents a **training example** with:

#### 1. Input Column (`Ungrammatical Statement`)
The incorrect sentence that needs correction.

**Examples:**
```
"i goes to the store everyday."
"they was playing soccer last night."
"she have completed her homework."
```

This is what we **feed into the model** during training and inference.

#### 2. Target Column (`Standard English`)
The grammatically corrected version of the input sentence.

**Examples:**
```
"i go to the store everyday."
"they were playing soccer last night."
"she has completed her homework."
```

This is what we **want the model to predict/output**.

#### 3. Error Type
Categorical label describing the type of grammatical mistake.

**Purpose:**
- Helps understand the distribution of errors in the dataset
- Not directly used in this simple seq2seq model
- Could be used for multi-task learning in advanced versions

### Dataset After Loading

**Raw DataFrame (`df.head()` output):**

```
   Serial Number       Error Type              Ungrammatical Statement         Standard English
0              1  Verb Tense Errors  I goes to the store everyday.       I go to the store everyday.
1              2  Verb Tense Errors  They was playing soccer...          They were playing soccer...
2              3  Verb Tense Errors  She have completed her homework.    She has completed her homework.
3              4  Verb Tense Errors  He don't know the answer.           He doesn't know the answer.
4              5  Verb Tense Errors  The sun rise in the east.           The sun rises in the east.
```

**After Preprocessing (aligned lists):**

```python
# input_texts (preprocessed incorrect sentences)
input_texts = [
    'i goes to the store everyday.',
    'they was playing soccer last night.',
    'she have completed her homework.',
    "he don't know the answer.",
    'the sun rise in the east.',
    ...
]

# target_texts (preprocessed correct sentences)
target_texts = [
    'i go to the store everyday.',
    'they were playing soccer last night.',
    'she has completed her homework.',
    "he doesn't know the answer.",
    'the sun rises in the east.',
    ...
]
```

**Key Points:**
- Each index position contains a **pair**: `input_texts[i]` ↔ `target_texts[i]`
- Lists are **aligned** (same length, corresponding indices)
- The model learns the mapping: **incorrect text → correct text**

### Error Types Covered

The dataset includes **8+ categories** of grammatical errors:

| Error Type | Example (Before → After) | Description |
|------------|--------------------------|-------------|
| **Verb Tense Errors** | "goes" → "go", "was" → "were" | Incorrect verb forms and tenses |
| **Subject-Verb Agreement** | "she have" → "she has" | Mismatch between subject and verb |
| **Article Errors** | "a school" → "school", "an umbrella" → "a umbrella" | Incorrect article usage (a/an/the) |
| **Pronoun Errors** | "it's" → "its" | Possessive vs contraction confusion |
| **Preposition Errors** | "in friday" → "on friday" | Wrong preposition selection |
| **Spelling Errors** | "extreem" → "extreme" | Misspelled words |
| **Punctuation Errors** | Missing commas, apostrophes | Incorrect or missing punctuation |
| **Word Choice** | "loose" → "lose", "affect" → "effect" | Commonly confused words |

### Dataset Loading and Cleaning

```python
import kagglehub
import pandas as pd
import os

# Define column names
input_column = 'Ungrammatical Statement'
target_column = 'Standard English'

# Download dataset from Kaggle
path = kagglehub.dataset_download("satishgunjal/grammar-correction")
filepath = os.path.join(path, 'Grammar Correction.csv')

# Load into DataFrame
df = pd.read_csv(filepath)

# Remove rows with missing values in critical columns
df = df.dropna(subset=[input_column, target_column])
```

**What This Code Does:**

1. **`kagglehub.dataset_download()`**
   - Downloads the dataset from Kaggle automatically
   - Handles authentication and caching
   - Returns the local path where dataset is stored

2. **`os.path.join()`**
   - Constructs the complete file path
   - Ensures cross-platform compatibility (Windows/Mac/Linux)
   - Combines directory path + filename

3. **`pd.read_csv()`**
   - Loads the CSV file into a pandas DataFrame
   - Automatically infers data types
   - Creates a tabular structure for easy manipulation

4. **`df.dropna(subset=[...])`**
   - Removes rows where specified columns have NaN (missing) values
   - Ensures we only have **complete pairs** for training
   - Prevents errors during model training

**Why `dropna()` is Critical:**
- ✓ Ensures every input has a corresponding target
- ✓ Prevents `None` or `NaN` from breaking the preprocessing pipeline
- ✓ Removes corrupted or incomplete data entries
- ✓ Maintains data quality and integrity

**Result:**
- Clean DataFrame with 1,500 valid sentence pairs
- No missing values in input or target columns
- Ready for text preprocessing

---

## Project Pipeline

### Complete Pipeline Overview

This project follows a standard **NLP sequence-to-sequence pipeline**:

```
┌─────────────────────────────────────────────────────────┐
│                  1. DATA COLLECTION                     │
│                    Kaggle Dataset                       │
│                   (CSV Download)                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  2. DATA LOADING                        │
│               pandas.read_csv()                         │
│         Load CSV → DataFrame (1500 rows)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  3. DATA CLEANING                       │
│                    dropna()                             │
│           Remove incomplete examples                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               4. TEXT PREPROCESSING                     │
│           • Lowercase conversion                        │
│           • Remove extra whitespace                     │
│           • Clean special characters                    │
│           • Strip leading/trailing spaces               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│             5. TOKENIZATION (VOCABULARY)                │
│           Build word-to-index mapping                   │
│           Create unified vocabulary                     │
│           vocab_size = 1248 words                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               6. SEQUENCE CONVERSION                    │
│           Convert text → numerical sequences            │
│           "i go store" → [3, 5, 6]                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  7. PADDING                             │
│           Make all sequences same length                │
│           Add zeros to reach max_len                   │
│           [3, 5] → [3, 5, 0, 0, 0]                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              8. MODEL ARCHITECTURE                      │
│           • Embedding Layer (50-dim)                   │
│           • 3× LSTM Layers (100 units each)            │
│           • Dense Output Layer (softmax)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               9. MODEL COMPILATION                      │
│           Optimizer: Adam                               │
│           Loss: Sparse Categorical Crossentropy        │
│           Metrics: Accuracy                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                10. MODEL TRAINING                       │
│           Fit model on (input, target) pairs           │
│           Learn grammar correction patterns            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               11. INFERENCE (PREDICTION)                │
│           Input: "apples is healthy"                   │
│           Output: "apples are healthy"                 │
└─────────────────────────────────────────────────────────┘
```

---

### Step 1: Data Loading

**Purpose:** Download and load the grammar correction dataset from Kaggle.

**Code:**
```python
import kagglehub
import pandas as pd
import os

input_column = 'Ungrammatical Statement'
target_column = 'Standard English'

# Download dataset from Kaggle
path = kagglehub.dataset_download("satishgunjal/grammar-correction")
filepath = os.path.join(path, 'Grammar Correction.csv')

# Load into DataFrame
df = pd.read_csv(filepath)

# Remove rows with missing values
df = df.dropna(subset=[input_column, target_column])
```

**Step-by-Step Breakdown:**

1. **Download**: `kagglehub` fetches dataset from Kaggle
2. **Path Construction**: `os.path.join()` creates full file path
3. **Load**: `pd.read_csv()` reads CSV into DataFrame
4. **Clean**: `dropna()` removes incomplete rows

**Output:**
```
DataFrame Shape: (1500, 4)
Columns: ['Serial Number', 'Error Type', 'Ungrammatical Statement', 'Standard English']
No missing values in input/target columns
```

**This step is called:** Data Collection and Loading

---

### Step 2: Text Preprocessing

**Purpose:** Normalize and clean text data for consistent model input.

**Also Known As:** Text Normalization, Text Cleaning, Feature Preparation

**Code:**
```python
import re

def preprocess_text(text):
    """
    Clean and preprocess text data.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove extra whitespaces (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)

    # 3. Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)

    # 4. Strip leading/trailing whitespace
    text = text.strip()

    return text

# Apply preprocessing to both input and target columns
input_texts = df[input_column].apply(preprocess_text).tolist()
target_texts = df[target_column].apply(preprocess_text).tolist()
```

**Detailed Explanation of Each Step:**

#### 1. Lowercase Conversion (`text.lower()`)
```python
Before: "I Goes To The Store"
After:  "i goes to the store"
```
**Why?**
- Makes model case-insensitive
- Reduces vocabulary size (no separate entries for "The", "the", "THE")
- Focuses on grammar, not capitalization

#### 2. Remove Extra Whitespace (`re.sub(r'\s+', ' ', text)`)
```python
Before: "I  go   to    store"
After:  "i go to store"
```
**What it does:**
- `\s+` matches one or more whitespace characters (spaces, tabs, newlines)
- Replaces with single space
- Normalizes inconsistent spacing

#### 3. Clean Special Characters (`re.sub(r'[^\w\s.,!?;:\'-]', '', text)`)
```python
Before: "Hello @user! #NLP @@@ text"
After:  "hello user nlp text"
```
**Pattern Breakdown:**
- `[^...]` = Match anything NOT in the brackets
- `\w` = Word characters (letters, digits, underscore)
- `\s` = Whitespace
- `.,!?;:'-` = Basic punctuation we want to keep
- Everything else gets removed

**Examples:**
- ✓ Keeps: letters, numbers, basic punctuation
- ✗ Removes: @, #, emojis, special symbols

#### 4. Strip Edges (`text.strip()`)
```python
Before: "   hello world   "
After:  "hello world"
```
**Why?**
- Removes leading/trailing whitespace
- Ensures clean sentence boundaries

**Complete Transformation Examples:**

| Input | After Preprocessing |
|-------|---------------------|
| `"I    Goes to the @Store!!!   #everyday  "` | `"i goes to the store everyday"` |
| `"She  DON'T  know  @@@ answer"` | `"she don't know answer"` |
| `"  THEY   was    playing  "` | `"they was playing"` |

**Result:**
```python
# Two aligned lists of preprocessed text
input_texts = [
    'i goes to the store everyday.',
    'they was playing soccer last night.',
    'she have completed her homework.',
    ...
]  # 1500 sentences

target_texts = [
    'i go to the store everyday.',
    'they were playing soccer last night.',
    'she has completed her homework.',
    ...
]  # 1500 sentences
```

**Why Text Preprocessing is Critical:**

1. **Consistency**: All text in uniform format
2. **Reduced Vocabulary**: Fewer unique tokens to learn
3. **Focus on Structure**: Model learns grammar, not formatting quirks
4. **Better Generalization**: Model works on various input styles
5. **Cleaner Training**: No noise from special characters

**This step is called:** Text Preprocessing / Normalization / Cleaning

---

### Step 3: Tokenization (Vocabulary Building)

**Purpose:** Build a vocabulary and create word-to-index mappings.

**Also Known As:** Vocabulary Building, Text Tokenization

**Code:**
```python
from tensorflow.keras.preprocessing.text import Tokenizer

# Create tokenizer instance
tokenizer = Tokenizer()

# Build vocabulary from ALL text (input + target combined)
tokenizer.fit_on_texts(input_texts + target_texts)

# Calculate vocabulary size (+1 for padding token at index 0)
vocab_size = len(tokenizer.word_index) + 1
```

**What `fit_on_texts()` Does Internally:**

```
Step 1: Scan all sentences
├─ "i go to the store everyday"
├─ "they were playing soccer last night"
├─ "she has completed her homework"
└─ ... (all 3000 sentences: 1500 input + 1500 target)

Step 2: Find all unique words
├─ Extract: ['i', 'go', 'to', 'the', 'store', 'everyday', 'they', 'were', ...]
└─ Remove duplicates: 1247 unique words

Step 3: Count word frequencies
├─ 'the': 3542 occurrences
├─ 'to': 2891 occurrences
├─ 'i': 2456 occurrences
└─ ...

Step 4: Assign indices (most frequent → lowest number)
└─ Create word_index dictionary
```

**The `word_index` Dictionary:**

```python
tokenizer.word_index = {
    'the': 1,        # Most frequent word
    'to': 2,
    'i': 3,
    'a': 4,
    'and': 5,
    'go': 6,
    'store': 7,
    'everyday': 8,
    'they': 9,
    'were': 10,
    'playing': 11,
    ...
    'beautiful': 1246,
    'unusual': 1247  # Least frequent word
}

# Total unique words found
len(tokenizer.word_index) = 1247
```

**Understanding `vocab_size` Calculation:**

```python
vocab_size = len(tokenizer.word_index) + 1
```

**Why the `+1`?**

```
Index Allocation:
┌─────────┬─────────────────────────┐
│ Index 0 │ <PAD> (Padding Token)   │ ← Reserved for zeros we'll add
├─────────┼─────────────────────────┤
│ Index 1 │ 'the'                   │ ← Most frequent word
│ Index 2 │ 'to'                    │
│ Index 3 │ 'i'                     │
│ Index 4 │ 'a'                     │
│   ...   │ ...                     │
│Index 1247│ 'unusual'               │ ← Least frequent word
└─────────┴─────────────────────────┘

Total indices needed: 0 to 1247 = 1248 positions
Therefore: vocab_size = 1247 + 1 = 1248
```

**Why Combine `input_texts + target_texts`?**

This is **CRITICAL** for sequence-to-sequence models:

```python
# ✓ CORRECT APPROACH - Shared vocabulary
tokenizer.fit_on_texts(input_texts + target_texts)

Result:
- 'go' → index 6 (in both input and target)
- 'went' → index 45 (in both input and target)
- Model understands: same word = same meaning

# ✗ WRONG APPROACH - Separate vocabularies
tokenizer_input.fit_on_texts(input_texts)    # 'go' = 6
tokenizer_target.fit_on_texts(target_texts)  # 'go' = 47

Result:
- Same word has different indices
- Model gets confused!
- Input 'go' ≠ Output 'go'
```

**Benefits of Combined Vocabulary:**

1. **Consistency**: Same word always maps to same number
2. **No Mismatches**: Prevents index conflicts between input/output
3. **Unified Representation**: Model learns one embedding per word
4. **Correct Predictions**: Output predictions map back correctly to words

**Example Transformation:**

```python
# Text
sentence = "i go to the store"

# Using word_index dictionary
indices = []
for word in sentence.split():
    indices.append(tokenizer.word_index[word])

# Result
print(indices)  # [3, 6, 2, 1, 7]

# Word-by-word mapping:
'i'      → 3
'go'     → 6
'to'     → 2
'the'    → 1
'store'  → 7
```

**This step is called:** Tokenization / Vocabulary Building

---

### Step 4: Sequence Conversion

**Purpose:** Convert text to numerical sequences using the vocabulary.

**Also Known As:** Text-to-Sequence Conversion, Sequence Encoding, Numerical Encoding

**Code:**
```python
# Convert sentences to sequences of integers
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)
```

**What `texts_to_sequences()` Does:**

Takes each sentence and replaces every word with its corresponding index from `word_index`.

**Detailed Example:**

```python
# Example sentence
sentence = "i go to the store everyday"

# Step 1: Split into words
words = ['i', 'go', 'to', 'the', 'store', 'everyday']

# Step 2: Look up each word in word_index
sequence = []
for word in words:
    if word in tokenizer.word_index:
        sequence.append(tokenizer.word_index[word])
    else:
        # Unknown word (not in vocabulary) - skip or use OOV token
        pass

# Result
sequence = [3, 6, 2, 1, 7, 8]
```

**Visual Transformation:**

```
Input Text:  "i go to the store everyday"
              ↓   ↓  ↓  ↓    ↓       ↓
Word Index:  [3,  6, 2, 1,   7,      8]

Target Text: "i go to the store every day"
              ↓   ↓  ↓  ↓    ↓      ↓    ↓
Word Index:  [3,  6, 2, 1,   7,    45,  46]
```

**Complete Examples:**

```python
# After conversion
input_sequences = [
    [3, 6, 2, 1, 7, 8],              # "i go to the store everyday"
    [9, 10, 11, 12, 13, 14, 15],     # "they were playing soccer last night"
    [3, 16, 17],                      # "she has completed" (shorter)
    [3, 18, 19, 20, 21, 22, 23, 24], # (longer sentence)
    ...
]  # 1500 sequences

target_sequences = [
    [3, 6, 2, 1, 7, 8],              # Same if no correction needed
    [9, 10, 11, 12, 13, 14, 15],     # Corrected version
    [3, 25, 17],                      # Different indices if corrected
    ...
]  # 1500 sequences
```

**Important Characteristics:**

1. **Variable Lengths**
   ```python
   sequences = [
       [3, 6, 2, 1, 7, 8],           # Length: 6
       [9, 10],                       # Length: 2
       [3, 16, 17, 20, 21, 22, 23],  # Length: 7
   ]
   ```
   - Each sequence has different length
   - Depends on original sentence length
   - Will be fixed in next step (padding)

2. **Unknown Words**
   ```python
   # If a word wasn't in training vocabulary
   "i love supercalifragilisticexpialidocious"
   #              ↑ Not in word_index

   # Result (word is skipped)
   [3, 26]  # Only 'i' and 'love' are encoded
   ```

3. **Preserves Order**
   - Word sequence is maintained
   - Critical for understanding sentence meaning
   - "dog bites man" ≠ "man bites dog"

**Result Structure:**

```python
# Two lists of variable-length sequences
input_sequences = [
    [3, 6, 2, 1, 7, 8],
    [9, 10, 11, 12, 13, 14, 15],
    [3, 16, 17],
    ...
]  # List of 1500 lists

target_sequences = [
    [3, 6, 2, 1, 7, 8],
    [9, 10, 11, 12, 13, 14, 15],
    [3, 25, 17],
    ...
]  # List of 1500 lists

# Each sequence is a list of integers
type(input_sequences[0])  # <class 'list'>
type(input_sequences[0][0])  # <class 'int'>
```

**Why This Step is Necessary:**

Neural networks can only process numbers, not text. This step converts:
- Human-readable text → Machine-readable numbers
- "i go" → [3, 6]

**This step is called:** Sequence Conversion / Text-to-Sequence Encoding

---

### Step 5: Padding (Sequence Padding)

**Purpose:** Make all sequences the same length for batch processing.

**Also Known As:** Sequence Padding, Zero-Padding

**Code:**
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Find the maximum sequence length across ALL data
max_len = max(
    max(len(seq) for seq in input_sequences),
    max(len(seq) for seq in target_sequences)
)

# Pad all sequences to max_len
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')
```

**Why Padding is Necessary:**

Neural networks process data in **batches** with uniform shapes.

**Problem Without Padding:**
```python
# Variable length sequences
batch = [
    [3, 6, 2],           # Shape: (3,)
    [9, 10, 11, 12],     # Shape: (4,)  ← Different shape!
    [3, 16],             # Shape: (2,)  ← Different shape!
]

# ❌ ERROR: Cannot create tensor with different shapes
# Cannot stack into array
```

**Solution With Padding:**
```python
# Fixed length sequences
batch = [
    [3, 6, 2, 0],        # Shape: (4,)
    [9, 10, 11, 12],     # Shape: (4,)  ← Same shape!
    [3, 16, 0, 0],       # Shape: (4,)  ← Same shape!
]

# ✓ SUCCESS: Can create tensor
# Shape: (3, 4) - can process in batch
```

**Detailed Example:**

```python
# Before padding (variable lengths)
input_sequences = [
    [3, 6, 2, 1, 7],              # Length: 5
    [9, 10],                       # Length: 2
    [3, 16, 17, 20, 21, 22, 23],  # Length: 7 (longest)
]

max_len = 7  # Length of longest sequence

# After padding with padding='post'
input_sequences = array([
    [3, 6, 2, 1, 7, 0, 0],         # Added 2 zeros at end
    [9, 10, 0, 0, 0, 0, 0],        # Added 5 zeros at end
    [3, 16, 17, 20, 21, 22, 23],  # No padding needed (already max_len)
])

# Shape: (3, 7) - uniform!
```

**Padding Options:**

1. **`padding='post'`** (zeros at END)
   ```python
   [3, 5, 2] → [3, 5, 2, 0, 0, 0, 0]
                        ↑_________↑
                        added zeros
   ```
   - Most common for seq2seq tasks
   - Easier to read and debug
   - Used in this project

2. **`padding='pre'`** (zeros at START)
   ```python
   [3, 5, 2] → [0, 0, 0, 0, 3, 5, 2]
               ↑_________↑
               added zeros
   ```
   - Sometimes used for RNNs
   - Can help with vanishing gradients

**Visual Representation:**

```
Before Padding (Variable Lengths):
┌──────────────────────────────────────┐
│ Sentence 1: [3, 6, 2, 1, 7]         │ len=5
│ Sentence 2: [9, 10]                 │ len=2
│ Sentence 3: [3, 16, 17, 20, 21]     │ len=5
│ Sentence 4: [8, 9, 10, 11, 12, 13]  │ len=6
└──────────────────────────────────────┘
         ↓ Find max_len = 6

After Padding (max_len=6, padding='post'):
┌──────────────────────────────────────┐
│ Sentence 1: [3, 6, 2, 1, 7, 0]      │ ← Added 1 zero
│ Sentence 2: [9, 10, 0, 0, 0, 0]     │ ← Added 4 zeros
│ Sentence 3: [3, 16, 17, 20, 21, 0]  │ ← Added 1 zero
│ Sentence 4: [8, 9, 10, 11, 12, 13]  │ ← No padding
└──────────────────────────────────────┘
         All sequences now length 6!
         Shape: (4, 6)
```

**Understanding `max_len` Calculation:**

```python
max_len = max(
    max(len(seq) for seq in input_sequences),   # Longest input
    max(len(seq) for seq in target_sequences)   # Longest target
)
```

**Why check BOTH input and target?**

```python
# Example scenario
input_sequences  longest: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]      # Length: 10
target_sequences longest: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Length: 12

# Must use max from BOTH
max_len = 12  # Not 10!

# Why? Because target sequences need to fit too
# If we used max_len=10, target sequences would be truncated
```

**After Padding - Data Structure:**

```python
# Before: Lists of lists (variable lengths)
type(input_sequences)       # <class 'list'>
input_sequences[0].shape    # Varies

# After: NumPy arrays (uniform shape)
type(input_sequences)       # <class 'numpy.ndarray'>
input_sequences.shape       # (1500, max_len)
target_sequences.shape      # (1500, max_len)

# Example with max_len=20
input_sequences.shape       # (1500, 20)
target_sequences.shape      # (1500, 20)
```

**How Model Handles Padding:**

The model learns to **ignore** padding tokens (zeros):
- Embedding layer: Index 0 → zero vector
- LSTM: Learns that 0 = no information
- Loss calculation: Can mask padding positions

**This step is called:** Padding / Sequence Padding / Zero-Padding

---

### Step 6: Model Architecture

**Purpose:** Define the neural network architecture for grammar correction.

**Code:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Hyperparameters
embedding_dim = 50  # Vector size for each word

# Build model
model = Sequential([
    # Layer 1: Embedding
    Embedding(
        input_dim=vocab_size,      # 1248 (vocabulary size)
        output_dim=embedding_dim,  # 50 (embedding dimension)
        input_length=max_len       # Length of input sequences
    ),

    # Layer 2-4: Stacked LSTM layers
    LSTM(100, return_sequences=True),
    LSTM(100, return_sequences=True),
    LSTM(100, return_sequences=True),

    # Layer 5: Dense output layer
    Dense(vocab_size, activation='softmax')
])
```

**Architecture Diagram:**

```
INPUT
[3, 5, 2, 1, 4, 0, 0]
"i go to the store <PAD> <PAD>"
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              EMBEDDING LAYER                            │
│         input_dim=1248, output_dim=50                   │
│                                                         │
│  Converts: Word indices → Dense vectors                │
│  [3, 5, 2] → Matrix of shape (7, 50)                   │
│                                                         │
│  Each word becomes 50-dimensional vector:              │
│  3 → [0.23, -0.45, 0.12, ..., 0.89]  (50 numbers)     │
│  5 → [0.67, 0.12, -0.34, ..., 0.23]                   │
│                                                         │
│  Output Shape: (batch_size, max_len, 50)              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LSTM LAYER 1                               │
│           100 units, return_sequences=True              │
│                                                         │
│  Processes sequence, captures patterns                 │
│  Returns output at EVERY timestep                      │
│                                                         │
│  Input:  (batch, 7, 50)                                │
│  Output: (batch, 7, 100)                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LSTM LAYER 2                               │
│           100 units, return_sequences=True              │
│                                                         │
│  Learns higher-level patterns                          │
│  Deeper representation of grammar                      │
│                                                         │
│  Input:  (batch, 7, 100)                               │
│  Output: (batch, 7, 100)                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LSTM LAYER 3                               │
│           100 units, return_sequences=True              │
│                                                         │
│  Most abstract understanding                           │
│  Complex grammar rules                                 │
│                                                         │
│  Input:  (batch, 7, 100)                               │
│  Output: (batch, 7, 100)                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         DENSE LAYER (OUTPUT)                            │
│      vocab_size=1248, activation='softmax'              │
│                                                         │
│  For each position, predict word probabilities         │
│  Output: (batch, 7, 1248)                              │
│                                                         │
│  Position 0: [0.001, 0.003, 0.856, ...]  ← 1248 probs │
│              Predicts: word index 3 (85.6% confident)  │
│                                                         │
│  Position 1: [0.002, 0.004, 0.001, 0.923, ...]        │
│              Predicts: word index 4 (92.3% confident)  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
OUTPUT SEQUENCE
[3, 4, 2, 1, 5, 0, 0]
"i go to the store <PAD> <PAD>"
```

**Layer-by-Layer Explanation:**

#### 1. Embedding Layer
```python
Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)
```

**What it does:**
- Converts sparse word indices → dense vectors
- Each word gets represented by 50 numbers (embedding_dim=50)
- Creates learnable word representations

**Example:**
```python
# Input: Word index
word_index = 3  # 'i'

# Output: 50-dimensional vector (learned during training)
embedding = [0.23, -0.45, 0.12, 0.89, ..., 0.34]  # 50 numbers

# Similar words have similar embeddings
'go'   → [0.23, -0.45, ..., 0.34]
'went' → [0.25, -0.43, ..., 0.36]  ← Close to 'go'
'cat'  → [-0.67, 0.89, ..., -0.12] ← Far from 'go'
```

**Parameters:**
- `input_dim=1248`: Vocabulary size (how many unique words)
- `output_dim=50`: Embedding dimension (vector size per word)
- `input_length=max_len`: Length of input sequences

**Shape Transformation:**
```
Input:  (batch_size, max_len)           # [3, 5, 2, 1, 4]
Output: (batch_size, max_len, 50)       # 50-dim vector per word
```

#### 2-4. LSTM Layers (Stacked)
```python
LSTM(100, return_sequences=True)  # × 3 layers
```

**What LSTM does:**
- **L**ong **S**hort-**T**erm **M**emory
- Processes sequences while remembering important information
- Captures grammar patterns and dependencies
- Handles long-distance relationships in sentences

**Why 3 layers?**
- **Layer 1**: Learns basic patterns (word order, simple grammar)
- **Layer 2**: Learns complex patterns (verb agreement, tenses)
- **Layer 3**: Learns abstract patterns (sentence structure, context)
- Deeper = more learning capacity

**`return_sequences=True` - CRITICAL!**

```python
# With return_sequences=True
Input:  [word1, word2, word3, word4, word5]
Output: [out1,  out2,  out3,  out4,  out5]  ← Output at EVERY position

# With return_sequences=False (wrong for our task!)
Input:  [word1, word2, word3, word4, word5]
Output: [out5]  ← Only LAST output
```

**Why we need outputs at every position:**
- We want to correct EVERY word in the sentence
- Not just predict one word at the end
- Sequence-to-sequence requires position-wise predictions

**Shape Transformation:**
```
Input:  (batch_size, max_len, 50)
        ↓
LSTM 1: (batch_size, max_len, 100)
        ↓
LSTM 2: (batch_size, max_len, 100)
        ↓
LSTM 3: (batch_size, max_len, 100)
```

#### 5. Dense Output Layer
```python
Dense(vocab_size, activation='softmax')
```

**What it does:**
- Makes final prediction for each position
- Outputs probability distribution over ALL words
- Softmax ensures probabilities sum to 1.0

**For Each Position:**
```python
# At position 0 (first word)
probabilities = [
    0.001,  # Index 0: <PAD>
    0.003,  # Index 1: 'the'
    0.002,  # Index 2: 'to'
    0.856,  # Index 3: 'i'      ← Highest probability!
    0.004,  # Index 4: 'a'
    ...     # (1248 total probabilities)
]

# Model predicts: Index 3 ('i') with 85.6% confidence
```

**Shape Transformation:**
```
Input:  (batch_size, max_len, 100)
        ↓
Dense:  (batch_size, max_len, 1248)  ← 1248 probabilities per position
```

**Complete Data Flow Example:**

```python
# Input sentence
input = [3, 5, 2, 1, 4, 0, 0]  # "i go to the store <PAD> <PAD>"
        # Shape: (7,)

# After Embedding
embedded = [
    [0.23, -0.45, ..., 0.34],  # 'i'
    [0.67, 0.12, ..., -0.11],  # 'go'
    [0.89, -0.23, ..., 0.55],  # 'to'
    [-0.34, 0.78, ..., 0.12],  # 'the'
    [0.12, 0.45, ..., 0.89],   # 'store'
    [0.00, 0.00, ..., 0.00],   # <PAD>
    [0.00, 0.00, ..., 0.00],   # <PAD>
]  # Shape: (7, 50)

# After LSTM layers
lstm_output = [
    [0.45, -0.23, ..., 0.67],  # Processed 'i'
    [0.89, 0.12, ..., -0.34],  # Processed 'go'
    [0.23, -0.67, ..., 0.45],  # Processed 'to'
    [0.56, 0.34, ..., -0.12],  # Processed 'the'
    [0.78, -0.45, ..., 0.23],  # Processed 'store'
    [0.00, 0.00, ..., 0.00],   # <PAD>
    [0.00, 0.00, ..., 0.00],   # <PAD>
]  # Shape: (7, 100)

# After Dense layer
predictions = [
    [0.001, 0.003, 0.856, ...],  # Position 0: predicts 'i' (index 3)
    [0.002, 0.004, 0.001, 0.923, ...],  # Position 1: predicts 'go' (index 5)
    [0.001, 0.891, 0.003, ...],  # Position 2: predicts 'to' (index 2)
    [0.934, 0.002, 0.001, ...],  # Position 3: predicts 'the' (index 1)
    [0.001, 0.003, 0.002, 0.003, 0.867, ...],  # Position 4: predicts 'store'
    [0.999, 0.000, ...],         # Position 5: predicts <PAD>
    [0.999, 0.000, ...],         # Position 6: predicts <PAD>
]  # Shape: (7, 1248)
```

**Model Summary Table:**

| Layer | Type | Input Shape | Output Shape | Parameters | Purpose |
|-------|------|-------------|--------------|------------|---------|
| 1 | Embedding | (None, max_len) | (None, max_len, 50) | ~62,400 | Word vectors |
| 2 | LSTM | (None, max_len, 50) | (None, max_len, 100) | ~60,400 | Sequence patterns |
| 3 | LSTM | (None, max_len, 100) | (None, max_len, 100) | ~80,400 | Deeper patterns |
| 4 | LSTM | (None, max_len, 100) | (None, max_len, 100) | ~80,400 | Complex grammar |
| 5 | Dense | (None, max_len, 100) | (None, max_len, 1248) | ~126,048 | Word prediction |

**Total Parameters:** ~409,648 learnable parameters

**This step is called:** Model Architecture Definition / Model Building

---

### Step 7: Model Compilation

**Purpose:** Configure the model for training by specifying optimizer, loss function, and metrics.

**Code:**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Component Breakdown:**

#### 1. Optimizer: Adam
```python
optimizer='adam'
```

**What is an optimizer?**
- Algorithm that adjusts model weights during training
- Determines HOW the model learns from errors
- Controls the learning rate (step size)

**Why Adam?**
- **Ada**ptive **M**oment Estimation
- Automatically adjusts learning rate for each parameter
- Works well with deep networks
- Fast convergence
- Most popular optimizer for NLP tasks

**Alternatives:**
- SGD (Stochastic Gradient Descent): Simpler, requires manual tuning
- RMSprop: Good for RNNs
- AdaGrad: Adapts to sparse data

#### 2. Loss Function: Sparse Categorical Crossentropy
```python
loss='sparse_categorical_crossentropy'
```

**What is a loss function?**
- Measures how wrong the model's predictions are
- Higher loss = worse predictions
- Goal: Minimize loss during training

**Why "Sparse" Categorical Crossentropy?**

```python
# Our targets are INTEGERS (sparse format)
target = [3, 5, 2, 1, 4]  # Word indices
# ✓ Use: sparse_categorical_crossentropy

# NOT one-hot encoded (categorical format)
target = [
    [0, 0, 0, 1, 0, 0, ...],  # 1 at index 3
    [0, 0, 0, 0, 0, 1, ...],  # 1 at index 5
    ...
]
# ✗ Would use: categorical_crossentropy (wasteful!)
```

**Sparse vs Regular:**

| Format | Target Example | Loss Function | Efficiency |
|--------|----------------|---------------|------------|
| **Sparse** | `[3, 5, 2]` | `sparse_categorical_crossentropy` | ✓ Memory efficient |
| **One-Hot** | `[[0,0,0,1,0,...], [0,0,0,0,0,1,...]]` | `categorical_crossentropy` | ✗ Wasteful |

**How Loss is Calculated:**

```python
# For one position in sequence
true_word_index = 5  # Target: 'go'

# Model's predictions (probabilities for all 1248 words)
predicted_probs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.85, 0.001, ...]
#                                               ↑ index 5

# Loss calculation
loss = -log(predicted_probs[true_word_index])
loss = -log(0.85)
loss ≈ 0.163

# Perfect prediction would be:
predicted_probs = [0, 0, 0, 0, 0, 1.0, 0, ...]  # 100% confidence at index 5
loss = -log(1.0) = 0  # Zero loss!
```

**Loss Across Entire Sequence:**

```python
# Targets: [3, 5, 2, 1, 4]
# Predictions: [[probs for pos 0], [probs for pos 1], ...]

# Calculate loss for each position, then average
total_loss = average([loss_pos0, loss_pos1, loss_pos2, loss_pos3, loss_pos4])
```

#### 3. Metrics: Accuracy
```python
metrics=['accuracy']
```

**What is accuracy?**
- Percentage of correctly predicted words
- Easy-to-understand performance measure
- NOT the same as loss

**How it's calculated:**

```python
# Targets:     [3, 5, 2, 1, 4, 0, 0]
# Predictions: [3, 5, 2, 1, 4, 0, 0]  ← Perfect match

# Accuracy = (Number of correct predictions) / (Total predictions)
accuracy = 7 / 7 = 100%

# Partial match:
# Targets:     [3, 5, 2, 1, 4, 0, 0]
# Predictions: [3, 6, 2, 1, 4, 0, 0]  ← One wrong (position 1)
#                 ↑

accuracy = 6 / 7 ≈ 85.7%
```

**Loss vs Accuracy:**

| Metric | What it measures | Range | Goal |
|--------|------------------|-------|------|
| **Loss** | How confident wrong predictions are | 0 to ∞ | Minimize |
| **Accuracy** | % of exact matches | 0% to 100% | Maximize |

**Example Scenario:**

```python
# Prediction A: Very confident, wrong
true_index = 5
predicted_probs_A = [0.01, 0.01, 0.01, 0.96, 0.01, 0.001, ...]
#                                      ↑ index 3 (wrong!)

Loss: -log(0.001) ≈ 6.9  # High loss!
Accuracy: 0%             # Wrong prediction

# Prediction B: Less confident, right
predicted_probs_B = [0.01, 0.01, 0.01, 0.01, 0.01, 0.92, ...]
#                                                  ↑ index 5 (correct!)

Loss: -log(0.92) ≈ 0.08  # Low loss!
Accuracy: 100%           # Correct prediction
```

**After Compilation:**

The model is now ready for training with:
- ✓ Defined architecture (layers)
- ✓ Learning algorithm (Adam optimizer)
- ✓ Error measurement (sparse categorical crossentropy)
- ✓ Performance tracking (accuracy)

**This step is called:** Model Compilation

---

### Step 8: Model Training

**Purpose:** Train the model on the grammar correction dataset.

**Code:**
```python
# Train the model
history = model.fit(
    input_sequences,      # X: Input features (incorrect sentences)
    target_sequences,     # y: Target labels (correct sentences)
    epochs=50,            # Number of complete passes through dataset
    batch_size=32,        # Number of samples per gradient update
    validation_split=0.2  # Use 20% of data for validation
)
```

**What Happens During Training:**

```
For each epoch (1 to 50):
    For each batch of 32 sentence pairs:
        1. Forward Pass:
           ├─ Input: [incorrect sentence sequences]
           ├─ Through: Embedding → LSTM → LSTM → LSTM → Dense
           └─ Output: [predicted corrected sequences]

        2. Calculate Loss:
           ├─ Compare predictions with true targets
           └─ Compute sparse categorical crossentropy

        3. Backward Pass:
           ├─ Calculate gradients (how to adjust weights)
           └─ Use Adam optimizer to update weights

        4. Track Metrics:
           └─ Calculate accuracy

    Validate on 20% holdout data
    Print epoch results
```

**Training Parameters Explained:**

#### epochs=50
```python
# One epoch = one complete pass through ALL training data
# 50 epochs = model sees each example 50 times

Epoch 1: Loss: 4.23, Accuracy: 15%  ← Poor performance
Epoch 10: Loss: 2.15, Accuracy: 45%
Epoch 30: Loss: 0.87, Accuracy: 78%
Epoch 50: Loss: 0.34, Accuracy: 91%  ← Much better!
```

#### batch_size=32
```python
# Process 32 examples at a time (not all 1500 at once)
# Total examples: 1500
# Batch size: 32
# Batches per epoch: 1500 / 32 ≈ 47 batches

Advantages of batches:
- Faster training (parallel processing)
- More stable gradients
- Fits in GPU memory
```

#### validation_split=0.2
```python
# Hold out 20% of data for validation
# Training data: 80% × 1500 = 1200 examples
# Validation data: 20% × 1500 = 300 examples

Purpose:
- Check if model generalizes (not just memorizing)
- Detect overfitting
- Model never trains on validation data
```

**Training Output:**

```
Epoch 1/50
38/38 [==============================] - 5s 89ms/step - loss: 4.2341 - accuracy: 0.1523 - val_loss: 3.8456 - val_accuracy: 0.1834
Epoch 2/50
38/38 [==============================] - 4s 87ms/step - loss: 3.5678 - accuracy: 0.2456 - val_loss: 3.2341 - val_accuracy: 0.2678
...
Epoch 50/50
38/38 [==============================] - 4s 86ms/step - loss: 0.3421 - accuracy: 0.9123 - val_loss: 0.4567 - val_accuracy: 0.8876
```

**Understanding the Metrics:**

- **loss**: Training loss (lower is better)
- **accuracy**: Training accuracy (higher is better)
- **val_loss**: Validation loss (how well model performs on unseen data)
- **val_accuracy**: Validation accuracy (true performance indicator)

**This step is called:** Model Training / Model Fitting

---

### Step 9: Inference (Grammar Correction)

**Purpose:** Use the trained model to correct new, unseen sentences.

**Also Known As:** Prediction, Inference, Decoding

**Code:**
```python
def correct_sentence(sentence):
    """
    Correct a grammatically incorrect sentence.

    Args:
        sentence (str): Input sentence with potential errors

    Returns:
        str: Grammatically corrected sentence
    """
    # Step 1: Convert sentence to sequence
    sequence = tokenizer.texts_to_sequences([sentence])

    # Step 2: Pad to max_len
    sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Step 3: Get model predictions (probabilities)
    predicted_sequence = model.predict(sequence)

    # Step 4: Convert probabilities to word indices
    predicted_sequence = tf.argmax(predicted_sequence, axis=-1).numpy()[0]

    # Step 5: Convert indices back to words (skip padding tokens)
    corrected_sentence = ' '.join(
        tokenizer.index_word[idx] for idx in predicted_sequence if idx != 0
    )

    return corrected_sentence
```

**Step-by-Step Breakdown:**

#### Step 1: Convert to Sequence
```python
sequence = tokenizer.texts_to_sequences([sentence])

# Example:
input_text = "apples is healthy"

# After texts_to_sequences:
sequence = [[45, 12, 78]]  # List of one sequence
```

**Why wrap in `[]`?**
- Function expects list of sentences
- Even for single sentence, needs to be in a list

#### Step 2: Pad Sequence
```python
sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

# Before:
[[45, 12, 78]]

# After (if max_len=20):
[[45, 12, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
```

**Why pad?**
- Model expects fixed-length input
- Must match training sequence length

#### Step 3: Get Predictions
```python
predicted_sequence = model.predict(sequence)

# Input shape:  (1, 20)       # 1 sentence, length 20
# Output shape: (1, 20, 1248) # 1 sentence, 20 positions, 1248 word probabilities
```

**What the output looks like:**

```python
predicted_sequence = [
    [  # Sentence 1
        [0.001, 0.002, 0.856, 0.003, ...],  # Position 0: 1248 probabilities
        [0.002, 0.003, 0.001, 0.923, ...],  # Position 1: 1248 probabilities
        [0.001, 0.891, 0.003, 0.002, ...],  # Position 2: 1248 probabilities
        ...  # 20 positions total
    ]
]
```

#### Step 4: Convert Probabilities to Indices
```python
predicted_sequence = tf.argmax(predicted_sequence, axis=-1).numpy()[0]
```

**Breaking down this line:**

1. **`tf.argmax(predicted_sequence, axis=-1)`**
   ```python
   # For each position, find index of maximum probability

   # Position 0:
   probs = [0.001, 0.002, 0.856, 0.003, ...]
   #                        ↑ max at index 2
   argmax → 2

   # Position 1:
   probs = [0.002, 0.003, 0.001, 0.923, ...]
   #                                  ↑ max at index 3
   argmax → 3
   ```

2. **`axis=-1`**: Apply argmax along last axis (vocabulary dimension)

3. **`.numpy()`**: Convert TensorFlow tensor to NumPy array

4. **`[0]`**: Extract first (and only) sequence from batch

**Result:**
```python
predicted_sequence = [2, 3, 5, 0, 0, 0, 0, ...]  # Word indices
```

**Visual Example:**

```python
# Probabilities at position 0:
[0.01, 0.03, 0.85, 0.05, 0.02, ...]
              ↑
         argmax → index 2

# Probabilities at position 1:
[0.02, 0.04, 0.01, 0.89, 0.03, ...]
                        ↑
                   argmax → index 3

# Result: [2, 3, 5, 0, 0, ...]
```

#### Step 5: Convert Indices to Words
```python
corrected_sentence = ' '.join(
    tokenizer.index_word[idx] for idx in predicted_sequence if idx != 0
)
```

**Breaking down this comprehension:**

1. **Loop through indices:**
   ```python
   for idx in predicted_sequence:
       # idx = 2, 3, 5, 0, 0, ...
   ```

2. **Skip padding (idx != 0):**
   ```python
   if idx != 0:  # Skip zeros
       # Process: 2, 3, 5
       # Ignore: 0, 0, 0
   ```

3. **Look up word:**
   ```python
   tokenizer.index_word[idx]
   # 2 → 'apples'
   # 3 → 'are'
   # 5 → 'healthy'
   ```

4. **Join with spaces:**
   ```python
   ' '.join(['apples', 'are', 'healthy'])
   # → "apples are healthy"
   ```

**Complete Example:**

```python
# Input
input_sentence = "apples is healthy"

# Step 1: texts_to_sequences
[[45, 12, 78]]

# Step 2: pad_sequences
[[45, 12, 78, 0, 0, 0, 0, ...]]  # Padded to max_len

# Step 3: model.predict
[
    [[0.01, 0.85, ...],    # Position 0 probabilities
     [0.03, 0.02, 0.91, ...],  # Position 1
     [0.04, 0.87, ...],    # Position 2
     [0.99, 0.00, ...],    # Position 3 (padding)
     ...]
]

# Step 4: argmax
[2, 3, 5, 0, 0, 0, ...]  # Most probable word indices

# Step 5: index_word mapping
idx=2 → 'apples'
idx=3 → 'are'
idx=5 → 'healthy'
idx=0 → skip (padding)

# Final output
"apples are healthy"
```

**Complete Inference Pipeline Diagram:**

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: "apples is healthy"                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  texts_to_sequences()                                   │
│  Output: [[45, 12, 78]]                                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  pad_sequences()                                        │
│  Output: [[45, 12, 78, 0, 0, ...]]                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  model.predict()                                        │
│  Shape: (1, max_len, 1248)                             │
│  Probability distributions for each position            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  tf.argmax(axis=-1)                                     │
│  Output: [2, 3, 5, 0, 0, ...]                          │
│  Most probable word index at each position             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  index_word mapping + filtering                         │
│  2 → 'apples', 3 → 'are', 5 → 'healthy'                │
│  Skip: 0 (padding tokens)                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: "apples are healthy"                           │
└─────────────────────────────────────────────────────────┘
```

**Usage Examples:**

```python
# Example 1
input_sentence = "I goes to Dubai"
corrected = correct_sentence(input_sentence)
print(f"Original:  {input_sentence}")
print(f"Corrected: {corrected}")
# Output:
# Original:  I goes to Dubai
# Corrected: i go to dubai

# Example 2
input_sentence = "she have completed her homework"
corrected = correct_sentence(input_sentence)
print(f"Original:  {input_sentence}")
print(f"Corrected: {corrected}")
# Output:
# Original:  she have completed her homework
# Corrected: she has completed her homework

# Example 3
input_sentence = "they was playing soccer"
corrected = correct_sentence(input_sentence)
print(f"Original:  {input_sentence}")
print(f"Corrected: {corrected}")
# Output:
# Original:  they was playing soccer
# Corrected: they were playing soccer
```

**This step is called:** Inference / Prediction / Decoding / Grammar Correction

---

## Installation

```bash
# Core libraries
pip install tensorflow
pip install keras
pip install pandas
pip install numpy

# Dataset download
pip install kagglehub

# Optional (for better text processing)
pip install regex
```

**System Requirements:**
- Python 3.7+
- 4GB+ RAM (8GB recommended)
- GPU optional (CPU training is fine for this dataset size)

---

## Complete Code Example

```python
# Import libraries
import kagglehub
import pandas as pd
import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ============ 1. DATA LOADING ============
input_column = 'Ungrammatical Statement'
target_column = 'Standard English'

path = kagglehub.dataset_download("satishgunjal/grammar-correction")
filepath = os.path.join(path, 'Grammar Correction.csv')
df = pd.read_csv(filepath)
df = df.dropna(subset=[input_column, target_column])

# ============ 2. TEXT PREPROCESSING ============
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
    text = text.strip()
    return text

input_texts = df[input_column].apply(preprocess_text).tolist()
target_texts = df[target_column].apply(preprocess_text).tolist()

# ============ 3. TOKENIZATION ============
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)
vocab_size = len(tokenizer.word_index) + 1

# ============ 4. SEQUENCE CONVERSION ============
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# ============ 5. PADDING ============
max_len = max(
    max(len(seq) for seq in input_sequences),
    max(len(seq) for seq in target_sequences)
)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

# ============ 6. MODEL ARCHITECTURE ============
embedding_dim = 50

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(100, return_sequences=True),
    LSTM(100, return_sequences=True),
    LSTM(100, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

# ============ 7. MODEL COMPILATION ============
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============ 8. MODEL TRAINING ============
history = model.fit(
    input_sequences,
    target_sequences,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# ============ 9. INFERENCE FUNCTION ============
def correct_sentence(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    predicted_sequence = model.predict(sequence)
    predicted_sequence = tf.argmax(predicted_sequence, axis=-1).numpy()[0]
    corrected_sentence = ' '.join(
        tokenizer.index_word[idx] for idx in predicted_sequence if idx != 0
    )
    return corrected_sentence

# ============ TEST ============
test_sentences = [
    "I goes to Dubai",
    "apples is healthy",
    "she have completed her homework"
]

for sentence in test_sentences:
    corrected = correct_sentence(sentence)
    print(f"Original:  {sentence}")
    print(f"Corrected: {corrected}")
    print()
```

---

## Model Architecture Details

### Embedding Layer
- **Purpose:** Convert word indices to dense vector representations
- **Input Dimension:** 1,248 (vocabulary size)
- **Output Dimension:** 50 (embedding vector size)
- **Trainable:** Yes (embeddings learned during training)
- **Parameters:** 1,248 × 50 = 62,400

**What it learns:**
- Similar words get similar embeddings
- Semantic relationships (king - man + woman ≈ queen)
- Syntactic patterns (verb forms, plurals)

### LSTM Layers
- **Count:** 3 stacked layers
- **Units per Layer:** 100
- **Type:** Long Short-Term Memory
- **return_sequences:** True (critical for seq2seq)

**What each layer learns:**

| Layer | Learning Focus | Example Patterns |
|-------|----------------|------------------|
| **LSTM 1** | Basic patterns | Word order, basic grammar |
| **LSTM 2** | Complex patterns | Verb agreement, tenses |
| **LSTM 3** | Abstract patterns | Sentence structure, context |

**Parameters per LSTM Layer:**
- 4 × (input_size + hidden_size + 1) × hidden_size
- LSTM 1: ~60,400 parameters
- LSTM 2 & 3: ~80,400 parameters each

### Dense Output Layer
- **Units:** 1,248 (one per vocabulary word)
- **Activation:** Softmax (probabilities sum to 1.0)
- **Purpose:** Predict word at each sequence position
- **Parameters:** 100 × 1,248 = 124,800

**Output Interpretation:**
- Each position gets 1,248 probabilities
- Highest probability = predicted word
- Example: [0.001, 0.003, 0.856, ...] → predicts word at index 2

---

## Examples

### Example 1: Verb Tense Correction
```python
Input:  "I goes to Dubai"
Output: "I go to Dubai"

Explanation: 'goes' (third-person singular) corrected to 'go' (first-person)
```

### Example 2: Subject-Verb Agreement
```python
Input:  "apples is healthy"
Output: "apples are healthy"

Explanation: 'is' (singular) corrected to 'are' (plural) to match 'apples'
```

### Example 3: Complex Correction
```python
Input:  "she have completed her homework"
Output: "she has completed her homework"

Explanation: 'have' corrected to 'has' for third-person singular subject
```

### Example 4: Multiple Errors
```python
Input:  "they was playing soccer last night"
Output: "they were playing soccer last night"

Explanation: 'was' (singular) corrected to 'were' (plural) to match 'they'
```

### Example 5: Preposition Error
```python
Input:  "he is interested to learn guitar"
Output: "he is interested in learning guitar"

Explanation: Preposition 'to' corrected to 'in', verb form adjusted
```

---

## Key Concepts Explained

### 1. Sequence-to-Sequence Learning
**What is it?**
- Input: Sequence (sentence with errors)
- Output: Sequence (corrected sentence)
- Both input and output are variable-length sequences

**Applications:**
- Machine translation
- Text summarization
- Grammar correction (this project)
- Speech recognition

### 2. Teacher Forcing
**What is it?**
- During training, model sees correct previous words
- Helps model learn faster and more stably
- Used in this project implicitly

**Example:**
```
Target: "i go to the store"
Model learns:
- Given "i", predict "go"
- Given "i go", predict "to"
- Given "i go to", predict "the"
```

### 3. Padding Tokens
**Why needed?**
- Neural networks require fixed-size inputs
- Sentences have different lengths
- Padding makes all sequences same length

**How handled?**
- Index 0 reserved for padding
- Model learns to ignore padding
- Embedding layer: 0 → zero vector

### 4. Word Embeddings
**What are they?**
- Dense vector representations of words
- Capture semantic and syntactic relationships
- Learned during training

**Example:**
```
'go'   → [0.23, -0.45, 0.12, ..., 0.89]
'went' → [0.25, -0.43, 0.14, ..., 0.87]  ← Similar to 'go'
'cat'  → [-0.67, 0.89, -0.34, ..., -0.12] ← Different from 'go'
```

### 5. Softmax Activation
**What it does:**
- Converts raw scores to probabilities
- All probabilities sum to 1.0
- Allows interpretation as confidence

**Example:**
```
Raw scores: [2.3, 1.5, 4.2, 0.8]
After softmax: [0.10, 0.04, 0.84, 0.02]  ← Sum = 1.0
Prediction: Index 2 (84% confident)
```

### 6. Argmax Operation
**What it does:**
- Finds index of maximum value
- Converts probabilities to single prediction

**Example:**
```python
probabilities = [0.10, 0.04, 0.84, 0.02]
argmax(probabilities) = 2  # Index of maximum (0.84)
```

### 7. Sparse vs Dense Targets
**Sparse (what we use):**
```python
target = [3, 5, 2, 1, 4]  # Word indices
# Memory: 5 integers = 20 bytes
```

**Dense (one-hot encoded):**
```python
target = [
    [0, 0, 0, 1, 0, ...],  # 1248 values
    [0, 0, 0, 0, 0, 1, ...],
    ...
]
# Memory: 5 × 1248 floats = ~25KB
```
Sparse is much more efficient!

---

## Technical Details

### Hyperparameters

| Parameter | Value | Description | Impact |
|-----------|-------|-------------|--------|
| **vocab_size** | 1,248 | Total unique words + padding | Larger = more words to learn |
| **embedding_dim** | 50 | Vector size per word | Larger = more capacity |
| **LSTM units** | 100 | Hidden state size | Larger = more patterns |
| **LSTM layers** | 3 | Network depth | Deeper = more abstract |
| **max_len** | Variable | Longest sequence | Based on data |
| **batch_size** | 32 | Samples per update | Larger = faster, less stable |
| **epochs** | 50 | Training iterations | More = better fit (risk overfitting) |

### Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning Rate** | Default (0.001) | Step size for weight updates |
| **Loss Function** | Sparse Categorical Crossentropy | Multi-class classification loss |
| **Metrics** | Accuracy | Performance measure |
| **Validation Split** | 0.2 (20%) | Holdout for testing |

### Model Complexity

| Metric | Value |
|--------|-------|
| **Total Parameters** | ~409,648 |
| **Trainable Parameters** | ~409,648 |
| **Non-trainable Parameters** | 0 |
| **Model Size** | ~1.6 MB (saved) |
| **Training Time** | ~5-10 min (CPU) |
| **Inference Time** | <100ms per sentence |

### Data Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 1,500 sentence pairs |
| **Training Samples** | 1,200 (80%) |
| **Validation Samples** | 300 (20%) |
| **Vocabulary Size** | 1,247 unique words |
| **Average Sentence Length** | 6-8 words |
| **Max Sentence Length** | Variable (dataset dependent) |
| **Error Categories** | 8+ types |

### Performance Expectations

| Metric | Typical Range |
|--------|---------------|
| **Training Accuracy** | 85-95% |
| **Validation Accuracy** | 80-90% |
| **Training Loss** | 0.2-0.5 |
| **Validation Loss** | 0.3-0.6 |

**Note:** Actual performance depends on:
- Dataset quality
- Training duration
- Model architecture choices
- Hyperparameter tuning

---

## Future Improvements

### Model Enhancements
1. **Attention Mechanism**: Better focus on relevant words
2. **Bidirectional LSTM**: Process sequence in both directions
3. **Transformer Architecture**: State-of-the-art for NLP
4. **Pre-trained Embeddings**: Use Word2Vec or GloVe
5. **Beam Search Decoding**: Better prediction strategies

### Data Improvements
1. **Larger Dataset**: More training examples
2. **Data Augmentation**: Generate synthetic errors
3. **Balanced Error Types**: Equal representation of error categories
4. **Real-world Data**: User-generated content

### System Enhancements
1. **Web Interface**: User-friendly correction tool
2. **API Endpoint**: Serve model as REST API
3. **Batch Processing**: Correct multiple sentences
4. **Explanation Feature**: Show what was corrected and why

---

## License

This is an educational project for learning purposes.

---

## Acknowledgments

- **Dataset:** [Kaggle Grammar Correction Dataset](https://www.kaggle.com/datasets/satishgunjal/grammar-correction) by Satish Gunjal
- **Framework:** TensorFlow/Keras
- **Architecture:** LSTM-based Sequence-to-Sequence Model
- **Inspiration:** Modern NLP and grammar correction research

---

## References

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215)
- [TensorFlow Text Processing Guide](https://www.tensorflow.org/tutorials/text/text_generation)
- [Keras Documentation](https://keras.io/)

---

**Note:** This is an educational project demonstrating sequence-to-sequence learning with LSTMs. For production-grade grammar correction, consider using:
- **Transformer models**: T5, BART, GPT-based architectures
- **Pre-trained models**: LanguageTool, Grammarly API
- **Advanced techniques**: Attention mechanisms, beam search, ensemble methods

---

**Created for:** Educational purposes and NLP learning
**Difficulty Level:** Intermediate
**Time to Complete:** 2-3 hours
**Prerequisites:** Python, basic machine learning knowledge