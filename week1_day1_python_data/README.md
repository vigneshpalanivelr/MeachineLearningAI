# 📘 Week 1 - Day 1: Python + ML Warm-up + Docker Kickstart

## 🧭 Table of Contents
- [📘 Week 1 - Day 1: Python + ML Warm-up + Docker Kickstart](#-week-1---day-1-python--ml-warm-up--docker-kickstart)
  - [🧭 Table of Contents](#-table-of-contents)
  - [Objective](#-objective)
  - [Project Structure](#-project-structure)
  - [⚙️ Dependencies Overview](#️-dependencies-overview)
  - [🧩 Key Concepts](#-key-concepts)
    - [What is `prog="data-tool"`](#what-is-progdata-tool)
    - [What is `parser.add_subparsers(dest="command")`](#what-is-parseradd_subparsersdestcommand)
    - [What is `main(argv=None)`](#what-is-mainargvnone)
      - [Why we use it:](#why-we-use-it)
    - [Why separate `cli.py` and `stats.py`](#why-separate-clipy-and-statspy)
    - [What is `@click.group()`](#what-is-clickgroup)
    - [Difference: `argparse` vs `click`](#difference-argparse-vs-click)
    - [Understanding Python Type Hints \& Function Signatures](#understanding-python-type-hints--function-signatures)
      - [📘 Anatomy Breakdown:](#-anatomy-breakdown)
      - [Example 1: `cmd_summarize(path: str, field: str) -> None`](#example-1-cmd_summarizepath-str-field-str---none)
      - [Example 2: `_make_numeric_pred(minv: Any = None, maxv: Any = None)`](#example-2-_make_numeric_predminv-any--none-maxv-any--none)
      - [Example 3: `cmd_filter(path: str, field: str, minv: Any = None, maxv: Any = None) -> None`](#example-3-cmd_filterpath-str-field-str-minv-any--none-maxv-any--none---none)
      - [🔍 Function Signatures Summary Table](#-function-signatures-summary-table)
      - [💬 Why Type Hints Matter](#-why-type-hints-matter)
      - [🧠 Type Hints Quick Reference](#-type-hints-quick-reference)
  - [⚙️ Setup Instructions](#️-setup-instructions)
  - [💻 Running the CLI](#-running-the-cli)
    - [Show help](#show-help)
    - [Summarize a field](#summarize-a-field)
    - [Filter by range](#filter-by-range)
  - [🧪 Testing](#-testing)
  - [🐳 Docker Usage](#-docker-usage)
  - [🧠 Interview Learnings](#-interview-learnings)
  - [📚 Stretch Goals](#-stretch-goals)
  - [Day 1 Checklist](#-day-1-checklist)

---

## Objective
Learn the **Python foundations for data workflows** by building a small but production-style project.

This includes:
- Strengthening Python reasoning (not just syntax)
- Using functions, error handling, and modular coding
- Learning both `argparse` and `click` for CLIs
- Writing and testing data-processing logic
- Running everything inside Docker

---

## Project Structure

```
week1_day1_python_data/
├── data/
│   └── sample.json
├── src/
│   ├── __init__.py
│   ├── stats.py        # Core logic (data processing)
│   └── cli.py          # CLI (argparse-based interface)
├── tests/
│   └── test_stats.py   # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Dependencies Overview

| Library | Purpose | Why We Use It |
|----------|----------|----------------|
| **numpy** | Fast numerical computing | For vectorized math and statistical operations |
| **scipy** | Scientific/statistical tools | For advanced stats and ML-related math |
| **click** | CLI framework | Makes professional command-line tools easy to build |
| **pytest** | Testing framework | Lightweight, clean, and auto-discovers test files |

---

## 🧩 Key Concepts

### What is `prog="data-tool"`

- When creating an `ArgumentParser`, you can specify a **program name** (`prog`).
- It defines how your command appears in help messages.

```python
parser = argparse.ArgumentParser(prog="data-tool", description="JSON Data CLI")
```

When you run:

```bash
python src/cli.py --help
```

You'll see:

```
usage: data-tool [OPTIONS] COMMAND [ARGS]...
```

This makes your CLI look polished and consistent, even when it's installed as a package later.

---

### What is `parser.add_subparsers(dest="command")`

This is how you add **subcommands** (like `git add`, `git commit`) using `argparse`.

```python
sub = parser.add_subparsers(dest="command")
parser_sum = sub.add_parser("summarize", help="Summarize numeric field")
parser_filter = sub.add_parser("filter", help="Filter JSON records")
```

Now you can run:

```bash
python src/cli.py summarize data/sample.json score
python src/cli.py filter data/sample.json score --min 80
```

**Internally:**

* `add_subparsers()` creates a container for subcommands.
* `add_parser()` adds one command.
* `dest="command"` saves the command name in `args.command`.

Example:

```python
args = parser.parse_args(["filter", "data.json", "score"])
print(args.command)  # → "filter"
```

Enables multiple subcommands under one tool, just like `git` or `docker`.

---

### What is `main(argv=None)`

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)
```

#### Why we use it:

* If you run from **terminal**, `argv` is `None`, and `argparse` reads from `sys.argv`.
* If you run from **Python**, you can pass custom arguments for testing:

  ```python
  from src.cli import main
  main(["summarize", "data/sample.json", "score"])
  ```

Makes your CLI testable and reusable — used in production tools like `pip`, `black`, and `aws-cli`.

---

### Why separate `cli.py` and `stats.py`

| File                  | Role                                              | Why separate                             |
| --------------------- | ------------------------------------------------- | ---------------------------------------- |
| `src/stats.py`        | Core data logic (no CLI)                          | Reusable in tests, notebooks, or APIs    |
| `src/cli.py`          | User interface (argparse, printing, parsing args) | Keeps logic clean and independent        |
| `tests/test_stats.py` | Verification                                      | Test logic directly without CLI overhead |

This follows the **Single Responsibility Principle**:

* `stats.py` = the chef 🍳 (core work)
* `cli.py` = the waiter 🧾 (handles user input)
* `tests/` = the critic 🧪 (checks quality)

Clean, testable, and ready for MLOps pipelines.

---

### What is `@click.group()`

If using **Click** instead of argparse:

```python
@click.group()
def cli():
    pass

@cli.command()
def summarize():
    ...
```

* `@click.group()` defines the **root CLI group** (like `git`).
* `@cli.command()` attaches subcommands to that group.

So:

```bash
python src/cli.py summarize data/sample.json score
```

is equivalent to:

```
data-tool summarize ...
```

In short: `@click.group()` = entry point; `@cli.command()` = subcommands under that entry.

---

### Difference: `argparse` vs `click`

| Feature        | `argparse`                                | `click`                        |
| -------------- | ----------------------------------------- | ------------------------------ |
| Built-in       | Yes                                     | Needs `pip install click`    |
| Syntax         | Imperative (`add_argument`, `parse_args`) | Declarative (`@click.command`) |
| Subcommands    | Manual (`add_subparsers`)                 | Easy (`@cli.group`)            |
| Help UI        | Basic                                     | Beautiful + colored            |
| Validation     | Manual                                    | Automatic                      |
| Learning Curve | Low                                       | Medium                         |
| Best For       | Scripts, internal tools                   | Production CLIs, DevOps tools  |

We used `argparse` for Day 1 to understand CLI fundamentals.

---

### Understanding Python Type Hints & Function Signatures

A function signature in Python can show:

```python
def function_name(param1: Type, param2: Type = default) -> ReturnType:
```

#### 📘 Anatomy Breakdown:

| Component                | Meaning                                      | Example                |
| ------------------------ | -------------------------------------------- | ---------------------- |
| `function_name`          | The name of the function                     | `cmd_summarize`        |
| `param1: Type`           | Parameter name and its **type hint**         | `path: str`            |
| `param2: Type = default` | Parameter with a **default value**           | `minv: Any = None`     |
| `-> ReturnType`          | The **expected return type**                 | `-> None` or `-> bool` |
| `:`                      | Start of the function body                   | –                      |

These are **type hints**, not strict rules — they don't enforce types, but help IDEs and tools like `mypy` understand what data is expected.

---

#### Example 1: `cmd_summarize(path: str, field: str) -> None`

```python
def cmd_summarize(path: str, field: str) -> None:
    """Summarize statistics for a numeric field in JSON data."""
    data = load_json(path)
    stats = calculate_stats(data, field)
    print(stats)
```

**📖 Meaning:**
* `path: str` — expects a **file path** (string)
* `field: str` — expects a **field name** (string, like `"score"`)
* `-> None` — **does not return anything**, just performs an action (printing)

**Example usage:**
```python
cmd_summarize("data/sample.json", "score")
# Output:
# Summary for field: score
#   count : 4
#   mean  : 84.0
#   ...
```

---

#### Example 2: `_make_numeric_pred(minv: Any = None, maxv: Any = None)`

```python
def _make_numeric_pred(minv: Any = None, maxv: Any = None):
    """Create a predicate function for filtering numeric values."""
    def predicate(value):
        if minv is not None and value < minv:
            return False
        if maxv is not None and value > maxv:
            return False
        return True
    return predicate
```

**📖 Meaning:**
* `_` prefix means it's **internal/private** (not meant for external use)
* `minv: Any = None` — optional parameter, can be **any type**, defaults to `None`
* `maxv: Any = None` — optional parameter, defaults to `None`
* No `->` annotation — but it **returns a function** (a closure)

**Example usage:**
```python
pred = _make_numeric_pred(minv=80, maxv=90)
pred(85)   # → True (85 is between 80 and 90)
pred(75)   # → False (75 is less than 80)
pred(95)   # → False (95 is more than 90)
```

**Full type hint (advanced):**
```python
from typing import Callable, Any

def _make_numeric_pred(minv: Any = None, maxv: Any = None) -> Callable[[Any], bool]:
    # Returns a function that takes Any and returns bool
```

---

#### Example 3: `cmd_filter(path: str, field: str, minv: Any = None, maxv: Any = None) -> None`

```python
def cmd_filter(path: str, field: str, minv: Any = None, maxv: Any = None) -> None:
    """Filter JSON records by numeric range."""
    data = load_json(path)
    predicate = _make_numeric_pred(minv, maxv)
    filtered = [item for item in data if predicate(item.get(field))]
    print(f"Found {len(filtered)} matching records:")
    for item in filtered[:5]:
        print(item)
```

**📖 Meaning:**
* `path: str` — JSON **file path**
* `field: str` — **field name** to filter on (like `"score"`)
* `minv: Any = None` — optional **minimum value**
* `maxv: Any = None` — optional **maximum value**
* `-> None` — just **prints output**, doesn't return anything

**Example usage:**
```python
cmd_filter("data/sample.json", "score", minv=85)
# Output:
# Found 2 matching records:
# {'id': 3, 'name': 'Carol', 'score': 91}
# {'id': 4, 'name': 'Dan', 'age': 28, 'score': 88}
```

---

#### 🔍 Function Signatures Summary Table

| Function Signature | Purpose | Returns | Key Notes |
| --- | --- | --- | --- |
| `cmd_summarize(path: str, field: str) -> None` | Calculate stats on a numeric field | Nothing (prints) | Command handler for CLI |
| `_make_numeric_pred(minv: Any = None, maxv: Any = None)` | Build a filtering function | Function | Internal helper; returns a closure |
| `cmd_filter(path: str, field: str, minv: Any = None, maxv: Any = None) -> None` | Filter JSON by numeric range | Nothing (prints) | Command handler for CLI |

---

#### 💬 Why Type Hints Matter

Type hints are **optional but powerful**:

**IDE Support** — Auto-complete and error detection
**Readability** — Self-documenting code
**Type Checking** — Tools like `mypy` catch bugs
**Collaboration** — Other engineers know what to pass in

**Comparison:**

Without type hints:
```python
def area(r):
    return 3.14 * r**2
```

With type hints:
```python
def area(radius: float) -> float:
    """Calculate circle area."""
    return 3.14 * radius**2
```

The second is instantly clearer! 📖

---

#### 🧠 Type Hints Quick Reference

| Type Hint | Meaning | Example |
| --- | --- | --- |
| `str` | String | `name: str` |
| `int` | Integer | `count: int` |
| `float` | Decimal number | `score: float` |
| `bool` | True/False | `is_active: bool` |
| `Any` | Could be anything | `value: Any` |
| `None` | No return value | `-> None` |
| `List[str]` | List of strings | `names: List[str]` |
| `Dict[str, int]` | Dictionary: str keys, int values | `scores: Dict[str, int]` |
| `Callable[[int], bool]` | Function taking int, returning bool | Function passed as argument |

Understanding these signatures makes you a better Python engineer!

---

## ⚙️ Setup Instructions

```bash
python -m venv .venv
source .venv/bin/activate    # mac/linux
pip install -r requirements.txt
```

---

## 💻 Running the CLI

### Show help

```bash
python src/cli.py --help
```

### Summarize a field

```bash
python src/cli.py summarize data/sample.json score
```

Output:

```
Summary for field: score
  count : 4
  mean  : 84.0
  median: 86.5
  mode  : None
  min   : 72.0
  max   : 91.0
```

### Filter by range

```bash
python src/cli.py filter data/sample.json score --min 85
```

Output:

```
Found 2 matching records (first 5 shown):
{'id': 3, 'name': 'Carol', 'score': 91}
{'id': 4, 'name': 'Dan', 'age': 28, 'score': 88}
```

---

## 🧪 Testing

Run:

```bash
pytest -q
```

Checks:

* Summary statistics
* Filtering logic
* JSON loading behavior

---

## 🐳 Docker Usage

Build the container:

```bash
docker build -t json-insights .
```

Run:

```bash
docker run --rm json-insights summarize data/sample.json score
```

Docker makes your CLI portable and reproducible.

---

## 🧠 Interview Learnings

| Concept        | Common Question                 | Strong Answer                                                                |
| -------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| JSON keys      | How to handle missing keys?     | Use `dict.get()` or `defaultdict`; skip or log missing values.               |
| Copies         | Shallow vs Deep copy?           | Shallow copies references; deep copy duplicates nested objects.              |
| Python lists   | How are lists stored in memory? | Lists store object references; Python over-allocates memory for performance. |
| Comprehensions | Why faster than loops?          | Implemented in C, fewer interpreter calls.                                   |
| CLI structure  | Why separate logic from CLI?    | Improves modularity, testability, and scaling.                               |
| `argparse`     | Purpose of `dest="command"`     | Stores subcommand name in `args.command`.                                    |

---

## 📚 Stretch Goals

* Add `--format csv` or `--format markdown`
* Handle large files using `ijson`
* Add `--verbose` and log levels
* Create a `train.py` (Week 2) for simple ML model
* Build FastAPI interface (Week 3)

---

## Day 1 Checklist

| Task                            | Done |
| ------------------------------- | ---- |
| Setup virtual environment       | ☐    |
| Install dependencies            | ☐    |
| Implemented `stats.py`          | ☐    |
| Implemented `cli.py` (argparse) | ☐    |
| Tested with `pytest`            | ☐    |
| Built Docker image              | ☐    |
| Read and understood this README | ☐    |

---

> **Tip:**
> Each "Day" in this 4-week plan builds upon this project.
> You're not just writing scripts — you're constructing the foundation of an **MLOps-ready AI system** step by step.