# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Machine Learning AI learning repository containing Python assignments, Jupyter notebooks, and structured weekly projects focused on ML/MLOps fundamentals. The repository follows a 4-week progressive learning plan building from Python foundations to production-ready ML systems.

## Development Environment

**Python Version:** 3.9+
**Virtual Environment:** `.venv/` (Python venv)
**Primary Stack:** NumPy, SciPy, pandas, Jupyter

### Setup Commands

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux

# Install dependencies (for week1_day1_python_data project)
cd week1_day1_python_data
pip install -r requirements.txt
```

## Project Structure

### `week1_day1_python_data/` - CLI Data Tool

A production-style Python CLI for JSON data processing, demonstrating clean separation of concerns.

**Architecture:**
- `src/stats.py`: Core data processing logic (reusable, testable)
- `src/cli.py`: CLI interface using argparse (command handlers)
- `tests/test_stats.py`: pytest-based unit tests

**Key Design Principles:**
- **Separation of Concerns:** Business logic in `stats.py`, UI/CLI in `cli.py`
- **Testability:** Core functions are pure and CLI-independent
- **Error Handling:** Graceful exits with logging for user-facing errors

**Running the CLI:**

```bash
cd week1_day1_python_data

# Show help
python src/cli.py --help

# Summarize a numeric field
python src/cli.py summarize data/sample.json score

# Filter records by range
python src/cli.py filter data/sample.json score --min 85
python src/cli.py filter data/sample.json age --max 30
```

**Testing:**

```bash
cd week1_day1_python_data
pytest -q                    # Quiet mode
pytest -v                    # Verbose mode
pytest tests/test_stats.py   # Run specific test file
```

**Docker:**

```bash
cd week1_day1_python_data

# Build image
docker build -t json-insights .

# Run commands
docker run --rm json-insights summarize data/sample.json score
docker run --rm json-insights filter data/sample.json score --min 85
```

### `Assignments/` - Jupyter Notebooks

Practice assignments covering:
- Basic Python (5 assignments including OOP and Exception Handling)
- NumPy operations
- Pandas data cleaning

**Running Notebooks:**

```bash
# Ensure virtual environment is activated
jupyter notebook Assignments/
```

## Code Architecture Patterns

### CLI Command Pattern (week1_day1_python_data)

The CLI uses argparse with subcommands:

```python
# Entry point with testable argv parameter
def main(argv=None):
    parser = argparse.ArgumentParser(prog="data-tool")
    sub = parser.add_subparsers(dest="command")

    # Each subcommand has its own parser
    sum_cmd = sub.add_parser("summarize", ...)
    filter_cmd = sub.add_parser("filter", ...)

    args = parser.parse_args(argv)
    # Dispatch to handler functions
```

**Why `argv=None`:** Enables testing by passing custom arguments programmatically while defaulting to `sys.argv` for CLI use.

### Data Processing Pattern

Core functions in `stats.py` follow functional patterns:

1. **Type validation and filtering:** `_to_numbers()` converts mixed-type lists to numeric values
2. **Safe access:** Uses `.get()` for dict access to handle missing keys
3. **Predicate pattern:** `filter_items()` accepts a callable for flexible filtering
4. **Error boundaries:** CLI layer (`cli.py`) handles errors; core layer (`stats.py`) raises them

### Type Hints

All functions use type hints:
- Parameters: `path: str`, `field: str`, `minv: Any = None`
- Return types: `-> None`, `-> List[Dict[str, Any]]`
- Complex types: `Callable[[Any], bool]` for predicate functions

## Testing Strategy

- **Unit tests focus on `stats.py`:** Test pure functions without CLI overhead
- **Test edge cases:** Missing fields, None values, NaN values, non-numeric data
- **Use pytest fixtures:** `tmp_path` fixture for file I/O tests
- Test isolation: Each test is independent and uses sample data

## Common Development Patterns

### Adding a New CLI Command

1. Add subparser in `cli.py`:
   ```python
   new_cmd = sub.add_parser("newcmd", help="Description")
   new_cmd.add_argument("path", help="...")
   ```

2. Create handler function in `cli.py`:
   ```python
   def handle_newcmd(path: str) -> None:
       data = stats.safe_load_json(path)
       result = stats.new_operation(data)
       print(result)
   ```

3. Add core logic to `stats.py`:
   ```python
   def new_operation(data: List[Dict[str, Any]]) -> Any:
       # Pure logic here
       pass
   ```

4. Add tests to `tests/test_stats.py`:
   ```python
   def test_new_operation():
       result = stats.new_operation(SAMPLE)
       assert result == expected
   ```

### Working with JSON Data

The project expects JSON arrays of objects:
```json
[
  {"id": 1, "name": "Alice", "score": 85},
  {"id": 2, "name": "Bob", "score": 72}
]
```

- Always handle missing keys with `.get(key, default)`
- Filter out None, NaN, and non-numeric values before statistical operations
- Use `isinstance(v, (int, float))` for type checking

## Dependencies

Key libraries in `week1_day1_python_data/requirements.txt`:
- **numpy:** Vectorized numerical operations
- **scipy:** Statistical functions
- **click:** Alternative CLI framework (for future reference)
- **pytest:** Testing framework
- **jupyter/ipykernel:** Notebook support

## Git Workflow

Current branch: `master`

Staged changes include new `week1_day1_python_data/` project. Use standard git commands for version control.

## Future Extensions (from README)

The weekly project is designed to evolve:
- Week 2: Add ML model training (`train.py`)
- Week 3: FastAPI REST interface
- Week 4: Full MLOps pipeline

When adding features, maintain the separation between core logic, CLI, and future API layers.
