"""
CLI using argparse. Commands:
  - summarize <path> <field>
  - filter <path> <field> [--min MIN] [--max MAX]
"""

import argparse
import logging
import stats
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def summarize_data(path: str, field: str) -> None:
    """Summarize the data"""
    data = stats.safe_load_json(path)
    stats_data = stats.summary_stats(data, field)
    logger.info("Summary for field: %s", field)
    for k, v in stats_data.items():
        print(f"  {k:6}: {v}")


def _make_numeric_pred(minv: Any = None, maxv: Any = None):
    def pred(v):
        try:
            if v is None:
                return False
            if isinstance(v, (int, float)) and (not (isinstance(v, float) and v != v)):  # skip NaN
                val = float(v)
            else:
                return False
            if minv is not None and val < float(minv):
                return False
            if maxv is not None and val > float(maxv):
                return False
            return True
        except Exception:
            return False
    return pred


def filter_data(path: str, field: str, minv: Any = None, maxv: Any = None) -> None:
    data = stats.safe_load_json(path)
    pred = _make_numeric_pred(minv, maxv)
    results = stats.filter_items(data, field, pred)
    print(f"Found {len(results)} matching records (first 5 shown):")
    for rec in results[:5]:
        print(rec)


def main(argv=None):
    """Main entry point of the script"""
    parser = argparse.ArgumentParser(prog="data-tool", description="Simple JSON data CLI")
    sub_parser = parser.add_subparsers(dest="command")

    sum_cmd = sub_parser.add_parser("summarize", help="Summarize numeric field from JSON file")
    sum_cmd.add_argument("path", help="Path to JSON file")
    sum_cmd.add_argument("field", help="Field name to summarize")

    filter_cmd = sub_parser.add_parser("filter", help="Filter records by numeric field range")
    filter_cmd.add_argument("path", help="Path to JSON file")
    filter_cmd.add_argument("field", help="Field name to filter on")
    filter_cmd.add_argument("--min", dest="min", default=None, help="Minimum value (inclusive)")
    filter_cmd.add_argument("--max", dest="max", default=None, help="Maximum value (inclusive)")

    args = parser.parse_args(argv)
    if args.command == "summarize":
        summarize_data(args.path, args.field)
    elif args.command == "filter":
        filter_data(args.path, args.field, args.min, args.max)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()