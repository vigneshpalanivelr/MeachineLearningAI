from src.stats import summary_stats, filter_items, load_json

SAMPLE = [
    {"x": 1},
    {"x": 2},
    {"x": 2},
    {"y": 5},
    {"x": None},
    {"x": 4.5},
    {"x": float("nan")},
]

def test_summary_x():
    stats = summary_stats(SAMPLE, "x")
    # numeric values are [1,2,2,4.5] -> count 4
    assert stats["count"] == 4
    assert abs(stats["mean"] - ((1 + 2 + 2 + 4.5) / 4)) < 1e-8
    assert stats["median"] == 2

def test_filter_items_min():
    res = filter_items(SAMPLE, "x", lambda v: isinstance(v, (int, float)) and v >= 2)
    # values >=2 : 2,2,4.5  => 3 records
    assert len(res) == 3

def test_load_json(tmp_path):
    p = tmp_path / "tmp.json"
    p.write_text('[{"a":1}]')
    data = load_json(str(p))
    assert isinstance(data, list)
    assert data[0]["a"] == 1
