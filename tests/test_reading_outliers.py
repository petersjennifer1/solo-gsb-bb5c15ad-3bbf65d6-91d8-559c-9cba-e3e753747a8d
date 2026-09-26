"""展柜采集质量审计（孤立异常读数）：纯算法单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.main import app
from app.outliers import detect_reading_outliers
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _at(minutes: float, t: float, h: float) -> Sample:
    """相对 BASE 的第 minutes 分钟读数。"""
    return Sample(BASE + dt.timedelta(minutes=minutes), t, h)


def _reading(sn, minutes, t, h):
    ts = (BASE + dt.timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


KW = dict(
    window_size=2,
    k=3.0,
    temp_min_delta=0.1,
    humidity_min_delta=0.5,
    max_gap_seconds=600.0,
)


# ---------- 基本判定 ----------


def test_isolated_temperature_spike_flagged():
    samples = [_at(i, 20.0, 50.0) for i in range(10)]
    samples[5] = _at(5, 35.0, 50.0)
    scan = detect_reading_outliers(samples, **KW)
    assert scan.sample_count == 10
    assert scan.segment_count == 1
    assert scan.evaluated_count == 6  # 下标 2..7（两侧各至少 2 个邻点）
    assert len(scan.outliers) == 1
    o = scan.outliers[0]
    assert o.ts == BASE + dt.timedelta(minutes=5)
    assert o.temperature == 35.0
    assert o.humidity == 50.0
    assert [f.dimension for f in o.findings] == ["temperature"]
    f = o.findings[0]
    assert f.median == 20.0
    assert f.deviation == pytest.approx(15.0)


def test_isolated_humidity_spike_flagged():
    samples = [_at(i, 20.0, 50.0) for i in range(10)]
    samples[4] = _at(4, 20.0, 90.0)
    scan = detect_reading_outliers(samples, **KW)
    assert len(scan.outliers) == 1
    o = scan.outliers[0]
    assert [f.dimension for f in o.findings] == ["humidity"]
    f = o.findings[0]
    assert f.median == 50.0
    assert f.deviation == pytest.approx(40.0)


def test_both_dimensions_flagged_temperature_first():
    samples = [_at(i, 20.0, 50.0) for i in range(10)]
    samples[6] = _at(6, 5.0, 5.0)
    scan = detect_reading_outliers(samples, **KW)
    assert len(scan.outliers) == 1
    o = scan.outliers[0]
    assert [f.dimension for f in o.findings] == ["temperature", "humidity"]
    temp, hum = o.findings
    assert temp.median == 20.0 and temp.deviation == pytest.approx(15.0)
    assert hum.median == 50.0 and hum.deviation == pytest.approx(45.0)


def test_no_outliers_on_flat_series():
    samples = [_at(i, 20.0, 50.0) for i in range(12)]
    scan = detect_reading_outliers(samples, **KW)
    assert scan.outliers == ()
    assert scan.evaluated_count == 8


def test_empty_input():
    scan = detect_reading_outliers([], **KW)
    assert scan.sample_count == 0
    assert scan.segment_count == 0
    assert scan.evaluated_count == 0
    assert scan.outliers == ()


# ---------- 邻窗规则 ----------


def test_segment_edges_not_evaluated():
    # 6 个读数：仅下标 2、3 两侧均满 2 个邻点；下标 1/4 的尖峰
    # 因一侧邻点不足 2 个而不判
    samples = [_at(i, 20.0, 50.0) for i in range(6)]
    samples[1] = _at(1, 99.0, 50.0)
    samples[4] = _at(4, 20.0, 1.0)
    scan = detect_reading_outliers(samples, **KW)
    assert scan.evaluated_count == 2
    assert scan.outliers == ()


def test_window_truncates_to_nearest_neighbors():
    # 下标 0=26 与 6=30 只在 window_size=3 时进入下标 3 的邻窗，
    # 抬高 MAD 使同一尖峰不再超限：证明只取左右各最近 window_size 个
    temps = [26.0, 25.0, 24.0, 32.0, 21.0, 20.0, 30.0]
    samples = [_at(i, t, 50.0) for i, t in enumerate(temps)]

    narrow = detect_reading_outliers(samples, **{**KW, "window_size": 2})
    assert len(narrow.outliers) == 1
    o = narrow.outliers[0]
    assert o.ts == BASE + dt.timedelta(minutes=3)
    assert o.findings[0].median == pytest.approx(22.5)  # (25,24,21,20) 的中位数

    wide = detect_reading_outliers(samples, **{**KW, "window_size": 3})
    assert wide.outliers == ()


def test_min_delta_floor_when_neighbors_identical():
    # 邻点完全相同 → MAD=0 → 尺度 = min_delta；k×min_delta 之外才标记
    samples = [_at(i, 20.0, 50.0) for i in range(10)]
    samples[5] = _at(5, 20.2, 50.0)  # 偏差 0.2 < 3×0.1：不标记
    scan = detect_reading_outliers(samples, **KW)
    assert scan.outliers == ()

    samples[5] = _at(5, 20.4, 50.0)  # 偏差 0.4 > 3×0.1：标记
    scan = detect_reading_outliers(samples, **KW)
    assert len(scan.outliers) == 1


def test_mad_scale_used_when_larger_than_min_delta():
    # 5 个读数、window_size=2：仅下标 2 可判。其邻点 [20,21,19,20]
    # 中位数 20、MAD 0.5 → 尺度 1.4826×0.5=0.7413；偏差 0.5 < 3×0.7413
    # 不标记（若错用 min_delta=0.1 作尺度则会误标）
    samples = [
        _at(0, 20.0, 50.0),
        _at(1, 21.0, 50.0),
        _at(2, 20.5, 50.0),
        _at(3, 19.0, 50.0),
        _at(4, 20.0, 50.0),
    ]
    scan = detect_reading_outliers(samples, **KW)
    assert scan.evaluated_count == 1
    assert scan.outliers == ()


def test_strictly_greater_than_k_times_scale():
    # 偏差恰好等于 k×尺度（0.5 = 2×0.25）不标记；严格大于才标记
    samples = [_at(i, 20.0, 50.0) for i in range(10)]
    samples[5] = _at(5, 20.5, 50.0)
    kw = {**KW, "k": 2.0, "temp_min_delta": 0.25}
    scan = detect_reading_outliers(samples, **kw)
    assert scan.outliers == ()

    samples[5] = _at(5, 20.5000001, 50.0)
    scan = detect_reading_outliers(samples, **kw)
    assert len(scan.outliers) == 1


# ---------- 连续段切断 ----------


def test_gap_cuts_segments_and_bounds_evaluation():
    # 第 4 分钟与第 25 分钟之间间隔 21 分钟 > max_gap=600s：切成两段
    samples = [_at(i, 20.0, 50.0) for i in range(5)]
    samples += [_at(i, 20.0, 50.0) for i in range(25, 30)]
    samples[7] = _at(27, 35.0, 50.0)  # 第二段中段尖峰
    scan = detect_reading_outliers(samples, **KW)
    assert scan.segment_count == 2
    assert scan.evaluated_count == 2  # 每段仅中段 1 点可判
    assert len(scan.outliers) == 1
    assert scan.outliers[0].ts == BASE + dt.timedelta(minutes=27)


def test_neighbors_never_cross_gap():
    # 段首尖峰（第 25 分钟）左侧在本段内没有任何邻点：不判；
    # 若错误跨段取邻点（左侧 20,20）则会误判为异常
    samples = [_at(i, 20.0, 50.0) for i in range(5)]
    samples += [_at(i, 20.0, 50.0) for i in range(25, 30)]
    samples[5] = _at(25, 99.0, 50.0)
    scan = detect_reading_outliers(samples, **KW)
    assert scan.segment_count == 2
    assert scan.outliers == ()


def test_gap_exactly_max_gap_stays_continuous():
    # 间隔恰好等于 max_gap=600s（10 分钟）仍连续：整段 5 点，中段可判
    samples = [_at(i * 10, 20.0, 50.0) for i in range(5)]
    samples[2] = _at(20, 35.0, 50.0)
    scan = detect_reading_outliers(samples, **KW)
    assert scan.segment_count == 1
    assert scan.evaluated_count == 1
    assert len(scan.outliers) == 1
    assert scan.outliers[0].ts == BASE + dt.timedelta(minutes=20)


# ---------- 顺序与确定性 ----------


def test_outliers_sorted_by_time_and_input_order_irrelevant():
    samples = [_at(i, 20.0, 50.0) for i in range(14)]
    samples[4] = _at(4, 35.0, 50.0)
    samples[9] = _at(9, 20.0, 95.0)
    ordered = detect_reading_outliers(samples, **KW)
    assert [o.ts for o in ordered.outliers] == [
        BASE + dt.timedelta(minutes=4),
        BASE + dt.timedelta(minutes=9),
    ]
    assert [f.dimension for f in ordered.outliers[0].findings] == ["temperature"]
    assert [f.dimension for f in ordered.outliers[1].findings] == ["humidity"]

    shuffled = detect_reading_outliers(list(reversed(samples)), **KW)
    assert ordered == shuffled


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def _params(**over):
    p = {
        "window_size": 2,
        "k": 3.0,
        "temp_min_delta": 0.1,
        "humidity_min_delta": 0.5,
        "max_gap_seconds": 600,
    }
    p.update(over)
    return p


def test_endpoint_flags_spike(client):
    rows = [_reading("RO-1", i, 20.0, 50.0) for i in range(10)]
    rows[5] = _reading("RO-1", 5, 35.0, 50.0)
    _ingest(client, "RO-1", rows)

    r = client.get("/api/v1/cases/RO-1/reading-outliers", params=_params())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "RO-1"
    assert body["window_size"] == 2
    assert body["k"] == 3.0
    assert body["temp_min_delta"] == 0.1
    assert body["humidity_min_delta"] == 0.5
    assert body["max_gap_seconds"] == 600.0
    assert body["sample_count"] == 10
    assert body["segment_count"] == 1
    assert body["evaluated_count"] == 6
    assert body["outlier_count"] == 1
    o = body["outliers"][0]
    assert o["ts"].startswith("2026-09-01T00:05:00")
    assert o["temperature"] == 35.0
    assert o["humidity"] == 50.0
    assert o["dimensions"] == ["temperature"]
    assert o["temperature_median"] == 20.0
    assert o["temperature_deviation"] == pytest.approx(15.0)
    assert o["humidity_median"] is None
    assert o["humidity_deviation"] is None


def test_endpoint_defaults(client):
    rows = [_reading("RO-2", i, 20.0, 50.0) for i in range(12)]
    _ingest(client, "RO-2", rows)
    body = client.get("/api/v1/cases/RO-2/reading-outliers").json()
    assert body["window_size"] == 5
    assert body["k"] == 3.0
    assert body["temp_min_delta"] == 0.1
    assert body["humidity_min_delta"] == 0.5
    assert body["max_gap_seconds"] == 600.0
    assert body["outlier_count"] == 0
    assert body["outliers"] == []


def test_endpoint_unknown_device_404(client):
    r = client.get("/api/v1/cases/NOPE/reading-outliers", params=_params())
    assert r.status_code == 404


def test_endpoint_param_validation_422(client):
    rows = [_reading("RO-3", i, 20.0, 50.0) for i in range(10)]
    _ingest(client, "RO-3", rows)
    url = "/api/v1/cases/RO-3/reading-outliers"
    # window_size 越界
    assert client.get(url, params=_params(window_size=1)).status_code == 422
    assert client.get(url, params=_params(window_size=51)).status_code == 422
    assert client.get(url, params=_params(window_size="abc")).status_code == 422
    # k 非正
    assert client.get(url, params=_params(k=0)).status_code == 422
    assert client.get(url, params=_params(k=-1)).status_code == 422
    # 两维 min_delta 非正
    assert client.get(url, params=_params(temp_min_delta=0)).status_code == 422
    assert client.get(url, params=_params(humidity_min_delta=-0.5)).status_code == 422
    # max_gap_seconds 非正
    assert client.get(url, params=_params(max_gap_seconds=0)).status_code == 422
    assert client.get(url, params=_params(max_gap_seconds=-10)).status_code == 422
    # 边界值合法
    assert client.get(url, params=_params(window_size=2)).status_code == 200
    assert client.get(url, params=_params(window_size=50)).status_code == 200


def test_endpoint_is_read_only(client):
    """审计查询不得改变读数、分段与版本号。"""
    from app.database import CaseState, Reading, SessionLocal

    rows = [_reading("RO-4", i, 20.0, 50.0) for i in range(10)]
    rows[5] = _reading("RO-4", 5, 35.0, 50.0)
    _ingest(client, "RO-4", rows)
    db = SessionLocal()
    try:
        version_before = db.get(CaseState, "RO-4").version
        count_before = db.query(Reading).filter(Reading.device_sn == "RO-4").count()
    finally:
        db.close()

    url = "/api/v1/cases/RO-4/reading-outliers"
    assert client.get(url, params=_params()).status_code == 200
    assert client.get(url, params=_params()).status_code == 200

    db = SessionLocal()
    try:
        db.expire_all()
        assert db.get(CaseState, "RO-4").version == version_before
        assert db.query(Reading).filter(Reading.device_sn == "RO-4").count() == count_before
    finally:
        db.close()


def test_endpoint_deterministic(client):
    rows = [_reading("RO-5", i, 20.0 + (i % 3), 50.0 + (i % 4)) for i in range(15)]
    rows[7] = _reading("RO-5", 7, 40.0, 90.0)
    _ingest(client, "RO-5", rows)
    url = "/api/v1/cases/RO-5/reading-outliers"
    first = client.get(url, params=_params()).json()
    second = client.get(url, params=_params()).json()
    assert first == second
