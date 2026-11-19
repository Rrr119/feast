from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import ibis
import pandas as pd

from feast import Entity, FeatureStore, FeatureView, Field, FileSource, RepoConfig
from feast.infra.offline_stores.dask import _time_range_filters
from feast.infra.offline_stores.duckdb import DuckDBOfflineStoreConfig
from feast.infra.offline_stores.ibis import get_historical_features_ibis
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.types import Float64
from feast.value_type import ValueType


def _write_partitioned_source(path: Path) -> None:
    pd.DataFrame(
        {
            "driver_id": [1, 1, 1],
            "event_timestamp": pd.to_datetime(
                [
                    "2026-01-01T10:00:00Z",
                    "2026-01-02T10:00:00Z",
                    "2026-01-03T10:00:00Z",
                ]
            ),
            "created": pd.to_datetime(
                [
                    "2026-01-01T10:05:00Z",
                    "2026-01-02T10:05:00Z",
                    "2026-01-03T10:05:00Z",
                ]
            ),
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]).date,
            "value": [10.0, 20.0, 30.0],
        }
    ).to_parquet(path, partition_cols=["date"], index=False)


def _feature_view(path: Path, ttl: timedelta) -> FeatureView:
    return FeatureView(
        name="driver_stats",
        entities=[
            Entity(name="driver", join_keys=["driver_id"], value_type=ValueType.INT64)
        ],
        schema=[Field(name="value", dtype=Float64)],
        source=FileSource(
            path=str(path),
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
            date_partition_column="date",
        ),
        ttl=ttl,
    )


def _historical_features(
    tmp_path: Path, feature_view: FeatureView, entity_df: pd.DataFrame, **kwargs
) -> tuple[pd.DataFrame, str]:
    job = get_historical_features_ibis(
        config=MagicMock(repo_path=tmp_path),
        feature_views=[feature_view],
        feature_refs=["driver_stats:value"],
        entity_df=entity_df,
        registry=MagicMock(),
        project="test",
        data_source_reader=lambda source, _: ibis.read_parquet(source.path),
        data_source_writer=MagicMock(),
        **kwargs,
    )
    return job.to_df(), job.table.compile()


def test_file_source_date_partition_column_round_trips_through_proto(tmp_path):
    source = FileSource(
        path=str(tmp_path),
        timestamp_field="event_timestamp",
        date_partition_column="date",
    )

    restored = FileSource.from_proto(source.to_proto())

    assert restored.date_partition_column == "date"


def test_historical_retrieval_prunes_to_the_point_in_time_window(tmp_path):
    _write_partitioned_source(tmp_path / "stats.parquet")
    entity_df = pd.DataFrame(
        {
            "driver_id": [1, 1],
            "event_timestamp": pd.to_datetime(
                ["2026-01-02T12:00:00Z", "2026-01-03T12:00:00Z"]
            ),
        }
    )

    result, query = _historical_features(
        tmp_path,
        _feature_view(tmp_path / "stats.parquet", timedelta(days=1)),
        entity_df,
    )

    assert result["value"].tolist() == [20.0, 30.0]
    assert '"date" >= MAKE_DATE(2026, 1, 1)' in query
    assert '"date" <= MAKE_DATE(2026, 1, 3)' in query


def test_created_timestamp_cutoff_keeps_a_backfill_from_the_future_out(tmp_path):
    path = tmp_path / "backfill.parquet"
    pd.DataFrame(
        {
            "driver_id": [1, 1],
            "event_timestamp": pd.to_datetime(
                ["2026-01-02T10:00:00Z", "2026-01-02T10:00:00Z"]
            ),
            "created": pd.to_datetime(["2026-01-02T11:00:00Z", "2026-01-02T13:00:00Z"]),
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]).date,
            "value": [20.0, 200.0],
        }
    ).to_parquet(path, partition_cols=["date"], index=False)
    entity_df = pd.DataFrame(
        {
            "driver_id": [1],
            "event_timestamp": pd.to_datetime(["2026-01-02T12:00:00Z"]),
        }
    )
    feature_view = _feature_view(path, timedelta(days=1))

    latest, _ = _historical_features(tmp_path, feature_view, entity_df)
    as_of, _ = _historical_features(
        tmp_path,
        feature_view,
        entity_df,
        filter_by_created_timestamp=True,
    )

    assert latest["value"].tolist() == [200.0]
    assert as_of["value"].tolist() == [20.0]


def test_feature_store_reads_partitioned_feature_view_as_of_each_timestamp(tmp_path):
    _write_partitioned_source(tmp_path / "stats.parquet")
    feature_view = _feature_view(tmp_path / "stats.parquet", timedelta(days=1))
    store = FeatureStore(
        config=RepoConfig(
            project="test",
            registry=str(tmp_path / "registry.db"),
            provider="local",
            offline_store=DuckDBOfflineStoreConfig(),
            online_store=SqliteOnlineStoreConfig(path=str(tmp_path / "online.db")),
            repo_path=tmp_path,
        )
    )
    store.apply([feature_view.entities[0], feature_view])
    entity_df = pd.DataFrame(
        {
            "driver_id": [1, 1],
            "event_timestamp": pd.to_datetime(
                ["2026-01-02T12:00:00Z", "2026-01-03T12:00:00Z"]
            ),
        }
    )

    result = store.get_historical_features(
        entity_df=entity_df, features=["driver_stats:value"]
    ).to_df()

    result = result.sort_values("event_timestamp").reset_index(drop=True)
    assert result["value"].tolist() == [20.0, 30.0]


def test_dask_partition_filters_use_date_only_bounds(tmp_path):
    source = FileSource(
        path=str(tmp_path),
        timestamp_field="event_timestamp",
        date_partition_column="date",
    )

    assert _time_range_filters(
        source,
        datetime(2026, 1, 1, 23, tzinfo=timezone.utc),
        datetime(2026, 1, 3, 1, tzinfo=timezone.utc),
    ) == [("date", ">=", "2026-01-01"), ("date", "<=", "2026-01-03")]
