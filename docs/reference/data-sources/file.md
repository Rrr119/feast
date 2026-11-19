# File source

## Description

File data sources are files on disk or on S3.
Currently only Parquet and Delta formats are supported.

## Example

```python
from feast import FileSource
from feast.data_format import ParquetFormat

parquet_file_source = FileSource(
    file_format=ParquetFormat(),
    path="file:///feast/customer.parquet",
    timestamp_field="event_timestamp",
)
```

## Point-in-time retrieval and partition pruning

Historical retrieval joins source rows to each entity row using the source's
`timestamp_field`, so feature backfills do not need to read an entire table.
When the data is stored in date-partitioned Parquet (for example
`date=2026-01-01/part.parquet`), set `date_partition_column` to let the offline
store prune partitions outside the entity dataframe's event-time window:

```python
customer_features = FileSource(
    path="s3://my-bucket/customer_features/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    date_partition_column="date",
)
```

The lower partition bound is derived from the earliest entity timestamp minus
the feature view TTL. The upper bound is the latest entity timestamp. With
`filter_by_created_timestamp=True`, rows created after the request timestamp
are also excluded, which keeps backfilled values from leaking into historical
training data.

The full set of configuration options is available [here](https://rtd.feast.dev/en/latest/index.html#feast.infra.offline_stores.file_source.FileSource).

## Supported Types

File data sources support all eight primitive types and their corresponding array types.
For a comparison against other batch data sources, please see [here](overview.md#functionality-matrix).
