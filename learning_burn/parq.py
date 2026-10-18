import polars as pl

pl.read_parquet("yellow_tripdata_2024-01.parquet").head(1000).write_csv(
    "yellow_tripdata_2024-01.csv"
)
