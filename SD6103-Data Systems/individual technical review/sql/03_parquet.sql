.echo on
COPY (SELECT * FROM retail ORDER BY sale_date,sale_id)
TO 'data/parquet/retail.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384);

SELECT region, sum(revenue) AS december_electronics_sgd
FROM read_parquet('data/parquet/retail.parquet')
WHERE sale_date >= DATE '2025-12-01'
  AND sale_date < DATE '2026-01-01'
  AND category = 'Electronics'
GROUP BY region ORDER BY december_electronics_sgd DESC;
.echo off
