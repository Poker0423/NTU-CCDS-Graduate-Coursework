.echo on
EXPLAIN
SELECT region, sum(revenue)
FROM read_parquet('data/parquet/retail.parquet')
WHERE sale_date >= DATE '2025-12-01'
  AND sale_date < DATE '2026-01-01'
  AND category = 'Electronics'
GROUP BY region;
.echo off
