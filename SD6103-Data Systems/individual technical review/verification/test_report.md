# Executed verification report

Run: 2026-11-17T18:20:43.860574+08:00
Python: 3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]

| Check | Status | Detail |
|---|---|---|
| Python DuckDB pinned | PASS | 2.0.0 |
| Official CLI version matches Python | PASS | v2.0.0 (Cyanoptera) d54cbbe3df |
| Official CLI archive SHA256 | PASS |  |
| Raw data hash: customers.csv | PASS |  |
| Raw data hash: products.csv | PASS |  |
| Raw data hash: sales.csv | PASS |  |
| Independent CSV oracle vs generator totals | PASS | 13746257120 |
| SELECT 42 | PASS |  |
| Persistent table count sales | PASS | 400000 |
| Persistent table count customers | PASS | 15000 |
| Persistent table count products | PASS | 200 |
| Explicit dates and monetary types | PASS | {'sale_id': 'INTEGER', 'sale_date': 'DATE', 'customer_id': 'INTEGER', 'product_id': 'INTEGER', 'quantity': 'SMALLINT', 'unit_price': 'DECIMAL(10,2)', 'discount_amount': 'DECIMAL(10,2)'} |
| Relational keys unique | PASS |  |
| No orphan keys | PASS |  |
| Revenue nonnegative | PASS |  |
| Python-written database read and modified by CLI | PASS |  |
| CLI-written row persists and Python reopens | PASS |  |
| Transaction rollback | PASS |  |
| Demo SQL executes via Python: 01_intro | PASS |  |
| Demo SQL executes via Python: 02_csv_analytics | PASS |  |
| Demo SQL executes via Python: 03_parquet | PASS |  |
| Demo SQL executes via Python: 04_explain | PASS |  |
| Demo SQL executes via Python: 05_analyze | PASS |  |
| Demo SQL executes via Python: 06_join_plan_optional | PASS |  |
| All 20 groups equal independent integer-cent CSV oracle | PASS |  |
| JOIN preserves cardinality and exact revenue | PASS |  |
| Parquet filtered result equals CSV oracle | PASS |  |
| Full multiset equality data/parquet/retail.parquet | PASS |  |
| Full multiset equality data/parquet/retail_reference.parquet | PASS |  |
| Like-for-like CSV/Parquet full equality | PASS |  |
| Parquet scan node | PASS |  |
| Projection pushed into scan | PASS |  |
| Date and category filters pushed into scan | PASS |  |
| Hash grouping and actual cardinalities | PASS |  |
| Actual JOIN operator | PASS |  |
| Multiple date-ordered row groups and pruning opportunity | PASS | (25, 4) |
| Complete native CLI dry run 1 | PASS | 0.490 seconds |
| Complete native CLI dry run 2 | PASS | 0.493 seconds |
| No database WAL after graceful close | PASS |  |

Automated checks: 39/39 PASS.

Evidence: results.json, demo_results.txt, cli_dry_run_1.txt, cli_dry_run_2.txt, *_plan.txt.

Timing is an observation on this laptop, not a comparative benchmark. Operator output rows are not physical rows read.

