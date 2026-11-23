# DuckDB technical review

**Topic:** DuckDB: An In-Process Analytical Database - From Raw CSV to Optimized SQL Analytics.

**Scope:** NTU SD6103 Technical Review; DuckDB is system 13 in the supplied PDF. The audience has studied relational databases and PostgreSQL. This repository contains the review (Step 1) and the reproducible hands-on experiment (Step 2).

## Evidence convention

**MEASURED** means executed locally with DuckDB CLI and Python 2.0.0. **DOCUMENTED** identifies a vendor design statement, not an experiment. **INTERPRETATION** connects a design to a workload. **NOT TESTED** identifies checks that require the presenter or were outside scope. See verification/test_report.md and docs/SOURCES.md.

## Data model, language and implementation

**DOCUMENTED:** DuckDB is a relational SQL DBMS, implemented primarily in C++, with embedded APIs including Python. SQL describes a result; the optimizer selects physical operators. Logical joins and aggregation therefore remain familiar from PostgreSQL. The dialect follows PostgreSQL conventions but is not identical; SQL compatibility does not imply extension or storage compatibility. [DuckDB overview](https://duckdb.org/why_duckdb), [SQL dialect](https://duckdb.org/docs/2.0/sql/dialect/overview).

**MEASURED:** Three persistent, typed relations implement a small star-shaped model:

| Relation | Cardinality | Columns and meaning |
|---|---:|---|
| customers | 15,000 | customer_id INTEGER primary key; region VARCHAR; segment VARCHAR |
| products | 200 | product_id INTEGER primary key; product_name VARCHAR; category VARCHAR; list_price DECIMAL(10,2) |
| sales | 400,000 | sale_id INTEGER primary key; sale_date DATE; customer_id/product_id INTEGER logical references; quantity SMALLINT; unit_price/discount_amount DECIMAL(10,2) |

The sales foreign-key relationships are logical and exhaustively checked, not declared FOREIGN KEY constraints. The parent dimensions and sale identifiers have enforced primary keys. Setup also enforces non-null values and basic sales-domain checks. The recording view adds region, category and revenue to seven sales columns, yielding ten columns. The temporary view is recomputed when queried and is scoped to the live connection.

## Architecture and persistence

**DOCUMENTED:** In this local configuration, the engine shares the application's process. The CLI and Python API are separate host processes, used sequentially. Calling an embedded engine avoids a separate service and client/server protocol for this workflow. Passing a database filename enables persistent storage; omitting it would use an in-memory database. [Connections](https://duckdb.org/docs/2.0/connect/overview), [CLI](https://duckdb.org/docs/2.0/clients/cli/overview).

**MEASURED:** Python creates the database and persistent tables, closes the connection, and the CLI reopens it. A round-trip test creates/inserts a probe in Python, reads/inserts through the CLI, then reopens in Python and verifies both rows. A transaction rollback test also passes. No Python writer remains open while a CLI writer owns this database.

**INTERPRETATION:** Embedding lowers the operational burden for a student's local analysis. PostgreSQL's separate server is useful when applications need a centralized access and administration boundary. DuckDB 2.0 also has remote/server mechanisms; “embedded by default” does not mean “incapable of server use.” [PostgreSQL architecture](https://www.postgresql.org/docs/current/tutorial-arch.html).

## Storage organization

**DOCUMENTED:** DuckDB's native persistent database uses compressed columnar organization. Keeping values of a column together favors reading a subset of columns across many rows. Typical PostgreSQL heap pages hold rows, supporting retrieval and modification of individual records. This concerns physical organization, not a different relational data model. [DuckDB design](https://duckdb.org/why_duckdb), [native format](https://duckdb.org/docs/2.0/internals/storage), [PostgreSQL page layout](https://www.postgresql.org/docs/current/storage-page-layout.html).

External Parquet is independently specified columnar storage, with row groups and per-column metadata. DuckDB can scan it without loading a persistent table. Native DuckDB storage, in-memory execution vectors and Parquet column chunks are three distinct concepts. [Parquet access](https://duckdb.org/docs/2.0/data/parquet/overview), [Apache Parquet format](https://parquet.apache.org/docs/file-format/).

**MEASURED:** The flattened CSV is 24,637,185 bytes; the corresponding date-sorted ZSTD Parquet is 4,748,869 bytes. Both have the same 400,000 rows and ten columns, checked using two-way EXCEPT ALL. CSV-to-Parquet size ratio is 5.188:1. This measures representation on this synthetic data, not runtime acceleration. The normalized sales CSV is 16,143,426 bytes and is deliberately not used as the denominator for the enriched export comparison.

## Query execution and optimization

**DOCUMENTED:** Vectorized execution passes batches of typed values between operators, amortizing per-tuple interpretation and function-call overhead. Column pruning reduces the data needed by a query; predicate pushdown lets a scanner apply restrictions earlier. These are physical optimizations of a declarative relational query. [Execution format](https://duckdb.org/docs/2.0/internals/vector), [Parquet partial reading](https://duckdb.org/docs/2.0/data/parquet/overview#partial-reading).

**MEASURED:** Experiment B joins fact/dimensions and groups the result. The optional join profile shows actual Hash Join and Hash Group By operators. Experiment D profiles December Electronics revenue by region directly from Parquet. The physical plan includes READ_PARQUET, scan-local date/category filters, projected region/revenue, and hash aggregation. Predicate columns must be accessed as well; “two projected columns” does not mean only two columns are physically needed.

The actual profiler reports 4/25 row groups scanned. Independent Parquet metadata inspection finds the same four groups whose date ranges overlap December. The scan emits 14,594 matching rows; grouping emits four rows. Operator output cardinality is not the count of physical rows examined or bytes read. The EXPLAIN estimates are substantially different from actual groups in this example; do not label an estimate as a measurement. The redundant upper date bound can disappear from the optimized scan because the file's maximum date is 2025-12-31. [EXPLAIN](https://duckdb.org/docs/2.0/guides/meta/explain), [EXPLAIN ANALYZE](https://duckdb.org/docs/2.0/guides/meta/explain_analyze).

## Indexing and trade-offs

**DOCUMENTED:** Min/max summaries (zone maps) make ordered values useful for ruling out groups. ART indexes support constraints and some selective access; maintaining indexes costs memory and write work. Our main evidence is ordered-file scan pruning, not an ART-index experiment. PostgreSQL's B-tree access is a natural counterpart when only a small fraction of rows is requested. PostgreSQL can also choose sequential scans and has BRIN indexes; the design comparison is about workload fit, not exclusive capabilities. [DuckDB indexing](https://duckdb.org/docs/2.0/guides/performance/indexing), [PostgreSQL indexes](https://www.postgresql.org/docs/current/indexes-types.html).

**INTERPRETATION:** Batch scans, joins and aggregations benefit from avoiding unnecessary columns and groups. Random point retrieval and frequent small updates have different access patterns; a scan-oriented design is not automatically optimal for them. Sorting and compression also consume preparation time; the project deliberately performs deterministic preparation before recording.

DuckDB supports ACID transactions. In-process read-write access is owned by one process, with supported concurrent threads inside it; multiple independent read-only processes are another mode. This differs from a shared PostgreSQL server accepting independent transactional clients. For a conventional many-client OLTP application, PostgreSQL is usually the more natural operational choice. This is a documented-design interpretation; no PostgreSQL installation, concurrency test or comparative benchmark was performed. [DuckDB concurrency](https://duckdb.org/docs/2.0/connect/concurrency), [transactions](https://duckdb.org/docs/2.0/sql/statements/transactions).

## Reproducibility and limits

Data are entirely synthetic: random.Random(6103), 2025 dates, five categories, four regions, weighted demand and prices. Electronics prices and APAC customer frequency intentionally produce interpretable differences. These are generator assumptions, not findings about any real market. All amounts are synthetic SGD, excluding tax/shipping; net revenue is quantity times unit price minus a whole-line discount rounded to cents with integer arithmetic. SQL imports monetary values as DECIMAL and sums an exact DECIMAL expression. Independent CSV recomputation checks every one of 20 groups and filtered December totals.

The current Windows x64 environment uses project Python 3.12.14 and DuckDB 2.0.0 in both clients. `setup.ps1` downloads the pinned official CLI and Python wheel and verifies their checksums. This demonstrates system behavior at a laptop-sized scale. It does not establish behavior at terabyte scale, universal performance superiority, distributed execution, or production reliability.
