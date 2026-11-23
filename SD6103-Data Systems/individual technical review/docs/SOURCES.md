# Source and provenance register

Checked 2026-10-24 (Asia/Singapore). Links are reference material; the demo itself runs fully offline once set up.

## Course authority

- The NTU SD6103 Technical Review brief (Prof. Melanie Herschel, AY26S1) is course material and is **not redistributed** in this repository.
- SHA256: `2A68BAA88133AAB08B29E57CC5B8E8355ADD23E37DC427632436F2C2060F84D4`.
- All three pages were text-extracted and rendered/visually inspected. Page 1: objectives and Steps 1-3. Page 2: deliverables and system list. Page 3: Appendix A, three equal-weight criteria. The PDF is the assignment authority.

## Official technical sources

| Source | Claim supported |
|---|---|
| [Installation](https://duckdb.org/install/) | Stable 2.0.0, Windows x64 CLI and VC++ runtime prerequisite |
| [CLI overview](https://duckdb.org/docs/2.0/clients/cli/overview) | Filename persistence, interactive SQL, .read and configuration |
| [Python overview](https://duckdb.org/docs/2.0/clients/python/overview) | Stable API and Python >=3.11 |
| [Release v2.0.0](https://github.com/duckdb/duckdb/releases/tag/v2.0.0) | Official release artifact; release date 2026-10-21 |
| [PyPI DuckDB](https://pypi.org/project/duckdb/2.0.0/) | Official Python distribution and compatible Windows wheel |
| [Why DuckDB](https://duckdb.org/why_duckdb) | Relational model, analytical orientation, embedding and native columnar design |
| [Connections](https://duckdb.org/docs/2.0/connect/overview) | Local/persistent/in-memory and remote deployment distinctions |
| [Concurrency](https://duckdb.org/docs/2.0/connect/concurrency) | In-process read-write access ownership and reader modes |
| [Transactions](https://duckdb.org/docs/2.0/sql/statements/transactions) | BEGIN, COMMIT, ROLLBACK and transactional semantics |
| [Execution format](https://duckdb.org/docs/2.0/internals/vector) | Vectors and batches |
| [Storage](https://duckdb.org/docs/2.0/internals/storage) | Native file format/version compatibility; 2.0 format needs 2.0 readers |
| [CSV](https://duckdb.org/docs/2.0/data/csv/overview) | Direct scans and explicit type binding |
| [Parquet](https://duckdb.org/docs/2.0/data/parquet/overview) | COPY, scanners, projections and predicates |
| [Indexes](https://duckdb.org/docs/2.0/guides/performance/indexing) | Zone maps, ordering and ART trade-offs |
| [EXPLAIN](https://duckdb.org/docs/2.0/guides/meta/explain) | Plan operators/estimates |
| [EXPLAIN ANALYZE](https://duckdb.org/docs/2.0/guides/meta/explain_analyze) | Execution, actual output rows and timings |
| [PostgreSQL architecture](https://www.postgresql.org/docs/current/tutorial-arch.html) | Typical client/server model |
| [PostgreSQL page layout](https://www.postgresql.org/docs/current/storage-page-layout.html) | Heap rows/pages |
| [PostgreSQL indexes](https://www.postgresql.org/docs/current/indexes-types.html) | B-tree selective access and other available types |
| [Apache Parquet](https://parquet.apache.org/docs/file-format/) | Independent external file format |

## Local measurements

`verification/results.json` contains sizes, schema, top groups, December results and row-group metadata. `verification/*_plan.txt` and `cli_dry_run_*.txt` preserve real execution plans. `data/dataset_manifest.json` records the seed, source-file hashes and exact integer-cent reference totals. All business data are synthetic; none came from an enterprise or public agency.

Official CLI archive SHA256: `91663c2f36c2a334b0e5e639006d8c29c29e013af0c477d80bb931df68870312`. Both current installed clients were checked, rather than assuming format compatibility from a name. `setup.ps1` downloads both into `tools/` (git-ignored) and checks the archive hash.

