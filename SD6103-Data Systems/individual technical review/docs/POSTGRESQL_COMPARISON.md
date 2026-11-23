# DuckDB versus the PostgreSQL course baseline

This is a scoped design comparison. PostgreSQL was not installed, run or benchmarked for this project. “Typical PostgreSQL” means a standard shared server with heap tables and ordinary indexes, not every extension or deployment.

| Design question | DuckDB in this project | PostgreSQL course baseline | Why the choice matters |
|---|---|---|---|
| Data model and language | Typed relations, declarative SQL | Typed relations, declarative SQL | JOIN and GROUP BY mean familiar logical operations; physical plans may differ. |
| A. Deployment | Engine inside CLI or Python process; local file | Clients connect to a separate database service | Embedded calls simplify local setup and access to application files. A service provides a common endpoint for independent users and administration. |
| B. Storage | Native compressed columnar storage; external Parquet scanned separately | Typical heap tables organize records into pages | A few analytical columns over many records favor column access. Whole-record lookup and small transactional updates fit row access. Neither physical layout changes the relational model. |
| Query execution | Vectorized batches and analytical operators | General-purpose query executor with multiple scan/join strategies | Batching can amortize overhead across values. This project observes operators, not CPU microarchitectural effects. |
| C. Access/optimization | Projection and filter pushdown; row-group metadata pruning; ART also available | B-tree selective lookup; sequential scans and other indexes also available | A broad SUM may process many values, making unnecessary-column/group elimination useful. A point lookup often benefits from a selective index path. |
| Transactions/concurrency | ACID; in-process read-write ownership by one process | Shared service coordinates independent transactional connections | Multiple writer threads inside one host are different from independent processes opening the same file. |
| File interoperability | CSV/Parquet scan and COPY export used live | COPY and extensions/connectors offer various integration paths | Direct local file analysis reduces import steps here. It is inaccurate to say PostgreSQL can never access Parquet. |
| Workload | Local exploration, batch processing, analytical SQL and Parquet | Often chosen for shared transactional application state | Workload and operating model drive selection; both can execute analytical SQL. |

## Three causal design arguments for the spoken review

1. **Deployment:** Co-locating the engine with the application removes the need to configure a separate local DB service for this task. The trade-off is a different coordination and administration boundary from a centralized server. Current DuckDB has remote options, so make this claim only about the selected local mode.
2. **Columns versus rows:** Summing revenue grouped by category needs a small column subset across many transactions. Column organization and batching match that access pattern. A customer-facing transaction looking up and updating one record has different locality and concurrency demands.
3. **Scans versus selective access:** Metadata can rule out nonoverlapping date groups; indexes can target a few identifiers. These are complementary approaches with different maintenance and memory costs. Do not claim that DuckDB has no indexes or that PostgreSQL always uses an index.

## Evidence boundaries

- **Executed:** exact aggregates, Parquet export/scan, pushdowns, four of twenty-five row groups scanned, actual join/group operators, persistence and rollback.
- **Documented:** vectorized engine design, native storage organization, ART and zone maps, concurrency modes, PostgreSQL architecture/heap/index concepts.
- **Interpretation:** suitability for the chosen analytical task and usual many-client OLTP deployments.
- **Not tested:** PostgreSQL speed, many-client transaction throughput, scale beyond 400,000 rows, any remote DuckDB deployment.

## Official sources

- [DuckDB connections](https://duckdb.org/docs/2.0/connect/overview) and [concurrency](https://duckdb.org/docs/2.0/connect/concurrency).
- [DuckDB execution](https://duckdb.org/docs/2.0/internals/vector), [storage format](https://duckdb.org/docs/2.0/internals/storage), [indexes](https://duckdb.org/docs/2.0/guides/performance/indexing).
- [PostgreSQL architecture](https://www.postgresql.org/docs/current/tutorial-arch.html), [heap/page layout](https://www.postgresql.org/docs/current/storage-page-layout.html), [index types](https://www.postgresql.org/docs/current/indexes-types.html).
