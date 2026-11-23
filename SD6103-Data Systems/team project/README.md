# SD6103 Data Systems — Team Project: HDB Resale Prices and Hawker Centre Proximity

Team project for NTU **SD6103 Data Systems** (AY26/27 Semester 1), built on PostgreSQL.

> **Code repository:** all source code, SQL, stored procedures, data-loading scripts and benchmark results live in
> **[yimingwang425/SD6103_Project](https://github.com/yimingwang425/SD6103_Project)**.
> This repository only holds the final report and design diagrams, so the code is not duplicated here.

**Team:** `SELECT name FROM student WHERE grade='A+';`
Members: How Gui Lin, Wang Yiming, Qiu Yina, Li Wanjing, Cui Jiawei, Ji Qiankun.

## Video walkthrough

[![SD6103 team project: video walkthrough](https://img.youtube.com/vi/LKpZm1_Gje4/hqdefault.jpg)](https://youtu.be/LKpZm1_Gje4)

A 4.5-minute animated summary of the report (Manim, with English voice-over): proximity definition, schema and normalization, data loading, query findings, performance and indexes. Watch on YouTube: <https://youtu.be/LKpZm1_Gje4>

## What the project does

The project designs and populates a relational database of Singapore HDB resale transactions and studies how resale prices relate to walking-distance proximity to hawker centres.

| Task | Content |
|---|---|
| 1. ER model, logical schema, implementation | 7 tables — `town`, `hdb_block`, `hdb_flat`, `resale_transaction`, `hawker_centre`, `proximity`, `proximity_level` — normalized to 3NF/BCNF |
| 2. Data acquisition and population | data.gov.sg resale files (1990–present), HDB block geolocation, hawker-centre locations and stall register; walking distances from OneMap routing; loaded by eight stored procedures |
| 3. SQL queries and performance | Six analytical queries (price per m² by town, remaining lease vs price, large flats, hawker proximity effect, fresh-market hawkers, hawker density) measured as the five resale files are loaded |
| 4. Indexes and runtime | Two targeted indexes on `resale_transaction`; up to −51 % runtime (Q3) on 976,714 transactions |

Final database size: 26 towns, 10,194 blocks, 145,700 flats, 976,714 resale transactions, 129 hawker centres, 3 proximity levels and 51,214 block–centre pairs.

## Contents

```text
report/SD6103_final_report.pdf   final project report (37 pages); video summary on YouTube (link above)
diagrams/er_model.drawio         conceptual ER model
diagrams/logical_schema.drawio   logical schema
```

Open the `.drawio` files with [diagrams.net](https://app.diagrams.net/) or the draw.io VS Code extension.

## Reproducing the results

Follow the instructions in the [code repository](https://github.com/yimingwang425/SD6103_Project): the database schema, stored procedures (`StoredProcedures/`), loading and benchmark scripts (`scripts/`) and raw benchmark results (`results/`) are maintained there.

Student ID numbers have been removed from the report cover for this public copy.
