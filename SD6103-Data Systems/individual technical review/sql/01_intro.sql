.echo on
SELECT version() AS version, current_database() AS database;
SELECT count(*) AS csv_rows
FROM read_csv('data/raw/sales.csv', header=true);
.echo off
