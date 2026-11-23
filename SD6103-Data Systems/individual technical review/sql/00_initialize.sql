-- PREPARATION ONLY. setup.ps1 executes this before recording, in one transaction.
BEGIN;
CREATE OR REPLACE TABLE customers (
  customer_id INTEGER PRIMARY KEY, region VARCHAR NOT NULL, segment VARCHAR NOT NULL
);
INSERT INTO customers SELECT * FROM read_csv('data/raw/customers.csv',
  columns={'customer_id':'INTEGER','region':'VARCHAR','segment':'VARCHAR'}, header=true);
CREATE OR REPLACE TABLE products (
  product_id INTEGER PRIMARY KEY, product_name VARCHAR NOT NULL,
  category VARCHAR NOT NULL, list_price DECIMAL(10,2) NOT NULL
);
INSERT INTO products SELECT * FROM read_csv('data/raw/products.csv',
  columns={'product_id':'INTEGER','product_name':'VARCHAR','category':'VARCHAR','list_price':'DECIMAL(10,2)'},header=true);
CREATE OR REPLACE TABLE sales (
  sale_id INTEGER PRIMARY KEY, sale_date DATE NOT NULL,
  customer_id INTEGER NOT NULL, product_id INTEGER NOT NULL,
  quantity SMALLINT NOT NULL CHECK(quantity>0),
  unit_price DECIMAL(10,2) NOT NULL CHECK(unit_price>=0),
  discount_amount DECIMAL(10,2) NOT NULL CHECK(discount_amount>=0)
);
INSERT INTO sales SELECT * FROM read_csv('data/raw/sales.csv',
  columns={'sale_id':'INTEGER','sale_date':'DATE','customer_id':'INTEGER','product_id':'INTEGER',
    'quantity':'SMALLINT','unit_price':'DECIMAL(10,2)','discount_amount':'DECIMAL(10,2)'},header=true);
CREATE OR REPLACE VIEW retail_base AS
SELECT s.*, c.region, p.category,
       CAST(s.quantity*s.unit_price-s.discount_amount AS DECIMAL(12,2)) AS revenue
FROM sales s JOIN customers c USING(customer_id)
JOIN products p USING(product_id);
-- Baseline flat CSV has exactly the same rows/columns/order as the Parquet export.
COPY (SELECT * FROM retail_base ORDER BY sale_date,sale_id)
TO 'data/raw/retail_flat.csv' (HEADER, DELIMITER ',');
COPY (SELECT * FROM retail_base ORDER BY sale_date,sale_id)
TO 'data/parquet/retail_reference.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384);
COMMIT;
CHECKPOINT;
