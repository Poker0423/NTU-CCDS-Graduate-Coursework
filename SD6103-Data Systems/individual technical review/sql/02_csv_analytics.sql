.echo on
CREATE OR REPLACE TEMP VIEW retail AS
SELECT s.*, c.region, p.category,
  CAST(quantity*unit_price-discount_amount AS DECIMAL(12,2)) AS revenue
FROM sales s JOIN customers c USING(customer_id)
JOIN products p USING(product_id);

SELECT region, category, count(*) AS transactions,
       sum(revenue) AS revenue_sgd
FROM retail
GROUP BY region, category
ORDER BY revenue_sgd DESC, region, category
LIMIT 5;
.echo off
