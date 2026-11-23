-- Optional rehearsal evidence; not needed in the timed recording.
.echo on
EXPLAIN ANALYZE
SELECT c.region,p.category,sum(s.quantity*s.unit_price-s.discount_amount)
FROM sales s JOIN customers c USING(customer_id)
JOIN products p USING(product_id)
GROUP BY c.region,p.category;
.echo off
