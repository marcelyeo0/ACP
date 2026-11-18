-- Presence article x point de vente sur la fenetre de reference.
-- Sert a construire l'assortiment theorique par typologie et la distribution effective.
-- Dialecte DuckDB. {raw} = repertoire des parquets bruts.
SELECT
    t.store_nbr,
    t.item_nbr,
    i.family,
    SUM(GREATEST(t.unit_sales, 0))       AS units,
    COUNT(DISTINCT t.date)               AS selling_days
FROM read_parquet('{raw}/train.parquet') AS t
JOIN read_parquet('{raw}/items.parquet') AS i USING (item_nbr)
WHERE CAST(t.date AS DATE) BETWEEN DATE '{reference_start}' AND DATE '{reference_end}'
  AND t.unit_sales > 0
GROUP BY ALL
ORDER BY t.store_nbr, t.item_nbr;
