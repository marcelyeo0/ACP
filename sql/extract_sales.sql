-- Ventes mensuelles par point de vente x famille de produits.
-- Base de la serie temporelle (prevision) : tout l'historique mensuel complet.
-- Les retours (unit_sales < 0) sont ramenes a 0 : ils ne disent rien de l'assortiment.
-- Dialecte DuckDB. {raw} = repertoire des parquets bruts.
SELECT
    t.store_nbr,
    i.family,
    date_trunc('month', CAST(t.date AS DATE))             AS month,
    SUM(GREATEST(t.unit_sales, 0))                         AS units,
    COUNT(DISTINCT t.item_nbr) FILTER (WHERE t.unit_sales > 0) AS n_items_sold,
    SUM(CASE WHEN t.onpromotion THEN GREATEST(t.unit_sales, 0) ELSE 0 END) AS units_promo
FROM read_parquet('{raw}/train.parquet') AS t
JOIN read_parquet('{raw}/items.parquet') AS i USING (item_nbr)
WHERE CAST(t.date AS DATE) <= DATE '{history_end}'
GROUP BY ALL
ORDER BY t.store_nbr, i.family, month;
