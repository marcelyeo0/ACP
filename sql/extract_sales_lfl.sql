-- Ventes mensuelles par point de vente a perimetre constant (like-for-like).
-- Le nombre d'articles presents dans la source varie fortement d'un mois a l'autre
-- (~2 000 a ~2 800 en 2014) : sommer tous les articles melange demande et couverture
-- catalogue. On ne garde que les articles vendus dans le reseau chaque mois de l'historique.
-- Dialecte DuckDB. {raw} = repertoire des parquets bruts.
WITH sales AS (
    SELECT
        store_nbr,
        item_nbr,
        date_trunc('month', CAST(date AS DATE)) AS month,
        GREATEST(unit_sales, 0)                 AS units
    FROM read_parquet('{raw}/train.parquet')
    WHERE CAST(date AS DATE) <= DATE '{history_end}'
),
panel AS (
    SELECT item_nbr
    FROM sales
    WHERE units > 0
    GROUP BY item_nbr
    HAVING COUNT(DISTINCT month) = (SELECT COUNT(DISTINCT month) FROM sales)
)
SELECT
    s.store_nbr,
    s.month,
    SUM(s.units)                AS units,
    COUNT(DISTINCT s.item_nbr)  AS n_items_sold,
    (SELECT COUNT(*) FROM panel) AS panel_size
FROM sales AS s
SEMI JOIN panel USING (item_nbr)
GROUP BY ALL
ORDER BY s.store_nbr, s.month;
