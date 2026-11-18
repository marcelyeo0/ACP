-- Dimension points de vente (ville, province, format Favorita, cluster Favorita)
-- et periode d'activite observee dans les ventes.
-- Dialecte DuckDB. {raw} = repertoire des parquets bruts.
WITH activity AS (
    SELECT
        store_nbr,
        MIN(CAST(date AS DATE)) AS first_sale,
        MAX(CAST(date AS DATE)) AS last_sale,
        COUNT(DISTINCT date_trunc('month', CAST(date AS DATE))) FILTER (
            WHERE CAST(date AS DATE) BETWEEN DATE '{reference_start}' AND DATE '{reference_end}'
        ) AS active_months_ref
    FROM read_parquet('{raw}/train.parquet')
    GROUP BY store_nbr
)
SELECT
    s.store_nbr,
    s.city,
    s.state,
    s.type   AS store_type,
    s.cluster AS favorita_cluster,
    a.first_sale,
    a.last_sale,
    a.active_months_ref
FROM read_parquet('{raw}/stores.parquet') AS s
LEFT JOIN activity AS a USING (store_nbr)
ORDER BY s.store_nbr;
