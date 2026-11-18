# Segmentation par ACP & prévision de distribution

Pipeline Python/SQL d'analyse d'assortiment sur les ventes réelles de **Corporación Favorita** (distributeur alimentaire équatorien : 54 magasins, ~4 000 articles, 125 M de lignes de ventes journalières 2013-2017).

ACP sur le mix et la profondeur de gamme des magasins → CAH (Ward) + consolidation k-means → typologies de points de vente → prévision à 3 mois par typologie → écart entre assortiment théorique (gamme cœur de la typologie) et distribution effective de chaque magasin. Restitution Streamlit + exports CSV en étoile pour Power BI.

## Problème

Un réseau n'applique jamais l'assortiment théorique tel quel. Comparer un hypermarché de Quito à une supérette de province n'a pas de sens : il faut d'abord des **groupes de magasins comparables**. L'ACP résume 58 variables d'assortiment en 6 axes interprétables ; la CAH regroupe les magasins en typologies homogènes, seule maille où une prévision et un écart d'assortiment ont du sens.

## Résultats (fenêtre de référence août 2016 → juillet 2017)

| Étape | Résultat |
|---|---|
| Données | 51 magasins actifs × 58 variables (29 familles × part de ventes + profondeur de gamme). 3 magasins ouverts/fermés en cours de fenêtre projetés en individus supplémentaires. |
| ACP | 6 axes, **81 % de variance**. F1 (54 %) = effet taille : profondeur de gamme. F2 = magasins à rayons non alimentaires (fournitures scolaires, auto) vs fêtes/presse/textile. |
| Typologies | **3 classes** (Ward + k-means, 0 réaffectation). ARI = **0,70** avec le format Favorita (A-E) : T1 = A+D (gamme large, 25 magasins), T2 = C (gamme courte, poids épicerie/entretien, 15), T3 = B+E (11). |
| Prévision 3 mois | Holt-Winters amorti multiplicatif à périmètre constant : **MAPE 4,4 % vs 7,6 %** pour la naïve saisonnière (**−42 %**), backtest 12 origines glissantes × 3 typologies. |
| Écarts | Couverture moyenne de la gamme cœur 98,4 %. Magasin 26 (Guayaquil) : 90,6 %, 330 références cœur manquantes, potentiel +18,5 % sur la gamme cœur. |

Figures : `outputs/figures/` (éboulis, cercles des corrélations, plan factoriel, dendrogramme, prévisions).

## Approche

1. **Extraction SQL** (DuckDB sur parquet, `sql/`) : ventes magasin × famille × mois, présence article × magasin, ventes à périmètre constant.
2. **Nettoyage** : retours (ventes négatives) ramenés à 0, mois partiels exclus, magasins sans 12 mois d'activité sortis de l'ACP.
3. **Variables** : part de chaque famille dans les ventes du magasin (mix, neutralise la taille) + % des références de la famille vendues (profondeur).
4. **ACP normée** (SVD maison : valeurs propres, coordonnées, contributions, cos², projection d'individus supplémentaires).
5. **CAH Ward** sur les axes retenus, coupe au plus grand saut du dendrogramme (k ≥ 3), consolidation k-means ; profils par valeurs-tests.
6. **Prévision** par typologie, unités par magasin actif à périmètre constant ; backtest par origine glissante contre naïve saisonnière ; intervalles empiriques à 80 % issus des erreurs du backtest.
7. **Écarts** : gamme cœur = articles vendus par ≥ 80 % des magasins de la typologie ; manquants valorisés au volume médian des magasins pairs.

Détail et justification des choix : [`docs/METHODOLOGIE.md`](docs/METHODOLOGIE.md). Argumentaire d'entretien : [`docs/DEFENSE_PROJET.md`](docs/DEFENSE_PROJET.md).

## Stack

Python 3.11+ · DuckDB (SQL) · pandas · NumPy · scikit-learn · SciPy · statsmodels · Matplotlib · Streamlit · pydantic-settings · pytest · ruff · GitHub Actions

## Structure

```
sql/                 # extractions DuckDB : dim_stores, extract_sales, extract_store_items, extract_sales_lfl
src/pca_segmentation/
├── config.py        # paramètres (.env)
├── pipeline.py      # orchestration de bout en bout
├── data/            # loaders (SQL, téléchargement), cleaning, aggregation
├── features/        # parts, profondeur, centrage-réduction
├── analysis/        # pca, clustering (CAH + k-means), interpretation (axes, valeurs-tests)
├── forecasting/     # baseline, Holt-Winters, backtest
├── reporting/       # gap_analysis, figures
└── app/             # dashboard Streamlit (+ pages/)
tests/               # invariants ACP, clustering, absence de fuite du backtest, écarts
outputs/exports/     # CSV en étoile (dim_stores + fact_*) pour Streamlit / Power BI
```

## Utilisation

```bash
uv sync --dev
cp .env.example .env
uv run python -m pca_segmentation.pipeline     # télécharge les parquets (~900 Mo) au premier lancement
uv run streamlit run src/pca_segmentation/app/streamlit_app.py
```

`--skip-extract` relance l'analyse sans refaire l'extraction SQL.

## Qualité

```bash
uv run ruff check .
uv run pytest
```

## Limites

- **Pas de stock ni de linéaire** : « non vendu sur 12 mois » ≠ « non référencé ». Un article cœur absent peut être en rupture, déréférencé ou simplement sans demande locale. L'écart est une liste de magasins/références à vérifier, pas un verdict.
- **F1 = effet taille** : la profondeur de gamme suit la surface. Assumé et interprété ; une ACP sur les seules parts donnerait une segmentation « mix pur ».
- **Unités hétérogènes** : les produits frais sont en kg, le reste en unités ; les parts de volume mélangent les deux (pas de prix dans la source).
- **Chocs non modélisés** : séisme d'avril 2016 (pic de dons), jours de paie. Une variable d'intervention améliorerait le backtest sur avril 2017.
- Prévision par typologie, pas par magasin : la maille utile pour piloter un assortiment, et des séries magasin trop bruitées.
