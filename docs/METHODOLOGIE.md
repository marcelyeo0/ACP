# Méthodologie

## 0. Données

Corporación Favorita Grocery Sales (compétition Kaggle 2017, miroir parquet Hugging Face `hsnalmasri/favorita`) :

| Table | Contenu |
|---|---|
| `train` | 125 M lignes : date × magasin × article → unités vendues (kg pour le frais), promotion |
| `items` | 4 100 articles : famille (33), classe, périssable |
| `stores` | 54 magasins : ville, province, format `type` (A-E), `cluster` Favorita (17) |

Pourquoi ce jeu : vrai réseau multi-formats, maille article × magasin (nécessaire pour la présence/assortiment), historique de 4,5 ans (saisonnalité annuelle estimable), et une segmentation métier existante (`type`) contre laquelle valider la nôtre.

## 1. Préparation (`sql/`, `data/`)

- Extraction en SQL DuckDB directement sur le parquet : 125 M lignes agrégées en quelques secondes, sans tout charger en mémoire pandas.
- Retours (ventes < 0) ramenés à 0 ; août 2017 exclu (mois partiel).
- **Fenêtre de référence** : 12 mois glissants complets (août 2016 → juillet 2017), pour capter un cycle saisonnier entier dans les parts.
- Magasins actifs < 12 mois sur la fenêtre (n°18 : 9 mois actifs, n°25 Salinas : 11 mois, n°52 : ouvert en avril 2017) exclus de l'ACP et **projetés en individus supplémentaires**.

## 2. Variables (`features/`)

Pour chaque famille ayant au moins 5 références (29 familles) :

- `part_<famille>` : part de la famille dans les volumes du magasin (%). Neutralise l'effet taille sur le mix.
- `pres_<famille>` : % des références de la famille (vendues dans le réseau) que le magasin vend. Mesure la profondeur de gamme.

## 3. ACP (`analysis/pca.py`)

- **Normée** (centrage-réduction) : parts et profondeurs n'ont ni la même unité ni la même dispersion.
- Implémentée par SVD de la matrice centrée-réduite pour exposer explicitement valeurs propres (somme = p), coordonnées, contributions (somme = 100 % par axe), cos² et projection de supplémentaires. Ces invariants sont testés.
- **Nombre d'axes** : min(critère de Kaiser λ > 1, variance cumulée ≥ 80 %), au moins 2 → 6 axes, 81 %.
- **Lecture des axes** : uniquement par les variables à contribution > 100/p, avec le signe de la corrélation.
  - F1 (54 %) : toutes les profondeurs du même côté = **effet taille**. Grands magasins à gamme large vs petits.
  - F2 (9 %) : profondeur auto, fournitures scolaires, maison vs part fêtes, presse, textile.
  - F3 (7 %) : hygiène/beauté/entretien vs boissons/alcools/œufs.

## 4. Classification (`analysis/clustering.py`)

- **CAH de Ward** sur les 6 coordonnées factorielles (bruit des derniers axes écarté).
- **Coupe** : plus grand saut de Ward pour k ≥ 3. Le saut est maximal à k = 2 mais cette partition ne fait que couper F1 (grands vs petits) ; k = 3 est le saut suivant (9,4 contre 2,4 pour k = 4) et sépare un groupe distinct (formats B/E). Silhouette 0,42 (k=2 : 0,47).
- **Consolidation k-means** initialisée sur les centres CAH : 0 magasin réaffecté → partition stable.
- **Caractérisation** par valeurs-tests (|v| > 1,96).
- **Validation externe** : ARI = 0,70 avec le format Favorita. T1 regroupe A + D, T3 regroupe B + E : du point de vue de l'assortiment, ces formats ne se distinguent pas.

## 5. Prévision (`forecasting/`)

- **Cible** : unités par magasin actif et par mois, par typologie (diviser par le nombre de magasins actifs évite de lire les ouvertures comme de la croissance).
- **Périmètre constant** : seuls les 1 487 articles vendus chaque mois de l'historique. Le nombre d'articles présents dans la source oscille entre ~2 000 et ~2 800 selon les mois de 2014 ; sans ce filtre, la série mesure la couverture du catalogue, pas la demande.
- **Baseline** naïve saisonnière (même mois l'an dernier) + variante avec dérive annuelle.
- **Modèle** Holt-Winters : saisonnalité multiplicative (le pic de décembre croît avec le niveau), tendance additive amortie (sur 3 mois, une tendance linéaire sur-réagit).
- **Backtest** par origine glissante : 12 origines × 3 pas × 3 typologies, entraînement strictement antérieur à l'origine (testé).
- **Intervalles** à 80 % : quantiles des erreurs relatives du backtest par pas d'horizon, plutôt que l'hypothèse gaussienne.

| Modèle | MAPE | RMSE | Gain MAPE vs baseline |
|---|---|---|---|
| Holt-Winters | 4,4 % | 11 251 | +42 % |
| Naïve saisonnière | 7,6 % | 16 633 | — |
| Naïve saisonnière + dérive | 8,0 % | 17 389 | −6 % |

## 6. Écarts d'assortiment (`reporting/gap_analysis.py`)

- **Assortiment théorique** d'une typologie = articles vendus par ≥ 80 % de ses magasins (gamme cœur : 3 501 / 2 432 / 2 979 références).
- **Distribution effective** = taux de couverture de cette gamme par magasin (distribution numérique).
- **Potentiel** d'un manquant = volume médian des magasins de la typologie qui le vendent.
- Couverture moyenne 98,4 %, min 90,6 % (magasin 26) ; potentiel total +0,6 % sur la gamme cœur, concentré sur quelques magasins.

## Pièges évités

- ACP sur données brutes : la famille GROCERY I écraserait tous les axes.
- Nommer un axe sans contributions.
- Prendre k = 2 parce que « le dendrogramme le dit » : partition triviale.
- Somme brute des ventes : mélange couverture catalogue, ouvertures de magasins et demande.
- MAPE sans baseline : 4,4 % ne dit rien seul ; −42 % vs naïve saisonnière, si.
- Backtest avec fuite : chaque modèle est réentraîné à chaque origine sur le seul passé.
