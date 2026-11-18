# Défendre le projet en entretien

## Pitch 30 secondes

> « J'ai pris les ventes réelles d'un distributeur alimentaire équatorien, 54 magasins et 125 millions de lignes. Question : comment comparer des magasins qui n'ont rien à voir entre eux, et où l'assortiment prévu n'est pas appliqué ? J'ai résumé 58 variables d'assortiment par une ACP, regroupé les magasins en 3 typologies par CAH, prévu les ventes à 3 mois par typologie avec 42 % d'erreur en moins qu'une prévision naïve, et listé magasin par magasin les références de la gamme cœur qui manquent, avec le volume en jeu. »

## Version 2 minutes (structure STAR)

- **Situation** : un réseau multi-formats. Le siège définit un assortiment, les magasins l'appliquent plus ou moins. Sans grille de lecture, on compare un hypermarché de Quito à une supérette de province.
- **Tâche** : construire des groupes de magasins comparables, prévoir leur activité, mesurer l'écart entre assortiment théorique et réel.
- **Action** :
  1. SQL (DuckDB) sur 125 M lignes → matrice magasin × famille : part des ventes (mix) et profondeur de gamme.
  2. ACP normée : 6 axes, 81 % de l'information. Axe 1 = taille/profondeur de gamme, axe 2 = rayons non alimentaires vs fêtes/textile.
  3. CAH Ward + consolidation k-means : 3 typologies. Contrôle : ARI 0,70 avec les formats de l'enseigne.
  4. Prévision Holt-Winters à périmètre constant, validée par backtest glissant contre une baseline.
  5. Gamme cœur par typologie, couverture par magasin, potentiel des manquants.
- **Résultat** : MAPE 4,4 % contre 7,6 % (−42 %). Les formats A et D d'un côté, B et E de l'autre ont le même profil d'assortiment. Un magasin (n°26, Guayaquil) couvre seulement 90,6 % de sa gamme cœur : 330 références manquantes, +18 % de potentiel. C'est la première visite terrain à faire.

## Chiffres à connaître par cœur

| | |
|---|---|
| Données | 54 magasins, ~4 000 articles, 125 M lignes, 2013-2017 |
| Matrice ACP | 51 magasins × 58 variables (29 familles × 2) |
| ACP | 6 axes, 81 % ; F1 = 54 % (effet taille) |
| Typologies | 3 : 25 / 15 / 11 magasins ; ARI 0,70 vs formats |
| Prévision | Holt-Winters MAPE 4,4 % vs naïve 7,6 % → −42 % ; 12 origines × 3 mois × 3 typologies |
| Écarts | couverture moyenne 98,4 % ; min 90,6 % (magasin 26) |

## Questions probables et réponses

**Pourquoi une ACP avant le clustering ?**
58 variables corrélées entre elles (un grand magasin est profond partout). Sans ACP, les distances sont dominées par ce bloc redondant et par le bruit. L'ACP garde 81 % de l'information sur 6 axes décorrélés et rend la classification lisible : chaque typologie se lit sur le plan factoriel.

**Pourquoi centrer-réduire ?**
Parts en % et profondeurs n'ont pas la même dispersion. Sans réduction, la famille Épicerie (la plus grosse part) ferait à elle seule le premier axe.

**Comment as-tu choisi 6 axes ?**
Minimum entre le critère de Kaiser (valeur propre > 1, soit plus qu'une variable d'origine) et 80 % de variance cumulée. Kaiser seul en garderait 8 avec autant de variables ; je préfère des axes que je sais interpréter.

**Comment as-tu nommé les axes ?**
Par les contributions, jamais à l'œil sur le cercle. Je ne retiens que les variables qui contribuent plus que la moyenne (100/p), avec le signe. F1 : toutes les profondeurs du même côté, donc effet taille.

**F1 est un effet taille, ce n'est pas un problème ?**
C'est une information réelle : la profondeur de gamme suit la surface. Je l'assume et je l'interprète. Si le métier voulait une segmentation sur le mix seul, je referais l'ACP sur les parts uniquement. C'est un choix de question, pas une erreur.

**Pourquoi 3 classes et pas 2 ?**
Le plus grand saut du dendrogramme est à 2, mais 2 classes ne font que couper l'axe 1 en grands vs petits magasins : on le voit sans classification. Le saut suivant est à 3 (9,4, contre 2,4 pour passer à 4) et isole un vrai groupe (formats B/E). Silhouette 0,42 contre 0,47 : je perds un peu de séparation pour gagner une classe utile.

**Pourquoi CAH puis k-means ?**
La CAH donne une hiérarchie lisible et le nombre de classes. Mais elle est gloutonne : une fusion n'est jamais remise en cause. Le k-means initialisé sur les centres CAH corrige ces affectations. Ici, 0 magasin a bougé : la partition est stable.

**Comment sais-tu que tes typologies valent quelque chose ?**
Trois contrôles : valeurs-tests significatives par classe ; stabilité CAH/k-means ; et une validation externe, l'indice de Rand ajusté de 0,70 avec les formats de l'enseigne, sans les avoir utilisés. Le désaccord est lui-même un résultat : les formats A et D ont le même profil d'assortiment.

**Pourquoi Holt-Winters et pas un modèle ML ?**
55 points mensuels par série. Un modèle à gradient boosting n'a rien à apprendre là-dessus. Holt-Winters a peu de paramètres, chacun interprétable (niveau, tendance, saison). Saisonnalité multiplicative, parce que le pic de décembre grossit avec le niveau. Tendance amortie, parce qu'à 3 mois une tendance linéaire extrapole trop.

**Pourquoi « périmètre constant » ?**
En 2014, le nombre d'articles présents dans la source passe de 2 000 à 2 800 selon les mois. Sommer tout mesurerait la couverture du catalogue, pas la demande. J'ai gardé les 1 487 articles vendus tous les mois : c'est la logique « à magasins/produits comparables » de la distribution.

**Comment as-tu validé la prévision ?**
Backtest par origine glissante : à chaque origine, le modèle est réentraîné sur le passé seul et prévoit 3 mois. 12 origines × 3 typologies. Un test unitaire vérifie qu'aucune date future n'entre dans l'entraînement. Et je compare toujours à une baseline : une MAPE de 4,4 % ne dit rien seule, −42 % contre « même mois l'an dernier » si.

**Ta baseline « avec dérive » fait pire que la naïve. Pourquoi la garder ?**
Pour montrer qu'ajouter de la complexité n'améliore pas mécaniquement. La croissance annuelle estimée sur 12 mois est bruitée par le séisme d'avril 2016. Holt-Winters, lui, amortit la tendance.

**Comment définis-tu l'assortiment théorique ?**
Je n'avais pas le plan d'assortiment officiel. Je l'ai reconstruit : un article est dans la gamme cœur d'une typologie s'il est vendu par au moins 80 % de ses magasins. Dans une entreprise, je remplacerais cette règle par le référentiel d'assortiment réel ; le reste du calcul ne change pas.

**Une référence non vendue, c'est une référence absente ?**
Non, et c'est la limite principale. Sans données de stock ni de linéaire, « zéro vente en 12 mois » peut vouloir dire déréférencée, en rupture ou sans demande locale. L'écart donne une liste priorisée à vérifier en magasin, pas un verdict.

**Comment as-tu valorisé le potentiel ?**
Volume médian des magasins de la même typologie qui vendent l'article. Médiane plutôt que moyenne pour ne pas laisser un magasin exceptionnel gonfler le potentiel. C'est un ordre de grandeur pour prioriser, pas une promesse de chiffre d'affaires (pas de prix dans la source).

**Qu'est-ce que tu ferais avec plus de temps ?**
- Une variable d'intervention pour le séisme d'avril 2016 et les jours de paie.
- Une prévision par typologie × famille, la maille d'une décision d'assortiment.
- Une analyse de stabilité de la typologie par bootstrap.
- Croiser l'écart avec les stocks pour séparer rupture et déréférencement.

**Pourquoi DuckDB et pas PostgreSQL ?**
125 M lignes en parquet : DuckDB les agrège en quelques secondes sans serveur ni chargement. Les requêtes sont en SQL standard (GROUP BY, CTE, SEMI JOIN, FILTER) et se portent sur PostgreSQL ou BigQuery presque telles quelles.

## Pièges à ne pas tendre soi-même

- Ne pas dire « l'ACP a trouvé 3 clusters » : l'ACP réduit la dimension, c'est la CAH qui classe.
- Ne pas présenter 98,4 % de couverture comme une performance du réseau : par construction, la gamme cœur est ce que 80 % des magasins vendent déjà. L'information est dans la dispersion et dans les magasins en bas du classement.
- Ne pas survendre le potentiel (+0,6 % au global) : il est concentré sur quelques magasins, c'est ça le message.
- Savoir redessiner de tête : éboulis, cercle des corrélations, dendrogramme avec la coupe.

## Cohérence avec le CV

- Le CV mentionne un **dashboard Power BI**. Le repo livre un dashboard **Streamlit** et des CSV en étoile (`outputs/exports/` : `dim_stores` + tables `fact_*`) prêts pour Power BI. Avant l'entretien : soit monter le rapport Power BI sur ces exports, soit corriger le CV en « dashboard Streamlit ». Ne pas laisser l'écart, il se voit en ouvrant le repo.
- Données : jeu public Kaggle/Favorita. Le dire d'emblée. Un projet sur données publiques bien défendu vaut mieux qu'un flou sur l'origine des données.
