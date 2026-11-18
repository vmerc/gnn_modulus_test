# Expériences historiques

Les chemins ci-dessous reproduisent l’arborescence utilisée avant la campagne Têt avec ghost nodes. Le commit `b8ea3a7` permet de retrouver les fichiers suivis à leurs chemins d’origine et de les exécuter depuis un checkout distinct avec les données et poids historiques sur Turpan.

| Chemin dans l’archive | Contenu et chemin d’origine |
| --- | --- |
| `Aube/` | Travaux Aube, tests Têt sur grille régulière et premières comparaisons ghost. Le notebook de génération d’hydrogrammes et ses huit fichiers `.liq` locaux ont été conservés avec le reste. |
| `bin/conf/papier1/` | Configurations historiques de la Têt, dont les six expériences du préprint. |
| `bin/conf/research/` | Transfert Têt/Aube, surapprentissage et premiers essais ghost. |
| `bin/` | Lanceurs, entraîneurs et préparation des données historiques déplacés depuis `bin/`. |
| `jupyter/` | Notebooks historiques de préparation et d’évaluation, notamment les rollouts Têt et le CSI sur la grille de 25 m. |

Cette archive est consultable dans le dépôt. Le lanceur Turpan exclut `archive/` des instantanés de calcul qu’il envoie par `git archive`.
