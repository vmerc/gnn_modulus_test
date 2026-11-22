# Campagne Têt avec ghost nodes

Le protocole suit le manuscrit Elsevier `main.pdf` fourni le 18 novembre 2026 (§ 3.4–3.5, § 4, tableau 2, § 5 et annexe A). Les six expériences utilisent un ghost node alimenté par le débit amont grâce à un encodeur distinct. La hauteur prescrite à l’aval reste imposée comme dans TELEMAC-2D. L’ablation de \(Q(t)\) porte sur sa diffusion supplémentaire aux nœuds physiques.

| YAML dans `bin/conf/papier_ghost/` | YAML historique | Graphe | \(Q(t)\) global | Fenêtre d’entraînement | Entraînement |
| --- | --- | --- | --- | --- | --- |
| `E1.yaml` | `Experience1` | ×8 | Non | 35–64 | Un pas |
| `E2.yaml` | `Experience7` | ×8 | Oui | 35–64 | Un pas |
| `E3.yaml` | `Experience9` | ×8 | Oui | 35–64 | Pushforward, warmup 150 |
| `E4.yaml` | `Experience2` | ×8 + ×16 + ×32 | Non | 35–64 | Un pas |
| `E5.yaml` | `Experience8` | ×8 + ×16 + ×32 | Oui | 35–64 | Un pas |
| `E6.yaml` | `Experience10` | ×8 + ×16 + ×32 | Oui | 35–64 | Pushforward, warmup 150 |

Chaque expérience utilise les mêmes 40 crues d’entraînement, avec 30 transitions d’une demi-heure, d’indices 35 à 64, dans la fenêtre de 15 h centrée sur le débordement. Les 16 crues de test sont séparées à l’échelle de l’événement.

Chaque entraînement couvre 900 époques. E3 reprend le checkpoint 600 d’E2 et E6 celui d’E5. Le pushforward s’applique des époques 601 à 900 avec une décroissance linéaire du teacher forcing sur 150 époques. Les reprises conservent la normalisation, les poids et les états de l’optimiseur, du scheduler et du scaler du checkpoint d’origine. Le lanceur réserve 8 GPU A100 de 80 Go sur 4 nœuds, comme au § 4.4. La partition `small` autorise jusqu’à 4 h par job. [Documentation Turpan](https://www.mesonet.fr/documentation/user-documentation/arch_exp/turpan/soumettre_calcul/)

## Stockage

`bin/conf/papier_ghost/base.yaml` définit `campaign_dir` à `/work/m24046/m24046mrcr/paper_ghost`. La campagne réutilise les trajectoires historiques dans `/work/m24046/m24046mrcr/dataset_Tet_short_ghost`. Les graphes physiques régénérés sont dans `paper_ghost/graphs/`, avec les mêmes nœuds ×8 pour les six expériences et les arêtes projetées ×16 et ×32 pour E4–E6. Les poids et la normalisation de chaque expérience sont dans `E1/Seed0/` à `E6/Seed0/`. Les métriques et données de figures sont séparées dans `article_results/E*/Seed0/eval-900/`. Un lancement normal protège son dossier de sortie existant ; le mode `resume` est le seul à le réutiliser.

## Préparer et vérifier les données

Le dataset historique a été inventorié sur Turpan avec 56 fenêtres courtes 35–64 et 56 trajectoires complètes 0–80. Il ne doit pas être régénéré pour cette campagne. Depuis `bin/` sur la frontale, régénère uniquement les deux graphes physiques depuis les fichiers SLF. Le job exige 96 984 arêtes dirigées pour le ×8 et 130 601 arêtes, dont 39 self-loops, pour le multimaillage ×8–×16–×32. Ce dernier diffère de quatre arêtes nettes par rapport aux 130 597 arêtes indiquées au § 3.5, tableau 1 de `main.pdf`.

```bash
sbatch prepare_tet_ghost_graphs.sh
```

Le job refuse de remplacer un dossier `paper_ghost/graphs` existant.

Après succès, soumets le contrôle. Il vérifie les chemins d’entraînement, les graphes, l’alignement du débit au ghost node et les incréments à l’entrée. Il exécute des batches à un pas et en pushforward sur un GPU et vérifie la conservation de la normalisation lors d’une reprise de test E5 vers E6.

```bash
sbatch test_turpan_gnn.sh
```

## Entraîner

Après succès du contrôle, soumets les quatre entraînements à un pas. E2 et E5 passent en premier pour produire les checkpoints de reprise. Les options `-J` identifient l’expérience dans les fichiers `.out` et `.err`.

```bash
for exp in E2 E5 E1 E4; do
  sbatch -J "$exp" train_push_source_nodes.sh "papier_ghost/$exp"
done
```

Après la sauvegarde du checkpoint 600 d’E2, soumets E3. Fais de même pour E6 à partir du checkpoint 600 d’E5. Les chemins des deux reprises sont définis dans leurs YAML.

```bash
sbatch -J E3 train_push_source_nodes.sh papier_ghost/E3
sbatch -J E6 train_push_source_nodes.sh papier_ghost/E6
```

Si Slurm interrompt un entraînement à la limite de quatre heures, attends la fin du job puis ajoute `resume` à la même commande. La reprise choisit la dernière époque possédant à la fois les poids du modèle et l’état d’entraînement, puis restaure le modèle, l’optimiseur, le scheduler et le scaler. Elle conserve aussi le début du pushforward à l’époque 601 pour E3 et E6.

```bash
sbatch -J E2-resume train_push_source_nodes.sh papier_ghost/E2 resume
```

Pour préparer la reprise avant la coupure sans faire tourner deux jobs simultanément, ajoute une dépendance au job actif :

```bash
sbatch --dependency=afterany:ID_JOB -J E2-resume train_push_source_nodes.sh papier_ghost/E2 resume
```

Ne lance jamais une reprise sans cette dépendance pendant que le job précédent écrit encore dans le même dossier. Les états des générateurs aléatoires ne sont pas sauvegardés : les paramètres d’entraînement et la graine restent identiques, mais la suite des batches après une reprise n’est pas garantie bit à bit.

## Analyser les graphes

Le tableau 1 et la figure 2 doivent être recalculés, car le multimaillage régénéré possède 130 601 arêtes au lieu des 130 597 du manuscrit. Ce job lit les graphes existants sans les modifier et écrit `article_results/graph/graph_summary_table1.csv` ainsi que `edge_lengths_figure2.npz`.

```bash
sbatch analyze_tet_ghost_graphs.sh
```

## Évaluer

Pour chaque expérience, soumets l’évaluation après la sauvegarde du checkpoint final. Le job lit l’époque 900 par défaut et écrit dans `/work/m24046/m24046mrcr/paper_ghost/article_results/E*/Seed0/eval-900/`, sans mélanger les analyses et les poids. Il refuse de remplacer un résultat existant. L’évaluation principale utilise les 16 crues de test et rapporte la MAE horaire — la norme \(L_1\) du manuscrit — sur \(h,u,v\) de 1 à 12 h. Elle rapporte aussi le CSI à 5 et 30 cm sur la grille de 25 m pour E2, E3, E5 et E6, par rapport aux sorties TELEMAC fines. Le contrôle exige la grille 689 × 748 du § 5.4 et ses 276 001 points dans le domaine TELEMAC. Le CSI utilise les 275 827 points également contenus dans la triangulation ×8, sans extrapoler les 174 points périphériques restants.

```bash
for exp in E1 E2 E3 E4 E5 E6; do
  sbatch -J "eval-$exp" eval_tet_ghost.sh "papier_ghost/$exp"
done
```

Chaque dossier contient `metrics_by_event.csv` et `metrics_summary.csv`. Les jobs E3 et E6 écrivent aussi `map_figure8.npz` pour le groupe 3 au pic de 3400 m³/s à 6 h et au seuil de 30 cm. `timing_6h.csv` contient trois mesures du rollout sur un GPU après échauffement. Le compteur démarre une fois les entrées chargées et s’arrête avant la production des cartes.

Les jobs E5 et E6 enchaînent avec l’annexe A sur les 16 crues de test complètes avec le même checkpoint. `metrics_long_by_event.csv` et `metrics_long_summary.csv` contiennent la MAE toutes les 30 minutes jusqu’à 40 h. Pour E5, `map_figure10.npz` fournit les masques sur le maillage ×8 à 6, 12, 24 et 36 h pour le groupe 3 au pic de 2800 m³/s et au seuil de 10 cm, avec le CSI nodal indiqué dans la figure.

Ces fichiers couvrent les données des figures 3 à 10 de `main.pdf`. Une fois les six dossiers récupérés, construis les figures à partir de ces sorties communes plutôt que depuis les anciens notebooks. Les temps TELEMAC de 27 min et 180 min ne sont pas mesurés par ce job et doivent garder leur provenance séparée.

## Tracer les courbes

Après les six évaluations, ce job CPU lit les CSV communs et crée les figures 3 à 7 et 9 en PDF et PNG dans `article_results/figures/`. Depuis le Mac, soumets le job après avoir commité les scripts.

```bash
bash bin/turpan.sh submit bin/plot_tet_ghost_curves.sh \
  /work/m24046/m24046mrcr/paper_ghost/article_results
```

Si `figures/` existe déjà, ajoute `--output-dir /work/…/autre-dossier` pour conserver la génération précédente. Les cartes des figures 8 et 10 restent dans leurs fichiers NPZ et font l’objet d’un tracé séparé.
