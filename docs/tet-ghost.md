# Campagne Têt avec ghost nodes

Le protocole suit le manuscrit Elsevier `main.pdf` fourni le 18 novembre 2026 (§ 4, tableau 2, § 5 et annexe A). Les six expériences utilisent un ghost node alimenté par le débit amont grâce à un encodeur distinct. La hauteur prescrite à l’aval reste imposée comme dans TELEMAC-2D. L’ablation de \(Q(t)\) porte sur sa diffusion supplémentaire aux nœuds physiques.

| YAML dans `bin/conf/papier_ghost/` | YAML historique | Graphe | \(Q(t)\) global | Fenêtre d’entraînement | Entraînement |
| --- | --- | --- | --- | --- | --- |
| `E1.yaml` | `Experience1` | ×8 | Non | 35–64 | Un pas |
| `E2.yaml` | `Experience7` | ×8 | Oui | 35–64 | Un pas |
| `E3.yaml` | `Experience9` | ×8 | Oui | 35–64 | Pushforward, warmup 150 |
| `E4.yaml` | `Experience2` | ×8 + ×32 | Non | 35–64 | Un pas |
| `E5.yaml` | `Experience8` | ×8 + ×32 | Oui | 35–64 | Un pas |
| `E6.yaml` | `Experience10` | ×8 + ×32 | Oui | 35–64 | Pushforward, warmup 150 |

Chaque expérience utilise les mêmes 40 crues d’entraînement, avec 30 transitions d’une demi-heure, d’indices 35 à 64, dans la fenêtre de 15 h centrée sur le débordement. Les 16 crues de test sont séparées à l’échelle de l’événement.

Chaque entraînement couvre 900 époques. E3 reprend le checkpoint 600 d’E2 et E6 celui d’E5. Le pushforward s’applique des époques 601 à 900 avec une décroissance linéaire du teacher forcing sur 150 époques. Les reprises conservent la normalisation, les poids et les états de l’optimiseur, du scheduler et du scaler du checkpoint d’origine. Le lanceur réserve 8 GPU A100 de 80 Go sur 4 nœuds, comme au § 4.4. La partition `small` autorise jusqu’à 4 h par job. [Documentation Turpan](https://www.mesonet.fr/documentation/user-documentation/arch_exp/turpan/soumettre_calcul/)

## Stockage

`bin/conf/papier_ghost/base.yaml` définit `campaign_dir` à `/work/m24046/m24046mrcr/paper_ghost`. Le dataset est partagé dans `data/`. Les poids et la normalisation de chaque expérience sont dans `E1/Seed0/` à `E6/Seed0/`. Cette structure reprend le rangement des anciennes expériences sous `paper/Experience*/Seed0/`. Chaque job protège son dossier de sortie existant. Pour relancer une expérience avec le même seed, déplace son dossier de sortie précédent sur Turpan.

## Préparer et vérifier les données

Depuis `bin/` sur la frontale, soumets la préparation. Le job projette les sorties TELEMAC ×8 et publie les deux graphes et les fenêtres dans `data/` une fois le dataset complet.

```bash
sbatch prepare_x8_short_ghost_dataset.sh
```

Après succès, soumets le contrôle. Il vérifie les chemins d’entraînement, l’alignement du débit au ghost node et les incréments à l’entrée. Il exécute des batches à un pas et en pushforward sur un GPU et vérifie la conservation de la normalisation lors d’une reprise de test E5 vers E6.

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

## Évaluer

Pour chaque expérience, soumets l’évaluation après la sauvegarde du checkpoint final. Le job lit l’époque 900 par défaut et écrit les résultats dans `EXPERIENCE/Seed0/eval-900/`. L’évaluation principale utilise les 16 crues de test et rapporte la MAE horaire sur \(h,u,v\) de 1 à 12 h. Elle rapporte aussi le CSI à 5 et 30 cm sur une grille de 25 m pour E2, E3, E5 et E6, par rapport aux sorties TELEMAC fines. Les CSV fournissent les scores par crue, leur moyenne et leur écart-type par horizon.

```bash
sbatch eval_tet_ghost.sh papier_ghost/E1
```

Les jobs E3 et E6 écrivent aussi `map_figure8.npz` pour le groupe 3 au pic de 3400 m³/s à 6 h et au seuil de 30 cm. `timing_6h.csv` contient trois mesures du rollout sur un GPU après échauffement. Le compteur démarre une fois les entrées chargées et s’arrête avant la production des cartes.

Les jobs E5 et E6 enchaînent avec l’annexe A sur les 16 crues de test complètes avec le même checkpoint. `metrics_long_by_event.csv` et `metrics_long_summary.csv` contiennent la MAE toutes les 30 minutes jusqu’à 40 h. Pour E5, `map_figure10.npz` fournit les masques sur le maillage ×8 à 6, 12, 24 et 36 h pour le groupe 3 au pic de 2800 m³/s et au seuil de 10 cm, avec le CSI nodal indiqué dans la figure.
