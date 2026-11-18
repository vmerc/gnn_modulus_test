# Tests et entraînements sur Turpan

Chaque calcul part d'un commit Git et d'un script Slurm choisi à la soumission. Une fois le job soumis, Slurm gère son exécution et le Mac peut se déconnecter.

## Configuration sur le Mac

Depuis le dépôt, configure ton alias SSH et un dossier absolu Turpan accessible aux nœuds de calcul. Remplace les deux premiers composants de l'exemple par les tiens.

```bash
git config --local turpan.host turpan-vscode
git config --local turpan.runRoot /work/PROJET/UTILISATEUR/gnn-runs
```

## Soumission et logs

Commite le code et le job à tester. Pour la préparation du dataset et les commandes de la campagne Têt, suis [le guide des expériences E1–E6](tet-ghost.md).

```bash
bash bin/turpan.sh submit CHEMIN_DU_JOB [ARGUMENTS...]
```

`submit` affiche un identifiant d'exécution local et le numéro du job Slurm. Pour la récupération, utilise l'identifiant d'exécution imprimé par le lanceur.

```bash
bash bin/turpan.sh logs ID_EXECUTION
```

Dans `.turpan-runs/ID_EXECUTION/job.log`, tu retrouves le SHA, le script, ses arguments, l'état Slurm disponible, stdout et stderr. Tu peux actualiser ce fichier avec la même commande. Le code de retour de `logs` indique le résultat de la récupération. L'état de l'entraînement et son code de sortie sont ceux de Slurm. Avant la disponibilité de la comptabilité ou des fichiers de sortie, relève le job dans la file et réessaie plus tard. En cas de coupure pendant `submit`, utilise l'identifiant affiché pour chercher le job avant de soumettre à nouveau.

## Préparer un autre job

- Indique le chemin du script versionné comme premier argument de `submit`. Les arguments suivants sont transmis à ce script. Le lanceur installe les fichiers du commit via `git archive` dans un nouveau dossier sur Turpan, puis soumet le job depuis le dossier contenant le script. Les attributs Git `export-ignore` s'appliquent, notamment à `archive/`.
- Pour un test, reprends le point d'entrée du code et limite le calcul à l'assertion à vérifier. Pour un entraînement, utilise un job et un YAML dédiés à l'expérience. Les directives `#SBATCH` du script règlent ses ressources. Le lanceur fixe les chemins stdout et stderr du run.
- Avec l'export d'environnement Slurm par défaut, les jobs reçoivent `TURPAN_RUN_DIR`, `TURPAN_REPO_DIR` et `TURPAN_COMMIT`. Utilise `TURPAN_REPO_DIR` pour retrouver le code depuis le job. Place les sorties de calcul dans `TURPAN_RUN_DIR` ou dans un dossier dédié explicitement choisi.
- L’entraîneur ghost compose sa configuration Hydra à partir du nom passé en premier argument. La campagne écrit ses poids dans `checkpoints/` sous `TURPAN_RUN_DIR`. Pour une reprise, le lanceur reçoit séparément le dossier du checkpoint initial.
- Fais remonter sur stdout ou stderr les métriques et erreurs utiles au diagnostic. `logs` lit les sorties Slurm du run. Les checkpoints, datasets et fichiers applicatifs restent dans leurs dossiers Turpan.
