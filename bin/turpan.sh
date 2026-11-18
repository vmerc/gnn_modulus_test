#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
    printf '%s\n' \
        'bash bin/turpan.sh submit CHEMIN_JOB.sh [ARGUMENTS...]' \
        'bash bin/turpan.sh logs ID_EXECUTION' \
        'Configuration locale : git config --local turpan.runRoot /work/PROJET/UTILISATEUR/gnn-runs' \
        'Alias SSH : git config --local turpan.host turpan-vscode'
}

fail() {
    printf '%s\n' "$1" >&2
    exit 2
}

remote_bash() {
    local host=$1
    shift
    local command='bash -c' argument
    for argument in "$@"; do
        # OpenSSH transmet une commande au shell distant, qui doit recevoir des arguments échappés.
        argument=${argument//\'/\'\\\'\'}
        command+=" '$argument'"
    done
    ssh -T -- "$host" "$command"
}

submit() (
    [[ $# -ge 1 ]] || fail 'Indique le chemin du job Slurm à soumettre.'
    job_path=${1#./}
    shift
    case "$job_path" in
        *.sh|*.sbatch) ;;
        *) fail 'Indique un script .sh ou .sbatch relatif à la racine Git.' ;;
    esac

    host=$(git config --local --get turpan.host || printf 'turpan-vscode')
    remote_root=$(git config --local --get turpan.runRoot || true)
    [[ $remote_root == /* ]] || fail 'Configure turpan.runRoot avec un chemin absolu partagé sur Turpan.'
    commit=$(git rev-parse --verify HEAD)
    if [[ $(git cat-file -t "$commit:$job_path" 2>/dev/null || true) != blob ]]; then
        fail "Le commit $commit ne contient pas le script $job_path."
    fi
    if [[ -n $(git status --porcelain) ]]; then
        printf 'Snapshot testé %s. Commite tes changements pour les inclure dans une exécution.\n' "$commit" >&2
    fi

    mkdir -p .turpan-runs
    local_run=$(mktemp -d ".turpan-runs/${commit:0:12}.XXXXXX")
    run_id=${local_run##*/}
    printf '%s\n' "$host" "$remote_root" > "$local_run/remote.txt"
    {
        printf 'Exécution %s\nCommit %s\nScript %s\n' "$run_id" "$commit" "$job_path"
        if [[ $# -gt 0 ]]; then
            printf 'Arguments'
            printf ' %q' "$@"
            printf '\n'
        fi
    } > "$local_run/commande.txt"

    archive=$(mktemp "$local_run/source.XXXXXX")
    trap 'rm -f -- "$archive"' EXIT
    git archive --format=tar --output="$archive" "$commit"

    printf 'Exécution %s\nCommit %s\n' "$run_id" "$commit"
    printf 'Récupération : bash bin/turpan.sh logs %s\n' "$run_id"
    remote_command=$(cat <<'REMOTE'
set -euo pipefail
umask 077
run_dir=$1
commit=$2
job_path=$3
shift 3

mkdir -p -- "$(dirname -- "$run_dir")"
mkdir -- "$run_dir"
repo_dir="$run_dir/code"
mkdir -- "$repo_dir"
tar -xf - -C "$repo_dir"

job="$repo_dir/$job_path"
if [[ ! -f "$job" || -L "$job" ]]; then
    printf 'Script absent du snapshot Git ou lien symbolique : %s\n' "$job_path" >&2
    exit 2
fi

cd -- "$(dirname -- "$job")"
export TURPAN_RUN_DIR="$run_dir" TURPAN_REPO_DIR="$repo_dir" TURPAN_COMMIT="$commit"
sbatch --parsable --open-mode=append \
    --output="$run_dir/slurm-%j.out" --error="$run_dir/slurm-%j.err" \
    "$job" "$@" > "$run_dir/job-id"
printf 'Job Slurm '
cat -- "$run_dir/job-id"
printf 'Dossier Turpan %s\n' "$run_dir"
REMOTE
)
    remote_bash "$host" "$remote_command" turpan "$remote_root/$run_id" \
        "$commit" "$job_path" "$@" < "$archive" 2>&1 | tee "$local_run/soumission.log"
)

logs() (
    [[ $# -eq 1 ]] || fail 'Indique l’identifiant imprimé par submit.'
    run_id=$1
    [[ $run_id =~ ^[[:xdigit:]]{12}\.[[:alnum:]]{6}$ ]] || fail 'Identifiant d’exécution invalide.'
    local_run=".turpan-runs/$run_id"
    [[ -r "$local_run/remote.txt" ]] || fail "Exécution inconnue sur ce Mac : $run_id."
    {
        IFS= read -r host
        IFS= read -r remote_root
    } < "$local_run/remote.txt"

    output=$(mktemp "$local_run/logs.XXXXXX")
    trap 'rm -f -- "$output"' EXIT
    cat -- "$local_run/commande.txt" > "$output"
    remote_command=$(cat <<'REMOTE'
set -euo pipefail
run_dir=$1
[[ -d "$run_dir" ]] || { printf 'Dossier Turpan absent : %s\n' "$run_dir" >&2; exit 2; }

if [[ -s "$run_dir/job-id" ]]; then
    IFS= read -r job_id < "$run_dir/job-id"
    job_id=${job_id%%;*}
    printf '\nJob Slurm %s\n' "$job_id"
    printf 'File Slurm — job, état, durée, raison\n'
    squeue --noheader --jobs="$job_id" --format='%i|%T|%M|%R' 2>/dev/null || true
    printf 'Comptabilité Slurm — job, état, code de sortie, durée\n'
    sacct --noheader --allocations --parsable2 --jobs="$job_id" \
        --format=JobID,State,ExitCode,Elapsed 2>/dev/null || true
else
    printf '\nSoumission sans identifiant Slurm enregistré. Consulte aussi soumission.log.\n'
fi

for file in "$run_dir"/slurm-*.out "$run_dir"/slurm-*.err; do
    name=${file##*/}
    if [[ $name =~ ^slurm-[0-9]+\.(out|err)$ && -f "$file" && ! -L "$file" ]]; then
        printf '\n--- %s ---\n' "$name"
        cat -- "$file"
    fi
done
REMOTE
)
    remote_bash "$host" "$remote_command" turpan "$remote_root/$run_id" </dev/null >> "$output"
    mv -f -- "$output" "$local_run/job.log"
    printf 'Log local %s/job.log\n' "$local_run"
)

case "${1:-}" in
    -h|--help|'') usage; exit 0 ;;
    submit|logs) subcommand=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"
"$subcommand" "$@"
