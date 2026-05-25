# Occupancy-Aware Inverse Folding

This repository contains experiments for testing whether a lambda coefficient in
multi-state inverse folding behaves like a universal occupancy control knob.

The working hypothesis is:

```text
There is no universal function lambda -> pi_i.
```

For each two-state conformer pair `(X0, X1)` and each lambda value, the inverse
model generates sequences. BioEmu then samples structures from each generated
sequence, and each sampled structure is assigned to the closer target basin. The
main response curve is:

```text
mu_hat_b(lambda) = mean_n pi_hat_1(S_{b,lambda,n})
```

The full experiment notes live in
`experiments/lambda_occupancy/README.md`. This top-level README keeps the
operational path visible: deployment, SSH access, persistent server layout,
virtual environments, GPU monitoring, and the commands most often needed on a
RunPod server.

## Repository Layout

```text
experiments/lambda_occupancy/
  configs/                 experiment configs
  scripts/                 stage runners, smoke tests, GPU monitor
  src/                     pipeline implementation
  conformer_pairs_*.csv    benchmark pair definitions
external/DynamicMPNN/      DynamicMPNN git submodule
outputs/lambda_occupancy/  generated local/server outputs, ignored by git
```

DynamicMPNN is integrated as a git submodule:

```bash
git clone --recurse-submodules https://github.com/donRumata03/occupancy-aware-inverse-folding
cd occupancy-aware-inverse-folding
git submodule update --init --recursive external/DynamicMPNN
```

## RunPod Deployment

Use GitHub as the source of truth for code movement to the server. Commit and
push locally, then pull on the pod:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/donRumata03/occupancy-aware-inverse-folding
cd /workspace/occupancy-aware-inverse-folding
git pull --ff-only origin master
git submodule update --init --recursive external/DynamicMPNN
```

If the checkout already exists:

```bash
cd /workspace/occupancy-aware-inverse-folding
git fetch origin master
git checkout master
git pull --ff-only origin master
git submodule update --init --recursive external/DynamicMPNN
```

Keep the repository, virtual environments, downloaded model assets, and outputs
under `/workspace`. RunPod's docs describe container/system storage as temporary
and the pod volume directory, usually `/workspace`, as the persistent location
for the pod lease. Files under `/workspace` survive pod stop/restart, but are
deleted if the pod itself is terminated. Network volumes are the option for data
that must survive pod deletion or move between pods.

Useful references:

- RunPod storage types: https://docs.runpod.io/pods/storage/types
- RunPod pod overview: https://docs.runpod.io/pods
- RunPod SSH guide: https://docs.runpod.io/pods/configuration/use-ssh

## SSH Access

Prefer SSH over exposed direct TCP. It supports `scp` and SFTP. The proxied
`ssh.runpod.io` SSH form shown in the RunPod UI does not support SCP/SFTP in the
same way.

When the RunPod UI shows a block like:

```text
SSH over exposed TCP
ssh root@87.120.211.205 -p 13327 -i ~/.ssh/id_ed25519
Direct TCP ports
87.120.211.205:13327 -> :22
```

extract only the host and port from the command:

```text
host = 87.120.211.205
port = 13327
```

Do not copy the UI's dummy key path blindly. On this Windows workstation, use:

```powershell
$key = "$env:USERPROFILE\.ssh\runpod_codex_ed25519"
ssh root@87.120.211.205 -p 13327 -i $key
```

File transfer uses the same direct TCP host and port:

```powershell
$key = "$env:USERPROFILE\.ssh\runpod_codex_ed25519"
scp -P 13327 -i $key .\local-file root@87.120.211.205:/workspace/
```

If SSH asks for a password, the public key is probably missing from
`/root/.ssh/authorized_keys` inside the pod.

## Python Environments

Use repo-local virtual environments under `/workspace/occupancy-aware-inverse-folding/.venvs`.
Avoid relying on system Python packages or globally installed dependencies
because they live outside the persistent workspace and may disappear when the
pod is rebuilt, edited, or restarted.

The current server layout uses separate environments:

```text
.venvs/dynamicmpnn/  DynamicMPNN, PyTorch, torch_geometric, graph dependencies
.venvs/bioemu/       BioEmu, mdtraj, matplotlib, plotting/runtime dependencies
```

Use environment variables so the stage runner can call the correct interpreter:

```bash
cd /workspace/occupancy-aware-inverse-folding
export DYNAMICMPNN_PYTHON="$PWD/.venvs/dynamicmpnn/bin/python3"
export BIOEMU_PYTHON="$PWD/.venvs/bioemu/bin/python3"
```

Run the main experiment driver with an environment that has plotting
dependencies. On the current pod, `.venvs/bioemu` has `matplotlib`; global
`/usr/bin/python` did not. If `matplotlib` is missing, install it into a
workspace venv, not system Python:

```bash
.venvs/bioemu/bin/python3 -m pip install matplotlib
```

Smoke checks:

```bash
.venvs/dynamicmpnn/bin/python3 experiments/lambda_occupancy/scripts/dynamicmpnn_smoke.py --device cuda
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/bioemu_smoke.py --num-samples 1
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/self_check.py
```

## Running Experiments

All outputs are written under:

```text
outputs/lambda_occupancy/{experiment_name}/
```

Main files:

```text
generated_sequences.csv
forward_samples.csv
assignment_scores.csv
sequence_occupancies.csv
response_curves.csv
run_metadata.json
config.resolved.yaml
pair_metadata.csv
figures/response_curves.png
figures/per_sequence_distributions.png
figures/delta_score_distributions.png
```

One-pair corner-lambda run:

```bash
cd /workspace/occupancy-aware-inverse-folding
export DYNAMICMPNN_PYTHON="$PWD/.venvs/dynamicmpnn/bin/python3"
export BIOEMU_PYTHON="$PWD/.venvs/bioemu/bin/python3"
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/run_all.py \
  --config experiments/lambda_occupancy/configs/one_pair_corner_dynamic.yaml
```

Smoke run:

```bash
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/run_all.py \
  --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

Two-pair small run:

```bash
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/run_all.py \
  --config experiments/lambda_occupancy/configs/two_pair_dynamic.yaml
```

Main micro-run:

```bash
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/run_all.py \
  --config experiments/lambda_occupancy/configs/micro_dynamic.yaml
```

Use `--overwrite` to recompute existing stage outputs.

## Background Run With Logs

For server runs, launch the experiment with a timestamped log:

```bash
cd /workspace/occupancy-aware-inverse-folding
OUT=outputs/lambda_occupancy/one_pair_corner_dynamic
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG=$LOGDIR/run_all_${TS}.log

export DYNAMICMPNN_PYTHON="$PWD/.venvs/dynamicmpnn/bin/python3"
export BIOEMU_PYTHON="$PWD/.venvs/bioemu/bin/python3"

nohup .venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/run_all.py \
  --config experiments/lambda_occupancy/configs/one_pair_corner_dynamic.yaml \
  > "$RUN_LOG" 2>&1 &

RUN_PID=$!
echo "$RUN_PID" > "$LOGDIR/run_all.pid"
printf '%s\n' "$RUN_LOG" > "$LOGDIR/latest.log.path"
```

Check progress:

```bash
cd /workspace/occupancy-aware-inverse-folding
ps -p "$(cat outputs/lambda_occupancy/one_pair_corner_dynamic/logs/run_all.pid)" -o pid,etime,stat,cmd
tail -120 "$(cat outputs/lambda_occupancy/one_pair_corner_dynamic/logs/latest.log.path)"
```

## GPU Monitoring

The GPU monitor samples `nvidia-smi`, writes raw CSV, computes averages and
p50/p90/p95/p99 percentiles, and writes a PNG plot.

For detailed monitoring, sample every second:

```bash
cd /workspace/occupancy-aware-inverse-folding
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/monitor_gpu.py \
  --interval 1 \
  --output-dir outputs/lambda_occupancy/one_pair_corner_dynamic/gpu_monitor
```

To monitor an experiment process and stop automatically when it exits:

```bash
cd /workspace/occupancy-aware-inverse-folding
RUN_PID=$(cat outputs/lambda_occupancy/one_pair_corner_dynamic/logs/run_all.pid)
.venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/monitor_gpu.py \
  --interval 1 \
  --pid "$RUN_PID" \
  --output-dir outputs/lambda_occupancy/one_pair_corner_dynamic/gpu_monitor
```

Launch monitor and experiment together:

```bash
cd /workspace/occupancy-aware-inverse-folding
OUT=outputs/lambda_occupancy/one_pair_corner_dynamic
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR" "$OUT/gpu_monitor"
TS=$(date -u +%Y%m%dT%H%M%SZ)

export DYNAMICMPNN_PYTHON="$PWD/.venvs/dynamicmpnn/bin/python3"
export BIOEMU_PYTHON="$PWD/.venvs/bioemu/bin/python3"

nohup .venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/run_all.py \
  --config experiments/lambda_occupancy/configs/one_pair_corner_dynamic.yaml \
  > "$LOGDIR/run_all_${TS}.log" 2>&1 &
RUN_PID=$!
echo "$RUN_PID" > "$LOGDIR/run_all.pid"
printf '%s\n' "$LOGDIR/run_all_${TS}.log" > "$LOGDIR/latest.log.path"

nohup .venvs/bioemu/bin/python3 experiments/lambda_occupancy/scripts/monitor_gpu.py \
  --interval 1 \
  --pid "$RUN_PID" \
  --output-dir "$OUT/gpu_monitor" \
  > "$LOGDIR/gpu_monitor_${TS}.log" 2>&1 &
MON_PID=$!
echo "$MON_PID" > "$LOGDIR/gpu_monitor.pid"
printf '%s\n' "$LOGDIR/gpu_monitor_${TS}.log" > "$LOGDIR/latest.monitor.log.path"
```

Monitor outputs:

```text
gpu_monitor/gpu_monitor.csv
gpu_monitor/gpu_monitor_summary.json
gpu_monitor/gpu_monitor.png
```

A 1-second interval is preferred for BioEmu because GPU compute can be bursty.
Longer intervals can under-report utilization while still capturing memory
pressure.

## Lambda Handling

DynamicMPNN's native public sampler uses symmetric conformation pooling and does
not expose a calibrated occupancy-control parameter. This repository adds an
optional inference-time weighted-pooling ablation:

```yaml
inverse:
  dynamicmpnn:
    apply_lambda_to_pooling: true
```

Lambda is mapped to two state weights:

```text
alpha = lambda / (1 + lambda)
weights = [1 - alpha, alpha]
```

This should be interpreted as a state-weighted pooled-feature ablation, not as a
trained thermodynamic occupancy controller.

## Compute Budgets

```text
one_pair_corner_dynamic:
  B = 1
  lambdas = [0.25, 1.0, 4.0]
  N = 3
  M = 10
  generated sequences = 9
  forward samples = 90

smoke_dynamic:
  B = 1
  lambdas = [0.5, 1.0, 2.0]
  N = 3
  M = 10
  generated sequences = 9
  forward samples = 90

two_pair_dynamic:
  B = 2
  lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
  N = 3
  M = 10
  generated sequences = 30
  forward samples = 300

micro_dynamic:
  B = 3
  lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
  N = 10
  M = 30
  generated sequences = 150
  forward samples = 4500
```

## Pulling Results Back To Windows

```powershell
$key = "$env:USERPROFILE\.ssh\runpod_codex_ed25519"
$remote = "root@87.120.211.205:/workspace/occupancy-aware-inverse-folding/outputs/lambda_occupancy/one_pair_corner_dynamic"
$local = "outputs\lambda_occupancy\one_pair_corner_dynamic"

scp -P 13327 -i $key "$remote/figures/response_curves.png" "$local\figures\response_curves.png"
scp -P 13327 -i $key "$remote/gpu_monitor/gpu_monitor.png" "$local\gpu_monitor\gpu_monitor.png"
scp -P 13327 -i $key "$remote/response_curves.csv" "$local\response_curves.csv"
scp -P 13327 -i $key "$remote/gpu_monitor/gpu_monitor_summary.json" "$local\gpu_monitor\gpu_monitor_summary.json"
```
