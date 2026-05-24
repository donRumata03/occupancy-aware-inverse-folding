# Lambda Occupancy Response Experiment

This module implements a minimal, resumable experiment for testing whether a coefficient used in multi-state inverse folding behaves like a universal occupancy knob.

The scientific hypothesis under test is:

```text
There is no universal function lambda -> pi_i.
```

For each two-state conformer pair `(X0, X1)` and each coefficient `lambda`, the inverse model generates sequences. A forward ensemble generator then samples structures from each sequence, and every sampled structure is assigned to the closer target basin. The main response curve is the mean estimated state-1 occupancy per conformer pair:

```text
mu_hat_b(lambda) = mean_n pi_hat_1(S_{b,lambda,n})
```

## What This Tests

The experiment tests whether response curves from different conformer pairs collapse onto a shared curve when the same coefficient schedule is used. If they do not collapse, coefficient-based multi-state control should be interpreted as generator bias rather than calibrated occupancy control.

## What This Does Not Test

`pi_hat_i` is not a calibrated thermodynamic population. It is a sampling-based proxy estimated under a fixed forward model and basin-assignment protocol.

This first implementation does not claim equilibrium sampling, does not perform a hierarchical bootstrap, and does not solve target alignment edge cases beyond using an external structure-alignment tool.

## Occupancy Estimate

For a generated sequence `S`, BioEmu is used to sample structures `Y_m`. Each `Y_m` is compared to both targets using TM-score:

```text
A_m0 = sim(Y_m, X0)
A_m1 = sim(Y_m, X1)
```

Hard assignment is:

```text
assigned_state = 1 if A_m1 > A_m0 else 0
```

Then:

```text
pi_hat_1(S) = mean_m 1[assigned_state == 1]
```

Raw per-sample scores are saved in `assignment_scores.csv`; the pipeline does not save only final means.

## Why BioEmu

BioEmu is the primary forward evaluator because it can be used as a sequence-conditioned ensemble generator:

```text
sequence S -> sampled structures Y_m
```

The configs support single-sequence A3M input to avoid uncontrolled external MSA retrieval where BioEmu supports that mode.

## Why Not aSAM as Primary Evaluator

aSAM/aSAMt-like samplers condition on an input structure. That is useful for some structure-conditioned questions, but it does not directly estimate sequence-dependent occupancy across target basins under the protocol used here. They are therefore not the primary evaluator in this experiment.

## Conformer Pair CSV

Prepare a CSV with:

```csv
pair_id,x0_pdb,x1_pdb,chain_id,length_hint,notes
pair_001,data/pairs/pair_001/X0.pdb,data/pairs/pair_001/X1.pdb,A,120,metamorphic/small
pair_002,data/pairs/pair_002/X0.pdb,data/pairs/pair_002/X1.pdb,A,150,hinge
pair_003,data/pairs/pair_003/X0.pdb,data/pairs/pair_003/X1.pdb,A,180,small two-state pair
```

A template is included at `experiments/lambda_occupancy/conformer_pairs_template.csv`.

## DynamicMPNN Integration

DynamicMPNN is integrated as a git submodule at:

```text
external/DynamicMPNN
```

Upstream: <https://github.com/Alex-Abrudan/DynamicMPNN>

Clone this repository with submodules on a server:

```bash
git clone --recurse-submodules https://github.com/donRumata03/occupancy-aware-inverse-folding
cd occupancy-aware-inverse-folding
```

If the repository was already cloned:

```bash
git submodule update --init --recursive external/DynamicMPNN
```

The default experiment config uses the bundled DynamicMPNN single-chain two-state checkpoint:

```yaml
inverse:
  dynamicmpnn:
    repo_path: external/DynamicMPNN
    model_ref: external/DynamicMPNN/checkpoints/single_chain_k2.ckpt
    device: auto
    sampling_mode: single
    refold_mode: single
```

If the two selected chains have different lengths, provide explicit 1-indexed residue mappings:

```yaml
inverse:
  dynamicmpnn:
    alignment_state0: "(1, 52)"
    alignment_state1: "(4, 55)"
```

For a Linux server, use DynamicMPNN's upstream conda environment:

```bash
cd external/DynamicMPNN
conda env create -f environment.yml
conda activate dynamicmpnn
pip install -e .
cd ../..
python experiments/lambda_occupancy/scripts/dynamicmpnn_smoke.py --device cuda
```

No DynamicMPNN training data or `.env` file is required for sequence sampling from the bundled checkpoints. The `.env` file is only needed for upstream training/data-regeneration paths and optional AF3 evaluation.

On this Windows workstation, the verified local install used the existing CUDA PyTorch in a repo-local virtual environment:

```powershell
uv venv --python 3.12 --system-site-packages .venvs/dynamicmpnn
.venvs\dynamicmpnn\Scripts\python.exe -m pip install -e external\DynamicMPNN biopython loguru beartype jaxtyping pytorch-lightning lovely-tensors
.venvs\dynamicmpnn\Scripts\python.exe -m pip install graphein==1.7.6 --no-deps
.venvs\dynamicmpnn\Scripts\python.exe -m pip install biopandas bioservices deepdiff looseversion multipledispatch plotly pydantic rich rich-click seaborn wget xarray
.venvs\dynamicmpnn\Scripts\python.exe -m pip install torch_geometric torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
.venvs\dynamicmpnn\Scripts\python.exe -m pip install "huggingface-hub>=0.34.0,<1.0"
.venvs\dynamicmpnn\Scripts\python.exe experiments\lambda_occupancy\scripts\dynamicmpnn_smoke.py --python .venvs\dynamicmpnn\Scripts\python.exe --device cuda
```

The smoke test runs one DynamicMPNN sample for the packaged `1BDT/1QTG` benchmark and writes:

```text
outputs/lambda_occupancy/dynamicmpnn_smoke/samples/samples.csv
```

To run the experiment generation stage from a Python environment that is not the DynamicMPNN environment, point the adapter at the DynamicMPNN Python:

```bash
export DYNAMICMPNN_PYTHON=/path/to/env/bin/python
python experiments/lambda_occupancy/scripts/run_generate.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

PowerShell:

```powershell
$env:DYNAMICMPNN_PYTHON=".venvs\dynamicmpnn\Scripts\python.exe"
python experiments\lambda_occupancy\scripts\run_generate.py --config experiments\lambda_occupancy\configs\smoke_dynamic.yaml
```

Alternatively, activate the DynamicMPNN environment and run the experiment scripts directly from it.

### Lambda Handling

Native upstream DynamicMPNN uses symmetric conformation pooling. It averages the encoded conformer features with a masked mean and does not define a calibrated occupancy-control parameter.

This repository adds an optional inference-time weighted-pooling ablation for DynamicMPNN. When enabled with:

```yaml
inverse:
  dynamicmpnn:
    apply_lambda_to_pooling: true
```

the adapter maps the experiment's lambda to two conformation weights:

```text
alpha = lambda / (1 + lambda)
weights = [1 - alpha, alpha]
```

The DynamicMPNN wrapper then applies those weights during inference-time pooling:

```text
pooled = sum_k mask_k * weight_k * feature_k / sum_k mask_k * weight_k
```

This is not native DynamicMPNN behavior and should be interpreted as an ablation of state-weighted pooled features, not as a trained or calibrated DynamicMPNN occupancy control mechanism. If `apply_lambda_to_pooling: false`, the original symmetric masked mean is preserved and lambda weights are saved only as metadata.

Generated sequence metadata records:

```text
lambda_value
effective_weight0
effective_weight1
lambda_applied_to_dynamic_pooling
pooling_mode
```

Generation uses the same DynamicMPNN sampling seed for all lambdas for a given conformer pair. This keeps the random stream matched across lambdas so differences are driven by weighted pooling rather than lambda-specific seed changes.

If you need a custom weighted sampler, keep using `inverse.command_template`. The command template may use:

```text
{x0_pdb} {x1_pdb} {pair_id} {lambda_value} {weight0} {weight1}
{n_sequences} {seed} {temperature} {output_csv}
```

If the DynamicMPNN implementation expects normalized state weights, this adapter uses:

```text
alpha = lambda / (1 + lambda)
weights = [1 - alpha, alpha]
```

The command must write a CSV with:

```csv
sequence_id,pair_id,lambda_value,inverse_model,seed,sequence,temperature
```

The exact command and effective weights are stored in per-sequence metadata.

Weighted-pooling self-check, from an environment with DynamicMPNN dependencies:

```bash
python experiments/lambda_occupancy/scripts/weighted_pooling_self_check.py
```

## Manual Sequence Fallback

Forward evaluation and plotting can be run before DynamicMPNN is integrated by adding this config field:

```yaml
manual_sequences_csv: path/to/generated_sequences.csv
```

Then run the generation stage; it will copy and validate those sequence records into the experiment output.

## BioEmu Setup

BioEmu is not assumed to be installed in the same environment. The upstream package is Linux-only and supports Python 3.10+. On a CUDA Linux server:

Upstream: <https://github.com/microsoft/bioemu>

```bash
python -m venv .venvs/bioemu
. .venvs/bioemu/bin/activate
pip install --upgrade pip
pip install "bioemu[cuda]" mdtraj
python experiments/lambda_occupancy/scripts/bioemu_smoke.py --num-samples 1
```

On first use BioEmu downloads model assets, including AlphaFold2/ColabFold assets used for embedding generation. Do this once on the server before launching a long run.

The experiment wrapper calls `bioemu.sample`, then converts BioEmu's `samples.xtc` trajectory into per-frame PDB files under `frames/` so the assignment stage can score individual samples. To use a separate BioEmu environment from the main experiment environment:

```bash
export BIOEMU_PYTHON=$PWD/.venvs/bioemu/bin/python
python experiments/lambda_occupancy/scripts/run_bioemu.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

For custom CLIs, use:

```yaml
bioemu:
  command_template: "python -m bioemu.sample --sequence {sequence} --output_dir {output_dir} --num_samples {n_samples} --filter_samples False"
```

Every BioEmu command is saved in `forward_samples.csv` metadata. If BioEmu is missing and `strict: false`, failed sample rows are written explicitly rather than corrupting the run.

## Step-by-Step Commands

DynamicMPNN install/model smoke, from the activated DynamicMPNN environment:

```bash
python experiments/lambda_occupancy/scripts/dynamicmpnn_smoke.py --device cuda
```

BioEmu install/model smoke, from the activated BioEmu environment:

```bash
python experiments/lambda_occupancy/scripts/bioemu_smoke.py --num-samples 1
```

Smoke generation:

```bash
python experiments/lambda_occupancy/scripts/run_generate.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

BioEmu sampling:

```bash
python experiments/lambda_occupancy/scripts/run_bioemu.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

Assignment:

```bash
python experiments/lambda_occupancy/scripts/run_assign.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

Aggregation:

```bash
python experiments/lambda_occupancy/scripts/run_aggregate.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

Plots:

```bash
python experiments/lambda_occupancy/scripts/run_plots.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml
```

All stages:

```bash
python experiments/lambda_occupancy/scripts/run_all.py --config experiments/lambda_occupancy/configs/smoke_dynamic.yaml --stages generate,bioemu,assign,aggregate,plots
```

Use `--overwrite` to recompute an existing stage.

## DynamicMPNN Micro-Run

After DynamicMPNN, `bioemu`, and `assignment.alignment_tool` are configured, and after `conformer_pairs_csv` points at real PDB files:

```bash
python experiments/lambda_occupancy/scripts/run_all.py --config experiments/lambda_occupancy/configs/micro_dynamic.yaml
```

## Optional ProtoMPNN Comparison

`micro_proto.yaml` is included, but this repository currently contains no ProtoMPNN or ProteinMPNN-MSD implementation. The adapter is a placeholder until an implementation or command template is added.

```bash
python experiments/lambda_occupancy/scripts/run_all.py --config experiments/lambda_occupancy/configs/micro_proto.yaml
```

## Compute Budget

Smoke test:

```text
B = 1
lambdas = [0.5, 1.0, 2.0]
N = 3
M = 10
generated sequences = 1 * 3 * 3 = 9
forward samples = 9 * 10 = 90
```

Main DynamicMPNN micro-run:

```text
B = 3
lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
N = 10
M = 30
generated sequences = 3 * 5 * 10 = 150
forward samples = 150 * 30 = 4500
```

DynamicMPNN + ProtoMPNN comparison:

```text
generated sequences = 300
forward samples = 9000
```

Stronger optional run:

```text
B = 5
lambdas = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
N = 15
M = 50
generated sequences per inverse model = 5 * 7 * 15 = 525
forward samples per inverse model = 525 * 50 = 26250
```

Hardware assumptions:

```text
RTX 3060 Ti 8GB: code development, DynamicMPNN generation, smoke tests, small BioEmu runs if they fit
A100 40GB: recommended for the main BioEmu micro-run
A100 80GB: not assumed
```

## Outputs

Outputs are written under:

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

`run_metadata.json` records timestamp, host, command line, Python version, git commit if available, torch version, CUDA availability, and GPU name if available.

## Self Check

The self-check exercises the local math and CSV mechanics without external tools:

```bash
python experiments/lambda_occupancy/scripts/self_check.py
```
