# Cluster Setup Scripts

These scripts are prepared for the UF Focus cluster workflow, but they are not yet executed remotely from Codex because the current SSH path is password-only. The non-interactive runner cannot issue remote `ssh` or `scp` commands without key-based access.

## Intended Remote Layout

`~/research/cmkc`

- `project/`: this repository
- `upstream/`: cloned benchmark repos
- `data/`: datasets and features
- `envs/`: Python or Conda environments
- `logs/`: job logs

## Recommended Order

1. Transfer this repository to the cluster.
2. Run `bootstrap_focus.sh` on the cluster.
3. Run `download_vqacl_assets.sh` on the cluster.
4. Launch `submit_quad_vqav2.slurm`.
5. Launch `submit_quad_nextqa.slurm`.
6. If the VQACL stage is stable, move on to `run_coin_moe_stage.sh`.

## Access Blocker

As of 2026-03-19, direct remote automation from this Codex session is blocked by SSH authentication mode:

- interactive `ssh` works in the app terminal with a password
- non-interactive `ssh` from the tool runner fails with `Permission denied (publickey,password)`

Once key-based auth is enabled, the transfer and remote execution can be automated with `sync_to_focus.ps1`.
