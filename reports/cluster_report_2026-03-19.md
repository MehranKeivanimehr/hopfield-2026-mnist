# Cluster Report: Benchmark Choice and Execution Plan

Date: 2026-03-19

## Executive Summary

I selected a two-stage benchmark strategy designed to maximize publishability while staying executable on a node with 8 x A6000 GPUs.

**Primary benchmark family**

- VQACL on VQAv2
- VQACL on NExT-QA

**Secondary extension benchmark**

- CoIN with LLaVA-1.5-7B plus LoRA or MoELoRA

This is the best balance between novelty, feasibility, and conference-level credibility. VQACL directly targets continual multimodal reasoning under dual-level task shifts, while CoIN extends the story to continual instruction tuning for multimodal LLMs.

## Why These Benchmarks

### 1. VQACL is the right primary benchmark

The VQACL setting was introduced specifically for continual visual question answering and explicitly models multimodal task evolution through both question-type and object-level task structure. That makes it more aligned with your project than generic image continual learning or text-only continual learning benchmarks.

Why it is the right first target:

- it is directly about continual vision-language learning
- it already has accepted evaluation protocols on VQAv2 and NExT-QA
- it is the benchmark used by later continual VQA methods such as QUAD
- it allows a strong paper narrative around dual-modality and compositional forgetting

### 2. QUAD is the strongest practical reproduction baseline

QUAD is the most useful starting point for real experiments because:

- it is recent and stronger than the original VQACL baseline family
- it already supports both VQAv2 and NExT-QA
- it includes question-only replay and attention distillation, which are directly relevant to our proposed CMKC extension
- it gives us a serious baseline to beat instead of building from a toy scaffold

### 3. CoIN is the best extension for top-tier scope

If the paper stops at VQACL, it can still be solid. If we also show transfer to CoIN, the paper becomes much stronger because it demonstrates that the method is not only useful for task-specific VQA, but also for multimodal LLM continual instruction tuning.

CoIN is attractive because:

- it is a dedicated benchmark for continual instruction tuning in MLLMs
- it spans 10 datasets across 8 task categories
- its official code already includes LLaVA-based LoRA and MoELoRA training scripts
- 8 x A6000 is enough for a serious 7B LoRA-based continual tuning study

## Recommended Paper Positioning

The publishable claim should be:

`A cross-modal consolidation method that reduces forgetting in both continual VQA and continual multimodal instruction tuning.`

That is stronger than:

- another replay method only for VQA
- another regularization loss only for MLLMs

The final paper should ideally contain:

1. Main results on VQACL-VQAv2
2. Main results on VQACL-NExT-QA
3. Ablations for visual anchor, language anchor, and alignment distillation
4. Novel composition results
5. Optional transfer results on CoIN with LLaVA-1.5-7B

## What I Did

### Local preparation

- created the project codebase scaffold for CMKC
- cloned the official upstream repositories locally for inspection:
  - VQACL
  - QUAD
  - CoIN
- created cluster bootstrap scripts in `scripts/cluster/`
- created SLURM launch templates for the first-stage QUAD reproductions
- created a sync helper for copying this repository to the cluster once SSH key access is enabled

### Access verification

I verified that:

- an interactive SSH login to `ece-focus-xg01.ad.ufl.edu` exists in the app terminal
- the non-interactive tool runner cannot issue remote commands because SSH currently requires password authentication and does not accept available public-key auth from this session

As a result, I could not honestly claim that I created directories or downloaded files on the cluster itself from the automation layer. I prepared the exact scripts for that step, but the final remote execution remains blocked by authentication mode.

## Immediate Run Plan

### Stage 1: Reproduce QUAD on VQACL

Goal:

- establish a strong accepted-method baseline
- verify dataset layout and job stability on the cluster
- get credible average accuracy and forgetting numbers before modifying the model

Run order:

1. bootstrap environment on cluster
2. download VQACL partitions and COCO/NExT-QA assets
3. run QUAD on VQAv2
4. run QUAD on NExT-QA
5. store logs, checkpoints, and summary tables

### Stage 2: Implement CMKC inside QUAD/VQACL code

After reproduction is stable, modify the QUAD code path to add:

- language-anchored visual consolidation
- visual-grounded language consolidation
- bidirectional prototype memory
- explicit multimodal alignment retention metrics

### Stage 3: Extend to CoIN

If Stage 2 produces gains on VQACL, port the method to the CoIN LLaVA path using LoRA-based continual tuning.

That extension is ambitious but realistic on 8 x A6000 if we stay in the 7B class and avoid full fine-tuning.

## Risk Assessment

### Low risk

- reproducing QUAD or VQACL on their official benchmarks
- running VQAv2 and NExT-QA continual experiments on A6000 hardware

### Medium risk

- dataset acquisition friction from Google Drive-hosted partitions
- adapting older codebases to a newer cluster software stack

### Higher risk

- full CoIN data assembly, because it spans many external datasets
- making the first paper depend entirely on MLLM-scale experiments

This is why VQACL should be the primary path and CoIN should be the extension path, not the reverse.

## Required Next Access Fix

To let me execute the cluster steps directly, enable one of these:

- SSH public-key login for `m.keivanimehr@ece-focus-xg01.ad.ufl.edu`
- a persistent mounted cluster workspace visible from this machine
- a Git remote for this project that the cluster can pull from directly

Without one of those, I can continue preparing code and scripts locally, but I cannot truthfully perform remote file transfer or remote job submission myself.
