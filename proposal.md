# Cross-Modal Knowledge Consolidation for Continual Vision-Language Learning

## 1. Motivation

Continual learning for vision-language systems is harder than continual learning for image-only or text-only models because the model must preserve three kinds of knowledge at once:

1. visual representations
2. language representations
3. the alignment between the two

This becomes especially visible in sequential visual question answering and multimodal instruction tuning, where each new task can shift objects, question styles, answer spaces, and reasoning patterns simultaneously. A model may retain some visual features yet still fail because the mapping from image evidence to language output has drifted. That failure mode is poorly addressed by standard continual learning methods built for a single modality.

## 2. Evidence of the Gap

Recent literature already points to the central problem.

- Zhang et al. (CVPR 2023) introduced VQACL and showed that established continual learning methods suffer from catastrophic forgetting and weak generalizability in a multimodal VQA setting.
- Qian et al. (ICCV 2023) argued that formulating continual VQA from only a vision or language perspective is incomplete, and proposed a multimodal prompt-based method to better capture modality interactions.
- Marouf et al. (ICCV 2025) further showed that continual VQA requires stability across both visual and textual domains, and proposed attention-based distillation to preserve essential visual-linguistic associations.
- Ni et al. (CVPR 2024) demonstrated that language-guided semantic targets reduce representation drift in visual continual learning, suggesting that language can act as a stabilizing anchor for evolving visual tasks.
- Pian et al. (arXiv 2024) showed that multimodal continual learning in MLLMs is affected not only by task shifts but also by inconsistent modalities, which intensifies forgetting.
- Zhang et al. (EMNLP 2025) identified an additional issue in modality-incremental learning for MLLMs: degradation comes from both forgetting and misalignment between modality-specific and modality-agnostic components.
- Jin et al. (ICCV 2025) showed that updating the visual side of generative VLMs can cause the model to underuse language instructions, reinforcing the need for explicit cross-modal balancing.

Taken together, these papers motivate a clear gap: existing methods partially preserve one modality, or preserve each modality independently, but do not directly optimize for mutual preservation of both modalities and their alignment.

## 3. Problem Statement

Given a sequence of vision-language tasks

`T1, T2, ..., Tn`

train a multimodal model so that it can learn each new task without significant degradation on earlier tasks. The goal is not only to reduce standard catastrophic forgetting, but to specifically minimize:

- visual forgetting: loss of previously learned image features or region-level grounding
- language forgetting: loss of previously learned question understanding, instructions, or answer generation behavior
- alignment forgetting: loss of correspondence between visual evidence and linguistic semantics

## 4. Proposed Direction

I propose a framework called **Cross-Modal Knowledge Consolidation (CMKC)**.

The central idea is simple: each modality should help preserve the other. Instead of regularizing only model weights or replaying old samples, the framework maintains structured cross-modal anchors that preserve both unimodal knowledge and multimodal alignment.

### 4.1 Core Components

#### A. Language-Anchored Visual Stabilization

For each task, the model extracts text-side semantic anchors such as:

- question-type embeddings
- answer-space prototypes
- instruction embeddings
- concept-level prompt summaries

These anchors are used to regularize the visual encoder or visual projector so that visual features for semantically related concepts do not drift excessively across tasks.

Rationale: Ni et al. show that language-derived semantic supervision helps stabilize visual continual learning. This project extends that idea from image classification to multimodal reasoning.

#### B. Visual-Grounded Language Stabilization

The framework also preserves visual prototypes linked to recurring concepts, objects, or scene relations. When learning a new task, these prototypes regularize the language side by constraining the model to generate or attend to text representations that remain grounded in compatible visual evidence.

Rationale: recent continual multimodal work shows that updating one side of the model can distort the other. Using visual grounding as a constraint should reduce language drift in instruction following and answer generation.

#### C. Cross-Modal Alignment Distillation

The method distills alignment information from a teacher snapshot of the model before each task transition. Distilled signals may include:

- image-text similarity structure
- cross-attention maps
- token-to-region relevance
- projector outputs before the language decoder

This step targets alignment forgetting directly rather than assuming it will be preserved indirectly.

#### D. Prototype and Question Replay

When raw replay is expensive or restricted, the system uses a lightweight replay buffer composed of:

- question-only replay for prior reasoning patterns
- text semantic prototypes
- visual centroids or compressed region features
- task-level prompt summaries

This borrows the efficiency intuition of question-only replay while adding explicit cross-modal memory.

#### E. Parameter-Efficient Adaptation

To keep the approach practical, the framework should update only a small subset of parameters:

- multimodal prompts
- LoRA adapters
- visual projectors
- small consolidation heads

This keeps training feasible on academic compute and limits destructive updates to the backbone.

## 5. Method Outline

Let the model contain a visual encoder `Ev`, a text encoder or LLM `El`, and a fusion module `F`. For task `t`, optimize:

`L_total = L_task + lambda1 * L_vis_anchor + lambda2 * L_lang_anchor + lambda3 * L_align + lambda4 * L_replay`

where:

- `L_task` is the current task loss
- `L_vis_anchor` preserves visual features relative to language anchors
- `L_lang_anchor` preserves language features relative to visual anchors
- `L_align` preserves cross-modal correspondence across tasks
- `L_replay` uses question replay and prototype replay for old tasks

The novelty is not any one loss alone. The novelty is the bidirectional consolidation structure, where both modalities act as memory supports for each other.

## 6. Experimental Plan

### 6.1 Base Models

The framework can be instantiated on top of established vision-language backbones such as:

- CLIP-style encoders with a fusion head
- BLIP-2 style architectures
- LLaVA-style multimodal LLMs

For a first implementation, BLIP-2 or a compact LLaVA-style model is a practical choice because both expose clear visual and language interfaces for distillation and adapter-based updates.

### 6.2 Benchmarks

A strong evaluation should cover both classical VQA continual learning and broader multimodal instruction learning.

Recommended benchmarks:

- VQACL task splits from Zhang et al. (2023)
- CL-VQA benchmarks from Qian et al. (2023)
- NExT-QA continual splits if video question answering is included
- a domain-shift extension such as medical VQA or science VQA if specialized robustness is desired

### 6.3 Baselines

Compare against:

- naive sequential fine-tuning
- regularization baselines such as EWC or LwF
- replay baselines
- VQACL (Zhang et al., 2023)
- TRIPLET (Qian et al., 2023)
- QUAD (Marouf et al., 2025)
- MoInCL (Pian et al., 2024)
- MERA (Zhang et al., 2025)
- instruction-grounded projector tuning (Jin et al., 2025)

### 6.4 Metrics

Use standard continual learning metrics:

- average incremental accuracy
- backward transfer
- forgetting score
- final average accuracy

Add multimodal-specific diagnostics:

- visual retention score: how much visual grounding quality degrades across tasks
- language retention score: how much question or instruction understanding degrades across tasks
- alignment retention score: how much cross-attention or image-text retrieval consistency degrades across tasks

These modality-specific retention metrics are important because average task accuracy alone can hide whether one modality has collapsed while the other compensates.

## 7. Expected Contributions

This project is expected to contribute:

1. a unified formulation of dual-modality and alignment forgetting in continual vision-language learning
2. a bidirectional consolidation method where language anchors vision and vision anchors language
3. a lightweight replay strategy based on questions and semantic prototypes rather than full raw data
4. a multimodal evaluation protocol with separate visual, language, and alignment retention measurements

## 8. Feasibility

The project is feasible for three reasons.

First, the problem is already measurable through existing continual VQA and multimodal continual learning benchmarks. Second, the building blocks already exist in the literature: prompt-based continual VQA, question-only replay, language-guided stabilization, and instruction-aware projector tuning. Third, the method can be implemented with parameter-efficient updates, which keeps compute demands manageable.

A realistic minimum viable implementation would:

1. start from a compact pretrained vision-language model
2. add LoRA or prompt adapters
3. maintain text and image prototype memories
4. distill cross-modal attention at each task transition
5. evaluate on a VQACL-style task sequence

## 9. Risks and Mitigations

**Risk:** Cross-modal losses may overconstrain learning and reduce plasticity.

**Mitigation:** Use adaptive weighting based on validation forgetting or task novelty.

**Risk:** Prototype memories may preserve high-level semantics but miss fine-grained grounding.

**Mitigation:** Store both global and region-level compressed features.

**Risk:** Multimodal benchmarks may not isolate which modality is forgetting.

**Mitigation:** Report separate modality retention metrics and ablations that freeze one side at a time.

## 10. Conclusion

The main limitation in current continual vision-language learning is not simply forgetting within vision or language alone. It is the failure to preserve their interaction. A strong next step is therefore a framework that consolidates knowledge across modalities in both directions. Cross-Modal Knowledge Consolidation directly targets that gap by treating language as a stabilizer for vision, vision as a stabilizer for language, and alignment as a first-class object to preserve throughout sequential learning.
