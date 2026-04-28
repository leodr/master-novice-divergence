# #11 Increasing Reward Quality & Density for Verifiable Rewards

**TUM Seminar — Chair of Software Engineering for Business Information Systems (sebis)**
**Department of Computer Science, School of Computation, Information and Technology (CIT)**
**Technical University of Munich**

> _"Propose a novel reinforcement learning framework that dynamically identifies challenging and meaningful actions, to increase reward density & quality for verifiable rewards."_

---

## Overview

Reinforcement Learning (RL) for LLMs often suffers from **reward sparsity**, particularly in reasoning tasks where a binary "correct/incorrect" signal is only available at the very end of a long chain of thought.

This project proposes a **Master/Novice framework** to densify rewards without expensive human annotation. By comparing the log-probabilities of a "Master" policy (the target model) against a "Novice" (a weaker or base model), we can identify **critical moments of divergence** — actions where the Master confidently deviates from the Novice during a successful trajectory are likely the pivotal reasoning steps that deserve high reward, whereas agreement implies trivial transitions.

The weaker Novice model is obtained via **tensor disentanglement** (singular value pruning), yielding a structurally comparable but capacity-reduced version of the same LLM.

---

## Key Contributions

1. **Implement** an RL environment to estimate baseline probabilities of actions during various tasks
2. **Design** a reward shaping function to assign dense, step-wise rewards
3. **Verify** the efficacy by using this dense signal and comparing convergence speed and final accuracy against a sparse baseline

---

## Approach

- Use **tensor disentanglement** (singular value pruning via the provided code) to create a weaker "Novice" model from a target LLM
- Treat inference as a hypothetical RL trajectory: each generated token is an **action**, previously generated text is the **state**
- **Observer pattern**: let the large (Master) model act and measure the level of disagreement at different pruning levels of the Novice to understand for how many states non-standard behavior is relevant
- **Performance analysis**: let the smaller (Novice) model act and measure downstream performance under pruned behavior to find at what point disagreement in paths measurably reduces reward

### Key Research Questions

- How do we generate a weaker version of the same model?
- What constitutes true "weakness" and how can it be reduced reliably?
- How do we ensure the method remains on-policy?
- How do we differentiate noise from genuine difficulty?
- How do we bound divergence on "easy" states?

---

## Deadlines

| Deliverable                         | Date       | Grade |
| ----------------------------------- | ---------- | ----- |
| Project Presentation Slides         | 25.07.2026 | 25%   |
| Seminar Paper [Peer Review version] | 25.07.2026 | —     |
| Tutorial Slides                     | 18.07.2026 | —     |
| Tutorial Notebook                   | 18.07.2026 | —     |
| Peer Reviews                        | 01.08.2026 | 10%   |
| Seminar Paper [Final]               | 08.08.2026 | 65%   |

---

## Team

| Name                     | Email                                           |
| ------------------------ | ----------------------------------------------- |
| Leonhard Driesch         | [leo.driesch@tum.de](mailto:leo.driesch@tum.de) |
| Ana Carolina Ramos Cunha | [ana.cunha@tum.de](mailto:ana.cunha@tum.de)     |
| Leonard Evers            | [ge95gih@mytum.de](mailto:ge95gih@mytum.de)     |

**Supervisor:** Tristan Till — sebis, TUM
