# System Architecture & Design Rationale

**Status:** draft for discussion — every row and every choice below is a proposal, not a lock-in. Mark up whatever you want to override.

## What this system is for

Two research questions drive every decision below:

1. Does a credit-assignment-aware RL algorithm (GiGPO) measurably outperform standard GRPO on a long-horizon, rule-heavy compliance task — and does that gap change under a mid-episode disruption?
2. Does a policy trained on a fixed set of hand-authored tasks generalize to structurally novel scenarios it has never seen, or does it just memorize the training distribution?

Everything else — the demo, the data layer, the infrastructure — exists to answer those two questions cleanly, cheaply, and reproducibly. Where the old hackathon build already got something right, it's kept. Where something was a hackathon-specific constraint rather than a real design choice, it's dropped.

## Quick-reference decision table

| # | Component | Chosen | Main alternative(s) rejected | One-line reason |
|---|---|---|---|---|
| 1 | Environment interface | Gym-style `reset` / `step` / `state` | Fully bespoke interface | Industry-standard shape; keeps the option to publish to the OpenEnv Hub later, for free |
| 2 | Rule & country data layer | Structured JSON/YAML fixtures | Hardcoded logic, relational DB | Matches the actual scale (a handful of countries); already proven in v1 |
| 3 | Reward design | Dense, milestone-shaped | Sparse, terminal-only reward | Matches recent literature (BEACON); already validated by v1's own results |
| 4 | Scenario generation | LLM-generated, held-out task pool | Hand-authoring only, templated randomization | The only way to actually test generalization instead of memorization |
| 5 | Training framework | verl-agent | TRL, Unsloth, hand-rolled | GRPO and GiGPO share plumbing there; avoids reimplementing GiGPO from a paper |
| 6 | RL algorithm(s) | GRPO (control) vs. GiGPO (treatment) | PPO, DPO-style, DAPO / Dr.GRPO / GVPO | GiGPO specifically fixes long-horizon credit assignment — this task's exact weak point |
| 7 | Base policy model | Qwen3-1.7B-Instruct | Qwen2.5-1.5B unchanged, 7B+, other families | Same size class as before (fair comparison) plus native tool-calling for free |
| 8 | Baseline agent | Current lightweight frontier model, rewritten prompt | Old GPT-4o-mini + prescriptive prompt | Dated model, and the old prompt handed the model the answer |
| 9 | Evaluation harness | Multi-seed (3-5), held-in + held-out | Single-seed point estimate | The team's own flagged gap; the only way a result survives scrutiny |
| 10 | Compute | RTX 4050 for iteration, free-tier cloud for real runs | Fully local only, paid cloud | 6GB VRAM is fine for debugging, tight for real multi-seed runs |
| 11 | Demo layer | Light FastAPI + Streamlit | Full NL-intake product, CLI-only, no demo | Matches the existing portfolio pattern; interview-ready without becoming the main project |
| 12 | RAG + poisoning | Deferred, optional Phase 4 | Built into core scope now | No ready regulation dataset exists — real curation cost, not core to the two research questions |

## Component detail

### 1. Environment state machine & action space

**What it is:** the core simulation — the object holding case state (documents, approvals, payroll flags), exposing the ~12 actions an agent can take, and advancing state one action at a time.

**What we're building:** kept from v1 almost as-is — a `reset()` / `step(action)` / `state()` interface, the same shape as Gymnasium and the OpenEnv spec.

**Alternatives weighed:**
- *A fully custom interface* (e.g. a chat-style "send a message, get a message back" loop instead of discrete actions) — more natural for a pure chat product, but it throws away clean, deterministic reward computation, which the whole experiment depends on.
- *A full discrete-event simulation framework* — overkill for a system with a few dozen state variables and at most a few hundred steps per episode.

**Why this choice:** `reset/step/state` isn't a hackathon artifact — it's the same shape ALFWorld, WebArena, and essentially every serious agent-training environment converges on, because it's the right abstraction for "an agent acts, the world responds, a trajectory gets scored." Keeping it costs nothing and leaves publishing to the OpenEnv Hub open as a free option later.

### 2. Rule & country data layer

**What it is:** how country-specific rules (visa types, tax treaties, PDPA-style data rules) are represented and loaded.

**What we're building:** kept from v1 — structured JSON fixtures per country, loaded by the rules engine at reset time. Possibly extended with one or two new countries, but only if they introduce a genuinely different rule-conflict shape (like the UAE's no-income-tax trap did), not just more volume.

**Alternatives weighed:**
- *Hardcoded if/else logic per country* — what a rushed build might do; v1 already avoided this, correctly.
- *A relational database or graph structure* — real overkill at 3-8 countries' worth of rules; adds infrastructure with no payoff at this scale.
- *Sourcing rules directly from real government regulation text as the primary representation* — more realistic, but there's no ready structured dataset for it. What's actually public is visa-issuance statistics (grant/refusal counts by year and nationality), not rule logic. This becomes the Phase 4 stretch goal, not the primary layer, because curating it properly for even one new country is real, non-trivial work.

**Why this choice:** fixtures are simple, fast to extend, easy to diff and version, and already proven to work. There's no scale pressure yet that justifies anything heavier.

### 3. Reward design

**What it is:** how a completed (or abandoned) episode becomes a training signal.

**What we're building:** kept from v1 — a dense, shaped reward with milestone bonuses (partial credit as categories of the task complete), rather than one reward at the very end.

**Alternatives weighed:**
- *Sparse, terminal-only reward* (1 if the case resolves correctly, 0 otherwise) — simpler to implement and easier to defend as "unbiased," but brutal for a small model to learn from across a 20-40 step episode, since most early attempts would look identically bad.

**Why this choice:** this is one of the nicer validated findings from the research pass — a method called BEACON, which partitions trajectories at milestone boundaries for exactly this kind of reward shaping, is part of the current RL-for-agents literature. The original team arrived at a similar idea independently under a 48-hour deadline, which is worth stating plainly in any write-up rather than treating as a hack.

### 4. Scenario & crisis generation layer — new

**What it is:** a process that generates new task variants and mid-episode disruption events, structurally different from the four hand-authored ones.

**What we're building:** an LLM prompted to produce new country-rule configurations and crisis scenarios, with most of its output held out as a never-trained-on evaluation set.

**Alternatives weighed:**
- *Hand-authoring more scenarios* — what v1 did, and what the team's own write-up named as the top priority to fix; doesn't scale, and doesn't test anything new about generalization if the same person designs both the training and the test tasks.
- *Templated random generation without an LLM* (randomly swap numbers and thresholds in existing templates) — cheaper and fully deterministic, but produces variations, not genuinely novel structures — a weaker test of real generalization.
- *An LLM that both generates and grades new scenarios* (a full self-play / auto-curriculum loop) — the more ambitious version of this idea, and a legitimate later extension, but it adds a second learned/prompted component whose own errors could contaminate the evaluation. Safer to start with LLM-as-generator only, reviewed once by you, then held fixed for evaluation.

**Why this choice:** this is the direct, well-grounded answer to "how do we meaningfully use an LLM here." It mirrors an active current research direction (LLM-generated environments and auto-curricula), and it's the only design that can actually answer research question #2 above, since a fixed four-task train/test split cannot.

### 5. Training framework

**What it is:** the library that runs rollouts, computes advantages, and updates model weights.

**What we're building:** verl-agent, an actively maintained extension of veRL purpose-built for training LLM agents via RL.

**Alternatives weighed:**
- *TRL* (what v1 used) — solid and well-documented, fine for a single algorithm like GRPO, but it doesn't have GiGPO's step-level anchor-state grouping built in; that logic would need to be implemented from the paper's pseudocode by hand.
- *Unsloth* — excellent for raw LoRA fine-tuning speed, but built more around single-turn supervised or preference fine-tuning than multi-turn agent rollouts against a live environment.
- *Hand-rolling both algorithms from scratch* — the most "pure" option, and not unreasonable as a learning exercise, but it turns a focused comparison into a multi-week implementation-and-debugging project for infrastructure that already exists and is peer-reviewed.

**Why this choice:** GRPO and GiGPO sit side by side on shared plumbing in this one framework, which is what makes a clean, controlled comparison actually feasible in the time available.

### 6. RL algorithm(s)

**What it is:** the specific policy-optimization method(s) being compared.

**What we're building:** GRPO as the control (keeps one result directly comparable to v1), GiGPO as the treatment.

**Alternatives weighed:**
- *Plain PPO* — the older standard; needs a learned critic/value network, adding memory and instability that group-based methods were specifically designed to avoid.
- *DPO-style offline preference optimization* — built for single-turn preference pairs, not naturally suited to an interactive, multi-step environment with a live reward signal.
- *Other current GRPO variants (DAPO, Dr.GRPO, GVPO)* — each fixes a real but different problem (token-length bias, entropy collapse, and so on); none specifically target the long-horizon, delayed-credit problem this task's mid-episode crisis creates. Worth knowing they exist, not worth splitting the experiment four ways instead of running one clean comparison.

**Why this choice:** GiGPO's entire pitch is fine-grained credit assignment for long, sparse-reward agent episodes — exactly the shape of the crisis task, where the "right" response happens somewhere in the middle of a 20-40 step episode. It's also validated at essentially this model scale (Qwen2.5-1.5B/7B) with double-digit percentage gains over GRPO on comparably long tasks, at no extra memory cost.

### 7. Base policy model

**What it is:** the model that actually gets trained.

**What we're building:** Qwen3-1.7B-Instruct as the primary model, with Qwen3-4B as an optional "does scale help" third arm if compute allows.

**Alternatives weighed:**
- *Keep Qwen2.5-1.5B unchanged* — keeps a literal apples-to-apples number against the hackathon result, but ships without native tool-calling and is a generation behind for no real benefit.
- *Jump straight to 7B+* — tempting since "bigger is better" is the easy intuition, but it confounds the experiment: if GiGPO wins, is it the algorithm or the extra capacity? It also sits right at the edge of what's comfortable even on free-tier cloud GPUs across many parallel seeds.
- *A different model family (Llama, Gemma, Phi)* — no strong reason to switch; breaks the "same size class as before" comparability story for no added benefit.

**Why this choice:** staying in the same parameter class as the original keeps the comparison honest and cheap to run many seeds of, while native tool-calling removes a real weakness — the JSON-schema-in-the-prompt workaround — for free.

### 8. Baseline / comparison agent

**What it is:** the prompted-only model used as the "what if nothing gets trained" reference point.

**What we're building:** a current lightweight frontier model (something in the GPT-5 Mini / Gemini Flash / Claude Haiku 4.5 tier), called through a leaner prompt — full rule and action schema, no exact step-sequence walkthroughs standing in for the model's own reasoning.

**Alternatives weighed:**
- *Keep GPT-4o-mini and the old prompt* — GPT-4o-mini is now a year-plus-old model, and the old prompt effectively hands the model the correct sequence for the hardest task, which makes any "trained beats prompted" claim hollow.
- *Drop the baseline agent entirely* — loses a genuinely useful reference point; "how far does a capable model get with zero training, given the same information" is a real, citable comparison.
- *Run several frontier baselines in parallel* — a nice-to-have for a richer table, not essential to either research question, and adds API cost and scope for marginal value.

**Why this choice:** one honestly-prompted, current baseline is enough to answer "does training actually matter here" without turning the baseline comparison into its own side-project.

### 9. Evaluation harness

**What it is:** how results get measured, aggregated, and reported.

**What we're building:** every condition (GRPO, GiGPO, each model size if used) run across 3-5 seeds, reported as mean ± spread, evaluated on both the original held-in tasks and the new held-out generated set.

**Alternatives weighed:**
- *Single-seed point estimates* — what v1 reported (0.585 → 0.665); fine for a 48-hour hackathon, not fine for a claim meant to stand behind in a write-up.
- *Formal statistical significance testing on top* — a legitimate extra layer (a simple t-test or bootstrap confidence interval across seeds), worth adding once the core runs exist, not a blocker to starting.

**Why this choice:** this is the team's own flagged gap from the hackathon, and it's non-negotiable if either research question is going to produce a claim that survives someone asking "on how many runs?"

### 10. Compute & infrastructure

**What it is:** where training and evaluation actually execute.

**What we're building:** the RTX 4050 (6GB VRAM) for fast local iteration — debugging the reward function, sanity-checking a handful of rollouts, tuning hyperparameters cheaply via QLoRA on the smaller model — and a free-tier cloud GPU (Colab, or Kaggle's free T4/P100 quota) for the real multi-seed training runs, the same role Colab played before, just without an artificial time limit.

**Alternatives weighed:**
- *Fully local on the RTX 4050* — QLoRA on a 7B-class model needs roughly 6-12GB depending on batch size and sequence length, a tight-to-failing fit on 6GB, and running several seeds sequentially on a laptop GPU would take a long time even for the smaller model.
- *Paid cloud compute* — unnecessary at this model scale; free tiers comfortably handle a 1.7-4B model with QLoRA.

**Why this choice:** this splits cleanly into "cheap fast iteration" and "the real experiment," matching how the original build actually worked, just without a compute ceiling that was never really yours to work around in the first place.

### 11. Demo / interface layer

**What it is:** the runnable, showable piece of the project.

**What we're building:** a light FastAPI backend plus a Streamlit front end to run and watch an episode, reusing the pattern from the rest of the portfolio.

**Alternatives weighed:**
- *A full natural-language case-intake layer* (freeform HR text parsed into structured state by an LLM) — a genuinely good idea, and a legitimate second phase of this whole project later, but it's real new product-engineering scope that doesn't serve either research question directly.
- *CLI-only* — technically sufficient to verify things work, weaker for showing anyone else what was built.
- *No demo at all* — a research project nobody outside you can see or run is a much weaker artifact, both for interviews and for your own motivation to finish it.

**Why this choice:** enough polish to be shown in an interview or a write-up, without becoming the main event.

### 12. Stretch — RAG-grounded rules & adversarial robustness (Phase 4)

**What it is:** grounding country rules in real regulation text via retrieval, and optionally testing what happens if that retrieved text is adversarially poisoned.

**What we're building:** nothing yet — this stays an explicit, optional Phase 4.

**Alternatives weighed:**
- *Build this into core scope now* — the piece with the strongest tie to the separate RAG poisoning research project, and the one most tempting to over-scope into this plan. But there's no ready-made structured dataset of real immigration rules — what's public is application and decision statistics, not rule logic — so doing this properly means the same kind of manual curation from official portals that built the original three countries, per new country, before any poisoning study can even start.
- *Skip it entirely* — leaves a genuinely interesting bridge between the two projects on the table for no good reason.

**Why deferred, not dropped:** worth doing once the core GRPO/GiGPO and generalization results are in and clean, with real time left over.

## Explicitly out of scope for now

- Multi-agent negotiation (separate LLM agents for HR/Legal/Finance instead of a single policy)
- The full natural-language product layer
- More than one or two additional countries
- Anything that requires a paid compute budget

None of this is dropped forever — it's just not what the first version of this project needs in order to prove its two claims.
