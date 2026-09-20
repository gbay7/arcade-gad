# The fusion problem: what is broken, and a plan to attack it

## What the evidence now says

On three unseen graphs the library already contains a view worth 62 to 64 AUROC
and the shipped gate returns a score near chance.

| graph | frozen gate | best single library view | best baseline |
| --- | --- | --- | --- |
| Tolokers | 52.3 | one-class 62.7 | DOMINANT 55.5 |
| Amazon fraud | 48.4 | one-class 64.4 | OCGNN 66.7 |
| Questions | 49.5 | reconstruction 62.4 | not run |

Diagnosis is no longer the binding constraint. Given a sourced description of the
graph and a catalog that says what each view computes, Opus names one-class on
both graphs where one-class is the answer, and on Amazon a mid-size model reaches
62.6 by selecting it. The gate then averages that view with weaker companions.

Two label-free rules for honouring the analyst's lead have been tested over all
399 stored decisions and both lose the study benchmark:

| rule | benchmark | held-out | verdict |
| --- | --- | --- | --- |
| lead view alone | 68.6 to 60.1, worse on 7 of 10 | 50.7 to 48.7 | rejected |
| lead view when it also tops a gate diagnostic | 69.1 to 64.6, worse on 8 of 10 | 50.8 to 49.2 | rejected |

So the gate's composition is worth roughly 8 points on the benchmark and costs
roughly 12 on the held-out graphs. Any rule that trades one for the other is not
progress. What is needed is a label-free quantity that separates the two regimes.

## Why averaging fails here, mechanically

Average-of-maxima assumes every input carries signal in the same direction. On
these graphs it does not hold. On Amazon fraud, four of the ten applicable views
score below chance (reconstruction 26.7, homophily violation 32.0, spectral 33.1,
adversarial 33.2) while one-class reaches 64.4. Every stored decision containing
one of the four lands near 50; the two decisions containing none scored 60.2 and
60.9. The gate's two diagnostics do not see this: on Tolokers they rank the
attribute-structure view almost exactly level with one-class.

A landmarker cannot reveal the sign of a signal. That is the root cause, and it
is why sign-alignment by principal component was catastrophic when tried
(Enron 83.7 to 26.3): flipping a view on a guess destroys the graphs that work.

## The ceiling we are chasing

For every stored decision, the best single selected view is the oracle any
fusion rule could reach without changing the selection. Phase 1 measures that
gap per graph. It bounds the whole effort: if the gap is small on the benchmark
and large on held-out graphs, a regime-dependent rule is worth building; if it
is large everywhere, selection and fusion have to be redesigned together.

## Plan

**Phase 0, one test bench.** Generalize the two rule-replay scripts into
`experiments/fusion_rules.py`: a rule is a function from (selected views, cached
scores, profile) to a fused vector, and the bench replays every stored decision
through it, reporting benchmark and held-out means, per-graph deltas, and how
often the rule changes the outcome. One artifact, many rules, no new LLM calls.

**Phase 1, measure the gap.** Per graph and per arm: shipped gate, best selected
view (oracle), worst selected view, and the fraction of decisions containing at
least one below-chance view. This says how much is recoverable at all.

**Phase 2, candidate rules.** Each must be computable without labels.

1. *Consensus filtering.* Build a pseudo-target from the aggregate of the
   selected views, drop views whose rank correlation with it is negative, fuse
   the rest. Literature-grounded (SELECT's vertical selection). Drops rather
   than flips, so a wrong guess costs a view instead of inverting the ranking.
2. *Mutual-agreement weighting.* Weight each view by its median rank correlation
   with the others, so a view that agrees with nothing contributes little. A
   softer form of 1 with no threshold.
3. *Analyst-order weighting.* Use the analyst's stated order as a prior with
   geometric decay instead of as a hard selection. The hard version failed; the
   soft version has not been tested, and it is free to test over stored decisions.
4. *Seed stability.* We hold five independent seed caches per graph. A view whose
   top-k set moves between seeds is unreliable; a view whose ranking is stable is
   not necessarily right but is at least reproducible. This favours closed-form
   detectors by construction, so it is only admissible as a tie-breaker, and that
   bias must be reported.
5. *The analyst as fusion adjudicator.* A second, cheap question to the model:
   given the pairwise rank-correlation matrix of the selected views and their
   score shapes, should one view carry the graph or should they be combined, and
   which one. This is the only candidate that adds an LLM call, and it is the one
   that fits the paper's thesis, since fusion is exactly where a reader would ask
   why the analyst is not consulted.

**Phase 3, admission test.** A rule ships only if it does not lose more than one
point on the ten study graphs and gains on the held-out graphs. Questions is held
back from rule design and used only to confirm, since Tolokers and Amazon fraud
have now been looked at extensively and are no longer clean tests.

**Phase 4, if a rule survives.** Re-run the affected tables, record the rule in
the gate behind a flag, and write it up with the negative results kept: four
rejected rules are the evidence that the surviving one is not arbitrary.

## Expected outcome, stated in advance

Candidates 1 and 2 are the most likely to work on Amazon fraud, where the
anti-signal views are a minority and disagree with the rest. Neither will help
Tolokers, where most views are below chance and the consensus itself is wrong.
If that is what happens, the honest conclusion is that label-free fusion can
remove contradictory views but cannot find a minority-correct one, and the
mechanism that fixes Tolokers is the domain description plus the analyst, not a
statistic. That would be a result worth publishing either way.

---

# Results, phases 0 to 3 (2026-09-18)

Phase 0 and 1 are done and phase 2 is done for every rule that needs no model
call. `experiments/fusion_rules.py` is the bench; `fusion_rules.json` holds one
row per stored decision with the outcome of every rule and the label-free
diagnostics of the selection.

## The gap, measured

Over the 181 decisions that reach the ensemble branch, the best single selected
view beats the gate by 7.2 on the benchmark graphs and 7.3 on the held-out ones,
and by 12.8 on the three hardest benchmark graphs. 140 of those 181 selections
contain a view that scores below chance on that graph.

## Every statistical rule failed

Ensemble-branch means against the shipped gate:

| rule | benchmark | held-out |
| --- | --- | --- |
| drop views that disagree with the rest | -0.7 | +0.3 |
| weight by median agreement | -0.4 | -1.9 |
| analyst order as a soft prior | -2.3 | -1.3 |
| analyst lead alone | -3.7 | +0.3 |
| top view by the gate's own diagnostic | -2.9 | +3.0 |
| drop the worst view by that diagnostic | -3.4 | +0.8 |

Each single-view rule gains exactly where the ensemble is failing and collapses
where it is working. The clearest case is the diagnostic-based rule: +5.4 on
Books, Disney and Enron, +3.0 on Tolokers, and -8.8 on Reddit, Weibo and the two
injected graphs, where it takes Weibo from 67.5 to 32.8.

## The regime is detectable; the right view is not

Mean agreement between the selected views correlates with how much the ensemble
dilutes (Spearman -0.38, p = 1e-7), and the number of selected views correlates
too (+0.33). The gate's own diagnostics do not (UED ratio +0.04, tail separation
+0.07, both insignificant). But conditioning any rule on an agreement threshold
buys at most 0.4 on the held-out graphs while losing on the benchmark, at every
threshold from 0.10 to 0.30. Weibo is why: it has the lowest agreement of all
eight graphs and the largest dilution gap, so the detector fires, and then the
rule picks the wrong view and destroys the graph.

So the problem splits cleanly:

- *Knowing that the ensemble is diluting* is partly solvable without labels.
- *Knowing which view to keep* is not solved by any statistic available to the
  gate: tail separation, UED tail quality, mutual agreement, or the analyst's
  own stated order.

## What is left

Candidate 5, the analyst as fusion adjudicator, is now the only untested route,
and it is the one the evidence points at: given a sourced description and a
catalog that says what each view computes, the analyst named the mechanism that
wins both held-out graphs, while no statistic could. It is implemented in
`experiments/fusion_adjudicator.py`, which asks a second, narrower question
(combine these views, or use exactly one, and which) with the pairwise agreement
matrix and each view's shape in front of it. Adjudication depends only on the
graph and the selection, so 181 decisions collapse to 93 model calls.

Candidate 4, seed stability across the five cached seeds, remains untested and is
admissible only as a tie-breaker, since it favours deterministic detectors by
construction.

---

# Results, phases 4 and 5 (2026-09-18): the candidate that asks the analyst

Candidate 5 was implemented in `experiments/fusion_adjudicator.py` and scored in
the bench as two columns. Amazon fraud was added to the bench as the clean test,
since it was never used to design any rule.

| rule | benchmark | held-out |
| --- | --- | --- |
| analyst adjudicates | +2.4 | -0.3 |
| drop views that disagree with the rest | -0.7 | +0.4 |
| weight by median agreement | -0.4 | -1.8 |
| analyst order as a soft prior | -2.3 | +0.4 |
| top view by the gate's own diagnostic | -2.9 | +2.7 |
| drop the worst view by that diagnostic | -3.4 | +2.1 |
| analyst lead alone | -3.7 | +2.4 |
| analyst adjudicates, quality statistics withheld | -1.2 | -5.0 |
| oracle, best selected view | +7.2 | +7.3 |

No rule gains on both sides. The adjudicator wins the benchmark by 2.4, gaining
11.2 on Enron, 11.1 on Weibo and 6.1 on Disney, and loses 14.8 on injected Cora
by discarding a genuine two-population union. Then it fails the clean test:
52.5 on Amazon fraud against a gate of 53.8. The statistical single-view rules do
the reverse, best on the held-out graphs and worst on the benchmark.

## A hypothesis that was wrong, in the informative direction

Twelve of the first fifteen Amazon adjudications cited tail separation or UED
tail quality, and the bench had already shown neither predicts which view to
keep. The obvious inference was that those numbers were misleading it. Withholding
them costs a further 8.4 on Amazon fraud and 5.0 across the held-out graphs. With
them the model found one-class once and local affinity three times, averaging
54.4 on the view it chose; without them it picked below-chance views four times
in fifteen, including reconstruction at 26.7 on the graph whose catalog entry
states that neighbour-relative views collapse at degree 736, averaging 45.0. The
weak statistics were propping the decision up, not corrupting it.

## Standing conclusion

Three things are established and two of them are negative.

1. The headroom is real: the best selected view beats the gate by about 7 points
   on both the benchmark and the held-out graphs, and by 12.8 on the three
   hardest benchmark graphs.
2. The regime in which the ensemble dilutes is partly detectable without labels,
   from the mean pairwise agreement of the selected views (Spearman -0.38,
   p = 1e-7) and from the number of views (+0.33). The gate's own diagnostics
   detect nothing (UED ratio +0.04, tail separation +0.07).
3. Neither fact converts into a rule. Eight candidates, six statistical and two
   that ask the analyst, all fail on one side or the other.

For the journal version the defensible claim is not a better gate but a measured
boundary: label-free fusion can remove contradictory views and cannot find a
minority-correct one, and the eight rejected rules are the evidence that the
shipped composition is not arbitrary. The remaining untested candidate is seed
stability across the five cached seeds, admissible only as a tie-breaker because
it favours deterministic detectors by construction.
