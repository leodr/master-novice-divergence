---
name: seminar-paper
description: Course rules and writing guidance for the TUM seminar paper — page limit, deadlines, grading criteria, structure, figure/caption style, and what to cut when over length. Use when writing, editing, restructuring, or reviewing the paper.
---

# Seminar paper (TUM, "Increasing Reward Quality and Density for Verifiable Rewards")

## Hard constraints

- **Maximum 8 pages, excluding references and appendix.** This is the one rule with no
  wiggle room. Everything else is guidance.
- Two-column conference layout, ~9pt body text, 1in margins, provided `rgstyle` template.
  Do not change margins, font size, or column spacing to buy space — fix length by cutting.
- No word count, no appendix length limit, no explicit AI-use policy, no filename or
  PDF-only restriction were stated in the course materials. Final upload allows files up
  to 100 MB and up to 50 attachments.
- The paper carries **65 %** of the grade (presentation 25 %, peer review 10 %).

### Deadlines

| Milestone | Date |
| --- | --- |
| Paper draft for peer review | 25 July 2026, 23:59 |
| Peer review submission | 1 August 2026, 23:59 |
| Revised final paper | 8 August 2026, 23:59 |

Only the **revised final paper** is graded (stated explicitly in the preliminary slides).
The draft exists so peers have something to review — it should still be complete enough
to get useful feedback.

## What the graders actually reward

Grading emphasizes **scientifically grounded work and good scientific practice**, not
impressive results. Experiments are expected to *verify the method*; they do not have to
produce novel insights. A negative or null result, honestly reported, is fine.

Two consequences that override any "make it sound exciting" instinct:

1. **Methodology must be transparent and simple enough that a bachelor's student could
   reproduce it** and understand why it is sound. Every hyperparameter, model, dataset,
   grading rule, and seed count that matters for reproduction belongs in the paper (body
   or appendix). If a reader would have to guess, write it down.
2. **Do not oversell.** Claims must be exactly as strong as the evidence. State what was
   tested, what was found, and what remains open. Marketing language costs points here in
   a way it would not at a conference.

## Page count

References and the appendix do not count, so the body must end on or before page 8 of the
built PDF — check where the bibliography starts, not the total page count. Re-check on
every structural edit.

## Structure

Suggested by the course:

Introduction · Related Work · Methodology · Experiments · Results · Discussion ·
Limitations · Conclusion

Limitations may be a subsection of Discussion rather than its own section.

## Writing guidance

The course recommends Maxwell Forbes, ["How to get a paper accepted"](https://maxwellforbes.com/posts/how-to-get-a-paper-accepted/).
Its thesis: **page 1 makes the reader want to accept the paper; the remaining pages remove
reasons to reject it.** Apply its clarity advice, not its salesmanship.

The full post is bundled at
[`references/phd-metagame-how-to-get-your-paper-accepted.md`](references/phd-metagame-how-to-get-your-paper-accepted.md).
Read it when you need the reasoning or the worked before/after examples behind the points
summarised below — in particular the rejected-vs-accepted title, abstract, Figure 1,
caption, and conclusion rewrites. The summary here is enough for routine edits.

**Page 1 (title, abstract, Figure 1, intro opening):**

- Title: specific and memorable, not broad and over-claiming.
- Figure 1 should communicate what the work does and why it matters *before* the caption
  is read.
- Abstract: concrete — problem, specific contributions, method, one brief result.
- Open the introduction with the specific problem and why it matters. Do not open with a
  generic survey of the field.
- Use a problem → difficulty → solution rhythm: establish the tension, then show how this
  work addresses it.

**The rest:**

- Design for skimmers. The paper should be intelligible from figures, captions, and the
  conclusion alone.
- **End every figure and table caption with the takeaway**, not a literal description of
  the axes. "…disagreement positions carry no more outcome leverage than depth-matched
  controls" beats "Accuracy versus rank fraction."
- State important findings early. Never withhold a result for dramatic effect.
- Anticipate the obvious criticism: include the baseline, ablation, control condition, or
  statistical test a skeptical reader will ask for. For a null result, the controls and the
  power of the test are the whole argument — give them room.
- Push extensive ablations, implementation detail, and supporting results to the appendix.

**Conclusion — keep it very short, three questions:**

1. What did you do?
2. Why is it good or useful?
3. Why does it matter?

## When over 8 pages

Cut in this order (Forbes: look at pages 5–7 first):

1. Repetitive qualitative examples — keep one, appendix the rest.
2. Secondary analysis that does not bear on the main hypothesis.
3. Related-work paragraphs that restate a neighbouring paper rather than positioning ours.
4. Detailed implementation prose → appendix, leaving a one-line pointer.

Never sacrifice the central methodology explanation or the controls that make the result
credible — those are what the grading criteria reward. Never shrink margins or fonts.

## Pre-submission checklist

- [ ] Body ends on or before page 8; references and appendix beyond that are fine.
- [ ] Every claim in the abstract and conclusion is supported by a specific result.
- [ ] Every caption ends with a takeaway.
- [ ] A bachelor's student could reproduce the experiments from the paper alone (models,
      data, grading rule, ranks, seeds, sample sizes, and statistics all stated).
- [ ] Limitations honestly state what the experiments cannot rule out.
- [ ] All citations resolve (no `[?]`), bibliography rebuilt.
- [ ] Figures regenerated from the current results, not stale.
