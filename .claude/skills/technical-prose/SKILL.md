---
name: technical-prose
description: Sentence-level rules for technical and academic writing — sentence length, em dashes, nominalization, old-to-new information flow, filler words, hedging, consistent naming, and framing a problem rather than a topic. Use when drafting or revising prose in a paper, report, README, or review, and when asked to tighten, clarify, or edit writing.
---

# Sentence-level rules for technical writing

These are revision rules. Draft first without them, then make a pass with them — applying
all ten while composing produces paralysis, not prose. The companion
[`seminar-paper`](../seminar-paper/SKILL.md) skill covers the level above this one:
structure, page limits, figure and caption strategy.

## The rules

1. **Prefer short sentences.** If a sentence runs past about 25 words, split it. But vary
   the length — a run of 20-word sentences flattens everything. Land the important claim in
   six words and it hits. A three-sentence conclusion is this same idea at paragraph scale.

2. **Prefer other structures to the em dash.** A period, a colon, a comma pair, or
   parentheses will usually do the job better. Reach for the dash only when nothing else
   works, and never twice in a paragraph.

3. **Put the actor in the subject slot and the action in the verb.** Most bad academic
   sentences hide the verb inside a noun.
   - Bad: "We perform an evaluation of the effect of masking on retrieval performance."
   - Good: "Masking hurts retrieval."
   - Test: circle your verbs. If they are all "is", "has", "performs", "provides",
     "exhibits", you have nominalized everything.

4. **Old information first, new information last.** English readers take the end of a
   sentence as the emphasis. Chain sentences by starting each one with something from the
   end of the last. This is what makes a paragraph feel like it flows rather than like a
   list. It is a mechanical fix, not a matter of taste.

5. **Keep subject and verb close.** "The model, which we train on 400k pairs collected from
   three sources and filtered for length, outperforms…" — thirteen words of interruption,
   and the reader has to hold the subject in memory. Move the qualifier out, or give it its
   own sentence.

6. **Passive is fine** when the agent is genuinely irrelevant, or when it puts old
   information first. "The dataset was collected from iNaturalist" beats a contorted active
   if the dataset is what the previous sentence ended on. Blanket passive-avoidance advice
   is wrong; the real rule is old-to-new.

7. **Cut filler words:** novel, leverage, utilize, in order to, it should be noted that,
   very, quite. Cut "interesting" — it is a tell that you have not said what is actually
   valuable.

8. **Hedge once, not twice.** "may potentially", "suggests that it is possible" — one hedge
   is honest, two is fear.

9. **Name your things and keep the name.** If a component appears three times, give it a
   short noun and use the same noun every time. Elegant variation — "the encoder", then
   "the upstream module", then "our representation network" — is a novelist's habit that
   costs a technical reader real effort.

10. **Use instability words to create a problem.** but, however, although, unfortunately,
    difficult, strain, while X / Y, surprising, inconsistent, anomaly, puzzling. If your
    opening paragraph contains none of these, you have described a topic, not posed a
    problem, and the reader has no reason to continue.

## Revision pass

Work outside-in; the later steps are wasted on text that gets cut.

1. **Openings first** (rule 10). Does the first paragraph of the paper, the section, or the
   README pose a problem? If it only announces a topic, rewrite it before touching
   anything else.
2. **Paragraph flow** (rule 4). Read the first six words of each sentence in order. If they
   introduce something new every time, the paragraph reads as a list — rechain it.
3. **Sentence surgery** (rules 1, 3, 5). Split the long ones, un-nominalize the verbs, move
   the interrupting qualifiers out.
4. **Word-level sweep** (rules 2, 7, 8, 9). Filler, double hedges, stray em dashes,
   inconsistent names for the same thing. Cheap, mechanical, do it last.

Rule 6 is not a step; it is the answer to anyone who tries to "fix" a passive that rule 4
put there on purpose.

## Editing someone else's text

Preserve the author's voice and terminology. Change what the rules identify as a defect,
not what merely differs from how you would have written it. If a sentence is long but
clear and deliberately rhythmic, leave it — rule 1 targets sentences that lose the reader,
not every sentence past a word count.

## Sources

- Williams, *Style: Toward Clarity and Grace* — rules 3, 4, 5
- Gopen & Swan, "The Science of Scientific Writing" — rules 4, 5
- McEnerney, "The Craft of Writing Effectively" (UChicago, 2014) — rule 10
- Forbes, "How to Get Your Paper Accepted" — rules 1, 7, 10
  (bundled at [`../seminar-paper/references/phd-metagame-how-to-get-your-paper-accepted.md`](../seminar-paper/references/phd-metagame-how-to-get-your-paper-accepted.md))
