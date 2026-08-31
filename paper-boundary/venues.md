# Venue analysis (read 2026-08-30, CFPs checked directly)

**Decision: two co-best fits, different framings.**
- **InterpScience (Sep 1-2)** for the interpretability / mechanism-vs-artifact framing. The current
  draft matches it with zero reframe; the paper's sharpest move (reconciling the three-way steering
  disagreement) is quintessential "interpretability as a science."
- **Pre-to-Post (Sep 5)** for the training-science framing. Its central question ("how does
  post-training reshape the base model beyond benchmark gains?") is answered almost verbatim by our
  margin-vs-lever result. Later deadline; needs a light intro/abstract reframe.

Both are non-archival. If concurrent non-archival submission is permitted by both (VERIFY each
policy; note reciprocal-reviewing obligations), submit to both: InterpScience with the current
framing, Pre-to-Post with a reframed intro. If picking one: InterpScience for zero-reframe and
sharper novelty, Pre-to-Post for the later deadline and a more literal thematic match to the empirics.

| Workshop | Deadline | Fit | Notes |
|---|---|---|---|
| **Interpretability as a Science (InterpScience)** | **Sep 1–2** | **Best** | CFP invites "experimental designs that distinguish mechanisms from artifacts", "measurement validity... and evaluation design", "causal/interventional methods". This paper IS a mechanism-vs-artifact measurement paper. 5/9 pp, non-archival. |
| **Transitioning from Pre-training to Post-training (Pre-to-Post)** | **Sep 5** | **Co-best** | Central CFP question: "how do different post-training procedures reshape the base model, beyond targeted benchmark gains?" This paper answers exactly that: post-training widens the margin, not the steering lever, across the full pre->post flow, 4 families, 2 scales. Later deadline than InterpScience. Needs a light reframe to foreground the training-transition question over the audit angle. Non-archival. |
| Foundations of LM Security (FLMSec) | Aug 27 (PASSED) | Strong (missed) | Wanted "reproducible, generalizable evaluation methodologies rather than fragile and hackable" + best-paper award, 8pp. Our two audit warnings fit dead-center. Deadline gone unless extended. |
| Can We Trust the Judge? (JUDGe) | Aug 30 (today) | Partial | Only the prefix-metric-inversion section fits; core lever/load is not judge-reliability. Rushed reframe not worth it. |
| TAI-Eval (Trustworthy AI Evaluation) | Aug 30 (today) | Partial | Same as JUDGe. |
| Interpretability for Discovery | Sep 3 | Weak | About discovering novel knowledge; we correct a measurement. Viable fallback only. |
| XAI4Science | Sep 6 | Weak | Interpretability-for-science/trust; off-core. |
| Trustworthy AI for Good (AI4GOOD) | Sep 2 | Poor | "AI for good" = beneficial applications (health, climate, accessibility). This is safety-audit methodology, not an application. |

**Fallback if InterpScience is missed:** Interpretability for Discovery (Sep 3), non-archival,
weaker theme. **Future full-paper / main-track:** the FLMSec framing (safety-audit reliability,
fragile evaluation methodology) is a genuinely strong second home; watch for its next edition or a
sibling security venue, and for a main-track submission (ICML/ICLR) lead with the audit-reliability
angle there.

Current draft's framing (interpretability: margin vs lever, mechanism vs artifact) matches
InterpScience as written — do not reframe for this submission.
