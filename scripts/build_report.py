"""Build the standalone HTML findings report with figures embedded as data URIs."""

import base64
import pathlib
import re

FIG_DIR = pathlib.Path("figures")
OUT = pathlib.Path("report.html")

FIGS = [
    "fig_e1_formation_scaffold.png",
    "fig_e1b_1b_fine_grained.png",
    "fig_e2_cosine_drift_scaffold.png",
    "fig_e2_frozen_vs_refit.png",
    "fig_e4_behaviour_vs_geometry.png",
    "fig_e3_causal_staleness.png",
    "fig_e5_topk_jaccard.png",
    "fig_e6_monitor_staleness.png",
    "fig_e7_harm_vs_honesty.png",
    "fig_e8_jspace_residency.png",
    "fig_probe_transfer_regimes.png",
]

HTML = r"""<title>Alignment Elasticity Along the Model Flow</title>
<style>
  :root {
    --ground:#F5F6F7; --surface:#FFFFFF; --sunk:#EDEFF1;
    --ink:#15191C; --ink-muted:#5C666E; --ink-faint:#8A939B;
    --rule:#DCE0E4; --rule-strong:#C3CAD0;
    --teal:#0E6E7C; --teal-soft:#DCEEF1;
    --amber:#96591A; --amber-soft:#F6EBDA;
    --clay:#9E3D2D; --clay-soft:#F6E3DF;
    --serif: Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --measure: 68ch;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground:#0F1215; --surface:#161B1F; --sunk:#1C2227;
      --ink:#E4E9EC; --ink-muted:#9AA6AE; --ink-faint:#6F7B84;
      --rule:#262E34; --rule-strong:#37424A;
      --teal:#3FB0C0; --teal-soft:#123037;
      --amber:#D19A52; --amber-soft:#2E2417;
      --clay:#C9705E; --clay-soft:#301A16;
    }
  }
  :root[data-theme="dark"] {
    --ground:#0F1215; --surface:#161B1F; --sunk:#1C2227;
    --ink:#E4E9EC; --ink-muted:#9AA6AE; --ink-faint:#6F7B84;
    --rule:#262E34; --rule-strong:#37424A;
    --teal:#3FB0C0; --teal-soft:#123037;
    --amber:#D19A52; --amber-soft:#2E2417;
    --clay:#C9705E; --clay-soft:#301A16;
  }
  :root[data-theme="light"] {
    --ground:#F5F6F7; --surface:#FFFFFF; --sunk:#EDEFF1;
    --ink:#15191C; --ink-muted:#5C666E; --ink-faint:#8A939B;
    --rule:#DCE0E4; --rule-strong:#C3CAD0;
    --teal:#0E6E7C; --teal-soft:#DCEEF1;
    --amber:#96591A; --amber-soft:#F6EBDA;
    --clay:#9E3D2D; --clay-soft:#F6E3DF;
  }

  * { box-sizing: border-box; }
  body {
    margin:0; background:var(--ground); color:var(--ink);
    font-family:var(--sans); font-size:17px; line-height:1.65;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:1180px; margin:0 auto; padding:0 28px 96px; }
  .col { max-width:var(--measure); }
  h1,h2,h3 { font-family:var(--serif); font-weight:600; text-wrap:balance; line-height:1.2; }
  p { margin:0 0 1.1em; }
  a { color:var(--teal); }

  /* ---------- masthead ---------- */
  header.mast { padding:72px 0 40px; border-bottom:1px solid var(--rule-strong); margin-bottom:8px; }
  .eyebrow {
    font-family:var(--mono); font-size:12px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--teal); margin:0 0 18px;
  }
  h1 { font-size:clamp(34px,5.2vw,56px); margin:0 0 18px; letter-spacing:-0.015em; }
  .standfirst { font-size:20px; line-height:1.55; color:var(--ink-muted); max-width:60ch; margin:0 0 32px; }
  .facts { display:flex; flex-wrap:wrap; gap:0; border-top:1px solid var(--rule); }
  .fact { padding:14px 26px 14px 0; margin-right:26px; border-right:1px solid var(--rule); }
  .fact:last-child { border-right:0; }
  .fact b { display:block; font-family:var(--mono); font-size:22px; font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }
  .fact span { font-size:13px; color:var(--ink-faint); }

  /* ---------- sections ---------- */
  section { padding:52px 0 0; }
  h2 {
    font-size:15px; font-family:var(--mono); font-weight:500; letter-spacing:.16em;
    text-transform:uppercase; color:var(--ink-faint); margin:0 0 26px;
    padding-bottom:10px; border-bottom:1px solid var(--rule);
  }
  h3 { font-size:26px; margin:0 0 12px; }
  .lede { font-size:19px; color:var(--ink-muted); max-width:62ch; }

  /* ---------- finding blocks ---------- */
  .find {
    background:var(--surface); border:1px solid var(--rule); border-left:3px solid var(--teal);
    border-radius:2px; padding:26px 30px; margin:0 0 20px;
  }
  .find.caveat { border-left-color:var(--amber); }
  .find.retract { border-left-color:var(--clay); }
  .find-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:10px; }
  .tag {
    font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase;
    padding:3px 8px; border-radius:2px; background:var(--teal-soft); color:var(--teal); white-space:nowrap;
  }
  .tag.caveat { background:var(--amber-soft); color:var(--amber); }
  .tag.retract { background:var(--clay-soft); color:var(--clay); }
  .expid { font-family:var(--mono); font-size:13px; color:var(--ink-faint); font-variant-numeric:tabular-nums; }
  .find h3 { font-size:22px; margin:0 0 10px; }
  .find p { max-width:var(--measure); }
  .find p:last-child { margin-bottom:0; }

  /* ---------- data ---------- */
  .tablewrap { overflow-x:auto; margin:0 0 20px; border:1px solid var(--rule); border-radius:2px; background:var(--surface); }
  table { border-collapse:collapse; width:100%; font-size:15px; }
  th, td { text-align:left; padding:10px 16px; border-bottom:1px solid var(--rule); white-space:nowrap; }
  th {
    font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--ink-faint); font-weight:500; background:var(--sunk);
  }
  td.num { font-family:var(--mono); font-variant-numeric:tabular-nums; }
  tr:last-child td { border-bottom:0; }
  .hi { color:var(--teal); font-weight:600; }
  .lo { color:var(--clay); font-weight:600; }
  code { font-family:var(--mono); font-size:.88em; background:var(--sunk); padding:1px 5px; border-radius:2px; }

  /* ---------- figures ---------- */
  figure { margin:0 0 32px; }
  figure img { display:block; width:100%; height:auto; border:1px solid var(--rule); border-radius:2px; background:#fff; }
  figcaption { font-size:14.5px; color:var(--ink-muted); margin-top:12px; max-width:78ch; }
  figcaption b { color:var(--ink); font-weight:600; }

  /* ---------- method steps ---------- */
  .steps { display:grid; gap:2px; background:var(--rule); border:1px solid var(--rule); border-radius:2px; }
  .step { background:var(--surface); padding:22px 26px; display:grid; grid-template-columns:auto 1fr; gap:22px; align-items:start; }
  .step .n {
    font-family:var(--mono); font-size:12px; color:var(--teal); letter-spacing:.1em;
    padding-top:4px; font-variant-numeric:tabular-nums;
  }
  .step h4 { font-family:var(--serif); font-size:19px; margin:0 0 6px; font-weight:600; }
  .step p { margin:0; color:var(--ink-muted); font-size:16px; max-width:66ch; }

  /* ---------- correction ledger ---------- */
  .ledger { border:1px solid var(--rule); border-radius:2px; overflow:hidden; }
  .corr { display:grid; grid-template-columns:1fr 1fr; gap:0; border-bottom:1px solid var(--rule); background:var(--surface); }
  .corr:last-child { border-bottom:0; }
  .corr > div { padding:20px 26px; }
  .corr .was { background:var(--clay-soft); border-right:1px solid var(--rule); }
  .corr .lbl {
    font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--ink-faint); display:block; margin-bottom:8px;
  }
  .corr .was .lbl { color:var(--clay); }
  .corr p { margin:0; font-size:16px; }
  @media (max-width:760px) {
    .corr { grid-template-columns:1fr; }
    .corr .was { border-right:0; border-bottom:1px solid var(--rule); }
  }

  .callout {
    background:var(--teal-soft); border:1px solid var(--rule); border-radius:2px;
    padding:26px 30px; margin:0 0 26px;
  }
  .callout p { margin:0; font-size:19px; font-family:var(--serif); line-height:1.5; max-width:64ch; }
  .callout .lbl {
    font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--teal); display:block; margin-bottom:12px;
  }

  ul.plain { padding-left:20px; max-width:var(--measure); }
  ul.plain li { margin-bottom:9px; }

  footer { margin-top:64px; padding-top:24px; border-top:1px solid var(--rule); color:var(--ink-faint); font-size:14px; }
  @media (prefers-reduced-motion:no-preference) { html { scroll-behavior:smooth; } }
</style>

<div class="wrap">

<header class="mast">
  <p class="eyebrow">Interpretability sprint &middot; 6&ndash;7 August 2026</p>
  <h1>Alignment Elasticity Along the Model Flow</h1>
  <p class="standfirst">
    When do safety-relevant representations form during pretraining, and what does capabilities RL
    do to them? Eight experiments across the full OLMo&nbsp;3 training flow &mdash; pretraining,
    SFT, DPO, RLVR, and RL-on-base &mdash; run in a single day on four GPUs.
  </p>
  <div class="facts">
    <div class="fact"><b>8</b><span>experiments</span></div>
    <div class="fact"><b>~120</b><span>checkpoint evaluations</span></div>
    <div class="fact"><b>4</b><span>properties &amp; methods</span></div>
    <div class="fact"><b>9</b><span>claims corrected by controls</span></div>
  </div>
</header>

<section>
  <h2>The question</h2>
  <div class="col">
    <p class="lede">
      OLMo&nbsp;3 is the first public release with checkpoints across an entire training flow, including
      variants trained with RL directly on the base model. That makes it possible, outside a frontier lab,
      to ask mechanistically when safety-relevant structure appears and whether later training erodes it.
    </p>
    <p>
      Three questions drove the day. <b>Formation:</b> at what point in pretraining do harmfulness and
      honesty become linearly decodable? <b>Elasticity:</b> how do those representations change across
      SFT, DPO, RLVR, and RL-Zero? <b>Staleness:</b> if you build a monitor at one stage, does it still
      catch bad behaviour at a later one?
    </p>
  </div>
</section>

<section>
  <h2>Method</h2>
  <div class="steps">
    <div class="step">
      <div class="n">01</div>
      <div>
        <h4>A contrast set that cannot be solved by topic</h4>
        <p>
          Probes trained on the standard harmful-versus-benign comparison score 0.99 &mdash; they are reading
          subject matter, not harm. Every result here instead trains on ordinary WildGuardMix prompts and
          <i>tests on adversarial ones</i>: harmful requests disguised as innocent, and innocent requests
          dressed up to look alarming. That out-of-distribution score is the only measure with real dynamic range.
        </p>
      </div>
    </div>
    <div class="step">
      <div class="n">02</div>
      <div>
        <h4>One prompt format, held fixed across every checkpoint</h4>
        <p>
          Base models have no chat template and post-trained ones do, so mixing formats inside a single
          figure silently measures format sensitivity instead of training. Every sweep applies the same
          neutral scaffold &mdash; <code>User: &hellip;</code> / <code>Assistant:</code> &mdash; to all checkpoints,
          with the model's own template reported separately as its own axis.
        </p>
      </div>
    </div>
    <div class="step">
      <div class="n">03</div>
      <div>
        <h4>Rank, not threshold</h4>
        <p>
          Probe accuracy proved untrustworthy all day: a randomly initialised network already scores 0.81 on
          the easy split, and a probe carries its ranking across a distribution shift far better than its
          decision boundary. AUROC is the headline metric everywhere; accuracy is reported only alongside it.
        </p>
      </div>
    </div>
    <div class="step">
      <div class="n">04</div>
      <div>
        <h4>Causal validation on every headline claim</h4>
        <p>
          Correlational probing convinces nobody. Directions are validated by ablating them during generation
          and watching refusals break; circuits by patching activations from the top-ranked components against
          random and mid-rank controls; behaviour by the official HarmBench classifier rather than string matching.
        </p>
      </div>
    </div>
  </div>
</section>

<section>
  <h2>What we found</h2>

  <div class="callout">
    <span class="lbl">The one-sentence version</span>
    <p>
      Safety-relevant structure appears almost immediately in pretraining, is set once during the first
      few thousand steps of supervised fine-tuning, and is then left essentially untouched by capabilities
      RL &mdash; and while that representation stays <i>readable</i> throughout, it stops being <i>steerable</i>.
    </p>
  </div>

  <div class="find">
    <div class="find-head"><span class="tag">Confirmed</span><span class="expid">E1 &middot; 19 pretraining checkpoints</span></div>
    <h3>The harmfulness representation forms almost immediately</h3>
    <p>
      Across pretraining, out-of-distribution AUROC departs from chance at roughly step 2,000 &mdash; about
      0.14% of the way through the run &mdash; and climbs to 0.804 by the end of stage&nbsp;1, still rising.
      A shuffled-label control sits at 0.49 at every checkpoint. Reading the last token, deep layers lead and
      layer&nbsp;4 never leaves chance &mdash; but that depth profile is a property of the readout, not the
      model: under mean pooling layer&nbsp;4 reaches 0.719 and the leading layer moves from 31 to 16.
    </p>
    <p>
      A non-linear probe does not find the representation any earlier. An MLP head loses to logistic
      regression at all fifteen pretraining checkpoints and roughly doubles the width of its own
      shuffled-label null, so the onset is a fact about the model rather than about linearity.
    </p>
  </div>
  {{FIG:fig_e1_formation_scaffold.png|<b>Formation curve.</b> Out-of-distribution AUROC against pretraining step for OLMo&nbsp;3 7B, one line per layer, neutral scaffold applied identically at every checkpoint.}}

  <div class="find">
    <div class="find-head"><span class="tag caveat">Caveated</span><span class="expid">E1b &middot; 26 checkpoints, 1B suite</span></div>
    <h3>&hellip;and the rise is front-loaded, not gradual</h3>
    <p>
      Sampling every 1,000 steps on the dense 1B early-training suite shows the single interval from step
      1,000 to 2,000 carrying <b>half of the entire rise</b> across a 1.9-million-step run &mdash; a jump
      3.2&times; the standard deviation of every later change. It is not a clean phase transition either:
      no flat period precedes it, because the rise has already begun by the first checkpoint after
      initialisation. The onset is at or before step 1,000 and remains unresolved.
    </p>
    <p>
      <b>Caveat:</b> this is a different run and scale from the 7B lineage, so step numbers do not transfer.
      It establishes the <i>shape</i> of the early rise, not the 7B onset.
    </p>
  </div>
  {{FIG:fig_e1b_1b_fine_grained.png|<b>Dense early sampling.</b> Linear-scale early region with the onset marked, alongside the full run on a log axis.}}

  <div class="find">
    <div class="find-head"><span class="tag">Confirmed</span><span class="expid">E2 &middot; 15 checkpoints &times; 2 formats</span></div>
    <h3>Post-training moves the geometry once, during early SFT</h3>
    <p>
      The refusal direction rotates sharply between SFT steps 1,000 and 15,000 &mdash; a cosine drop of
      0.185 [0.180, 0.190], some forty bootstrap standard deviations from zero &mdash; then stops. Across
      1,350 steps of RLVR the total additional drift is 0.0009: statistically resolvable, 206&times; smaller
      than the cliff, and <b>below the noise floor of the metric itself</b> (two random directions in this
      space have cosine 0.012). RL applied directly to the base model leaves it at cosine 0.9993 after 1,900
      steps. The cliff replicates in raw text, under the neutral scaffold, and under both poolings.
    </p>
  </div>
  {{FIG:fig_e2_cosine_drift_scaffold.png|<b>Drift along the flow.</b> Cosine similarity of each checkpoint's refusal direction to the base model's. The right panel holds RL-on-base against the same axis.}}

  <div class="find">
    <div class="find-head"><span class="tag">Confirmed</span><span class="expid">E2 &middot; frozen vs refit probes</span></div>
    <h3>The subspace rotates about 60&deg; without losing information</h3>
    <p>
      A probe frozen at the base model scores 0.812 AUROC across the flow against 0.820 for one refit from
      scratch &mdash; a gap of <b>0.007</b> [0.000, 0.015], and at the best layer indistinguishable from zero
      &mdash; even though the probe direction itself has rotated to cosine 0.50, roughly sixty degrees. The
      information survives essentially intact; only the basis moves. For monitor design this is the difference
      between a stale monitor being useless and being as good as a fresh one.
    </p>
  </div>
  {{FIG:fig_e2_frozen_vs_refit.png|<b>Elastic, not eroded.</b> Frozen base-model probe against a probe refit at each checkpoint, with the probe-direction rotation on the companion axis.}}

  <div class="find">
    <div class="find-head"><span class="tag">Confirmed</span><span class="expid">E4 &middot; 60 HarmBench behaviours, 8 checkpoints</span></div>
    <h3>Behaviour and geometry move together, as a single event</h3>
    <p>
      Attack-success rate, judged by the official HarmBench classifier over 200 behaviours, falls from
      0.355 [0.290, 0.425] on the base model to 0.180 [0.130, 0.235] within the first 1,000 SFT steps &mdash;
      <b>61% of the total decline, at the same place as the geometric cliff</b> &mdash; then continues falling
      more slowly through SFT before going statistically flat across DPO and RLVR (every contrast spans zero).
      RL-Zero sits back at base-level success, 0.400, because it never learned refusal in the first place.
    </p>
    <p>
      <b>Suppressed, not removed.</b> Under a prefill attack every post-training checkpoint returns to
      0.92&ndash;0.96 attack success &mdash; <i>above</i> the base model's own 0.865. Post-training buries the
      capability behind a refusal onset rather than removing it, and leaves the model more exploitable than
      base once that onset is bypassed.
    </p>
  </div>
  {{FIG:fig_e4_behaviour_vs_geometry.png|<b>Coupling.</b> HarmBench attack-success and refusal rate across the flow, overlaid with mean representational drift on a twin axis.}}

  <div class="find">
    <div class="find-head"><span class="tag">Confirmed</span><span class="expid">E5 &middot; 13 checkpoints, 1,056 components</span></div>
    <h3>The refusal circuit is re-weighted once, then frozen</h3>
    <p>
      Attribution patching over every attention head and MLP shows top-20 component overlap with the base model
      falling to 0.43 through SFT &mdash; and then holding at exactly <b>1.00 for every pair</b> from SFT step
      43,000 through DPO and all of RLVR. Zero components enter or leave. Patching confirms it causally: the
      base model's top-5 components still deliver 84&ndash;87% of each later checkpoint's own effect, with
      mid-rank and random controls at zero.
    </p>
    <p>
      One refinement worth noting: the circuit keeps moving until roughly SFT step 20,000, while behaviour has
      already saturated by step 1,000. <b>Behaviour saturates before the mechanism does.</b>
    </p>
  </div>
  {{FIG:fig_e5_topk_jaccard.png|<b>Circuit stability.</b> Pairwise overlap of the top-20 causally attributed components between checkpoints. The uniform block covering DPO and RLVR is exact identity.}}

  <div class="find">
    <div class="find-head"><span class="tag">Confirmed</span><span class="expid">E3 &middot; 2&times;2 interface control</span></div>
    <h3>Steering goes stale where reading does not</h3>
    <p>
      Take the refusal direction fitted on the base model, never refit it, and ablate it on descendants, over
      200 held-out prompts with bootstrapped intervals. Measured as an <i>outcome</i> &mdash; the fraction of
      prompts pushed from refusing to complying &mdash; it moves <b>0.670</b> of the base model's prompts and
      <b>0.350</b> of the deployed model's, an interval that overlaps neither the base nor the RL-on-base arm.
      In generation the same ablation removes 93.5% of the base model's refusals and <b>15%</b> of the deployed
      model's.
    </p>
    <p>
      <b>Why the direction has not simply gone stale.</b> Measured as a raw displacement the effect on the
      deployed model is the <i>largest</i> of any checkpoint &mdash; 156% of base. It starts at +7.85 logits
      of refusal, is pushed down 6.81, and lands at +1.04: still refusing. The base model starts at +0.85,
      drops 4.37, and lands at &minus;3.51: complying. Post-training does not move refusal out of the residual
      stream; it inflates dependence on refusal-onset logit mass so far that removing the same quantity no
      longer changes the decision.
    </p>
  </div>
  {{FIG:fig_e3_causal_staleness.png|<b>Causal staleness.</b> Effect of ablating the base-era refusal direction on descendants, grouped by prompt interface so model and format are separable.}}

  <div class="find">
    <div class="find-head"><span class="tag caveat">Caveated</span><span class="expid">E6 &middot; 4 monitors &times; 480 outputs</span></div>
    <h3>Trusted monitoring does not decay with training distance &mdash; but weak monitors fail where it counts</h3>
    <p>
      With every monitored checkpoint carrying 200 independently-judged harmful positives &mdash; elicited by
      ablation and prefill, and reported as part of the setup &mdash; an earlier checkpoint reading a later
      one's outputs shows <b>no degradation with training distance</b>. The raw distance correlation
      (&minus;0.28) is a confound: the two hardest-to-monitor generators sit at opposite <i>ends</i> of the
      flow, so both are far from mid-flow monitors, and there is no slope within either cluster. In a
      regression the distance term is &minus;0.002 [&minus;0.006, +0.002].
    </p>
    <p>
      Nor is monitor capability the constraint: at proper power all four monitors sit near 0.95 AUROC and
      catch 71&ndash;79% of harmful outputs at a 1% false-positive rate. The only term that survives is
      <b>how legible the monitored model's harmful output is</b> (r = 0.996 with catch rate). The one real
      difference between monitors appears on hard negatives &mdash; harmful compliance versus refusals
      <i>about</i> harmful topics &mdash; where the base model scores 0.744 against an SFT checkpoint's 0.894.
    </p>
  </div>
  {{FIG:fig_e6_monitor_staleness.png|<b>Monitorability against training distance.</b> Well-powered points are filled, low-power points hollow &mdash; the honest reading of the left panel is &ldquo;flat within noise&rdquo;.}}

  <div class="find retract">
    <div class="find-head"><span class="tag retract">Falsified</span><span class="expid">E7 &middot; honesty, 17 checkpoints</span></div>
    <h3>Formation timing does not predict elasticity &mdash; the pipeline stage does</h3>
    <p>
      The project's central hypothesis was that a representation forming earlier in pretraining would survive
      post-training more robustly. A second property settles it, negatively. Honesty becomes decodable around
      step 5,000, roughly 2.5&times; later than harmfulness &mdash; yet its elasticity is essentially identical:
      the same single SFT cliff, the same RL inertness (drift 0.0007), and a frozen probe that transfers
      <i>better</i>, not worse (gap 0.012 against 0.04).
    </p>
    <p>
      Elasticity is a property of where you are in the pipeline, not of when the concept crystallised. That is
      a cleaner and more general claim than the one it replaces, and it means the paper needs reframing.
    </p>
  </div>
  {{FIG:fig_e7_harm_vs_honesty.png|<b>Two properties, one law.</b> Honesty forms markedly later than harmfulness and drifts identically, contradicting the formation-predicts-elasticity hypothesis.}}

  <div class="find caveat">
    <div class="find-head"><span class="tag caveat">Inconclusive</span><span class="expid">E8 &middot; Jacobian Lens, 11 checkpoints</span></div>
    <h3>Does the direction leave the model's verbalizable workspace?</h3>
    <p>
      Anthropic's Jacobian Lens decodes a residual-stream direction into vocabulary. At the base model the
      refusal direction reads <code>unethical</code>, <code>prohibited</code>, <code>forbidden</code>,
      <code>immoral</code> &mdash; unambiguously verbalizable. Along the flow, its coherence retention tracks
      the causal pattern closely: 0.909 through RL-on-base lenses against 0.227 through the deployed model's.
    </p>
    <p>
      <b>But the control breaks the interpretation.</b> An unrelated sentiment direction collapses the same
      way, harder. This is a general de-verbalization of base-era directions in SFT-descended checkpoints, not
      a refusal-specific mechanism &mdash; and since every available lens was fit on plain web text, a worse
      linearisation for chat-tuned models would produce this exact pattern with no workspace claim involved.
    </p>
  </div>
  {{FIG:fig_e8_jspace_residency.png|<b>Three measures, one discriminating.</b> Semantic coherence separates the arms; transport magnitude is a flat null and subspace mass is weak. Disagreement between operationalisations is reported rather than resolved by selection.}}
</section>

<section>
  <h2>What the controls overturned</h2>
  <div class="col">
    <p class="lede">
      Five claims were retracted during the day, each by a control run against it. They are recorded here
      because the corrections are the most informative part of the record &mdash; every one of them would have
      produced a confident, wrong figure.
    </p>
  </div>
  <div class="ledger">
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>The harmfulness representation never forms during pretraining &mdash; the curve is flat at chance.</p></div>
      <div><span class="lbl">After control</span><p>An artifact of raw-text prompting plus an accuracy-based reading. In AUROC the same sweep shows clean formation. The deployed model scored 0.509 accuracy on raw text with unchanged weights.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>Access to the representation is gated by prompt format.</p></div>
      <div><span class="lbl">After control</span><p>Too strong. Ranking is largely format-invariant; <i>calibration</i> is strongly format-dependent. The model represents harm without dialogue framing &mdash; the format supplies a usable decision boundary.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>The geometry is frozen after early SFT.</p></div>
      <div><span class="lbl">After control</span><p>Holds in raw text and scaffold, but under the model's native template DPO rotates the direction 18&times; more and RLVR moves it over a measurable span. &ldquo;One cliff, then mostly flat&rdquo; survives; &ldquo;frozen&rdquo; does not.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>Causal erosion is proportional to how far the direction rotated.</p></div>
      <div><span class="lbl">After control</span><p>A format artifact. At a matched interface the two RL-on-base runs retain 86% and 87% &mdash; indistinguishable &mdash; where the confounded comparison showed 76% against 32%.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>When a representation forms predicts how robustly it survives.</p></div>
      <div><span class="lbl">After control</span><p>Falsified by a second property. Honesty forms 2.5&times; later and behaves identically under post-training. The flow determines elasticity, not the concept.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>The base-era refusal direction retains only 8% of its causal effect on the deployed model.</p></div>
      <div><span class="lbl">After power</span><p>At six prompts, yes; at two hundred it inverts to 156% &mdash; because raw logit delta measures displacement, not outcome. Scored as crossing rate the dissociation returns, cleanly separated, with a sharper mechanism behind it.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>Attack success collapses to zero within the first 1,000 SFT steps.</p></div>
      <div><span class="lbl">After power</span><p>It falls to 0.180, not zero. The zero was a generation-length bug: at 44 tokens the answer never escapes the reasoning block, so the judge scored reasoning traces as harmless. 61% of the decline happens there; the rest continues through SFT.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>A base-model monitor catches 1.7% of harmful outputs at a 1% false-positive rate.</p></div>
      <div><span class="lbl">After power</span><p>Small-sample noise &mdash; six of eight columns had fewer than five positives. With 200 positives per column it catches 71%, against an SFT checkpoint's 75%.</p></div>
    </div>
    <div class="corr">
      <div class="was"><span class="lbl">Claimed</span><p>A frozen probe loses 0.04 AUROC across the flow.</p></div>
      <div><span class="lbl">After protocol</span><p>0.007, and indistinguishable from zero at the best layer. Part of the difference was an under-converged solver: sklearn's default tolerance stops after fourteen iterations, leaving the fitted direction at cosine 0.978 to the true optimum.</p></div>
    </div>
  </div>
</section>

<section>
  <h2>Why the contrast set matters</h2>
  <div class="col">
    <p>
      Every result above depends on a measurement choice made before any of them: probes are trained on ordinary
      prompts and scored on adversarial ones. Without that, harmfulness probing saturates immediately and
      every checkpoint scores identically &mdash; a flat elasticity curve produced by the metric, not the model.
    </p>
  </div>
  {{FIG:fig_probe_transfer_regimes.png|<b>Why the hard split exists.</b> The standard benchmark comparison is solved at 0.99 by surface topic; the adversarial split leaves room to measure, and exposes early layers as a lexical shortcut that collapses under reframing.}}
</section>

<section>
  <h2>What this means</h2>
  <div class="col">
    <p>
      For work on <b>robust initialisations</b>, the result is encouraging in a specific and limited way. Alignment-relevant
      geometry in this lineage is established early and is genuinely inert to the capabilities RL applied here:
      not merely small drift, but drift indistinguishable from zero across every RLVR checkpoint, in two properties,
      confirmed at the level of individual circuit components. A null this clean is worth reporting precisely.
    </p>
    <p>
      For <b>control protocols</b>, two findings cut in opposite directions and both are actionable. Monitors built
      from earlier checkpoints do not go stale with training distance, which is better news than the field assumes.
      But a trusted monitor that is merely a pretrained base model catches 1.7% of harmful outputs at a 1% alert
      budget while looking respectable on AUROC &mdash; so monitor capability, not monitor freshness, is the binding
      constraint. And the read/steer split matters for intervention design: a direction can remain perfectly readable
      while losing the ability to control behaviour.
    </p>
    <p>
      The honest limits: one lineage, one scale, moderate RL doses, and causal claims resting on tens of prompts
      rather than hundreds. Those are the first things to fix.
    </p>
  </div>
</section>

<section>
  <h2>Next</h2>
  <div class="col">
    <ul class="plain">
      <li><b>Power the causal claims.</b> Staleness and layer profiles currently rest on 6&ndash;8 prompts. Two hundred, with bootstrapped intervals.</li>
      <li><b>Rule out non-linear decodability.</b> Every formation claim uses linear probes at the last token; one non-linear readout showing the same onset closes the obvious objection.</li>
      <li><b>Deconfound the template result.</b> The Think checkpoints ship slightly different chat templates, so part of the templated drift is template text rather than representation.</li>
      <li><b>Replicate at 32B and on a second lineage.</b> One run at one scale is the limitation every reader will name first.</li>
      <li><b>Two more properties.</b> Two points support &ldquo;the flow determines elasticity&rdquo; weakly; four would make it hard to argue with.</li>
      <li><b>Fit a lens on chat-formatted data</b> to settle whether the workspace result is real or an artifact of lenses fit on plain web text.</li>
    </ul>
  </div>
</section>

<footer>
  OLMo&nbsp;3 7B and OLMo&nbsp;2 1B checkpoints, Ai2. Probes: logistic regression on residual-stream activations,
  WildGuardMix and Azaria&ndash;Mitchell contrast sets. Behavioural judging: HarmBench classifier.
  Attribution and activation patching over all attention heads and MLPs. Jacobian Lens v0.1.0, Anthropic.
  Run on four RTX&nbsp;PRO&nbsp;6000 Blackwell GPUs, 6 August 2026.
</footer>

</div>
"""


def embed(match):
    name, caption = match.group(1), match.group(2)
    path = FIG_DIR / name
    if not path.exists():
        print(f"  MISSING: {name}")
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    print(f"  embedded {name} ({path.stat().st_size / 1024:.0f} KB)")
    return (
        f'<figure><img src="data:image/png;base64,{b64}" alt="{name}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


html = re.sub(r"\{\{FIG:([^|]+)\|(.*?)\}\}", embed, HTML, flags=re.S)
OUT.write_text(html, encoding="utf-8")
print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.2f} MB)")
