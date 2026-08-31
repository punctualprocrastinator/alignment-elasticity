import json, numpy as np, io, re

d = json.load(open('results/gateA_gen_instruct.json'))
pr, ut, st = d['prompts'], d['unsteered_text'], d['steered_text']
ur, sr = d['unsteered_refusal'], d['steered_refusal']
flip = [i for i in range(len(pr)) if ur[i] and not sr[i]]
rng = np.random.default_rng(42)
samp = sorted(rng.choice(flip, size=6, replace=False).tolist())

ESC = [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
       ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
       ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]

def esc(t):
    t = re.sub(r'\s+', ' ', t).strip()
    for a, b in ESC:
        t = t.replace(a, b)
    return t

def clip(t, n):
    return (t[:n] + r'\ldots') if len(t) > n else t

L = []
L.append('% Auto-generated from results/gateA_gen_instruct.json (seed 42). Do not hand-edit.')
L.append(r'\section*{Appendix B\quad Randomly selected aligned-model completions}')
L.append(r'\addcontentsline{toc}{section}{Appendix B}')
L.append('')
L.append("Six examples drawn at random (seed 42) from the OLMo-3-7B-Instruct prompts whose refusal")
L.append(r"\emph{onset} flipped under the base-model refusal direction at the model's own crossing dose")
L.append(r"($c=-1.22$), out of the 82 of 100 held-out harmful prompts where the onset flipped. The sample")
L.append("is not curated: it is the first six indices the seeded generator returns. Unsteered, the model")
L.append("refuses every prompt; steered, the onset flips but the continuation degenerates or deflects,")
L.append(r"producing no genuine harm. This is the \S6 dissociation at the level of individual generations")
L.append("(aggregate HarmBench harm at this dose: 0.05).")
L.append('')
L.append(r'\begin{table}[h]\centering\footnotesize')
L.append(r'\renewcommand{\arraystretch}{1.3}')
L.append(r'\begin{tabular}{p{0.30\linewidth} p{0.28\linewidth} p{0.34\linewidth}}')
L.append(r'\toprule')
L.append(r'Harmful prompt & Unsteered onset & Steered onset (flipped) \\')
L.append(r'\midrule')
for i in samp:
    p = esc(clip(pr[i], 52)); u = esc(clip(ut[i], 44)); s = esc(clip(st[i], 60))
    L.append('``%s\'\' & ``%s\'\' & ``%s\'\' \\\\' % (p, u, s))
L.append(r'\bottomrule')
L.append(r'\end{tabular}')
L.append(r'\end{table}')
L.append('')
L.append(r"Generated at 512 new tokens, greedy; full texts in \texttt{results/gateA\_gen\_instruct.json}.")
L.append("The steered completions are degenerate or evasive rather than harmful, consistent with the")
L.append(r"HarmBench-judged genuine-harm rate of 0.05 at this dose (Figure~\ref{fig:onset}).")

io.open('paper-boundary/appendix-completions.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('wrote appendix-completions.tex; sampled', samp)
