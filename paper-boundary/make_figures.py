"""Regenerate all paper figures from results JSON, one shared style.
Palette: dataviz validated categorical set (CVD- and normal-vision-checked).
Design: minimal in-figure annotation; captions carry the explanation.
"""
import json, os, numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "results"; OUT = "paper-boundary/figures"
os.makedirs(OUT, exist_ok=True)

# --- validated categorical palette (light mode) ---
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; YELLOW="#eda100"; MAGENTA="#e87ba4"
INK="#0b0b0b"; INK2="#52514e"; MUT="#9a998f"; GRID="#e7e6e1"; SURF="#ffffff"
# semantic, consistent across figures
C_MARGIN=ORANGE      # the load that grows
C_LEVER=BLUE         # the lever that stays flat
C_REFUSAL=BLUE; C_HONESTY=AQUA
FAM={"OLMo-3":BLUE,"Qwen3-8B":ORANGE,"Llama-3.1-8B":AQUA,"Gemma-2-9B":YELLOW}

mpl.rcParams.update({
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "font.family":"DejaVu Sans","font.size":11,
    "axes.edgecolor":INK2,"axes.linewidth":0.8,"axes.labelcolor":INK,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.color":INK2,"ytick.color":INK2,"text.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,
    "legend.frameon":False,"legend.fontsize":9.5,
})
def load(p): return json.load(open(os.path.join(R,p)))
def save(fig,name):
    fig.savefig(os.path.join(OUT,name),dpi=200,bbox_inches="tight")
    plt.close(fig); print("wrote",name)

def efficacy_margin(sweep):
    g=np.array(sweep["gaps_massmean"]); c=np.array(sweep["c_grid_massmean"]); zi=sweep["zero_index_massmean"]
    g0=g[zi]; absc=np.abs(c); near=absc<=2.0; disp=(g0[None,:]-g).mean(axis=1)
    A=np.vstack([absc[near],np.ones(near.sum())]).T
    return float(g0.mean()), float(np.linalg.lstsq(A,disp[near],rcond=None)[0][0])

def sweep(label):
    for p in [f"gateA_traj/gateA_sweep_{label}.json", f"gateA_sweep_{label}.json"]:
        if os.path.exists(os.path.join(R,p)): return load(p)

# ============================ FIGURE 1: three instruments, three rankings
def fig1():
    models=["base","Instruct","RL-Zero Math","RL-Zero Code"]
    keys=["base","instruct","rlz-math","rlz-code"]
    col={"base":MUT,"Instruct":ORANGE,"RL-Zero Math":MUT,"RL-Zero Code":MUT}
    inst=["fixed\nablation","fixed-dose\nsteering","per-dose\nefficacy"]
    vals={m:[] for m in models}
    raw={}
    for m,k in zip(models,keys):
        d=sweep(k); g=np.array(d["gaps_massmean"]); c=np.array(d["c_grid_massmean"]); zi=d["zero_index_massmean"]
        ai=int(np.argmin(np.abs(c-(-1.0)))); ci=int(np.argmin(np.abs(c-(-0.5))))
        fabl=float((g[ai]<0).mean()); fdose=float((g[ci]<0).mean()); _,eff=efficacy_margin(d)
        raw[m]=[fabl,fdose,eff]
    # min-max normalise WITHIN each instrument so the three are comparable on one axis
    arr=np.array([raw[m] for m in models])  # [4,3]
    norm=(arr-arr.min(0))/(arr.max(0)-arr.min(0)+1e-9)
    fig,ax=plt.subplots(figsize=(6.2,4.4))
    x=[0,1,2]
    for i,m in enumerate(models):
        lw=2.6 if m=="Instruct" else 1.6
        z=3 if m=="Instruct" else 1
        ax.plot(x,norm[i],"-o",color=col[m],lw=lw,ms=7 if m=="Instruct" else 5,zorder=z,
                markerfacecolor=col[m],markeredgecolor=SURF,markeredgewidth=1.2)
        ax.annotate(m,(x[-1],norm[i][-1]),xytext=(8,0),textcoords="offset points",
                    va="center",fontsize=9.5,color=col[m] if m=="Instruct" else INK2,
                    fontweight="bold" if m=="Instruct" else "normal")
    ax.set_xticks(x); ax.set_xticklabels(inst); ax.set_xlim(-0.25,2.85)
    ax.set_ylim(-0.08,1.12)
    ax.set_ylabel("steerability (normalised within instrument)")
    ax.set_yticks([0,1]); ax.set_yticklabels(["least","most"])
    save(fig,"fig1_three_instruments.png")

# ============================ FIGURE 2: lever flat, load grows (fold-change, ONE axis)
def fig2():
    order=[("base","base"),("SFT 1k","think-sft-1000"),("SFT 15k","think-sft-15000"),
           ("SFT 43k","think-sft-43000"),("DPO","think-dpo"),
           ("RLVR first","think-rlvr-first"),("RLVR last","think-rlvr-last"),
           ("Instruct","instruct")]
    labels=[l for l,_ in order]; M=[];E=[]
    for _,k in order:
        m,e=efficacy_margin(sweep(k)); M.append(m); E.append(e)
    M=np.array(M); E=np.array(E)
    fig,ax=plt.subplots(figsize=(6.6,4.3))
    x=np.arange(len(labels))
    ax.plot(x,M/M[0],"-o",color=C_MARGIN,lw=2.4,ms=6,markeredgecolor=SURF,markeredgewidth=1.2,label="behavioural margin (load)")
    # per-checkpoint efficacy 95% CIs (bootstrap over prompts), in fold-change units
    try:
        tost=load("efficacy_tost.json"); ef=tost["efficacy"]
        keymap={"base":"base","SFT 1k":"SFT 1k","SFT 15k":"SFT 15k","SFT 43k":"SFT 43k",
                "DPO":"DPO","RLVR first":"RLVR first","RLVR last":"RLVR last","Instruct":"Instruct"}
        elo=np.array([ef[keymap[l]]["ci"][0] for l in labels])/E[0]
        ehi=np.array([ef[keymap[l]]["ci"][1] for l in labels])/E[0]
        yerr=np.vstack([E/E[0]-elo, ehi-E/E[0]])
        ax.errorbar(x,E/E[0],yerr=yerr,fmt="-s",color=C_LEVER,lw=2.4,ms=6,capsize=3,
                    markeredgecolor=SURF,markeredgewidth=1.2,label="steering efficacy (lever), 95% CI")
    except Exception:
        ax.plot(x,E/E[0],"-s",color=C_LEVER,lw=2.4,ms=6,markeredgecolor=SURF,markeredgewidth=1.2,label="steering efficacy (lever)")
    ax.axhline(1,color=MUT,lw=0.8,ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("fold change vs base")
    ax.set_ylim(0,10.5)
    ax.axvspan(6.5,7.5,color="#eda100",alpha=0.06)
    ax.legend(loc="upper left")
    save(fig,"fig2_lever_vs_load.png")

# ============================ FIGURE 3: margin universal (4 families) + honesty control
def fig3():
    # left: four-family margin fold-change base->instruct
    # format-matched (neutral scaffold) margin fold-change, base -> instruct, all families
    def nfold(f):
        m=load(f)["margins"]; return m["instruct-neutral"]/m["base-neutral"]
    om,_=efficacy_margin(sweep("base")); im,_=efficacy_margin(sweep("instruct"))
    fams=[("OLMo-3",im/om),("Qwen3-8B",nfold("E5_second_family.json")),
          ("Llama-3.1-8B",nfold("E5_llama.json")),("Gemma-2-9B",nfold("E5_gemma.json"))]
    # right: refusal vs honesty margin fold (OLMo), lever flat both
    hon=load("E4_honesty.json"); hg=hon["margin_growth_base_to_instruct"]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,4.3),gridspec_kw={"width_ratios":[1.25,1]})
    # panel a
    names=[f for f,_ in fams]; folds=[v for _,v in fams]; cols=[FAM[n] for n in names]
    y=np.arange(len(names))[::-1]
    a1.barh(y,folds,color=cols,height=0.62,edgecolor=SURF,linewidth=1.5)
    for yi,v in zip(y,folds): a1.annotate(f"{v:.1f}x",(v,yi),xytext=(5,0),textcoords="offset points",va="center",fontsize=9.5,color=INK2)
    a1.set_yticks(y); a1.set_yticklabels(names); a1.set_xlim(0,11)
    a1.set_xlabel("margin growth base -> instruct"); a1.axvline(1,color=MUT,lw=0.8,ls=":")
    a1.set_title("margin grows in every family",fontsize=11,color=INK,pad=8)
    # panel b: refusal vs honesty (OLMo) margin fold
    b_names=["refusal","honesty"]; b_folds=[im/om,hg]; b_cols=[C_REFUSAL,C_HONESTY]
    yb=np.arange(2)[::-1]
    a2.barh(yb,b_folds,color=b_cols,height=0.5,edgecolor=SURF,linewidth=1.5)
    for yi,v in zip(yb,b_folds): a2.annotate(f"{v:.1f}x",(v,yi),xytext=(5,0),textcoords="offset points",va="center",fontsize=9.5,color=INK2)
    a2.set_yticks(yb); a2.set_yticklabels(b_names); a2.set_xlim(0,11)
    a2.set_xlabel("margin growth (OLMo-3)"); a2.axvline(1,color=MUT,lw=0.8,ls=":")
    a2.set_title("the control concept",fontsize=11,color=INK,pad=8)
    save(fig,"fig3_margin_universal_honesty.png")

# ============================ FIGURE 4: onset-control != harm-control
def fig4():
    e2=load("E2_behavioural_dose.json")
    cps=e2["checkpoints"]; xlab=["base","SFT 1k","DPO","RLVR last","Instruct"]
    x=np.arange(len(cps))
    # refusal_endpoints[c] = [unsteered_refusal, steered_refusal]; onset flipped = drop
    onset=[e2["refusal_endpoints"][c][0]-e2["refusal_endpoints"][c][1] for c in cps]
    # genuine harm = peak HarmBench harmful rate across doses at each checkpoint
    harm=[max(d["harmbench_harmful_rate"] for d in e2["harmbench"][c]) for c in cps]
    corr=[e2["harmbench_vs_prefix_corr"][c] for c in cps]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,4.3))
    # panel a: onset steerability (stays high) vs genuine harm (decays)
    a1.plot(x,onset,"-o",color=C_LEVER,lw=2.4,ms=6,markeredgecolor=SURF,markeredgewidth=1.2,label="refusal-onset flipped")
    a1.plot(x,harm,"-s",color=C_MARGIN,lw=2.4,ms=6,markeredgecolor=SURF,markeredgewidth=1.2,label="genuine harm (HarmBench)")
    a1.set_xticks(x); a1.set_xticklabels(xlab,rotation=20,ha="right"); a1.set_ylim(0,1.05)
    a1.set_ylabel("rate at crossing dose"); a1.legend(loc="center left")
    a1.set_title("onset flips; harm does not follow",fontsize=11,color=INK,pad=8)
    # panel b: prefix-vs-harm correlation flips sign
    corr=np.array(corr,dtype=float)
    cbar=[C_LEVER if v>=0 else C_MARGIN for v in corr]
    a2.axhline(0,color=INK2,lw=1)
    a2.bar(x,corr,color=cbar,width=0.6,edgecolor=SURF,linewidth=1.5)
    a2.set_xticks(x); a2.set_xticklabels(xlab,rotation=20,ha="right"); a2.set_ylim(-1,1.05)
    a2.set_ylabel("corr(prefix-refusal, true harm)")
    a2.set_title("prefix metric inverts",fontsize=11,color=INK,pad=8)
    save(fig,"fig4_onset_vs_harm.png")

for f in (fig1,fig2,fig3,fig4):
    try: f()
    except Exception as e:
        import traceback; print("ERR in",f.__name__); traceback.print_exc()
print("done")
