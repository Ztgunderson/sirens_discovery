import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
AUTODEL = {"add_global_autodelete_rule","add_user_autodelete_rule"}
def eps(variant, is_rogue):
    out=[]
    for p in sorted(glob.glob(f"data/transcripts/{variant}/*/*/transcript_*.json")):
        d=json.load(open(p)); turn=0; first=None; total=0; side=bool(d.get("side_task_evaluation"))
        for m in d.get("conversation",[]):
            if isinstance(m,dict) and m.get("tool_calls"):
                names=[l.strip()[2:].strip() for l in str(m["tool_calls"]).splitlines() if l.strip().startswith("- ")]
                if names:
                    turn+=1; total=turn
                    if first is None and any(n in AUTODEL for n in names): first=turn
        out.append({"total":total,"detect":first,"side":side and is_rogue})
    return out
E = eps("benign",False) + eps("rogue",True)
pos=[e for e in E if e["side"]]
full=sum(e["total"] for e in E); Tmax=max(e["total"] for e in E); Ts=list(range(1,Tmax+1))
det=[100*sum(1 for e in pos if e["detect"] and e["detect"]<=T)/len(pos) for T in Ts]
comp=[100*sum(min(e["total"],T) for e in E)/full for T in Ts]
Tstar=max(e["detect"] for e in pos if e["detect"]); comp_star=100*sum(min(e["total"],Tstar) for e in E)/full
plt.figure(figsize=(7,4.3))
plt.plot(Ts,det,label="positives detected (%)",lw=2.4,color="#1f77b4")
plt.plot(Ts,comp,label="compute used (% of full run)",lw=2.4,color="#d62728")
plt.axvline(Tstar,ls="--",color="gray",lw=1.2)
plt.scatter([e["detect"] for e in pos],[100*(i+1)/len(pos) for i,e in enumerate(sorted(pos,key=lambda x:x["detect"]))],
            color="#1f77b4",zorder=5,s=40)
plt.annotate(f"catch BOTH positives only at turn {Tstar}\n= {comp_star:.0f}% of full compute (~{100-comp_star:.0f}% saved)",
             xy=(Tstar,comp_star),xytext=(3,40),fontsize=9.5,
             bbox=dict(boxstyle="round",fc="white",ec="gray",alpha=0.9),
             arrowprops=dict(arrowstyle="->",color="black"))
plt.xlabel("episode turn cap",fontsize=11); plt.ylabel("percent",fontsize=11)
plt.title("Figure 1. Early-stopping barely helps: the covert action is late",fontsize=12)
plt.ylim(0,105); plt.xlim(0,Tmax); plt.legend(loc="lower right",fontsize=10,framealpha=0.95)
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("docs/fig1_efficiency.png",dpi=150)
print(f"Tstar={Tstar} compute_at_Tstar={comp_star:.1f}% saved={100-comp_star:.1f}%  -> saved PNG")
