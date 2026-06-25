"""Build the stacked AIdea submission: apply T2 evidence head + T3 clarity head
overlays onto aidea_rt_all12_knn.csv (12-way). Mirrors combined_overlay.py logic
on the 2000 test, using saved test probs."""
import numpy as np, pandas as pd
base=pd.read_csv("official_sub/aidea_rt_all12_knn.csv",keep_default_na=False)
test=pd.read_csv("final_data/vpesg4k_test_2000.csv",keep_default_na=False)
assert base["id"].tolist()==test["id"].tolist()
N=len(base)
T2H=["Yes","No"]; t2=np.load("agent_cache/t2_test_probs.npz")["probs"]; t2a=t2.argmax(1); t2c=t2.max(1)
T3H=["Clear","Not Clear","Misleading"]; cp=np.load("agent_cache/clarity_test_probs.npz")["probs"]; ca=cp.argmax(1); cc=cp.max(1)
t1=base["promise_status"].tolist(); t2b=base["evidence_status"].tolist(); t3b=base["evidence_quality"].tolist(); t4=base["verification_timeline"].tolist()
T2THR,T3THR=0.8,0.7
nt2=nt3=0
o2=list(t2b); o3=list(t3b)
for i in range(N):
    if t1[i]=="Yes" and t2b[i] in T2H and t2c[i]>=T2THR and T2H[t2a[i]]!=t2b[i]:
        o2[i]=T2H[t2a[i]]; nt2+=1
for i in range(N):
    if o2[i]=="Yes" and o3[i] in T3H and cc[i]>=T3THR and T3H[ca[i]]!="Misleading" and T3H[ca[i]]!=o3[i]:
        o3[i]=T3H[ca[i]]; nt3+=1
# cascade re-enforce
for i in range(N):
    if t1[i]=="No": o2[i]="N/A"; o3[i]="N/A"; t4[i]="N/A"
    elif o2[i]=="No": o3[i]="N/A"
sub=base.copy(); sub["evidence_status"]=o2; sub["evidence_quality"]=o3; sub["verification_timeline"]=t4
sub.to_csv("official_sub/aidea_stacked.csv",index=False)
from collections import Counter
print(f"[combine] T2 changed {nt2}, T3 changed {nt3}")
print(f"  T2 dist: {dict(Counter(o2))}")
print(f"  T3 dist: {dict(Counter(o3))}  (12-way base T3: {dict(Counter(t3b))})")
print(f"-> official_sub/aidea_stacked.csv")
