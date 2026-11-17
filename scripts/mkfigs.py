import json,numpy as np,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size":7,"font.family":"serif"})
R="results/";O="figs/";import os;os.makedirs(O,exist_ok=True)
e2=json.load(open(R+"e2_diag_withqwen.json"))
print(list(e2))
names=[("magnitude_only","Magnitude only"),("clip_l14","CLIP"),("siglip_so400m","SigLIP"),("dinov2_l","DINOv2"),("qwen25vl_vis","Qwen2.5-VL vis.")]
names=[n for n in names if n[0] in e2]
lab=["Mat.","Size","Ratio","Dir.","Kelvin"]
fig,ax=plt.subplots(1,len(names),figsize=(7,1.7))
for a,(k,t) in zip(ax,names):
    cm=np.array(e2[k]["5way"]["cm"]);a.imshow(cm,vmin=0,vmax=1,cmap="Blues")
    for i in range(5):
        for j in range(5):a.text(j,i,f"{cm[i,j]:.2f}"[1:] if cm[i,j]<1 else "1.0",ha="center",va="center",fontsize=5,color="w" if cm[i,j]>.5 else "k")
    a.set_xticks(range(5));a.set_xticklabels(lab,rotation=45,fontsize=5.5);a.set_yticks(range(5));a.set_yticklabels(lab if a is ax[0] else [],fontsize=5.5)
    a.set_title(f"{t} ({e2[k]['5way']['bal_acc']:.3f})",fontsize=6.5)
ax[0].set_ylabel("true change");fig.text(.5,.0,"predicted change",ha="center")
plt.tight_layout();plt.savefig(O+"cm.pdf",bbox_inches="tight")
# k curve
P=json.load(open(R+"xfer_proj_withqwen.json"));ov=json.load(open(R+"subspace_overlap.json"))
enc=[("clip_l14","CLIP"),("siglip_so400m","SigLIP"),("dinov2_l","DINOv2"),("qwen25vl_vis","Qwen vis.")]
fig,ax=plt.subplots(1,2,figsize=(3.4,1.5))
for k,t in enc:
    c=P[k]["xfer_single"]["heldout_curve"];ks=sorted(map(int,c));
    ax[0].plot(range(len(ks)),[c[str(x)]["mae"] for x in ks],marker="o",ms=2,label=t)
ax[0].set_xticks(range(len(ks)));ax[0].set_xticklabels(ks,fontsize=5.5);ax[0].set_xlabel("$k$");ax[0].set_ylabel("held-out MAE");ax[0].axhline(1.6,ls=":",c="gray",lw=.6);ax[0].legend(fontsize=5,frameon=False)
for i,(k,t) in enumerate(enc[:3]):
    ks=sorted(map(int,ov[k]));col=f"C{i}"
    ax[1].plot(ks,[ov[k][str(x)]["material"] for x in ks],c=col,marker="o",ms=2,label=t)
    ax[1].plot(ks,[ov[k][str(x)]["random"] for x in ks],c=col,ls=":",lw=.7)
ax[1].set_xscale("log",base=2);ax[1].set_xlabel("$k$");ax[1].set_ylabel("material energy share");ax[1].legend(fontsize=5,frameon=False)
plt.tight_layout();plt.savefig(O+"kcurve.pdf",bbox_inches="tight")
# teaser
D="renders_diag/";fs=[("00264_pink_velvet_sphere_0_base1.png","base"),("00266_pink_velvet_sphere_0_material2.png","material: sheen $\\uparrow$"),("00270_pink_velvet_sphere_0_ratio2.png","lighting: key/fill $\\uparrow$"),("00274_pink_velvet_sphere_0_kelvin2.png","lighting: Kelvin $\\uparrow$")]
fig,ax=plt.subplots(1,4,figsize=(3.4,1.05))
for a,(f,t) in zip(ax,fs):a.imshow(Image.open(D+f));a.set_title(t,fontsize=5.5);a.axis("off")
plt.tight_layout(pad=.2);plt.savefig(O+"teaser.pdf",bbox_inches="tight",dpi=300)
print("ok")
