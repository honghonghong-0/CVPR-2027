import json,numpy as np,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,seaborn as sns
from PIL import Image
sns.set_theme(style="ticks",context="paper",font="serif",rc={"font.size":7,"axes.titlesize":7,"axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":6,"axes.linewidth":.6,"xtick.major.width":.6,"ytick.major.width":.6})
R="results/";O="figs/"
pal=dict(zip(["CLIP","SigLIP","DINOv2","Qwen vis."],sns.color_palette("Set2",4)))
e2=json.load(open(R+"e2_diag_withqwen.json"))
names=[("magnitude_only","Magnitude only"),("clip_l14","CLIP"),("siglip_so400m","SigLIP"),("dinov2_l","DINOv2"),("qwen25vl_vis","Qwen2.5-VL vis.")]
lab=["Mat.","Size","Ratio","Dir.","Kelvin"]
fig,ax=plt.subplots(1,5,figsize=(7,1.75),gridspec_kw={"wspace":.12})
cmap=sns.light_palette("#3b6ea8",as_cmap=True)
for a,(k,t) in zip(ax,names):
    cm=np.array(e2[k]["5way"]["cm"])
    annot=np.vectorize(lambda v:("%.2f"%v)[1:] if v<1 else "1.0")(cm)
    sns.heatmap(cm,ax=a,vmin=0,vmax=1,cmap=cmap,annot=annot,fmt="",cbar=False,square=True,linewidths=.5,linecolor="white",annot_kws={"fontsize":5.2},xticklabels=lab,yticklabels=lab if a is ax[0] else False)
    a.add_patch(plt.Rectangle((0,0),5,1,fill=False,ec="#c0392b",lw=1))  # material row
    a.set_title(f"{t}  ({e2[k]['5way']['bal_acc']:.3f})",pad=3)
    a.tick_params(length=0);plt.setp(a.get_xticklabels(),rotation=40,ha="right")
ax[0].set_ylabel("true change");fig.supxlabel("predicted change",fontsize=7,y=-.08)
plt.savefig(O+"cm.pdf",bbox_inches="tight")
P=json.load(open(R+"xfer_proj_withqwen.json"));ov=json.load(open(R+"subspace_overlap.json"))
enc=[("clip_l14","CLIP"),("siglip_so400m","SigLIP"),("dinov2_l","DINOv2"),("qwen25vl_vis","Qwen vis.")]
fig,ax=plt.subplots(1,2,figsize=(3.4,1.55),gridspec_kw={"wspace":.45})
for k,t in enc:
    c=P[k]["xfer_single"]["heldout_curve"];ks=sorted(map(int,c))
    ax[0].plot(range(len(ks)),[c[str(x)]["mae"] for x in ks],marker="o",ms=2.5,lw=1.1,c=pal[t],label=t)
ax[0].set_xticks(range(len(ks)));ax[0].set_xticklabels(ks);ax[0].set_xlabel("$k$");ax[0].set_ylabel("held-out MAE")
ax[0].axhline(1.6,ls=":",c="gray",lw=.7);ax[0].text(0,1.57,"chance",fontsize=5,color="gray",va="top")
ax[0].legend(frameon=False,handlelength=1.2,loc="lower left",fontsize=5)
for k,t in enc[:3]:
    ks=sorted(map(int,ov[k]))
    ax[1].plot(ks,[ov[k][str(x)]["material"] for x in ks],c=pal[t],marker="o",ms=2.5,lw=1.1,label=t)
    ax[1].plot(ks,[ov[k][str(x)]["random"] for x in ks],c=pal[t],ls=":",lw=.8)
ax[1].set_xscale("log",base=2);ax[1].set_xticks([1,4,16,64]);ax[1].set_xticklabels([1,4,16,64])
ax[1].set_xlabel("$k$");ax[1].set_ylabel("material energy share");ax[1].text(64,.09,"random",fontsize=5,color="gray",ha="right")
for a in ax: sns.despine(ax=a); a.grid(axis="y",lw=.3,alpha=.5)
plt.savefig(O+"kcurve.pdf",bbox_inches="tight")
D="renders_diag/";fs=[("00264_pink_velvet_sphere_0_base1.png","Base"),("00266_pink_velvet_sphere_0_material2.png","Material: sheen $\\uparrow$"),("00270_pink_velvet_sphere_0_ratio2.png","Light: key/fill $\\uparrow$"),("00274_pink_velvet_sphere_0_kelvin2.png","Light: Kelvin $\\uparrow$")]
cols=["#555555","#c0392b","#3b6ea8","#3b6ea8"]
fig,ax=plt.subplots(1,4,figsize=(3.4,1.05),gridspec_kw={"wspace":.04})
for a,(f,t),c in zip(ax,fs,cols):
    a.imshow(Image.open(D+f));a.set_title(t,fontsize=5.5,color=c,pad=2);a.set_xticks([]);a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(c);s.set_linewidth(1.2)
plt.savefig(O+"teaser.pdf",bbox_inches="tight",dpi=300)
