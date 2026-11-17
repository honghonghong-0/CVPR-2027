import sys,glob,json,torch,lpips,numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(16)
d=sys.argv[1]; M=[json.loads(l) for f in glob.glob(d+'/meta_*.jsonl') for l in open(f)]
img=lambda f:np.asarray(Image.open(d+'/'+f).convert('RGB'),np.float32)/255
T=lambda a:torch.from_numpy(a).permute(2,0,1)[None]*2-1
net=lpips.LPIPS(net='alex',verbose=False)
base={m['pair_id']:img(m['file']) for m in M if m['changed']=='base'}
rows=[]
with torch.no_grad():
  for m in M:
    if m['changed']=='base': continue
    a,b=base[m['pair_id']],img(m['file'])
    rows.append(dict(mat=m['mat'],geom=m['geom'],changed=m['changed'],level=m['level'],
      pix=float(abs(a-b).mean()),lp=float(net(T(a),T(b)))))
json.dump(rows,open('overlap_rows.json','w'))
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,k in enumerate(['pix','lp']):
  mt=[r[k] for r in rows if r['changed']=='material']; lt=[r[k] for r in rows if r['changed']!='material']
  bins=np.logspace(np.log10(min(mt+lt)),np.log10(max(mt+lt)),30)
  ax[i].hist(mt,bins,alpha=.6,label=f'material (n={len(mt)})'); ax[i].hist(lt,bins,alpha=.6,label=f'lighting (n={len(lt)})')
  ax[i].set_xscale('log'); ax[i].set_title(k); ax[i].legend()
plt.tight_layout(); plt.savefig('overlap_hist.png',dpi=120)
for k in ['pix','lp']:
  print('==',k)
  for c in ['material','size','ratio','dir','kelvin']:
    v=np.array([r[k] for r in rows if r['changed']==c]); print(f'{c:9s} n={len(v):3d} min {v.min():.4f} med {np.median(v):.4f} max {v.max():.4f}')
  mt=np.array([r[k] for r in rows if r['changed']=='material']); lt=np.array([r[k] for r in rows if r['changed']!='material'])
  lo,hi=max(mt.min(),lt.min()),min(mt.max(),lt.max())
  print(f'overlap range [{lo:.4f},{hi:.4f}]  material in: {((mt>=lo)&(mt<=hi)).sum()}/{len(mt)}  lighting in: {((lt>=lo)&(lt<=hi)).sum()}/{len(lt)}')
  # size-only baseline: best threshold accuracy (balanced)
  best=max((( (mt<t).mean()+(lt>=t).mean())/2,t) for t in np.unique(np.r_[mt,lt]))
  print(f'size-only balanced acc {best[0]:.3f} @ {best[1]:.4f}')
