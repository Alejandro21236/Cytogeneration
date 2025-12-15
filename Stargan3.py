# Stargan2_patched.py
# retains original functionality; adds sharper upsampling, perceptual loss, richer validation CSVs and plots
import os,random,argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image,make_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from torchvision.models import resnet18,ResNet18_Weights,vgg16,VGG16_Weights
    _HW=True
except Exception:
    from torchvision.models import resnet18,vgg16
    ResNet18_Weights=None;VGG16_Weights=None;_HW=False

# ---------------- utils -----------------
def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def to01(t):
    t=(t+1)/2
    return t.clamp_(0,1)

def save_img01(t,path):
    save_image(to01(t.detach().cpu()),path)

# --------------- data -------------------
class PNGStack(Dataset):
    def __init__(self,root,labels_csv,frames,classes,transform=None,max_per_frame=64,use_best_focus=True,edf_dir=None,recursive=True):
        self.root=Path(root)
        df=pd.read_csv(labels_csv)
        df["frame"]=df["frame"].astype(str).str.strip()
        want=[str(f).strip() for f in frames]
        self.df=df[df.frame.isin(want)].copy()
        self.cls=classes;self.c2i={c:i for i,c in enumerate(self.cls)}
        self.transform=transform;self.maxpf=max_per_frame;self.use_best=use_best_focus;self.edf_dir=edf_dir;self.recursive=recursive
        self.samples=[];self.diag=[]
        def resolve_dir(root,f):
            cand=[root/f]
            if not f.endswith("_stack"):cand.append(root/f"{f}_stack")
            cand+=list(root.glob(f"{f}*"))
            for d in cand:
                if d.is_dir():return d
            return None
        for _,r in self.df.iterrows():
            f=str(r.frame);l=str(r.label)
            fd=resolve_dir(self.root,f)
            if fd is None:
                self.diag.append((f,"NO_DIR",0));continue
            it=fd.rglob if self.recursive else fd.glob
            files=sorted(list(it("*.png"))+list(it("*.PNG")))
            n=len(files)
            if n==0:
                self.diag.append((f,"NO_PNG",0));continue
            if self.use_best:
                pick=files if n<=128 else random.sample(files,128)
                best=None;bs=-1e9
                for p in pick:
                    im=Image.open(p).convert("L");g=np.array(im,dtype=np.float32)
                    lap=(-4*g+np.roll(g,1,0)+np.roll(g,-1,0)+np.roll(g,1,1)+np.roll(g,-1,1))
                    s=float(lap.var())
                    if s>bs:bs=s;best=p
                files=[best]
            if self.maxpf and len(files)>self.maxpf:files=random.sample(files,self.maxpf)
            for p in files:self.samples.append((str(p),l,f))
            self.diag.append((f,"OK",n))
        random.shuffle(self.samples)
        if len(self.samples)==0:
            lines=["[PNGStack] zero samples; check paths/labels:"]
            for f,st,n in self.diag:lines.append(f"{f}\t{st}\t{n}")
            raise SystemExit("\n".join(lines))
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        p,l,f=self.samples[i]
        if self.edf_dir:
            dname=f if f.endswith("_stack") else f"{f}_stack"
            edf=self.root/self.edf_dir/f"{Path(dname).name}.png"
            if edf.is_file():p=str(edf)
        im=Image.open(p).convert("RGB")
        im=self.transform(im) if self.transform else T.ToTensor()(im)
        return im,self.c2i[l],{"png":p,"frame":f,"label":l}

# -------------- model -------------------
class AdaIN(nn.Module):
    def __init__(self,c,sd):super().__init__();self.norm=nn.InstanceNorm2d(c,affine=False,eps=1e-5);self.fc=nn.Linear(sd,c*2)
    def forward(self,x,s):
        y=self.norm(x);h=self.fc(s);g,b=h.chunk(2,1);g=g.view(g.size(0),g.size(1),1,1);b=b.view(b.size(0),b.size(1),1,1);return y*g+b

class ResBlk(nn.Module):
    def __init__(self,cin,cout,sd,down=False):super().__init__();self.a1=AdaIN(cin,sd);self.c1=nn.Conv2d(cin,cout,3,1,1);self.a2=AdaIN(cout,sd);self.c2=nn.Conv2d(cout,cout,3,1,1);self.sk=nn.Conv2d(cin,cout,1,1,0) if cin!=cout else nn.Identity();self.down=down;self.pool=nn.AvgPool2d(2)
    def forward(self,x,s):
        h=self.a1(x,s);h=F.leaky_relu(h,0.2);h=self.c1(h);h=self.a2(h,s);h=F.leaky_relu(h,0.2);h=self.c2(h);h=h+self.sk(x);return self.pool(h) if self.down else h

class UpBlock(nn.Module):
    def __init__(self,cin,cout):super().__init__();self.m=nn.Sequential(nn.Upsample(scale_factor=2,mode="nearest"),nn.Conv2d(cin,cout,3,1,1))
    def forward(self,x):return self.m(x)

class Gen(nn.Module):
    def __init__(self,sd=64,ch=64):super().__init__();self.f=nn.Conv2d(3,ch,3,1,1);self.b1=ResBlk(ch,ch*2,sd,True);self.b2=ResBlk(ch*2,ch*4,sd,True);self.b3=ResBlk(ch*4,ch*4,sd,True);self.u1=UpBlock(ch*4,ch*2);self.u2=UpBlock(ch*2,ch);self.u3=UpBlock(ch,ch);self.t=nn.Conv2d(ch,3,1,1,0)
    def forward(self,x,s):
        h=self.f(x);h=self.b1(h,s);h=self.b2(h,s);h=self.b3(h,s);h=self.u1(h);h=F.leaky_relu(h,0.2);h=self.u2(h);h=F.leaky_relu(h,0.2);h=self.u3(h);h=F.leaky_relu(h,0.2);return torch.tanh(self.t(h))

class Disc(nn.Module):
    def __init__(self,ch=64,k=4):super().__init__();self.m=nn.Sequential(nn.Conv2d(3,ch,4,2,1),nn.LeakyReLU(0.2,True),nn.Conv2d(ch,ch*2,4,2,1),nn.LeakyReLU(0.2,True),nn.Conv2d(ch*2,ch*4,4,2,1),nn.LeakyReLU(0.2,True),nn.Conv2d(ch*4,ch*4,4,2,1),nn.LeakyReLU(0.2,True));self.rf=nn.Conv2d(ch*4,1,1,1,0);self.cls=nn.Conv2d(ch*4,k,1,1,0)
    def forward(self,x):
        h=self.m(x);return self.rf(h),self.cls(h)

class PatchDiscSmall(nn.Module):
    def __init__(self,ch=32,k=4):super().__init__();self.m=nn.Sequential(nn.Conv2d(3,ch,3,2,1),nn.LeakyReLU(0.2,True),nn.Conv2d(ch,ch*2,3,2,1),nn.LeakyReLU(0.2,True),nn.Conv2d(ch*2,ch*4,3,2,1),nn.LeakyReLU(0.2,True));self.rf=nn.Conv2d(ch*4,1,1,1,0);self.cls=nn.Conv2d(ch*4,k,1,1,0)
    def forward(self,x):
        h=self.m(x);return self.rf(h),self.cls(h)

class MapNet(nn.Module):
    def __init__(self,z=16,sd=64,k=4):super().__init__();self.emb=nn.Embedding(k,32);self.net=nn.Sequential(nn.Linear(z+32,128),nn.ReLU(True),nn.Linear(128,sd))
    def forward(self,z,c):e=self.emb(c);return self.net(torch.cat([z,e],1))

# --------- losses / helpers -------------
def adv_out(d):return d.view(d.size(0),-1).mean(1,True)

def hD(r,f):return F.relu(1-r).mean()+F.relu(1+f).mean()

def hG(f):return -f.mean()

def ssim(x,y):
    K1=0.01;K2=0.03;L=1.0
    mux=x.mean([2,3]);muy=y.mean([2,3])
    vx=x.var([2,3],False);vy=y.var([2,3],False)
    vxy=((x*y).mean([2,3])-(mux*muy))
    C1=(K1*L)**2;C2=(K2*L)**2
    return (((2*mux*muy+C1)*(2*vxy+C2))/((mux**2+muy**2+C1)*(vx+vy+C2))).mean()

class VGGPerceptual(nn.Module):
    def __init__(self,device):
        super().__init__()
        if _HW and VGG16_Weights is not None:
            v=vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval().to(device)
        else:
            v=vgg16(weights=None).features.eval().to(device)
        for p in v.parameters():p.requires_grad=False
        self.v=v;self.slices=[3,8,15] # relu1_2, relu2_2, relu3_3 approx
        self.reg=nn.InstanceNorm2d(3,affine=False)
    def forward(self,x,y):
        x=to01(x);y=to01(y)
        feats_x=[];feats_y=[];h1=x;h2=y
        for i,layer in enumerate(self.v):
            h1=layer(h1);h2=layer(h2)
            if i in self.slices:
                feats_x.append(h1);feats_y.append(h2)
        loss=0.0
        for a,b in zip(feats_x,feats_y):loss+=F.l1_loss(a,b)
        return loss/len(self.slices)

# -------- validation / reporting --------
def evaluate_generated_index(gen_csv_path,outdir,classes):
    outdir=Path(outdir);df=pd.read_csv(gen_csv_path)
    if df.empty:
        pd.DataFrame([{"n_rows":0,"tss_rate":np.nan,"mean_delta_target":np.nan,"soft_cross_entropy":np.nan,"kl_divergence":np.nan,"argmax_accuracy":np.nan}]).to_csv(outdir/"metrics.csv",index=False);return
    real_cols=[f"real_p_{c}" for c in classes];pred_cols=[f"p_{c}" for c in classes];c2i={c:i for i,c in enumerate(classes)}
    real=df[real_cols].to_numpy(float);pred=df[pred_cols].to_numpy(float);eps=1e-9
    tgt_idx=df["tgt_label"].map(c2i).to_numpy();before=real[np.arange(len(df)),tgt_idx];after=pred[np.arange(len(df)),tgt_idx];delta=after-before;success=(delta>0).astype(np.float32)
    log_pred=np.log(pred+eps);log_real=np.log(real+eps);soft_ce=float(np.mean(-np.sum(real*log_pred,1)));kl=float(np.mean(np.sum(real*(log_real-log_pred),1)))
    y_true=np.argmax(real,1);y_pred=np.argmax(pred,1);acc=float(np.mean(y_true==y_pred))
    K=len(classes);cm=np.zeros((K,K),dtype=int)
    for t,p in zip(y_true,y_pred):cm[t,p]+=1
    pd.DataFrame(cm,index=[f"true_{c}" for c in classes],columns=[f"pred_{c}" for c in classes]).to_csv(outdir/"confusion_matrix.csv")
    dfo=df.copy();dfo["target_index"]=tgt_idx;dfo["delta_target"]=delta;dfo["tss_success"]=success;dfo.to_csv(outdir/"generated_index_with_tss.csv",index=False)
    # per-class TSS/delta
    pc=[]
    for i,c in enumerate(classes):
        mask=(tgt_idx==i)
        if mask.any():
            pc.append({"class":c,"n":int(mask.sum()),"tss_rate":float(success[mask].mean()),"mean_delta_target":float(delta[mask].mean())})
        else:
            pc.append({"class":c,"n":0,"tss_rate":np.nan,"mean_delta_target":np.nan})
    pd.DataFrame(pc).to_csv(outdir/"per_class_tss.csv",index=False)
    pd.DataFrame([{"n_rows":int(len(df)),"tss_rate":float(success.mean()),"mean_delta_target":float(delta.mean()),"soft_cross_entropy":soft_ce,"kl_divergence":kl,"argmax_accuracy":acc}]).to_csv(outdir/"metrics.csv",index=False)
    # plots
    figs=outdir/"figs";figs.mkdir(exist_ok=True,parents=True)
    plt.figure();plt.hist(delta,bins=30);plt.xlabel("delta_target");plt.ylabel("count");plt.title("Delta(target) histogram");plt.tight_layout();plt.savefig(figs/"delta_hist.png");plt.close()
    plt.figure();
    try:
        import itertools
        im=plt.imshow(cm,aspect="auto");plt.colorbar(im);plt.xticks(range(K),classes,rotation=45,ha="right");plt.yticks(range(K),classes);plt.title("Confusion (argmax)");plt.tight_layout();plt.savefig(figs/"confusion.png");plt.close()
    except Exception:
        plt.close()
    pcdf=pd.DataFrame(pc)
    plt.figure();plt.bar(pcdf["class"],pcdf["tss_rate"]);plt.xticks(rotation=45,ha="right");plt.ylabel("TSS rate");plt.title("Per-class TSS");plt.tight_layout();plt.savefig(figs/"tss_per_class.png");plt.close()

# --------- backbone classifier ----------
def feats_backbone(device):
    if _HW and ResNet18_Weights is not None:
        m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    else:
        m=resnet18(weights=None).to(device)
    return m

def train_clf(frames,root,labels,classes,device,img,bs,maxpf,epochs,seed):
    set_seed(seed)
    tf=T.Compose([T.Resize(img),T.CenterCrop(img),T.RandomHorizontalFlip(),T.ToTensor()])
    ds=PNGStack(root,labels,frames,classes,tf,max_per_frame=min(maxpf,32),use_best_focus=True,edf_dir="EDF",recursive=True)
    if len(ds)==0:return None
    nw=0;pin=(device.type=="cuda")
    dl=DataLoader(ds,batch_size=min(bs,16),shuffle=True,num_workers=nw,pin_memory=pin,drop_last=True)
    m=feats_backbone(device);m.fc=nn.Linear(m.fc.in_features,len(classes));m=m.to(device);m.train()
    opt=torch.optim.Adam(m.parameters(),lr=1e-4);ce=nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x,y,_ in dl:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad();p=m(x);l=ce(p,y);l.backward();opt.step()
    m.eval();return m

def clf_probs(model,img,device,classes):
    if model is None:return {f"p_{c}":float("nan") for c in classes}
    with torch.no_grad():
        x=(img+1)/2;x=T.Resize(224)(x.cpu());x=x.unsqueeze(0).to(device);p=torch.softmax(model(x),1)[0].cpu().numpy().tolist()
    return {f"p_{c}":float(p[i]) for i,c in enumerate(classes)}

# --------------- train fold -------------
def train_fold(args,device,frames_train,frame_test,classes,outdir):
    set_seed(args.seed)
    tf=T.Compose([T.Resize(args.img),T.CenterCrop(args.img),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
    ds=PNGStack(args.root,args.labels,frames_train,classes,tf,max_per_frame=args.maxpf,use_best_focus=True,edf_dir="EDF",recursive=True)
    if len(ds)==0:raise SystemExit("no training images found")
    nw=0;pin=(device.type=="cuda");dl=DataLoader(ds,batch_size=args.bs,shuffle=True,num_workers=nw,pin_memory=pin,drop_last=True)
    G=Gen(args.sdim,args.ch).to(device);D=Disc(args.ch,len(classes)).to(device);M=MapNet(args.zdim,args.sdim,len(classes)).to(device)
    D2=PatchDiscSmall(args.ch//2,len(classes)).to(device) if args.multi_disc else None
    perc=VGGPerceptual(device) if args.w_perc>0 else None
    oG=torch.optim.Adam(list(G.parameters())+list(M.parameters()),lr=args.lr,betas=(0.0,0.99));paramsD=list(D.parameters())+(list(D2.parameters()) if D2 is not None else [])
    oD=torch.optim.Adam(paramsD,lr=args.lr,betas=(0.0,0.99))
    for _ in range(args.epochs):
        for x,y,_ in dl:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            ct=torch.randint(0,len(classes),(x.size(0),),device=device)
            z=torch.randn(x.size(0),args.zdim,device=device);s=M(z,ct);x1=G(x,s)
            dr,dc=D(x);fr,fc=D(x1);dr=adv_out(dr);fr=adv_out(fr)
            ld=hD(dr,fr)+F.cross_entropy(dc.mean([2,3]),y)+F.cross_entropy(fc.mean([2,3]),ct)
            if D2 is not None:
                dr2,dc2=D2(x);fr2,fc2=D2(x1);dr2=adv_out(dr2);fr2=adv_out(fr2)
                ld+=hD(dr2,fr2)+0.5*(F.cross_entropy(dc2.mean([2,3]),y)+F.cross_entropy(fc2.mean([2,3]),ct))
            oD.zero_grad();ld.backward();oD.step()
            z=torch.randn(x.size(0),args.zdim,device=device);s=M(z,ct);x2=G(x,s);fr,fc=D(x2);fr=adv_out(fr)
            lg=hG(fr)+F.cross_entropy(fc.mean([2,3]),ct)
            if D2 is not None:
                fr2,fc2=D2(x2);fr2=adv_out(fr2);lg+=hG(fr2)+0.5*F.cross_entropy(fc2.mean([2,3]),ct)
            xi=M(torch.randn(x.size(0),args.zdim,device=device),y);idim=G(x,xi);lid=F.l1_loss(idim,x)
            zb=torch.randn(x.size(0),args.zdim,device=device);back=G(x2,M(zb,y));lcy=F.l1_loss(back,x)
            ldiv=-F.l1_loss(x1.detach(),x2)
            lp=perc(x2,x) if perc is not None else 0.0
            L=lg+args.w_id*lid+args.w_cyc*lcy+args.w_div*ldiv+args.w_perc*lp
            oG.zero_grad();L.backward();oG.step()
    clf=train_clf(frames_train,args.root,args.labels,classes,device,args.img,args.bs,args.maxpf,args.clf_epochs,args.seed)
    fdir=Path(outdir)/f"fold_{frame_test}";(fdir/"gen").mkdir(parents=True,exist_ok=True);(fdir/"figs").mkdir(parents=True,exist_ok=True)
    te_tf=T.Compose([T.Resize(args.img),T.CenterCrop(args.img),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
    dtest=PNGStack(args.root,args.labels,[frame_test],classes,te_tf,max_per_frame=1,use_best_focus=True,edf_dir="EDF",recursive=True)
    if len(dtest)==0:raise SystemExit(f"no test images in {frame_test}")
    dtl=DataLoader(dtest,batch_size=1,shuffle=False,num_workers=0)
    rows=[]
    with torch.no_grad():
        for x,y,meta in dtl:
            x=x.to(device);y=y.to(device)
            src=classes[int(y.item())]
            others=[c for c in classes if c!=src]
            idx=sorted(random.sample(range(len(others)),min(2,len(others)))) if len(others)>0 else []
            targets=[others[i] for i in idx]
            ims=[x[0].cpu()]
            rp=clf_probs(clf,x[0],device,classes);rp={f"real_{k}":v for k,v in rp.items()}
            sp=Path(meta["png"][0])
            for tgt in targets:
                tid=torch.tensor([classes.index(tgt)],device=device);z=torch.randn(1,args.zdim,device=device);s=M(z,tid);g=G(x,s)
                ims.append(g[0].cpu())
                fn=f"{frame_test}__{sp.name}__to__{tgt}.png";save_img01(g,(fdir/"gen"/fn))
                gp=clf_probs(clf,g[0],device,classes)
                row={"src_frame":frame_test,"src_png":sp.name,"src_label":src,"tgt_label":tgt,"out_png":fn}
                row.update(rp);row.update(gp);rows.append(row)
            ims01=[to01(im) for im in ims];grid=make_grid(torch.stack(ims01,0),nrow=len(ims01));save_image(grid,(fdir/"figs"/f"{frame_test}__{sp.stem}.png"))
    pd.DataFrame(rows).to_csv(fdir/"generated_index.csv",index=False)
    evaluate_generated_index(fdir/"generated_index.csv", fdir, classes)
    id_ds=PNGStack(args.root,args.labels,frames_train[:min(8,len(frames_train))],classes,te_tf,max_per_frame=4,use_best_focus=True,edf_dir="EDF",recursive=True)
    id_dl=DataLoader(id_ds,batch_size=min(8,args.bs),shuffle=False,num_workers=0)
    svals=[]
    for x,y,_ in id_dl:
        x=x.to(device);xi=M(torch.randn(x.size(0),args.zdim,device=device),y.to(device));g=G(x,xi);svals.append(ssim((x+1)/2,(g+1)/2).item())
    with open(fdir/"identity_ssim.txt","w") as f:f.write(str(float(np.mean(svals)) if len(svals)>0 else float("nan")))
    torch.save({"G":G.state_dict(),"M":M.state_dict()},fdir/"ckpt.pt")

# --------------- main -------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=str,required=True)
    p.add_argument("--labels",type=str,required=True)
    p.add_argument("--outdir",type=str,required=True)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--img",type=int,default=256)
    p.add_argument("--bs",type=int,default=8)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--ch",type=int,default=64)
    p.add_argument("--sdim",type=int,default=64)
    p.add_argument("--zdim",type=int,default=16)
    p.add_argument("--maxpf",type=int,default=64)
    p.add_argument("--seed",type=int,default=1337)
    p.add_argument("--device",type=str,default="cuda")
    p.add_argument("--w_id",type=float,default=5.0)
    p.add_argument("--w_cyc",type=float,default=2.0)
    p.add_argument("--w_div",type=float,default=0.1)
    p.add_argument("--w_perc",type=float,default=0.1)
    p.add_argument("--clf_epochs",type=int,default=3)
    p.add_argument("--multi_disc",action="store_true")
    args=p.parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir,exist_ok=True)
    df=pd.read_csv(args.labels);df["frame"]=df["frame"].astype(str).str.strip();classes=sorted(df.label.unique().tolist());frames=df.frame.tolist()
    device=torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    fold_rows=[]
    for ft in frames:
        fr_train=[f for f in frames if f!=ft]
        train_fold(args,device,fr_train,ft,classes,args.outdir)
        m=pd.read_csv(Path(args.outdir)/f"fold_{ft}"/"metrics.csv")
        m.insert(0,"fold",ft);fold_rows.append(m.iloc[0])
    agg=pd.DataFrame(fold_rows);agg.to_csv(Path(args.outdir)/"fold_summary.csv",index=False)
    # aggregate plots
    try:
        plt.figure();plt.bar(agg["fold"].astype(str),agg["tss_rate"].astype(float));plt.ylabel("TSS rate");plt.title("Fold-level TSS");plt.tight_layout();plt.savefig(Path(args.outdir)/"tss_by_fold.png");plt.close()
    except Exception:
        pass

if __name__=="__main__":main()
