"""
CLUSTERING RESEARCH LAB — DATA DISTRIBUTION COMPARISON
-------------------------------------------------------

This framework lets you:

1. Compare clustering algorithms numerically (BATCH mode)
2. Visualize clustering geometry and errors (VISUAL mode)
3. Test algorithms against different dataset geometries

You can change everything from the SWITCHES section below.
"""

import numpy as np
import pygame
import time

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================================================
# ======================== SWITCHES =======================
# =========================================================
"""
MODE OPTIONS:
- "BATCH"  -> prints table comparison in terminal
- "VISUAL" -> shows animated clustering window
"""
MODE = "BATCH"

"""
ALGORITHM OPTIONS (used only in VISUAL mode):
- "K-Means"
- "DBSCAN"
- "OPTICS"
- "Agglomerative Hierarchical"
- "Gaussian Mixture Models"
- "Fuzzy C-Means"
- "Soft K-Means"
"""
ALGORITHM = "K-Means"

"""
DATA_MODE OPTIONS:
- "UNIFORM"          -> perfect circles, equal size
- "SPHERICAL"        -> circles, different radii
- "NON_UNIFORM"      -> unequal size + noise
- "ELLIPTICAL"       -> stretched clusters
- "OVERLAP"          -> mixed clusters
- "DENSITY_VARIANT"  -> dense + sparse clusters
- "HEAVY_NOISE"      -> strong outliers
- "MOONS"            -> non-convex half circles
- "SPIRAL"           -> spiral arms
"""
DATA_MODE = "SPHERICAL"

# =========================================================
# ======================== CONFIG =========================
# =========================================================
WIDTH, HEIGHT = 820, 820
MAP_SIZE = 100
SCALE = WIDTH / MAP_SIZE
ANIM_DURATION = 8.0
np.random.seed(42)

TRUE_CENTERS = np.array([[20,20],[80,25],[30,75],[70,70]])

# =========================================================
# ================= DATA GENERATION =======================
# =========================================================
def generate_data():
    pts = []

    if DATA_MODE == "UNIFORM":
        for c in TRUE_CENTERS:
            for _ in range(25):
                r = 10*np.sqrt(np.random.rand())
                t = 2*np.pi*np.random.rand()
                pts.append(c + [r*np.cos(t), r*np.sin(t)])

    elif DATA_MODE == "SPHERICAL":
        radii = [6,12,18,8]
        for i,c in enumerate(TRUE_CENTERS):
            for _ in range(30):
                r = radii[i]*np.sqrt(np.random.rand())
                t = 2*np.pi*np.random.rand()
                pts.append(c + [r*np.cos(t), r*np.sin(t)])

    elif DATA_MODE == "NON_UNIFORM":
        sizes=[30,10,25,15]
        radii=[5,15,20,8]
        for i,c in enumerate(TRUE_CENTERS):
            for _ in range(sizes[i]):
                r=radii[i]*np.sqrt(np.random.rand())
                t=2*np.pi*np.random.rand()
                pts.append(c+[r*np.cos(t),r*np.sin(t)])
        pts.extend(np.random.uniform(0,MAP_SIZE,(12,2)))

    elif DATA_MODE == "ELLIPTICAL":
        for c in TRUE_CENTERS:
            for _ in range(30):
                pts.append(c+[np.random.normal(0,15),np.random.normal(0,4)])

    elif DATA_MODE == "OVERLAP":
        for c in TRUE_CENTERS:
            for _ in range(30):
                pts.append(c+[np.random.normal(0,12),np.random.normal(0,12)])

    elif DATA_MODE == "DENSITY_VARIANT":
        densities=[80,15,40,8]
        radii=[6,18,10,25]
        for i,c in enumerate(TRUE_CENTERS):
            for _ in range(densities[i]):
                r=radii[i]*np.sqrt(np.random.rand())
                t=2*np.pi*np.random.rand()
                pts.append(c+[r*np.cos(t),r*np.sin(t)])

    elif DATA_MODE == "HEAVY_NOISE":
        for c in TRUE_CENTERS:
            for _ in range(20):
                r=8*np.sqrt(np.random.rand())
                t=2*np.pi*np.random.rand()
                pts.append(c+[r*np.cos(t),r*np.sin(t)])
        pts.extend(np.random.uniform(0,MAP_SIZE,(80,2)))

    elif DATA_MODE == "MOONS":
        for _ in range(120):
            t=np.pi*np.random.rand()
            pts.append([40+20*np.cos(t),40+20*np.sin(t)])
        for _ in range(120):
            t=np.pi*np.random.rand()
            pts.append([60+20*np.cos(t),40-20*np.sin(t)])

    elif DATA_MODE == "SPIRAL":
        for arm in range(3):
            for i in range(80):
                ang=i/10+arm*2*np.pi/3
                r=2*i
                pts.append([50+r*np.cos(ang)/4,50+r*np.sin(ang)/4])

    return np.array(pts)

X = generate_data()

# =========================================================
# =============== SOFT + FUZZY ============================
# =========================================================
def soft_kmeans(X,k=4,beta=1.2,iters=40):
    C=X[np.random.choice(len(X),k,replace=False)]
    for _ in range(iters):
        d=np.linalg.norm(X[:,None]-C[None,:],axis=2)
        W=np.exp(-beta*d); W/=W.sum(axis=1,keepdims=True)
        C=(W.T@X)/W.sum(axis=0)[:,None]
    return C,W.argmax(axis=1)

def fuzzy_c_means(X,c=4,m=2,iters=60):
    U=np.random.rand(len(X),c); U/=U.sum(axis=1,keepdims=True)
    for _ in range(iters):
        um=U**m
        C=(um.T@X)/um.sum(axis=0)[:,None]
        d=np.linalg.norm(X[:,None]-C[None,:],axis=2)+1e-6
        U=1/(d**(2/(m-1))); U/=U.sum(axis=1,keepdims=True)
    return C,U.argmax(axis=1)

# =========================================================
# ================= METRICS ===============================
# =========================================================
def center_error(C):
    return np.mean([min(np.linalg.norm(c-t) for t in TRUE_CENTERS) for c in C])

def safe_metrics(labels):
    if len(set(labels))<2 or -1 in labels:
        return 0,999,0
    return (
        silhouette_score(X,labels),
        davies_bouldin_score(X,labels),
        calinski_harabasz_score(X,labels)
    )

# =========================================================
# ================= ENGINE ================================
# =========================================================
def run_algorithm(name):
    t0=time.time()

    if name=="K-Means":
        m=KMeans(4,n_init=10).fit(X)
        C,labels=m.cluster_centers_,m.labels_

    elif name=="DBSCAN":
        m=DBSCAN(eps=10,min_samples=5).fit(X)
        labels=m.labels_
        C=np.array([X[labels==i].mean(0) for i in set(labels) if i!=-1])

    elif name=="OPTICS":
        m=OPTICS(min_samples=5).fit(X)
        labels=m.labels_
        C=np.array([X[labels==i].mean(0) for i in set(labels) if i!=-1])

    elif name=="Agglomerative Hierarchical":
        m=AgglomerativeClustering(4).fit(X)
        labels=m.labels_
        C=np.array([X[labels==i].mean(0) for i in range(4)])

    elif name=="Gaussian Mixture Models":
        m=GaussianMixture(4).fit(X)
        labels=m.predict(X)
        C=m.means_

    elif name=="Fuzzy C-Means":
        C,labels=fuzzy_c_means(X)

    elif name=="Soft K-Means":
        C,labels=soft_kmeans(X)

    runtime=(time.time()-t0)*1000
    sil,dbi,ch=safe_metrics(labels)
    err=center_error(C)
    return C,labels,runtime,sil,dbi,ch,err

# =========================================================
# ================= BATCH MODE ============================
# =========================================================
ALGORITHMS=[
"K-Means","DBSCAN","OPTICS",
"Agglomerative Hierarchical",
"Gaussian Mixture Models",
"Fuzzy C-Means","Soft K-Means"
]

if MODE=="BATCH":
    print("\nCLUSTERING ALGORITHM COMPARISON\n")
    header = f"{'Algorithm':30s} | {'Silhouette Score':16s} | {'Davies–Bouldin Index':20s} | {'Calinski–Harabasz Score':23s} | {'Runtime (ms)':12s} | {'Center Error':12s}"
    print(header)
    print("-"*len(header))
    for a in ALGORITHMS:
        _,_,rt,s,d,c,e=run_algorithm(a)
        print(f"{a:30s} | {s:16.3f} | {d:20.3f} | {c:23.1f} | {rt:12.1f} | {e:12.2f}")
    exit()

# =========================================================
# ================= VISUAL MODE ===========================
# =========================================================
pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption(f"{ALGORITHM} — {DATA_MODE}")
clock=pygame.time.Clock()
font=pygame.font.SysFont("Arial",17)

def ts(p): return int(p[0]*SCALE),int((MAP_SIZE-p[1])*SCALE)

COLORS=[(52,152,219),(231,76,60),(46,204,113),(155,89,182)]

C,L,RT,SIL,DBI,CH,ERR=run_algorithm(ALGORITHM)
start=time.time()

running=True
while running:
    clock.tick(60)
    for e in pygame.event.get():
        if e.type==pygame.QUIT: running=False

    prog=min((time.time()-start)/ANIM_DURATION,1)

    screen.fill((240,240,240))

    # points
    for i,p in enumerate(X):
        col=COLORS[L[i]%4] if L[i]>=0 else (120,120,120)
        pygame.draw.circle(screen,col,ts(p),3)

    # true centers
    for t in TRUE_CENTERS:
        pygame.draw.circle(screen,(0,150,0),ts(t),7)

    # estimated centers
    for i,c in enumerate(C):
        t=TRUE_CENTERS[i%4]
        pygame.draw.circle(screen,(0,0,0),ts(c),10,2)
        pygame.draw.line(screen,(200,0,0),ts(c),ts(t),2)

    # ===== METRICS PANEL =====
    panel=pygame.Surface((330,160))
    panel.set_alpha(230)
    panel.fill((255,255,255))
    screen.blit(panel,(480,20))

    lines=[
        f"Silhouette Score        : {SIL*prog:.2f}",
        f"Davies–Bouldin Index    : {DBI*prog:.2f}",
        f"Calinski–Harabasz Score : {CH*prog:.1f}",
        f"Center Recovery Error   : {ERR*prog:.2f} m",
        f"Runtime                 : {RT*prog:.1f} ms"
    ]

    for i,t in enumerate(lines):
        screen.blit(font.render(t,True,(20,20,20)),(500,35+i*26))

    pygame.display.flip()

pygame.quit()