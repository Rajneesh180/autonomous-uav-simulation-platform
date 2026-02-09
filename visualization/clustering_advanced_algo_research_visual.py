"""
CLUSTERING LAB — UNIFORM vs NON-UNIFORM DATASETS
------------------------------------------------

This script allows comparison of clustering algorithms
on simple geometric datasets.

MODES
-----
BATCH  -> Terminal comparison table
VISUAL -> Animated visualization with live metrics overlay

You can change everything from the SWITCHES section.
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
MODE OPTIONS
------------
"BATCH"  : Prints algorithm comparison table
"VISUAL" : Shows animated clustering window
"""
MODE = "VISUAL"

"""
ALGORITHM OPTIONS (used only in VISUAL mode)
--------------------------------------------
"K-Means"
"DBSCAN"
"OPTICS"
"Agglomerative Hierarchical"
"Gaussian Mixture Models"
"Fuzzy C-Means"
"Soft K-Means"
"""
ALGORITHM = "K-Means"

"""
DATA_MODE OPTIONS
-----------------
"UNIFORM"      -> Equal cluster sizes and radii
"NON_UNIFORM"  -> Unequal sizes, radii, and noise
"""
DATA_MODE = "UNIFORM"

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
        sizes = [25,25,25,25]
        radii = [10,10,10,10]
        noise = 0
    else:
        sizes = [30,10,25,15]
        radii = [5,15,20,8]
        noise = 12

    for i,c in enumerate(TRUE_CENTERS):
        for _ in range(sizes[i]):
            r = radii[i]*np.sqrt(np.random.rand())
            t = 2*np.pi*np.random.rand()
            pts.append(c + [r*np.cos(t), r*np.sin(t)])

    pts.extend(np.random.uniform(0,MAP_SIZE,(noise,2)))
    return np.array(pts)

X = generate_data()

# =========================================================
# ================= SOFT K-MEANS ==========================
# =========================================================
def soft_kmeans(X, k=4, beta=1.2, iters=40):
    C = X[np.random.choice(len(X),k,replace=False)]
    for _ in range(iters):
        d = np.linalg.norm(X[:,None]-C[None,:],axis=2)
        W = np.exp(-beta*d)
        W /= W.sum(axis=1,keepdims=True)
        C = (W.T @ X) / W.sum(axis=0)[:,None]
    return C, W.argmax(axis=1)

# =========================================================
# ================= FUZZY C-MEANS =========================
# =========================================================
def fuzzy_c_means(X, c=4, m=2, iters=60):
    U=np.random.rand(len(X),c)
    U/=U.sum(axis=1,keepdims=True)
    for _ in range(iters):
        um=U**m
        C=(um.T@X)/um.sum(axis=0)[:,None]
        d=np.linalg.norm(X[:,None]-C[None,:],axis=2)+1e-6
        U=1/(d**(2/(m-1)))
        U/=U.sum(axis=1,keepdims=True)
    return C,U.argmax(axis=1)

# =========================================================
# ================= METRIC UTILITIES ======================
# =========================================================
def center_error(C):
    return np.mean([min(np.linalg.norm(c-t) for t in TRUE_CENTERS) for c in C])

def safe_metrics(labels):
    if len(set(labels)) < 2 or -1 in labels:
        return 0,999,0
    return (
        silhouette_score(X,labels),
        davies_bouldin_score(X,labels),
        calinski_harabasz_score(X,labels)
    )

# =========================================================
# ================= ALGORITHM ENGINE ======================
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
# ======================= BATCH ===========================
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
# ======================= VISUAL ==========================
# =========================================================
pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption(f"{ALGORITHM} — {DATA_MODE}")
clock=pygame.time.Clock()
font=pygame.font.SysFont("Arial",17)

def ts(p): return int(p[0]*SCALE), int((MAP_SIZE-p[1])*SCALE)

COLORS=[(52,152,219),(231,76,60),(46,204,113),(155,89,182)]

C,L,RT,SIL,DBI,CH,ERR=run_algorithm(ALGORITHM)
start_time=time.time()

running=True
while running:
    clock.tick(60)
    for e in pygame.event.get():
        if e.type==pygame.QUIT: running=False

    prog=min((time.time()-start_time)/ANIM_DURATION,1.0)

    screen.fill((240,240,240))

    for i,p in enumerate(X):
        col=COLORS[L[i]%4] if L[i]>=0 else (120,120,120)
        pygame.draw.circle(screen,col,ts(p),3)

    for t in TRUE_CENTERS:
        pygame.draw.circle(screen,(0,150,0),ts(t),7)

    for i,c in enumerate(C):
        t=TRUE_CENTERS[i%4]
        pygame.draw.circle(screen,(0,0,0),ts(c),10,2)
        pygame.draw.line(screen,(200,0,0),ts(c),ts(t),2)

    # ===== METRICS PANEL =====
    panel=pygame.Surface((340,160))
    panel.set_alpha(230)
    panel.fill((255,255,255))
    screen.blit(panel,(470,20))

    lines=[
        f"Silhouette Score        : {SIL*prog:.2f}",
        f"Davies–Bouldin Index    : {DBI*prog:.2f}",
        f"Calinski–Harabasz Score : {CH*prog:.1f}",
        f"Center Recovery Error   : {ERR*prog:.2f} m",
        f"Runtime                 : {RT*prog:.1f} ms"
    ]

    for i,t in enumerate(lines):
        screen.blit(font.render(t,True,(20,20,20)),(490,35+i*26))

    pygame.display.flip()

pygame.quit()