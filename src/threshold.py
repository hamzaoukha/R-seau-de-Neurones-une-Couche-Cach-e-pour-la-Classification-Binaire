import numpy as np

def class_weights_from(y):
    yv = y.ravel()
    n = len(yv); n_pos = int(yv.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0: return {0:1.0, 1:1.0}
    return {0: n/(2*n_neg), 1: n/(2*n_pos)}

def tune_threshold(prob_val, y_val, grid=np.linspace(0.2, 0.8, 61)):
    yv = y_val.ravel().astype(int)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        yhat = (prob_val > t).astype(int).ravel()
        tp = ((yhat==1)&(yv==1)).sum()
        fp = ((yhat==1)&(yv==0)).sum()
        fn = ((yhat==0)&(yv==1)).sum()
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), float(best_f1)
