# NumPy MLP from scratch with optional L2, Adam, and class weights
import numpy as np

def relu(z): return np.maximum(0.0, z)
def drelu(z): return (z > 0).astype(float)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, l2=0.0,
                 use_adam=False, class_weights=None, seed=42):
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2
        self.layer_sizes = layer_sizes
        self.lr = float(learning_rate)
        self.l2 = float(l2)
        self.use_adam = bool(use_adam)
        self.class_weights = class_weights  # {0: w0, 1: w1} or None

        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(layer_sizes)-1):
            fan_in = layer_sizes[i]
            self.W.append(rng.standard_normal((fan_in, layer_sizes[i+1])) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros((1, layer_sizes[i+1])))

        if self.use_adam:
            self.beta1, self.beta2, self.eps, self.t = 0.9, 0.999, 1e-8, 0
            self.mW = [np.zeros_like(W) for W in self.W]
            self.vW = [np.zeros_like(W) for W in self.W]
            self.mb = [np.zeros_like(b) for b in self.b]
            self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X):
        A = X
        self.A, self.Z = [X], []
        for l in range(len(self.W)-1):
            Z = A @ self.W[l] + self.b[l]
            A = relu(Z)
            self.Z.append(Z); self.A.append(A)
        Z = A @ self.W[-1] + self.b[-1]
        A = sigmoid(Z)
        self.Z.append(Z); self.A.append(A)
        return A

    @staticmethod
    def _bce(y, p, w=None):
        eps = 1e-8
        if w is None:
            return -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
        return -np.mean(w*(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)))

    def loss(self, y, p):
        m = y.shape[0]
        w = None
        if self.class_weights is not None:
            w = np.where(y == 1, self.class_weights[1], self.class_weights[0])
        base = self._bce(y, p, w=w)
        reg = 0.0 if self.l2 == 0.0 else (self.l2/(2*m)) * sum((Wi**2).sum() for Wi in self.W)
        return float(base + reg)

    def backward(self, y, p):
        m = y.shape[0]
        dW = [np.zeros_like(Wi) for Wi in self.W]
        db = [np.zeros_like(bi) for bi in self.b]

        if self.class_weights is None:
            dZ = (p - y)
        else:
            w = np.where(y == 1, self.class_weights[1], self.class_weights[0])
            dZ = (p - y) * w

        dW[-1] = (self.A[-2].T @ dZ) / m
        db[-1] = dZ.mean(axis=0, keepdims=True)

        for l in range(len(self.W)-2, -1, -1):
            dA = dZ @ self.W[l+1].T
            dZ = dA * drelu(self.Z[l])
            dW[l] = (self.A[l].T @ dZ) / m
            db[l] = dZ.mean(axis=0, keepdims=True)

        if self.l2 > 0.0:
            for l in range(len(self.W)):
                dW[l] += (self.l2 / m) * self.W[l]

        if self.use_adam:
            self.t += 1
            for l in range(len(self.W)):
                self.mW[l] = 0.9*self.mW[l] + 0.1*dW[l]
                self.vW[l] = 0.999*self.vW[l] + 0.001*(dW[l]**2)
                self.mb[l] = 0.9*self.mb[l] + 0.1*db[l]
                self.vb[l] = 0.999*self.vb[l] + 0.001*(db[l]**2)
                mWh = self.mW[l] / (1 - 0.9**self.t)
                vWh = self.vW[l] / (1 - 0.999**self.t)
                mbh = self.mb[l] / (1 - 0.9**self.t)
                vbh = self.vb[l] / (1 - 0.999**self.t)
                self.W[l] -= self.lr * mWh / (np.sqrt(vWh) + self.eps)
                self.b[l] -= self.lr * mbh / (np.sqrt(vbh) + self.eps)
        else:
            for l in range(len(self.W)):
                self.W[l] -= self.lr * dW[l]
                self.b[l] -= self.lr * db[l]

    def train(self, X, y, Xv, yv, epochs=100, batch_size=32, seed=42):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        trL, vaL = [], []
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch_size):
                b = idx[s:s+batch_size]
                p = self.forward(X[b]); self.backward(y[b], p)
            p_tr = self.forward(X);  p_va = self.forward(Xv)
            trL.append(self.loss(y, p_tr)); vaL.append(self.loss(yv, p_va))
        return np.array(trL), np.array(vaL)

    def predict_proba(self, X): return self.forward(X)
    def predict(self, X, thresh=0.5): return (self.forward(X) > thresh).astype(int)
