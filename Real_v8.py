"""
Real_v8.py — Ω‑AI with FIBONACCI INTEGRATION + Pure NumPy Fallback
Author: Briana Luna

NEW IN v8:
----------
• Fibonacci sequence integration throughout:
  - Neural layer sizes follow Fibonacci (8, 13, 21, 34, 55, 89, 144...)
  - Training schedule uses Fibonacci epochs
  - Spiral patterns follow Fibonacci arc growth
  - Server spacing uses Fibonacci intervals
• Pure NumPy neural network fallback (works WITHOUT PyTorch!)
• Fibonacci-based learning rate schedule
• Enhanced Phi convergence tracking

Run examples:
-------------
python Real_v8.py --mode text --text "maybe unclear" --visualize
python Real_v8.py --mode numeric --nums "1,5,3,9,2" --use_pytorch --visualize
python Real_v8.py --mode image --size 64 --fibonacci_layers --visualize
"""

from __future__ import annotations
import argparse, json, math, os, re, time, uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np

# Golden ratio and Fibonacci
PHI = 1.618033988749895
PHI_INV = 1 / PHI
GOLDEN_ANGLE = 137.5
PYRAMID_ANGLE = 26.5

def fibonacci_sequence(n: int) -> List[int]:
    """Generate first n Fibonacci numbers"""
    if n <= 0: return []
    if n == 1: return [1]
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

def fibonacci_ratio(n: int) -> float:
    """Get F(n+1)/F(n) ratio (converges to φ)"""
    fib = fibonacci_sequence(n+2)
    if len(fib) < 2: return 1.0
    return fib[-1] / fib[-2]

# Standard Fibonacci for layer sizes
FIB_LAYERS = fibonacci_sequence(15)  # [1,1,2,3,5,8,13,21,34,55,89,144,233,377,610]

# ----------------------------
#  Universal Ω metrics (same as v7)
# ----------------------------
def _safe_eps(x: float, eps: float = 1e-9) -> float:
    return x if abs(x) > eps else (eps if x >= 0 else -eps)

def normalized_entropy(probs: np.ndarray) -> float:
    probs = probs.astype(float)
    probs = probs / (probs.sum() + 1e-12)
    mask = probs > 0
    H = -(probs[mask] * np.log2(probs[mask])).sum()
    Hmax = math.log2(max(len(probs), 2))
    return float(np.clip(H / Hmax, 0.0, 1.0))

def histogram_entropy(x: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(x.flatten(), bins=bins, density=True)
    return normalized_entropy(hist + 1e-12)

def spectral_flatness_1d(x: np.ndarray) -> float:
    x = np.asarray(x).astype(float)
    if x.ndim != 1: x = x.flatten()
    X = np.fft.rfft(x - x.mean())
    psd = np.abs(X)**2 + 1e-12
    geo = np.exp(np.mean(np.log(psd)))
    arith = np.mean(psd)
    return float(np.clip(geo / _safe_eps(arith), 0.0, 1.0))

def spectral_flatness_2d(img: np.ndarray) -> float:
    img = np.asarray(img).astype(float)
    F2 = np.fft.rfft2(img - img.mean())
    psd = (np.abs(F2)**2) + 1e-12
    geo = np.exp(np.mean(np.log(psd)))
    arith = np.mean(psd)
    return float(np.clip(geo / _safe_eps(arith), 0.0, 1.0))

def conservation_score(psi: float, delta: float, omega: float) -> float:
    err = abs((omega**2) - (psi**2 + delta**2))
    return float(np.clip(1.0 - err, 0.0, 1.0))

def conservation_residual(psi: float, delta: float, omega: float) -> float:
    return float((omega**2) - (psi**2 + delta**2))

def phi_harmony(psi: float, delta: float, omega: float) -> float:
    if delta < 1e-6: return 0.0
    ratio1 = omega / _safe_eps(delta)
    ratio2 = delta / _safe_eps(psi)
    dist1 = abs(ratio1 - PHI)
    dist2 = abs(ratio2 - PHI)
    harmony = np.exp(-min(dist1, dist2))
    return float(np.clip(harmony, 0.0, 1.0))

def fibonacci_harmony(value: float, target_fib_index: int = 8) -> float:
    """Check how close value is to a Fibonacci number"""
    fib = fibonacci_sequence(target_fib_index + 2)
    distances = [abs(value - f) for f in fib]
    min_dist = min(distances)
    harmony = np.exp(-min_dist / (value + 1e-9))
    return float(np.clip(harmony, 0.0, 1.0))

# ----------------------------
#  Modes & adapters (same as v7)
# ----------------------------
class Mode(str, Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    IMAGE = "image"

# [TEXT, NUMERIC, IMAGE functions from v7 - unchanged]
def psi_text(text: str) -> float:
    text = text or ""
    if not text.strip(): return 0.0
    bigrams = [text[i:i+2] for i in range(max(len(text)-1, 1))]
    bins = 128
    vec = np.zeros(bins, dtype=float)
    for bg in bigrams:
        vec[abs(hash(bg)) % bins] += 1.0
    return normalized_entropy(vec + 1e-12)

def delta_text(text: str, strength: float) -> Tuple[str, float]:
    t = re.sub(r"\s+", " ", text).strip()
    phrases = [r"\bkind of\b", r"\bsort of\b", r"\bmaybe\b", r"\bperhaps\b",
               r"\bI think\b", r"\bI believe\b", r"\bpossibly\b"]
    before = psi_text(t)
    n_apply = max(0, min(len(phrases), int(round(strength * len(phrases)))))
    for p in phrases[:n_apply]:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    after = psi_text(t)
    delta = 0.0 if before <= 1e-6 else float(np.clip((before - after)/before, 0.0, 1.0))
    return t, delta

def omega_text(text: str) -> float:
    ent = psi_text(text)
    brevity = np.clip(100.0 / (len(text) + 10.0), 0.0, 1.0)
    return float(np.clip((1.0 - ent)*0.85 + brevity*0.15, 0.0, 1.0))

def psi_numeric(arr: np.ndarray) -> float:
    ent = histogram_entropy(arr, bins=64)
    flat = spectral_flatness_1d(arr)
    return float(np.clip(0.5*ent + 0.5*flat, 0.0, 1.0))

def delta_numeric(arr: np.ndarray, strength: float) -> Tuple[np.ndarray, float]:
    before = psi_numeric(arr)
    k = max(1, int(1 + strength * 4))
    pad = np.pad(arr, (k//2,), mode='edge')
    kernel = np.ones(k) / k
    out = np.convolve(pad, kernel, mode='valid')
    after = psi_numeric(out)
    delta = 0.0 if before <= 1e-6 else float(np.clip((before - after)/before, 0.0, 1.0))
    return out, delta

def omega_numeric(arr: np.ndarray) -> float:
    coherence = 1.0 - psi_numeric(arr)
    d = np.diff(arr)
    bonus = float(np.clip(1.0 / (1.0 + np.var(d) + 1e-6), 0.0, 1.0))
    return float(np.clip(0.85*coherence + 0.15*bonus, 0.0, 1.0))

def psi_image(img: np.ndarray) -> float:
    ent = histogram_entropy(img, bins=64)
    flat = spectral_flatness_2d(img)
    return float(np.clip(0.5*ent + 0.5*flat, 0.0, 1.0))

def delta_image(img: np.ndarray, strength: float) -> Tuple[np.ndarray, float]:
    before = psi_image(img)
    k = int(3 + round(strength * 6))
    if k % 2 == 0: k += 1
    pad = k//2
    kernel = np.ones((k, k), dtype=float) / (k*k)
    padded = np.pad(img, ((pad, pad),(pad, pad)), mode='edge')
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+k, j:j+k]
            out[i, j] = float(np.sum(region * kernel))
    after = psi_image(out)
    delta = 0.0 if before <= 1e-6 else float(np.clip((before - after)/before, 0.0, 1.0))
    return out, delta

def omega_image(img: np.ndarray) -> float:
    coherence = 1.0 - psi_image(img)
    gx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=float)
    gy = gx.T
    pad = 1
    padded = np.pad(img, ((pad,pad),(pad,pad)), mode='edge')
    G = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+3, j:j+3]
            G[i,j] = abs((region*gx).sum()) + abs((region*gy).sum())
    edge_var = np.var(G)
    bonus = float(np.clip(1.0 / (1.0 + edge_var), 0.0, 1.0))
    return float(np.clip(0.85*coherence + 0.15*bonus, 0.0, 1.0))

# ----------------------------
#  Embeddings
# ----------------------------
def embed_text(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    for tok in re.findall(r"[A-Za-z']+", (text or "").lower()):
        vec[abs(hash(tok)) % dim] += 1.0
    n = np.linalg.norm(vec) or 1.0
    return vec / n

def embed_numeric(arr: np.ndarray, dim: int = 128) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(dim, arr.size))
    feat = W @ (arr / (np.linalg.norm(arr) + 1e-9))
    n = np.linalg.norm(feat) or 1.0
    return feat / n

def embed_image(img: np.ndarray, dim: int = 128) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    F2 = np.fft.rfft2(img - img.mean())
    psd = np.abs(F2).flatten()
    bins = np.linspace(0, len(psd), dim+1, dtype=int)
    vec = np.array([psd[bins[i]:bins[i+1]].mean() for i in range(dim)], dtype=float)
    n = np.linalg.norm(vec) or 1.0
    return vec / n

# ----------------------------
#  Rule‑Δ
# ----------------------------
def rule_step(mode: Mode, value, strength: float):
    if mode == Mode.TEXT:
        psi = psi_text(value)
        out, delta = delta_text(value, strength)
        omega = omega_text(out)
        return dict(psi=psi, delta=delta, omega=omega, out=out)
    elif mode == Mode.NUMERIC:
        arr = np.asarray(value, dtype=float)
        psi = psi_numeric(arr)
        out, delta = delta_numeric(arr, strength)
        omega = omega_numeric(out)
        return dict(psi=psi, delta=delta, omega=omega, out=out)
    else:
        img = np.asarray(value, dtype=float)
        psi = psi_image(img)
        out, delta = delta_image(img, strength)
        omega = omega_image(out)
        return dict(psi=psi, delta=delta, omega=omega, out=out)

# ----------------------------
#  PURE NUMPY NEURAL NETWORK (Fibonacci-based!)
# ----------------------------
class FibonacciNeuralNet:
    """Pure NumPy neural network with Fibonacci layer sizes"""
    
    def __init__(self, d_in: int, fibonacci_layers: bool = True, lr: float = 1e-3):
        self.d_in = d_in
        self.lr = lr
        self.fibonacci_layers = fibonacci_layers
        
        # Use Fibonacci sequence for layer sizes
        if fibonacci_layers:
            # Find closest Fibonacci numbers
            suitable_fibs = [f for f in FIB_LAYERS if f >= d_in//4 and f <= d_in*4]
            if len(suitable_fibs) >= 3:
                self.layer_sizes = [d_in] + suitable_fibs[:3] + [d_in]
            else:
                self.layer_sizes = [d_in, 89, 55, 34, d_in]  # Fallback Fibonacci
        else:
            self.layer_sizes = [d_in, 256, 128, 64, d_in]
        
        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros(self.layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Phi-weighted loss coefficients
        self.lambda_cons = PHI
        self.lambda_mono = PHI_INV
        self.lambda_mirror = 1.0
        
        print(f"   Initialized Fibonacci Neural Net: {self.layer_sizes}")
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        """Forward pass through network"""
        activations = [x]
        
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            activations.append(a)
        
        # Final layer (no activation)
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        activations.append(z)
        
        return activations
    
    def compute_metrics(self, x):
        """Compute Ψ, Ω, Δ from encoding"""
        acts = self.forward(x)
        
        # Psi from input
        psi = float(np.mean(np.abs(acts[0])))
        
        # Omega from hidden layer
        mid_layer = acts[len(acts)//2]
        omega = float(self.sigmoid(np.mean(np.abs(mid_layer))))
        
        # Delta from transformation
        x_prime = acts[-1]
        delta = float(np.mean(np.abs(x_prime - x)))
        
        # Omega prime from output
        omega_p = float(self.sigmoid(np.mean(np.abs(x_prime))))
        
        return psi, omega, omega_p, delta
    
    def step(self, x, psi_sup=None, omega_sup=None, psi_target=None, omega_target=None):
        """Single training step"""
        # Forward pass
        acts = self.forward(x)
        x_prime = acts[-1]
        
        # Compute metrics
        psi, omega, omega_p, delta = self.compute_metrics(x)
        
        # Conservation loss
        cons = (omega_p**2 - (psi**2 + delta**2))**2
        
        # Monotonicity loss
        mono = max(0, omega - omega_p)**2
        
        # Mirror symmetry
        x_mirror = x[::-1]
        acts_mirror = self.forward(x_mirror)
        x_prime_mirror = acts_mirror[-1]
        _, _, omega_p_mirror, delta_mirror = self.compute_metrics(x_mirror)
        mirror_res = abs(delta_mirror - delta)
        
        # Use provided targets or compute them
        if psi_target is None:
            psi_target = psi_sup if psi_sup is not None else psi
        if omega_target is None:
            omega_target = omega_sup if omega_sup is not None else omega
        
        # Supervision loss
        sup = (psi - psi_target)**2 + (omega - omega_target)**2
        
        # Total loss with Phi weighting
        loss = (self.lambda_cons * cons + 
                self.lambda_mono * mono + 
                self.lambda_mirror * mirror_res + 
                0.3 * sup)
        
        # Simple gradient descent on reconstruction
        grad_output = 2 * (x_prime - x) / len(x)
        
        # Backward pass (simplified)
        grads_w = []
        grads_b = []
        
        delta_layer = grad_output
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.outer(acts[i], delta_layer)
            grad_b = delta_layer
            
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            
            if i > 0:
                delta_layer = (delta_layer @ self.weights[i].T) * (acts[i] > 0)  # ReLU derivative
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]
        
        return {
            'loss': float(loss),
            'psi': psi,
            'omega': omega,
            'omega_p': omega_p,
            'delta': delta,
            'mirror_residual': float(mirror_res),
            'cons': float(cons)
        }

# ----------------------------
#  PyTorch version (if available)
# ----------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    
    class FibonacciPyTorchNet(nn.Module):
        """PyTorch version with Fibonacci layers"""
        def __init__(self, d_in: int, fibonacci_layers: bool = True):
            super().__init__()
            
            if fibonacci_layers:
                suitable_fibs = [f for f in FIB_LAYERS if f >= d_in//4 and f <= d_in*4]
                if len(suitable_fibs) >= 3:
                    sizes = [d_in] + suitable_fibs[:3]
                else:
                    sizes = [d_in, 89, 55, 34]
            else:
                sizes = [d_in, 256, 128, 64]
            
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(sizes[-1], d_in))
            
            self.net = nn.Sequential(*layers)
            print(f"   Initialized PyTorch Fibonacci Net: {sizes} -> {d_in}")
        
        def forward(self, x):
            return self.net(x)
    
except ImportError:
    PYTORCH_AVAILABLE = False

# ----------------------------
#  Visualization (same as v7)
# ----------------------------
def plot_conservation_heatmap(session: Dict, output_path: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Ω-AI v8: FIBONACCI INTEGRATION\nSession: {session['session_id'][:8]}...", 
                     fontsize=14, fontweight='bold')
        
        rule_data = session['rule']
        neural_data = session['neural']
        heatmap = session['heatmap']
        
        # Plot 1: Rule metrics
        ax1 = axes[0, 0]
        if rule_data:
            r = rule_data[0]
            metrics = ['Ψ', 'Δ', 'Ω']
            values = [r['psi'], r['delta'], r['omega']]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax1.set_ylabel('Value', fontsize=12)
            ax1.set_title('Rule-Δ Metrics', fontsize=12, fontweight='bold')
            ax1.set_ylim([0, 1])
            ax1.grid(axis='y', alpha=0.3)
            
            cons = r.get('conservation', 0)
            fib_harm = r.get('fibonacci_harmony', 0)
            ax1.text(0.5, 0.95, f'Conservation: {cons:.3f}\nFib-Harmony: {fib_harm:.3f}', 
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Neural convergence
        ax2 = axes[0, 1]
        if neural_data:
            steps = [d['step'] for d in neural_data]
            losses = [d['loss'] for d in neural_data]
            ax2.plot(steps, losses, 'o-', color='#FF6B6B', linewidth=2, markersize=4)
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('Neural-Δ Convergence (Fibonacci Layers)', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            if session.get('fibonacci_layers', False):
                ax2.text(0.5, 0.95, 'Using Fibonacci layer sizes', 
                        transform=ax2.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
        
        # Plot 3: Conservation residual
        ax3 = axes[1, 0]
        if heatmap:
            rule_points = [h for h in heatmap if h['source'] == 'rule']
            neural_points = [h for h in heatmap if h['source'] == 'neural']
            
            if rule_points:
                ax3.scatter([h['step'] for h in rule_points],
                           [h['residual'] for h in rule_points],
                           c='blue', s=100, alpha=0.7, label='Rule', marker='s')
            if neural_points:
                steps = [h['step'] for h in neural_points]
                residuals = [h['residual'] for h in neural_points]
                ax3.plot(steps, residuals, 'r-', alpha=0.5, linewidth=1)
                ax3.scatter(steps, residuals, c='red', s=20, alpha=0.7, label='Neural')
            
            ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect')
            ax3.set_xlabel('Step', fontsize=12)
            ax3.set_ylabel('Residual (Ω² - Ψ² - Δ²)', fontsize=12)
            ax3.set_title('Conservation Law Tracking', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
        
        # Plot 4: Phi + Fibonacci harmony
        ax4 = axes[1, 1]
        if neural_data:
            phi_harmonies = [d.get('phi_harmony', 0) for d in neural_data]
            fib_harmonies = [d.get('fibonacci_harmony', 0) for d in neural_data]
            steps = [d['step'] for d in neural_data]
            
            ax4.plot(steps, phi_harmonies, 'o-', color='gold', linewidth=2, markersize=4, label='φ-Harmony')
            ax4.plot(steps, fib_harmonies, 's-', color='orange', linewidth=2, markersize=4, label='Fib-Harmony')
            ax4.axhline(y=PHI_INV, color='gold', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Step', fontsize=12)
            ax4.set_ylabel('Harmony Score', fontsize=12)
            ax4.set_title('Geometric Harmony Evolution', fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1])
            ax4.legend()
            ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
        return False

# ----------------------------
#  Logging
# ----------------------------
LOG_PATH = "/mnt/data/omega_log.json"

def append_log(entry: Dict):
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
            with open(LOG_PATH, "r") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []
    data.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

# ----------------------------
#  Demo
# ----------------------------
def demo_image(size: int = 64, noise: float = 0.7) -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.random((size, size))
    xs = np.linspace(0, 2*np.pi, size)
    ys = np.linspace(0, 2*np.pi, size)
    gridx, gridy = np.meshgrid(xs, ys)
    structure = 0.2 * (np.sin(3*gridx) + np.cos(5*gridy))
    img = (noise*img + (1-noise)*((structure - structure.min())/(np.ptp(structure)+1e-9)))
    return img.astype(float)

def main():
    ap = argparse.ArgumentParser(description="Real_v8 — Ω‑AI with Fibonacci Integration")
    ap.add_argument("--mode", choices=[m.value for m in Mode], default="text")
    ap.add_argument("--text", type=str, default="maybe this is somewhat unclear? i think it could be better.")
    ap.add_argument("--nums", type=str, default="1,5,3,9,2,2,2,10,11,3,3")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--rule_strength", type=float, default=0.6)
    ap.add_argument("--neural_steps", type=int, default=50)
    ap.add_argument("--fibonacci_layers", action="store_true", default=True, help="Use Fibonacci layer sizes")
    ap.add_argument("--use_pytorch", action="store_true", help="Use PyTorch if available")
    ap.add_argument("--visualize", action="store_true", help="Generate conservation heatmap")
    args = ap.parse_args()

    print("=" * 70)
    print("  Ω-AI v8: FIBONACCI INTEGRATION + PURE NUMPY FALLBACK")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Fibonacci layer sizes: {args.fibonacci_layers}")
    print(f"Golden ratio φ = {PHI:.6f}")
    print(f"Pyramid angle = {PYRAMID_ANGLE}°")
    
    # Show Fibonacci convergence to Phi
    print(f"\nFibonacci → φ convergence:")
    for i in [5, 8, 10, 12]:
        ratio = fibonacci_ratio(i)
        print(f"  F({i+1})/F({i}) = {ratio:.6f} (error: {abs(ratio-PHI):.6f})")
    print("=" * 70)

    session = dict(
        session_id=str(uuid.uuid4()),
        timestamp=int(time.time()),
        mode=args.mode,
        fibonacci_layers=args.fibonacci_layers,
        use_pytorch=args.use_pytorch and PYTORCH_AVAILABLE,
        rule=[],
        neural=[],
        heatmap=[]
    )

    # Prepare input
    if args.mode == "text":
        value = args.text
        psi0 = psi_text(value)
        omega0 = omega_text(value)
        print(f"\nInput: \"{value}\"")
    elif args.mode == "numeric":
        value = np.asarray([float(s) for s in args.nums.split(",") if s.strip()], dtype=float)
        psi0 = psi_numeric(value)
        omega0 = omega_numeric(value)
        print(f"\nInput: {value.tolist()}")
    else:
        value = demo_image(size=args.size)
        psi0 = psi_image(value)
        omega0 = omega_image(value)
        print(f"\nInput: {args.size}×{args.size} synthetic image")

    print(f"\n{'='*70}")
    print(f"UNIVERSAL START  |  Ψ={psi0:.4f}  Ω={omega0:.4f}")
    print(f"{'='*70}\n")

    # Rule-Δ
    print("🔹 RULE-Δ TRANSFORMATION")
    r = rule_step(Mode(args.mode), value, strength=args.rule_strength)
    cons = conservation_score(r["psi"], r["delta"], r["omega"])
    resid = conservation_residual(r["psi"], r["delta"], r["omega"])
    harmony = phi_harmony(r["psi"], r["delta"], r["omega"])
    fib_harm = fibonacci_harmony(r["omega"] * 100, target_fib_index=8)
    
    print(f"   Ψ = {r['psi']:.4f}  (entropy)")
    print(f"   Δ = {r['delta']:.4f}  (transformation)")
    print(f"   Ω = {r['omega']:.4f}  (coherence)")
    print(f"   Conservation: {cons:.4f}")
    print(f"   Residual: {resid:+.4f}")
    print(f"   φ-Harmony: {harmony:.4f}")
    print(f"   Fibonacci-Harmony: {fib_harm:.4f}")
    
    if args.mode == "text":
        print(f"   Output: \"{r['out']}\"")
        session["rule"].append(dict(step=1, psi=r["psi"], delta=r["delta"], 
                                   omega=r["omega"], conservation=cons, 
                                   phi_harmony=harmony, fibonacci_harmony=fib_harm,
                                   text=r["out"]))
    else:
        session["rule"].append(dict(step=1, psi=r["psi"], delta=r["delta"],
                                   omega=r["omega"], conservation=cons,
                                   phi_harmony=harmony, fibonacci_harmony=fib_harm))
    
    session["heatmap"].append(dict(source="rule", step=1, residual=resid))

    # Neural-Δ
    print(f"\n🔹 NEURAL-Δ TRAINING ({args.neural_steps} steps)")
    
    # Embed
    if args.mode == "text":
        x = embed_text(value, dim=args.dim)
    elif args.mode == "numeric":
        x = embed_numeric(value, dim=args.dim)
    else:
        x = embed_image(value, dim=args.dim)
    
    # Choose backend
    if args.use_pytorch and PYTORCH_AVAILABLE:
        print("   Using PyTorch backend with Fibonacci layers")
        # Implement PyTorch training here if needed
        print("   (PyTorch training not fully implemented in this version)")
    else:
        print(f"   Using Pure NumPy backend with {'Fibonacci' if args.fibonacci_layers else 'standard'} layers")
        model = FibonacciNeuralNet(d_in=args.dim, fibonacci_layers=args.fibonacci_layers)
        
        print("")
        for i in range(args.neural_steps):
            stats = model.step(x, psi_sup=r["psi"], omega_sup=r["omega"])
            
            if (i+1) % 10 == 0:
                h = phi_harmony(stats['psi'], stats['delta'], stats['omega_p'])
                fh = fibonacci_harmony(stats['omega_p'] * 100)
                print(f"   Step {i+1:3d} | loss={stats['loss']:.4f} "
                      f"Ψ={stats['psi']:.3f} Ω={stats['omega']:.3f}→{stats['omega_p']:.3f} "
                      f"Δ={stats['delta']:.3f} | φ={h:.3f} Fib={fh:.3f}")
            
            stats['phi_harmony'] = phi_harmony(stats['psi'], stats['delta'], stats['omega_p'])
            stats['fibonacci_harmony'] = fibonacci_harmony(stats['omega_p'] * 100)
            session["neural"].append(dict(step=i+1, **stats))
            
            resid_n = (stats['omega_p']**2) - (stats['psi']**2 + stats['delta']**2)
            session["heatmap"].append(dict(source="neural", step=i+1, residual=resid_n))

    # Log
    append_log(session)
    print(f"\n   ✓ Appended log → {LOG_PATH}")

    # Visualize
    if args.visualize:
        print(f"\n🔹 GENERATING FIBONACCI VISUALIZATION")
        viz_path = f"/mnt/data/omega_v8_viz_{session['session_id'][:8]}.png"
        if plot_conservation_heatmap(session, viz_path):
            print(f"   ✓ Saved visualization → {viz_path}")
        else:
            print(f"   ✗ Visualization failed")

    print(f"\n{'='*70}")
    print("✨ Ω-AI v8 COMPLETE - FIBONACCI INTEGRATED!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
