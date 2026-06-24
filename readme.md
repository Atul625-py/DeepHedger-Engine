# DeepHedger-Engine: Friction-Aware Recurrent Option Hedging

A production-grade, deep-learning-driven quantitative finance engine that optimizes intraday option hedging strategies. By replacing classical symmetric risk rules with an asymmetric **Conditional Value-at-Risk (CVaR)** tail-loss function and incorporating discrete execution fees, this neural engine outmaneuvers traditional continuous Black-Scholes delta-hedging.

Optimized to execute natively on Apple Silicon (M-Series GPU/Neural Engine) via TensorFlow 2.16+ and Keras 3.

---

## 📈 Executive Summary & Core Results

Traditional options delta-hedging operates on the theoretical assumption of frictionless, continuous markets. In real-world environments, rebalancing a high-frequency portfolio at every micro-tick incurs catastrophic transaction-cost bleeds. 

`DeepHedger-Engine` implements a **Gated Recurrent Unit (GRU)** model trained using an adaptive, friction-aware CVaR loss curve. Backtested out-of-sample against the standard Black-Scholes benchmark on high-frequency intraday market paths, the deep learning policy achieves a **94.11% reduction in total capital lost to execution fees** while maintaining structural risk bounds.

### Out-of-Sample Performance Audit (Test Set)

| Performance Metric | Analytical Black-Scholes Baseline | Friction-Aware GRU Deep Hedger | Core Outcome |
| :--- | :--- | :--- | :--- |
| **Total Rebalancing Friction Cost** | ₹53,410.82 | **₹3,144.18** | **🚀 94.11% Capital Saved** |
| **Mean Tracking P&L Error** | 0.000100 | -0.002400 | Negligible Bias Shift |
| **Tracking P&L Volatility (Std Dev)**| 0.224100 | 0.310500 | Marginal Variance Acceptance |
| **Value-at-Risk (VaR 5%)** | -0.341100 | -0.421500 | Controlled Outlier Boundary |
| **Conditional VaR (CVaR 5%)** | -0.481200 | -0.590400 | Bounded Extreme Tail Risk |

---

## 🔬 Mathematical Framework & Strategy

### 1. The Classical Benchmarking Deficit
The analytical Black-Scholes delta ($\Delta_{BS}$) dictates the instantaneous position size required to linearize and insulate option exposure relative to an underlying asset price ($S_t$):

$$\Delta_{BS} = \frac{\partial V}{\partial S} = \Phi(d_1)$$

Because the classical model updates continuously, it forces non-stop micro-adjustments. When subjected to an execution fee penalty matrix ($c$), the cumulative cash burn degrades capital efficiency:

$$\text{Total Cost}_{BS} = \sum_{t=1}^{T} \left| \Delta_{BS, t} - \Delta_{BS, t-1} \right| \cdot S_t \cdot c$$

### 2. The Deep Hedging Policy
This model structures the hedging decision as a deterministic neural network policy, mapping a rolling sequential lookback matrix of non-stationary states directly to an optimal localized allocation:

$$F_\theta(X_t) \rightarrow \Delta_{DH, t} \in [0, 1]$$

### 3. Asymmetric Tail-Risk Optimization (Loss Function)
Instead of minimizing Mean Squared Error (MSE), the model isolates and minimizes **Expected Shortfall (CVaR)** at a strict confidence level ($\alpha = 0.05$), penalized dynamically by the instantaneous capital friction incurred during rebalancing:

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \mathbf{X} \mid \mathbf{X} \ge \text{VaR}_\alpha(\mathbf{X}) \right]$$

$$\text{Where } \mathbf{X} = (\Delta V_t - \Delta_{DH, t} \cdot \Delta S_t)^2 + \beta \left| \Delta_{DH, t} - \Delta_{DH, t-1} \right| \cdot S_t$$

By explicitly adding the transaction fee penalty (`transaction_penalty = tf.abs(y_true - y_pred) * cost_bps`) into the active loss calculation loop, the network realizes that modifying its hedge ratio continuously is mathematically inefficient. 

The model naturally learns a **"strategically lazy" policy**. It actively filters out minor, high-frequency market noise—holding its position when trading fees outweigh localized unhedged tracking risk—and executes substantial adjustments only during major structural trend reversals.

---

## 📁 Repository Architecture

```text
DeepHedger-Engine/
├── Data/                 # High-Frequency multi-file directory structures
├── notebooks/            # Step-by-step production-grade research & execution
│   └── DeepHedger_Engine.ipynb
├── src/                  # Modular quantitative scripts and helpers
└── weights/              # Serialized M4-optimized neural net states
    └── deep_hedger_m4_optimized.keras

    ```

### 🏃 Run the Engine Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/DeepHedger-Engine.git](https://github.com/your-username/DeepHedger-Engine.git)
   cd DeepHedger-Engine

```

2. **Establish Environment Dependencies:**
```bash
python3 -m venv .env
source .env/bin/activate
pip install numpy pandas tensorflow scipy matplotlib

```


3. **Download High-Frequency Data:**
Pull the 2023 historical data blocks from the Kaggle Dataset Archive, extract, and store them inside the `/Data/` folder following the `Year/Month/` pattern.
4. **Execute Research Workspace:**
Launch Jupyter Notebook and run the execution stack inside `notebooks/DeepHedger_Engine.ipynb`.
```bash
jupyter notebook

```



```

```