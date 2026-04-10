# Baseline Encoder–Decoder RNN (Sequence-to-Sequence)

**Laboratory Activity** — *"Hanapin Mo Ang Bit"*  

---

## Overview

This is a **pure NumPy implementation** of a baseline Encoder–Decoder RNN architecture for sequence-to-sequence learning. It demonstrates the fundamental architecture that serves as the foundation for all group protocols in the lab activity.

The baseline uses a simple **Identity (Echo) task** where the model learns to output the same sequence it receives as input. This allows you to see every component of the architecture clearly without domain-specific complexity.

### Key Features

✓ **100% NumPy implementation** (no PyTorch, TensorFlow, or Keras)  
✓ **SGD optimizer** with fixed learning rate (0.01)  
✓ **One-hot encoding** for inputs and outputs  
✓ **BPTT (Backpropagation Through Time)** with gradient clipping  
✓ **Real-time visualization** with 4-panel training dashboard  
✓ **Automatic PNG export** of results  

---

## Architecture

The model consists of two main components:

### Encoder
- Processes input sequence token-by-token
- Maintains hidden state across time steps
- Final hidden state = **context vector** (compressed representation of input)

### Decoder
- Uses teacher forcing during training
- Generates output tokens conditioned on context vector
- Outputs probability distribution at each step via softmax

```
INPUT → [Encoder RNN cells] → Context Vector → [Decoder RNN cells] → OUTPUT
                                                        ↓
                                                   Predictions
```

### Parameters

| Layer | Shape | Count |
|-------|-------|-------|
| Wx_enc | (8, 4) | 32 |
| Wh_enc | (8, 8) | 64 |
| bh_enc | (8, 1) | 8 |
| Wx_dec | (8, 4) | 32 |
| Wh_dec | (8, 8) | 64 |
| bh_dec | (8, 1) | 8 |
| Wy_dec | (4, 8) | 32 |
| by_dec | (4, 1) | 4 |
| **Total** | — | **244** |

---

## Installation & Requirements

```bash
pip install numpy matplotlib
```

**Requirements:**
- Python 3.7+
- NumPy
- Matplotlib

---

## How to Run

```bash
python baseline_enc_dec_rnn.py
```

The script will:
1. Generate 20 random training samples (identity task)
2. Initialize model parameters
3. Train for 300 epochs with progress logging
4. Evaluate on full training set
5. Display interactive visualization window
6. Save PNG plot to `baseline_rnn_results.png`

### Expected Runtime
~30-60 seconds on standard hardware

---

## Output & Visualization

After training completes, you'll see:

### 1. **Training Logs** (Terminal)
```
EPOCH    1 / 300
  >  Avg loss: 4.155681
  >    Sample 1: input=[2, 3, 0]  target=[2, 3, 0]  pred=[3, 2, 3]  [FAIL]
  ...
EPOCH  300 / 300
  >  Avg loss: 0.077815
  >    Sample 1: input=[2, 3, 0]  target=[2, 3, 0]  pred=[2, 3, 0]  [OK]
  ...
ACCURACY: 20/20  (100.0%)
```

### 2. **Interactive Plot Window** (Real-time display)

Four subplots show:

- **[LOSS] Loss vs. Iteration** — Training loss decreasing over epochs
- **[STATE] Encoder Hidden State Heatmap** — Neuron activations during encoding
- **[STATE] Decoder Hidden State Heatmap** — Neuron activations during decoding
- **[ARCH] Architecture Overview** — Visual diagram of the network structure

### 3. **PNG Export**
Saved as `baseline_rnn_results.png` for documentation/reports.

---

## Configuration

Edit these parameters at the top of `baseline_enc_dec_rnn.py`:

```python
VOCAB_SIZE   = 4          # Number of symbols (0, 1, 2, 3)
SEQ_LEN      = 3          # Fixed sequence length
HIDDEN_SIZE  = 8          # Hidden units in encoder & decoder
LEARNING_RATE = 0.01      # SGD learning rate
EPOCHS       = 300        # Training iterations
LOG_EVERY    = 50         # Log interval
NUM_SAMPLES  = 20         # Training examples
SEED         = 42         # Reproducibility
```

---

## Training Details

### Optimizer
- **Algorithm:** Stochastic Gradient Descent (SGD)
- **Learning Rate:** Fixed at 0.01
- **Gradient Clipping:** ±5.0 (prevents exploding gradients)
- **Batch Update:** One sample at a time

### Loss Function
- **Cross-Entropy Loss** computed at each decoder time step
- **Average loss** reported per epoch across all samples

### Training Dynamics
- **Epoch 1:** High loss (random initialization)
- **Epoch 100:** Loss drops (~1.6), model starts learning patterns
- **Epoch 300:** Loss converges (~0.08), 100% accuracy achieved

---

## Code Structure

```
Section 1:  Utility functions (one_hot, softmax, tanh, logging)
Section 2:  Dataset generation (identity/echo task)
Section 3:  Model parameter initialization
Section 4:  Forward pass (encoder + decoder)
Section 5:  Backward pass (BPTT)
Section 6:  SGD weight updates
Section 7:  Inference (greedy decoding)
Section 8:  Training loop
Section 9:  Evaluation
Section 10: Visualization
Section 11: Main entry point
```

---

## Mathematical Overview

### Forward Pass

**Encoder (time step t):**
```
a_t = Wx_enc · x_t + Wh_enc · h_{t-1} + bh_enc
h_t = tanh(a_t)
```

**Decoder (time step t):**
```
a_t = Wx_dec · x_t + Wh_dec · h_{t-1} + bh_dec
h_t = tanh(a_t)
y_t = Wy_dec · h_t + by_dec
probs_t = softmax(y_t)
```

### Loss

```
L = -Σ log(probs_t[y_t])   for all time steps
```

### Backward Pass
- Backprop through decoder first (gradient flows backward in time)
- Then backprop through encoder (gradient flows from context vector)
- Gradients accumulated for all parameters

---

## Expected Results

After 300 epochs of training on the echo task:

| Metric | Value |
|--------|-------|
| Final Loss | ~0.078 |
| Accuracy | 100% |
| Training Time | ~40 seconds |

The model successfully learns the identity mapping:
- Input: `[2, 3, 0]` → Output: `[2, 3, 0]` ✓
- Input: `[1, 2, 2]` → Output: `[1, 2, 2]` ✓
- All 20 training samples predicted correctly

---

## Extending for Your Group's Protocol

To adapt this baseline for your specific task:

1. **Replace `generate_dataset()`** with your protocol generator
2. **Adjust `VOCAB_SIZE`** and `SEQ_LEN` as needed
3. **Modify loss function** if using different target format
4. **Tune hyperparameters** (learning rate, hidden size, epochs)
5. **Keep architecture** (encoder–decoder structure is fixed per lab spec)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution:** Install dependencies
```bash
pip install numpy matplotlib
```

### Issue: Plot window doesn't appear
**Solution:** Ensure X11 forwarding is enabled (Linux/Mac) or use headless mode by changing one line:
```python
matplotlib.use("Agg")  # Uncomment to save only, no display
```

### Issue: Very high loss (not decreasing)
**Solution:** Try increasing learning rate or hidden size in configuration

### Issue: Slow training
**Solution:** Reduce `NUM_SAMPLES` or `EPOCHS` during experimentation

---

## References

- **RNN Basics:** Hochreiter et al. (1997) - LSTM paper
- **Seq2Seq:** Sutskever et al. (2014) - Sequence to Sequence Learning
- **BPTT:** Werbos (1990) - Backpropagation Through Time

---

## License

Laboratory Activity Material | BSCS 4-5 Group Project

---

**Last Updated:** April 2026  
**Status:** ✓ Working (tested on Python 3.13)