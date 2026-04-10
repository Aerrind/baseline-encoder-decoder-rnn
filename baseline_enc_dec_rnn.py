"""
=============================================================================
BASELINE: Encoder–Decoder RNN (Sequence-to-Sequence)
Laboratory Activity — "Hanapin Mo Ang Bit"
Group 5 | BSCS 4-5

PURPOSE:
    This is the BASELINE implementation. It demonstrates the general
    Encoder–Decoder RNN architecture that all group protocols are built upon.
    No group-specific task is solved here — instead, it uses a simple
    IDENTITY (echo) task so you can see every moving part clearly.

CONSTRAINTS (as per lab):
    • NumPy only (no PyTorch, TensorFlow, Keras, etc.)
    • SGD optimizer
    • Fixed learning rate = 0.01
    • One-hot encoding for all inputs/outputs

HOW TO RUN:
    pip install numpy matplotlib
    python baseline_enc_dec_rnn.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (tweak these to experiment)
# ─────────────────────────────────────────────────────────────────────────────
# VOCAB_SIZE depends on protocol:
#   • mirror       : 10
#   • secret acct  : 13
#   • broken typewrt: 26
#   • liar         : 2
VOCAB_SIZE   = 10         # set to appropriate size for your protocol
SEQ_LEN      = 5          # fixed sequence length
HIDDEN_SIZE  = 128        # number of hidden units in encoder & decoder
LEARNING_RATE = 0.01      # SGD learning rate (fixed per lab spec)
EPOCHS       = 1000       # training iterations
LOG_EVERY    = 20         # print a log line every N epochs (captures ~50 snapshots)
NUM_SAMPLES  = 1000       # training examples to generate
TEST_SIZE    = 250        # test set size
SEED         = 42         # reproducibility

OUTPUT_DIR   = "."        # where plots are saved
PLOT_FILE    = os.path.join(OUTPUT_DIR, "baseline_rnn_results.png")

np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def one_hot(index: int, size: int) -> np.ndarray:
    """Return a one-hot column vector of length `size` for `index`."""
    v = np.zeros((size, 1))
    v[index] = 1.0
    return v


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis 0."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of tanh: 1 - tanh(x)^2."""
    return 1.0 - np.tanh(x) ** 2


def cross_entropy_loss(probs: np.ndarray, target_idx: int) -> float:
    """Scalar cross-entropy loss for a single time step."""
    return -float(np.log(probs[target_idx, 0] + 1e-9))


def log_section(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def log_step(msg: str) -> None:
    print(f"  >  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATASET  (identity / echo task)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(num_samples: int, seq_len: int, vocab_size: int):
    """
    BASELINE TASK — Identity / Echo
    Input  : [a, b, c]
    Output : [a, b, c]   (same sequence)

    This is the simplest possible task so the baseline shows the architecture
    without any domain-specific logic.  Each group will replace this function
    with their own protocol generator.
    """
    data = []
    for _ in range(num_samples):
        seq = [np.random.randint(0, vocab_size) for _ in range(seq_len)]
        data.append((seq, seq[:]))        # input == output for echo task
    return data


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def init_params(vocab_size: int, hidden_size: int) -> dict:
    """
    Xavier-ish random init for all weight matrices.

    ENCODER weights:
        Wx_enc  : input  -> hidden   (hidden_size x vocab_size)
        Wh_enc  : hidden -> hidden   (hidden_size x hidden_size)
        bh_enc  : bias              (hidden_size × 1)

    DECODER weights:
        Wx_dec  : input  -> hidden   (hidden_size x vocab_size)
        Wh_dec  : hidden -> hidden   (hidden_size x hidden_size)
        bh_dec  : bias              (hidden_size x 1)
        Wy_dec  : hidden -> output   (vocab_size  x hidden_size)
        by_dec  : output bias       (vocab_size  × 1)
    """
    scale = 0.1
    p = {
        # encoder
        "Wx_enc": np.random.randn(hidden_size, vocab_size)  * scale,
        "Wh_enc": np.random.randn(hidden_size, hidden_size) * scale,
        "bh_enc": np.zeros((hidden_size, 1)),
        # decoder
        "Wx_dec": np.random.randn(hidden_size, vocab_size)  * scale,
        "Wh_dec": np.random.randn(hidden_size, hidden_size) * scale,
        "bh_dec": np.zeros((hidden_size, 1)),
        "Wy_dec": np.random.randn(vocab_size,  hidden_size) * scale,
        "by_dec": np.zeros((vocab_size, 1)),
    }
    return p


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

def encoder_forward(input_seq: list, params: dict, vocab_size: int):
    """
    Read the input sequence one token at a time and return:
        h_enc  : final hidden state = context vector   (hidden_size × 1)
        h_hist : hidden states at every step            list of arrays
        x_hist : one-hot inputs at every step           list of arrays
        a_hist : pre-activation values at every step    list of arrays
    """
    h = np.zeros((params["Wx_enc"].shape[0], 1))   # h_0 = zeros
    h_hist, x_hist, a_hist = [], [], []

    for idx in input_seq:
        x = one_hot(idx, vocab_size)
        a = (params["Wx_enc"] @ x
             + params["Wh_enc"] @ h
             + params["bh_enc"])
        h = tanh(a)

        x_hist.append(x)
        a_hist.append(a)
        h_hist.append(h)

    return h, h_hist, x_hist, a_hist


def decoder_forward(context: np.ndarray, target_seq: list,
                    params: dict, vocab_size: int):
    """
    Generate output tokens conditioned on the context vector.
    Teacher forcing is used during training (ground-truth token fed each step).

    Returns:
        probs_hist : softmax output at each step   list of arrays
        h_hist     : hidden state at each step     list of arrays
        x_hist     : one-hot inputs at each step   list of arrays
        a_hist     : pre-activations               list of arrays
    """
    h = context
    probs_hist, h_hist, x_hist, a_hist = [], [], [], []

    # First decoder input: a zero vector (START token)
    prev_token = np.zeros((vocab_size, 1))

    for target_idx in target_seq:
        a = (params["Wx_dec"] @ prev_token
             + params["Wh_dec"] @ h
             + params["bh_dec"])
        h = tanh(a)
        y = params["Wy_dec"] @ h + params["by_dec"]
        probs = softmax(y)

        x_hist.append(prev_token)
        a_hist.append(a)
        h_hist.append(h)
        probs_hist.append(probs)

        # teacher forcing: next input = ground-truth current token
        prev_token = one_hot(target_idx, vocab_size)

    return probs_hist, h_hist, x_hist, a_hist


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BACKWARD PASS  (BPTT through decoder, then encoder)
# ─────────────────────────────────────────────────────────────────────────────

def backward(input_seq, target_seq,
             enc_h_hist, enc_x_hist, enc_a_hist,
             dec_probs_hist, dec_h_hist, dec_x_hist, dec_a_hist,
             context,
             params, vocab_size):
    """
    Full BPTT.  Returns a gradient dict with the same keys as `params`.
    """
    grads = {k: np.zeros_like(v) for k, v in params.items()}

    # ── 5a. Decoder backward ──────────────────────────────────────────────
    dh_next = np.zeros_like(context)   # gradient flowing from future steps

    for t in reversed(range(len(target_seq))):
        # output gradient: d(CE) / d(logits) = probs - one_hot(target)
        dy = dec_probs_hist[t].copy()
        dy[target_seq[t]] -= 1.0                   # shape (vocab_size, 1)

        grads["Wy_dec"] += dy @ dec_h_hist[t].T
        grads["by_dec"] += dy

        dh = params["Wy_dec"].T @ dy + dh_next     # gradient into hidden
        da = dh * tanh_grad(dec_a_hist[t])

        grads["Wx_dec"] += da @ dec_x_hist[t].T
        grads["Wh_dec"] += da @ (dec_h_hist[t - 1].T if t > 0 else context.T)
        grads["bh_dec"] += da

        dh_next = params["Wh_dec"].T @ da

    # gradient of context = sum of dh flowing into first decoder step
    # (dh_next after the loop covers t=0, which propagates into context)
    d_context = dh_next

    # ── 5b. Encoder backward ──────────────────────────────────────────────
    dh = d_context

    for t in reversed(range(len(input_seq))):
        da = dh * tanh_grad(enc_a_hist[t])

        grads["Wx_enc"] += da @ enc_x_hist[t].T
        grads["Wh_enc"] += da @ (enc_h_hist[t - 1].T if t > 0
                                  else np.zeros_like(enc_h_hist[0]).T)
        grads["bh_enc"] += da

        dh = params["Wh_enc"].T @ da

    return grads


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SGD UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def sgd_update(params: dict, grads: dict, lr: float) -> None:
    """In-place SGD: θ ← θ − lr · ∇θ"""
    for key in params:
        # gradient clipping to prevent exploding gradients
        np.clip(grads[key], -5.0, 5.0, out=grads[key])
        params[key] -= lr * grads[key]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — INFERENCE  (greedy decoding, no teacher forcing)
# ─────────────────────────────────────────────────────────────────────────────

def predict(input_seq: list, output_len: int,
            params: dict, vocab_size: int) -> list:
    """
    Run the model end-to-end without teacher forcing.
    Returns the predicted token indices.
    """
    context, _, _, _ = encoder_forward(input_seq, params, vocab_size)

    h = context
    prev_token = np.zeros((vocab_size, 1))
    predictions = []

    for _ in range(output_len):
        a = (params["Wx_dec"] @ prev_token
             + params["Wh_dec"] @ h
             + params["bh_dec"])
        h = tanh(a)
        y = params["Wy_dec"] @ h + params["by_dec"]
        probs = softmax(y)

        pred_idx = int(np.argmax(probs))
        predictions.append(pred_idx)
        prev_token = one_hot(pred_idx, vocab_size)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset, params, vocab_size, epochs, lr, log_every):
    loss_history = []
    hidden_state_snapshots = []          # for heatmap

    log_section("TRAINING STARTED")
    log_step(f"Samples      : {len(dataset)}")
    log_step(f"Epochs       : {epochs}")
    log_step(f"Learning rate: {lr}")
    log_step(f"Hidden size  : {params['Wx_enc'].shape[0]}")
    log_step(f"Vocab size   : {vocab_size}")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for input_seq, target_seq in dataset:

            # ── Forward ──────────────────────────────────────────────────
            context, enc_h_hist, enc_x_hist, enc_a_hist = \
                encoder_forward(input_seq, params, vocab_size)

            dec_probs, dec_h_hist, dec_x_hist, dec_a_hist = \
                decoder_forward(context, target_seq, params, vocab_size)

            # ── Loss ──────────────────────────────────────────────────────
            step_loss = sum(
                cross_entropy_loss(dec_probs[t], target_seq[t])
                for t in range(len(target_seq))
            )
            epoch_loss += step_loss

            # ── Backward ─────────────────────────────────────────────────
            grads = backward(
                input_seq, target_seq,
                enc_h_hist, enc_x_hist, enc_a_hist,
                dec_probs, dec_h_hist, dec_x_hist, dec_a_hist,
                context, params, vocab_size,
            )

            # ── SGD Update ───────────────────────────────────────────────
            sgd_update(params, grads, lr)

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

        # capture hidden states of the first sample for heatmap
        if epoch % log_every == 0 or epoch == 1:
            ctx, enc_h, _, _ = encoder_forward(
                dataset[0][0], params, vocab_size)
            hidden_state_snapshots.append(
                (epoch, np.hstack(enc_h))           # (hidden, seq_len)
            )

            log_section(f"EPOCH {epoch:>4d} / {epochs}")
            log_step(f"Avg loss: {avg_loss:.6f}")

            # show a couple of sample predictions
            for i, (inp, tgt) in enumerate(dataset[:3]):
                pred = predict(inp, len(tgt), params, vocab_size)
                correct = "[OK]" if pred == tgt else "[FAIL]"
                log_step(
                    f"  Sample {i+1}: input={inp}  "
                    f"target={tgt}  pred={pred}  {correct}"
                )

    return loss_history, hidden_state_snapshots


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(dataset, params, vocab_size):
    log_section("EVALUATION ON FULL TRAINING SET")
    correct = 0
    for inp, tgt in dataset:
        pred = predict(inp, len(tgt), params, vocab_size)
        status = "[OK]" if pred == tgt else "[FAIL]"
        log_step(f"input={inp}  target={tgt}  pred={pred}  {status}")
        if pred == tgt:
            correct += 1
    acc = correct / len(dataset) * 100
    log_section(f"ACCURACY: {correct}/{len(dataset)}  ({acc:.1f}%)")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(loss_history, hidden_state_snapshots,
                 params, dataset, vocab_size, output_path):
    """
    Produces a 2×2 figure:
      [0,0]  Loss vs. Iteration
      [0,1]  Encoder hidden-state heatmap (last epoch)
      [1,0]  Decoder hidden-state heatmap (last epoch, first sample)
      [1,1]  Architecture diagram (schematic)
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f1a")
    fig.suptitle(
        "Baseline Encoder–Decoder RNN  ·  Hanapin Mo Ang Bit",
        fontsize=15, color="white", fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.96,
                           top=0.93, bottom=0.07)

    DARK_BG  = "#0f0f1a"
    PANEL_BG = "#1a1a2e"
    ACCENT   = "#7f5af0"
    GREEN    = "#2cb67d"
    TEXT     = "#fffffe"
    SUBTEXT  = "#94a1b2"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)

    # ── Plot 1: Loss curve ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    style_ax(ax0, "[LOSS]  Loss vs. Iteration")
    epochs_range = np.arange(1, len(loss_history) + 1)
    ax0.plot(epochs_range, loss_history, color=ACCENT, linewidth=1.5)
    ax0.fill_between(epochs_range, loss_history,
                     alpha=0.15, color=ACCENT)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Cross-Entropy Loss")
    ax0.grid(True, color="#333355", linewidth=0.5)

    # ── Plot 2: Encoder hidden-state heatmap (final epoch) ───────────────
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1, "[STATE]  Encoder Hidden State Heatmap (Final Epoch)")

    ctx, enc_h_hist, _, _ = encoder_forward(
        dataset[0][0], params, vocab_size)
    heatmap_data = np.hstack(enc_h_hist)          # (hidden_size, seq_len)

    im1 = ax1.imshow(heatmap_data, aspect="auto", cmap="plasma",
                     vmin=-1, vmax=1)
    ax1.set_xlabel("Encoder Time Step")
    ax1.set_ylabel("Hidden Unit")
    ax1.set_xticks(range(heatmap_data.shape[1]))
    ax1.set_xticklabels([f"t={i}" for i in range(heatmap_data.shape[1])])
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors=SUBTEXT, labelsize=7)

    # ── Plot 3: Decoder hidden-state heatmap (final epoch) ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, "[STATE]  Decoder Hidden State Heatmap (Final Epoch)")

    dec_probs, dec_h_hist, _, _ = decoder_forward(
        ctx, dataset[0][1], params, vocab_size)
    dec_heatmap = np.hstack(dec_h_hist)

    im2 = ax2.imshow(dec_heatmap, aspect="auto", cmap="viridis",
                     vmin=-1, vmax=1)
    ax2.set_xlabel("Decoder Time Step")
    ax2.set_ylabel("Hidden Unit")
    ax2.set_xticks(range(dec_heatmap.shape[1]))
    ax2.set_xticklabels([f"t={i}" for i in range(dec_heatmap.shape[1])])
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors=SUBTEXT, labelsize=7)

    # ── Plot 4: Architecture schematic ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL_BG)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis("off")
    ax3.set_title("[ARCH]  Architecture Overview", color=TEXT,
                  fontsize=10, fontweight="bold", pad=8)

    def box(ax, x, y, w, h, label, color, fontsize=7.5):
        rect = plt.Rectangle((x, y), w, h,
                              facecolor=color, edgecolor="white",
                              linewidth=0.8, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold")

    def arrow(ax, x1, y1, x2, y2, color="#aaaacc"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->",
                                   color=color, lw=1.2))

    # encoder cells
    enc_color = "#7f5af0"
    for i in range(SEQ_LEN):
        xi = 0.5 + i * 2.0
        box(ax3, xi, 3.5, 1.5, 0.9, f"Enc\nt={i}", enc_color)
        if i > 0:
            arrow(ax3, xi - 0.5, 3.95, xi, 3.95)

    # context vector
    box(ax3, 6.3, 3.5, 1.4, 0.9, "Context\nVector", "#e53170")
    arrow(ax3, 5.5, 3.95, 6.3, 3.95, color="#e53170")

    # decoder cells
    dec_color = "#2cb67d"
    for i in range(SEQ_LEN):
        xi = 0.5 + i * 2.0
        box(ax3, xi, 1.2, 1.5, 0.9, f"Dec\nt={i}", dec_color)
        if i > 0:
            arrow(ax3, xi - 0.5, 1.65, xi, 1.65)

    # context -> first decoder
    arrow(ax3, 7.0, 3.5, 1.25, 2.1, color="#e53170")

    # output labels
    for i in range(SEQ_LEN):
        xi = 0.5 + i * 2.0 + 0.75
        ax3.text(xi, 0.85, f"ŷ{i}", ha="center",
                 color=GREEN, fontsize=8, fontweight="bold")
        arrow(ax3, xi, 1.2, xi, 1.0)

    # labels
    ax3.text(2.5, 5.0, "ENCODER", ha="center",
             color=enc_color, fontsize=9, fontweight="bold")
    ax3.text(2.5, 0.4, "DECODER", ha="center",
             color=dec_color, fontsize=9, fontweight="bold")
    ax3.text(7.0, 4.7, "SGD·lr=0.01\none-hot inputs",
             ha="center", color=SUBTEXT, fontsize=7)

    # ── Save & Display ────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    print(f"\n  [OK]  Plot saved -> {os.path.abspath(output_path)}")
    plt.show()  # Display plot in real-time
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log_section("BASELINE ENCODER–DECODER RNN  |  Hanapin Mo Ang Bit")
    log_step("Task          : Identity / Echo  (input = output)")
    log_step("Architecture  : Encoder–Decoder RNN (NumPy only)")
    log_step("Optimizer     : SGD")
    log_step(f"Learning rate : {LEARNING_RATE}")
    log_step("One-hot input : Yes")

    # 1. Dataset
    log_section("STEP 1 — GENERATING DATASET")
    dataset = generate_dataset(NUM_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    log_step(f"Generated {len(dataset)} samples | "
             f"vocab={VOCAB_SIZE} | seq_len={SEQ_LEN}")
    for i, (inp, tgt) in enumerate(dataset[:5]):
        log_step(f"  Sample {i+1}: input={inp}  ->  target={tgt}")
    log_step("  ... (showing first 5 of dataset)")

    # 2. Model init
    log_section("STEP 2 — INITIALISING MODEL PARAMETERS")
    params = init_params(VOCAB_SIZE, HIDDEN_SIZE)
    total_params = sum(v.size for v in params.values())
    for name, w in params.items():
        log_step(f"  {name:10s}  shape={str(w.shape):18s}  "
                 f"mean={w.mean():.4f}  std={w.std():.4f}")
    log_step(f"\n  Total trainable parameters: {total_params}")

    # 3. Training
    loss_history, hidden_snapshots = train(
        dataset, params, VOCAB_SIZE, EPOCHS, LEARNING_RATE, LOG_EVERY)

    # 4. Evaluation
    accuracy = evaluate(dataset, params, VOCAB_SIZE)

    # 5. Visualisation
    log_section("STEP 5 — GENERATING PLOTS")
    plot_results(loss_history, hidden_snapshots,
                 params, dataset, VOCAB_SIZE, PLOT_FILE)

    # 6. Summary
    log_section("SUMMARY")
    log_step(f"Final avg loss : {loss_history[-1]:.6f}")
    log_step(f"Accuracy       : {accuracy:.1f}%")
    log_step(f"Plot saved to  : {os.path.abspath(PLOT_FILE)}")
    log_section("DONE — extend this baseline for your group's protocol!")


if __name__ == "__main__":
    main()
