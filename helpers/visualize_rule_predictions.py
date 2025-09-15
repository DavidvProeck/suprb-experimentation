import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.gridspec as gridspec

def visualize_rule_predictions_1D(X, y, rules, runtime, estimator=None, filename=None, show_params = True, subtitle=None):
    # Create figure with 2 rows:  top for the plot (4 units high), bottom for text (1 unit high)
    fig = plt.figure(figsize=(12, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.2)

    # ── TOP AXIS: your scatter + lines ─────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.scatter(X, y, s=10, color="lightgray", label="Eggholder samples")
    colors = plt.cm.tab10(np.linspace(0, 1, len(rules)))
    for i, rule in enumerate(rules):
        if not rule.is_fitted_:
            continue
        mask      = rule.match(X)
        Xm        = X[mask]
        idx       = np.argsort(Xm[:, 0])
        Xm_sorted = Xm[idx]
        y_pred    = rule.predict(Xm_sorted)
        vol, err  = rule.volume_, rule.error_
        ax.plot(Xm_sorted, y_pred,
                label=f"Rule {i+1} (V:{vol:.3f}, E:{err:.3f})",
                color=colors[i])
    if subtitle is not None:
        ax.set( title=f"Rule Predictions on Eggholder Samples" + subtitle,
            xlabel="X",
            ylabel="Predicted y" )
    else:
        ax.set( title="Rule Predictions on Eggholder Samples",
                xlabel="X",
                ylabel="Predicted y" )
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True)

    # ── BOTTOM AXIS: parameter block ─────────────────────────────────
    if show_params:
        ax_txt = fig.add_subplot(gs[1])
        ax_txt.axis("off")  # hide ticks

        if estimator is not None:
            params = estimator.get_params()
            lines = [f"{k} = {v}" for k, v in sorted(params.items())]
            lines.append(f"runtime = {runtime}")
            block = "\n".join(lines)
            # place text at top-left of this small axes
            ax_txt.text(
                0, 1, block,
                va="top", ha="left",
                fontsize="x-small", family="monospace"
            )

    # ── SAVE/SHOW ────────────────────────────────────────────────
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def build_filename(estimator):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cls = estimator.__class__.__name__
    return f"results/{cls}_{ts}"
