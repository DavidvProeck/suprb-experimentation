from datetime import datetime
from matplotlib import pyplot as plt, gridspec
from suprb.utils import estimate_bounds
import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from suprb.solution.base import Solution
from suprb.solution.fitness import ComplexityWu
from suprb.solution.mixing_model import ErrorExperienceHeuristic


def estimate_and_set_bounds(rule_discovery, X):
    bounds = estimate_bounds(X)
    for key, value in rule_discovery.get_params().items():
        if key.endswith("bounds") and value is None:
            print(f"Setting bounds for {key} based on data")
            rule_discovery.set_params(**{key: bounds})


def load_eggholder(n_samples=1000, noise=0.0, random_state=None):
    random_state = check_random_state(random_state)

    X = np.linspace(0, 20, num=n_samples)
    y = -(X + 47) * np.sin(np.sqrt(np.abs(X + 0.5 * X + 47))) \
        - X * np.sin(np.sqrt(np.abs(X - (X + 47)))) \
        + 5 * np.sin(1.5 * X)
    y += random_state.normal(scale=noise, size=y.shape)
    X = X.reshape(-1, 1)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))
    return X, y


def init_rule_discovery_env(rule_discovery):
    rule_discovery.pool_ = []
    rule_discovery.elitist_ = Solution([0, 0, 0], [0, 0, 0], ErrorExperienceHeuristic(), ComplexityWu())


def visualize_rule_predictions(X, y, rules, runtime, estimator=None, filename=None, show_params=True, subtitle=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.2)

    ax = fig.add_subplot(gs[0])
    ax.scatter(X, y, s=10, color="lightgray", label="Eggholder samples")
    colors = plt.cm.tab10(np.linspace(0, 1, len(rules)))
    for i, rule in enumerate(rules):
        if not rule.is_fitted_:
            continue
        mask = rule.match(X)
        Xm = X[mask]
        idx = np.argsort(Xm[:, 0])
        Xm_sorted = Xm[idx]
        y_pred = rule.predict(Xm_sorted)
        vol, err = rule.volume_, rule.error_
        ax.plot(Xm_sorted, y_pred,
                label=f"Rule {i+1} (V:{vol:.3f}, E:{err:.3f})",
                color=colors[i])
    title = "Rule Predictions on Eggholder Samples"
    if subtitle is not None:
        title += subtitle
    ax.set(title=title, xlabel="X", ylabel="Predicted y")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True)

    if show_params:
        ax_txt = fig.add_subplot(gs[1])
        ax_txt.axis("off")  # hide ticks
        if estimator is not None:
            params = estimator.get_params()
            lines = [f"{k} = {v}" for k, v in sorted(params.items())]
            lines.append(f"runtime = {runtime}")
            ax_txt.text(0, 1, "\n".join(lines), va="top", ha="left",
                        fontsize="x-small", family="monospace")

    if filename:
        pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_filename, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Vector PDF saved to '{pdf_filename}'")
    else:
        plt.show()


def build_filename(estimator):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cls = estimator.__class__.__name__
    return f"results/{cls}_{ts}"
