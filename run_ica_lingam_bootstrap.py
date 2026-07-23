from __future__ import annotations

from pathlib import Path
import json
import time
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from causallearn.search.FCMBased import lingam


# ============================================================
# CẤU HÌNH THỰC NGHIỆM
# ============================================================

RANDOM_STATE = 42
MAX_ITER = 3000

EDGE_THRESHOLD = 0.05
N_BOOTSTRAP = 500
STABILITY_THRESHOLD = 0.80
SIGN_CONSISTENCY_THRESHOLD = 0.80

FAIL_ON_CONVERGENCE_WARNING = False
SHOW_ISOLATED_NODES_IN_STABLE_GRAPH = False

OUTPUT_DIR = Path("outputs_ica_lingam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REMOVED_COLUMNS = [
    "Overall Height",
    "Orientation",
    "Glazing Area Distribution",
]

ANALYSIS_COLUMNS = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Glazing Area",
    "Heating Load",
    "Cooling Load",
]


# ============================================================
# 1. TẢI DATASET
# ============================================================

def load_energy_efficiency() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=242)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    df = pd.concat([features, targets], axis=1)

    rename_map = {
        "X1": "Relative Compactness",
        "X2": "Surface Area",
        "X3": "Wall Area",
        "X4": "Roof Area",
        "X5": "Overall Height",
        "X6": "Orientation",
        "X7": "Glazing Area",
        "X8": "Glazing Area Distribution",
        "Y1": "Heating Load",
        "Y2": "Cooling Load",
    }

    df = df.rename(columns=rename_map)

    expected_columns = [
        "Relative Compactness",
        "Surface Area",
        "Wall Area",
        "Roof Area",
        "Overall Height",
        "Orientation",
        "Glazing Area",
        "Glazing Area Distribution",
        "Heating Load",
        "Cooling Load",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Không tìm thấy đầy đủ các cột mong đợi.\n"
            f"Các cột bị thiếu: {missing_columns}\n"
            f"Các cột hiện có: {df.columns.tolist()}"
        )

    return df[expected_columns]


# ============================================================
# 2. TÓM TẮT BIẾN
# ============================================================

def summarize_variables(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for column in df.columns:
        unique_values = np.sort(df[column].dropna().unique())

        records.append(
            {
                "Variable": column,
                "Data type": str(df[column].dtype),
                "Number of unique values": int(df[column].nunique()),
                "Minimum": float(df[column].min()),
                "Maximum": float(df[column].max()),
                "Unique values": (
                    unique_values.tolist()
                    if len(unique_values) <= 20
                    else "More than 20 values"
                ),
            }
        )

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_DIR / "variable_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result


# ============================================================
# 3. TIỀN XỬ LÝ
# ============================================================

def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_removed_columns = [
        column
        for column in REMOVED_COLUMNS
        if column not in df.columns
    ]

    if missing_removed_columns:
        raise ValueError(
            "Không tìm thấy các cột cần loại: "
            f"{missing_removed_columns}"
        )

    filtered = df.drop(columns=REMOVED_COLUMNS).copy()

    missing_analysis_columns = [
        column
        for column in ANALYSIS_COLUMNS
        if column not in filtered.columns
    ]

    if missing_analysis_columns:
        raise ValueError(
            "Không tìm thấy các biến phân tích: "
            f"{missing_analysis_columns}"
        )

    filtered = filtered[ANALYSIS_COLUMNS]
    filtered = filtered.apply(
        pd.to_numeric,
        errors="raise",
    )

    if filtered.empty:
        raise ValueError("Dữ liệu bị rỗng sau tiền xử lý.")

    if filtered.isna().any().any():
        missing_counts = filtered.isna().sum()
        raise ValueError(
            "Dữ liệu có missing values:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    values = filtered.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Dữ liệu chứa NaN hoặc infinity.")

    return filtered


# ============================================================
# 4. CHẨN ĐOÁN MA TRẬN
# ============================================================

def save_matrix_diagnostics(df: pd.DataFrame) -> dict:
    X = df.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(X))

    diagnostics = {
        "n_samples": int(X.shape[0]),
        "n_variables": int(X.shape[1]),
        "matrix_rank": rank,
        "full_column_rank": bool(rank == X.shape[1]),
        "condition_number": float(np.linalg.cond(X)),
    }

    with open(
        OUTPUT_DIR / "matrix_diagnostics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)

    df.corr(method="pearson").to_csv(
        OUTPUT_DIR / "pearson_correlation.csv",
        encoding="utf-8-sig",
    )

    return diagnostics


# ============================================================
# 5. CHUẨN HÓA
# ============================================================

def standardize_array(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if not np.isfinite(X_scaled).all():
        raise ValueError("Dữ liệu không hữu hạn sau chuẩn hóa.")

    return X_scaled, scaler


# ============================================================
# 6. CHẠY ICA-LINGAM
# ============================================================

def fit_ica_lingam(
    X: np.ndarray,
    random_state: int,
    max_iter: int,
) -> dict:
    model = lingam.ICALiNGAM(
        random_state=random_state,
        max_iter=max_iter,
    )

    convergence_warning_messages: list[str] = []
    start_time = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter(
            "always",
            category=ConvergenceWarning,
        )
        model.fit(X)

    runtime_seconds = time.perf_counter() - start_time

    for warning_item in caught_warnings:
        if issubclass(warning_item.category, ConvergenceWarning):
            convergence_warning_messages.append(str(warning_item.message))

    converged = len(convergence_warning_messages) == 0

    if FAIL_ON_CONVERGENCE_WARNING and not converged:
        raise RuntimeError(
            "FastICA không hội tụ.\n" + "\n".join(convergence_warning_messages)
        )

    adjacency_matrix = np.asarray(
        model.adjacency_matrix_,
        dtype=float,
    )

    expected_shape = (X.shape[1], X.shape[1])

    if adjacency_matrix.shape != expected_shape:
        raise ValueError(
            "adjacency_matrix_ có kích thước không hợp lệ.\n"
            f"Kỳ vọng: {expected_shape}\n"
            f"Nhận được: {adjacency_matrix.shape}"
        )

    if not np.isfinite(adjacency_matrix).all():
        raise ValueError("adjacency_matrix_ chứa NaN hoặc infinity.")

    np.fill_diagonal(adjacency_matrix, 0.0)

    causal_order = [int(index) for index in model.causal_order_]

    return {
        "model": model,
        "adjacency_matrix": adjacency_matrix,
        "causal_order": causal_order,
        "runtime_seconds": float(runtime_seconds),
        "converged": converged,
        "convergence_warnings": convergence_warning_messages,
    }


# ============================================================
# 7. MA TRẬN KỀ -> DANH SÁCH CẠNH
# ============================================================

def adjacency_to_edges(
    adjacency_matrix: np.ndarray,
    variable_names: list[str],
    threshold: float,
) -> pd.DataFrame:
    records = []
    n_variables = len(variable_names)

    for effect_index in range(n_variables):
        for cause_index in range(n_variables):
            if effect_index == cause_index:
                continue

            coefficient = float(adjacency_matrix[effect_index, cause_index])

            if abs(coefficient) < threshold:
                continue

            records.append(
                {
                    "Cause": variable_names[cause_index],
                    "Effect": variable_names[effect_index],
                    "Coefficient": coefficient,
                    "Absolute coefficient": abs(coefficient),
                    "Sign": "Positive" if coefficient > 0 else "Negative",
                }
            )

    columns = [
        "Cause",
        "Effect",
        "Coefficient",
        "Absolute coefficient",
        "Sign",
    ]

    result = pd.DataFrame(records, columns=columns)

    if not result.empty:
        result = result.sort_values(
            by="Absolute coefficient",
            ascending=False,
        ).reset_index(drop=True)

    return result


# ============================================================
# 8. TÌM CẠNH HAI CHIỀU TRONG KẾT QUẢ BOOTSTRAP
# ============================================================

def find_bidirectional_stable_edges(
    stable_edges: pd.DataFrame,
) -> pd.DataFrame:
    edge_lookup = {
        (row["Cause"], row["Effect"]): row
        for row in stable_edges.to_dict("records")
    }

    records = []
    processed_pairs = set()

    for cause, effect in edge_lookup:
        pair = tuple(sorted([cause, effect]))

        if pair in processed_pairs:
            continue

        reverse = (effect, cause)

        if reverse in edge_lookup:
            forward_row = edge_lookup[(cause, effect)]
            reverse_row = edge_lookup[reverse]

            records.append(
                {
                    "Variable 1": cause,
                    "Variable 2": effect,
                    "Frequency 1->2": float(forward_row["Selection_frequency"]),
                    "Frequency 2->1": float(reverse_row["Selection_frequency"]),
                    "Median coefficient 1->2": float(forward_row["Median_coefficient"]),
                    "Median coefficient 2->1": float(reverse_row["Median_coefficient"]),
                }
            )

        processed_pairs.add(pair)

    return pd.DataFrame(records)


# ============================================================
# 9. BỐ CỤC ĐỒ THỊ DỄ ĐỌC
# ============================================================

def build_layered_positions(
    ordered_nodes: list[str],
) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}

    n_nodes = len(ordered_nodes)
    split_index = int(np.ceil(n_nodes / 2))

    first_row = ordered_nodes[:split_index]
    second_row = ordered_nodes[split_index:]

    for index, node in enumerate(first_row):
        positions[node] = (float(index), 1.0)

    offset = (
        (len(first_row) - len(second_row)) / 2
        if len(second_row) > 0
        else 0.0
    )

    for index, node in enumerate(second_row):
        positions[node] = (float(index + offset), 0.0)

    return positions


# ============================================================
# 10. VẼ ĐỒ THỊ DỄ ĐỌC
# ============================================================

def plot_causal_graph_readable(
    edges_df: pd.DataFrame,
    variable_names: list[str],
    output_path: Path,
    title: str,
    causal_order: list[str] | None = None,
    frequency_column: str | None = None,
    show_isolated_nodes: bool = True,
) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    for row in edges_df.to_dict("records"):
        attributes = {"coefficient": float(row["Coefficient"])}

        if frequency_column is not None and frequency_column in row:
            attributes["frequency"] = float(row[frequency_column])

        graph.add_edge(row["Cause"], row["Effect"], **attributes)

    if not show_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    if graph.number_of_nodes() == 0:
        return

    if causal_order is not None:
        ordered_nodes = [node for node in causal_order if node in graph.nodes]
        ordered_nodes.extend(
            node for node in graph.nodes if node not in ordered_nodes
        )
    else:
        ordered_nodes = list(graph.nodes())

    positions = build_layered_positions(ordered_nodes)
    n_top = int(np.ceil(len(ordered_nodes) / 2))

    figure_width = max(12, 2.7 * max(n_top, 1))
    plt.figure(figsize=(figure_width, 7.5))

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=3300,
        node_color="lightblue",
        edgecolors="black",
        linewidths=1.2,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9,
        font_weight="bold",
    )

    bidirectional_pairs = set()
    for cause, effect in graph.edges():
        if graph.has_edge(effect, cause):
            bidirectional_pairs.add(tuple(sorted([cause, effect])))

    for cause, effect, attributes in graph.edges(data=True):
        coefficient = float(attributes["coefficient"])
        edge_color = "green" if coefficient > 0 else "red"
        edge_style = "solid" if coefficient > 0 else "dashed"

        pair = tuple(sorted([cause, effect]))
        radius = 0.24 if pair in bidirectional_pairs and cause < effect else -0.24 if pair in bidirectional_pairs else 0.06

        edge_width = (
            1.4 + 2.6 * float(attributes["frequency"])
            if "frequency" in attributes
            else 2.0
        )

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(cause, effect)],
            edge_color=edge_color,
            style=edge_style,
            width=edge_width,
            arrows=True,
            arrowsize=22,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={radius}",
            min_source_margin=22,
            min_target_margin=22,
        )

    edge_labels = {}
    for cause, effect, attributes in graph.edges(data=True):
        coefficient = float(attributes["coefficient"])
        if "frequency" in attributes:
            edge_labels[(cause, effect)] = (
                f"β={coefficient:.2f}\nf={attributes['frequency']:.2f}"
            )
        else:
            edge_labels[(cause, effect)] = f"β={coefficient:.2f}"

    label_artists = nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False,
        label_pos=0.55,
    )

    for label in label_artists.values():
        label.set_bbox(
            {
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 1.4,
            }
        )

    plt.plot([], [], color="green", linewidth=2.0, label="Positive coefficient")
    plt.plot(
        [],
        [],
        color="red",
        linestyle="dashed",
        linewidth=2.0,
        label="Negative coefficient",
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=18)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    plt.axis("off")

    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]

    plt.xlim(min(x_values) - 0.7, max(x_values) + 0.7)
    plt.ylim(min(y_values) - 0.55, max(y_values) + 0.55)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 11. BOOTSTRAP ICA-LINGAM
# ============================================================

def bootstrap_ica_lingam(
    X_original: np.ndarray,
    variable_names: list[str],
    n_bootstrap: int,
    edge_threshold: float,
    random_state: int,
    max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    n_samples, n_variables = X_original.shape

    edge_records = []
    run_records = []

    print("\n" + "=" * 80)
    print(f"BOOTSTRAP ICA-LINGAM: {n_bootstrap} LẦN")
    print("=" * 80)

    for bootstrap_index in range(n_bootstrap):
        sample_indices = rng.integers(low=0, high=n_samples, size=n_samples)

        X_bootstrap_raw = X_original[sample_indices, :]
        X_bootstrap, _ = standardize_array(X_bootstrap_raw)

        result = fit_ica_lingam(
            X=X_bootstrap,
            random_state=(random_state + bootstrap_index),
            max_iter=max_iter,
        )

        adjacency_matrix = result["adjacency_matrix"]
        causal_order_names = [
            variable_names[index] for index in result["causal_order"]
        ]

        run_records.append(
            {
                "Bootstrap ID": bootstrap_index + 1,
                "Runtime seconds": result["runtime_seconds"],
                "Converged": result["converged"],
                "Number of convergence warnings": len(
                    result["convergence_warnings"]
                ),
                "Convergence warnings": " | ".join(
                    result["convergence_warnings"]
                ),
                "Causal order": " -> ".join(causal_order_names),
            }
        )

        for effect_index in range(n_variables):
            for cause_index in range(n_variables):
                if effect_index == cause_index:
                    continue

                coefficient = float(
                    adjacency_matrix[effect_index, cause_index]
                )

                edge_records.append(
                    {
                        "Bootstrap ID": bootstrap_index + 1,
                        "Cause": variable_names[cause_index],
                        "Effect": variable_names[effect_index],
                        "Coefficient": coefficient,
                        "Absolute coefficient": abs(coefficient),
                        "Selected": abs(coefficient) >= edge_threshold,
                        "Positive selected": coefficient >= edge_threshold,
                        "Negative selected": coefficient <= -edge_threshold,
                        "Run converged": result["converged"],
                    }
                )

        completed = bootstrap_index + 1
        if completed == 1 or completed % 10 == 0 or completed == n_bootstrap:
            print(f"Đã hoàn thành {completed}/{n_bootstrap} lần.")

    raw_results = pd.DataFrame(edge_records)
    run_results = pd.DataFrame(run_records)

    edge_summary = (
        raw_results
        .groupby(["Cause", "Effect"], as_index=False)
        .agg(
            Selection_frequency=("Selected", "mean"),
            Selection_count=("Selected", "sum"),
            Mean_coefficient=("Coefficient", "mean"),
            Median_coefficient=("Coefficient", "median"),
            Coefficient_std=("Coefficient", "std"),
            Mean_absolute_coefficient=("Absolute coefficient", "mean"),
            Positive_frequency=("Positive selected", "mean"),
            Negative_frequency=("Negative selected", "mean"),
            Converged_run_frequency=("Run converged", "mean"),
        )
    )

    edge_summary["Sign_consistency"] = edge_summary[
        ["Positive_frequency", "Negative_frequency"]
    ].max(axis=1)

    edge_summary["Dominant_sign"] = np.where(
        edge_summary["Positive_frequency"] > edge_summary["Negative_frequency"],
        "Positive",
        np.where(
            edge_summary["Negative_frequency"] > edge_summary["Positive_frequency"],
            "Negative",
            "Mixed/None",
        ),
    )

    edge_summary = edge_summary.sort_values(
        by=["Selection_frequency", "Sign_consistency", "Mean_absolute_coefficient"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return raw_results, edge_summary, run_results


# ============================================================
# 12. LỌC CẠNH ỔN ĐỊNH
# ============================================================

def select_stable_edges(
    bootstrap_summary: pd.DataFrame,
    stability_threshold: float,
    sign_consistency_threshold: float,
) -> pd.DataFrame:
    stable = bootstrap_summary[
        (bootstrap_summary["Selection_frequency"] >= stability_threshold)
        & (bootstrap_summary["Sign_consistency"] >= sign_consistency_threshold)
    ].copy()

    stable["Coefficient"] = stable["Median_coefficient"]

    if stable.empty:
        return stable

    stable["Absolute representative coefficient"] = stable["Coefficient"].abs()

    stable = stable.sort_values(
        by=[
            "Selection_frequency",
            "Sign_consistency",
            "Absolute representative coefficient",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return stable


# ============================================================
# 13. BỔ SUNG: HÀM ĐÁNH GIÁ ĐỘ ÔN ĐỊNH CAUSAL ORDER (KENDALL TAU)
# ============================================================

def compute_kendall_tau(order1: list[str], order2: list[str]) -> float:
    pos2 = {var: i for i, var in enumerate(order2)}
    ranks = [pos2[var] for var in order1 if var in pos2]
    n = len(ranks)
    if n <= 1:
        return 1.0

    concordant, discordant = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] < ranks[j]:
                concordant += 1
            elif ranks[i] > ranks[j]:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0


def summarize_causal_order_stability(
    bootstrap_runs: pd.DataFrame,
    full_data_causal_order: list[str],
) -> dict:
    full_order_str = " -> ".join(full_data_causal_order)
    orders = bootstrap_runs["Causal order"].tolist()
    n_runs = len(orders)

    exact_matches = sum(1 for o in orders if o == full_order_str)
    order_counts = pd.Series(orders).value_counts()

    most_frequent_order = order_counts.index[0]
    most_frequent_count = int(order_counts.iloc[0])

    kendall_taus = [
        compute_kendall_tau(full_data_causal_order, o.split(" -> "))
        for o in orders
    ]

    return {
        "full_data_causal_order": full_order_str,
        "exact_match_with_full_order_count": exact_matches,
        "exact_match_with_full_order_rate": float(exact_matches / n_runs),
        "most_frequent_causal_order": most_frequent_order,
        "most_frequent_order_count": most_frequent_count,
        "most_frequent_order_rate": float(most_frequent_count / n_runs),
        "n_unique_causal_orders": int(len(order_counts)),
        "mean_kendall_tau_with_full_order": float(np.mean(kendall_taus)),
        "std_kendall_tau_with_full_order": float(np.std(kendall_taus)),
    }


# ============================================================
# 14. BỔ SUNG: XUẤT TỔNG HỢP 5 NHÓM CHỈ SỐ HIỆU NĂNG
# ============================================================

def export_comprehensive_metrics(
    full_runtime: float,
    full_converged: bool,
    bootstrap_runs: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    stable_edges_df: pd.DataFrame,
    bidirectional_edges_df: pd.DataFrame,
    n_original_edges: int,
    n_variables: int,
    full_data_causal_order: list[str],
    matrix_diagnostics: dict,
) -> dict:
    max_possible_edges = n_variables * (n_variables - 1)
    converged_runs = int(bootstrap_runs["Converged"].sum())
    n_runs = len(bootstrap_runs)

    # 1. Hiệu năng & Hội tụ
    comp_metrics = {
        "algorithm": "ICA-LiNGAM",
        "full_data_runtime_seconds": float(full_runtime),
        "full_data_converged": full_converged,
        "bootstrap_mean_runtime_seconds": float(bootstrap_runs["Runtime seconds"].mean()),
        "bootstrap_median_runtime_seconds": float(bootstrap_runs["Runtime seconds"].median()),
        "bootstrap_std_runtime_seconds": float(bootstrap_runs["Runtime seconds"].std()),
        "converged_runs_count": converged_runs,
        "convergence_rate": float(converged_runs / n_runs if n_runs > 0 else 0.0),
    }

    # 2. Độ ổn định Cấu trúc
    struct_metrics = {
        "n_stable_edges": int(len(stable_edges_df)),
        "n_bidirectional_stable_pairs": int(len(bidirectional_edges_df)),
        "mean_selection_frequency": float(bootstrap_summary["Selection_frequency"].mean()),
        "median_selection_frequency": float(bootstrap_summary["Selection_frequency"].median()),
    }

    # 3. Độ ổn định Hệ số Tác động
    path_metrics = {
        "mean_sign_consistency": float(bootstrap_summary["Sign_consistency"].mean()),
        "median_sign_consistency": float(bootstrap_summary["Sign_consistency"].median()),
        "mean_coefficient_std": float(bootstrap_summary["Coefficient_std"].mean()),
        "median_coefficient_std": float(bootstrap_summary["Coefficient_std"].median()),
    }

    # 4. Đặc trưng Ma trận Kề & Chẩn đoán
    adj_metrics = {
        "n_variables": n_variables,
        "max_possible_directed_edges": max_possible_edges,
        "n_original_edges": n_original_edges,
        "original_graph_sparsity": float(1.0 - (n_original_edges / max_possible_edges)),
        "stable_graph_sparsity": float(1.0 - (len(stable_edges_df) / max_possible_edges)),
        "matrix_rank": matrix_diagnostics.get("matrix_rank"),
        "condition_number": matrix_diagnostics.get("condition_number"),
    }

    # 5. Độ ổn định Causal Order
    order_metrics = summarize_causal_order_stability(
        bootstrap_runs, full_data_causal_order
    )

    comprehensive_report = {
        "1_computational_and_convergence": comp_metrics,
        "2_bootstrap_structural_stability": struct_metrics,
        "3_path_coefficient_stability": path_metrics,
        "4_adjacency_matrix_characteristics": adj_metrics,
        "5_causal_order_stability": order_metrics,
    }

    # Lưu JSON
    with open(
        OUTPUT_DIR / "comprehensive_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

    # Lưu CSV phẳng
    flat_rows = []
    for group_name, group_dict in comprehensive_report.items():
        for metric_key, metric_val in group_dict.items():
            flat_rows.append(
                {
                    "Metric Group": group_name,
                    "Metric Name": metric_key,
                    "Value": str(metric_val),
                }
            )

    pd.DataFrame(flat_rows).to_csv(
        OUTPUT_DIR / "comprehensive_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return comprehensive_report


# ============================================================
# 15. MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("ICA-LINGAM + BOOTSTRAP TRÊN ENERGY EFFICIENCY")
    print("=" * 80)

    print("\n[1] Đang tải bộ dữ liệu Energy Efficiency...")
    raw_df = load_energy_efficiency()

    print("\n[2] Đang tạo bảng mô tả biến...")
    summarize_variables(raw_df)

    print("\n[3] Đang loại ba biến theo yêu cầu...")
    analysis_df = prepare_analysis_data(raw_df)
    assert analysis_df.shape == (768, 7)

    analysis_df.to_csv(
        OUTPUT_DIR / "energy_efficiency_filtered.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[4] Đang kiểm tra rank và condition number...")
    diagnostics = save_matrix_diagnostics(analysis_df)

    print("\n[5] Đang chuẩn hóa toàn bộ dữ liệu...")
    X_original = analysis_df.to_numpy(dtype=float)
    X_standardized, _ = standardize_array(X_original)
    variable_names = analysis_df.columns.tolist()

    print("\n[6] Đang chạy ICA-LiNGAM trên toàn bộ dữ liệu...")
    full_result = fit_ica_lingam(
        X=X_standardized,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
    )

    causal_order_names = [
        variable_names[index] for index in full_result["causal_order"]
    ]
    adjacency_matrix = full_result["adjacency_matrix"]

    original_edges = adjacency_to_edges(
        adjacency_matrix=adjacency_matrix,
        variable_names=variable_names,
        threshold=EDGE_THRESHOLD,
    )
    original_edges.to_csv(
        OUTPUT_DIR / "ica_lingam_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_causal_graph_readable(
        edges_df=original_edges,
        variable_names=variable_names,
        output_path=OUTPUT_DIR / "ica_lingam_causal_graph_readable.png",
        title=f"ICA-LiNGAM causal graph\n|coefficient| >= {EDGE_THRESHOLD}",
        causal_order=causal_order_names,
        show_isolated_nodes=True,
    )

    print("\n[7] Đang chạy bootstrap...")
    (
        bootstrap_raw,
        bootstrap_summary,
        bootstrap_runs,
    ) = bootstrap_ica_lingam(
        X_original=X_original,
        variable_names=variable_names,
        n_bootstrap=N_BOOTSTRAP,
        edge_threshold=EDGE_THRESHOLD,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
    )

    bootstrap_raw.to_csv(
        OUTPUT_DIR / "ica_lingam_bootstrap_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        OUTPUT_DIR / "ica_lingam_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_runs.to_csv(
        OUTPUT_DIR / "ica_lingam_bootstrap_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[8] Đang lọc các cạnh bootstrap ổn định...")
    stable_edges = select_stable_edges(
        bootstrap_summary=bootstrap_summary,
        stability_threshold=STABILITY_THRESHOLD,
        sign_consistency_threshold=SIGN_CONSISTENCY_THRESHOLD,
    )
    stable_edges.to_csv(
        OUTPUT_DIR / "ica_lingam_stable_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bidirectional_edges = find_bidirectional_stable_edges(stable_edges)
    bidirectional_edges.to_csv(
        OUTPUT_DIR / "ica_lingam_bidirectional_stable_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_causal_graph_readable(
        edges_df=stable_edges,
        variable_names=variable_names,
        output_path=OUTPUT_DIR / "ica_lingam_bootstrap_stable_graph_readable.png",
        title=f"Bootstrap-stable ICA-LiNGAM graph\nfreq >= {STABILITY_THRESHOLD}, sign cons >= {SIGN_CONSISTENCY_THRESHOLD}",
        causal_order=causal_order_names,
        frequency_column="Selection_frequency",
        show_isolated_nodes=SHOW_ISOLATED_NODES_IN_STABLE_GRAPH,
    )

    print("\n[9] Đang tính toán và xuất tổng hợp 5 nhóm chỉ số hiệu năng...")
    export_comprehensive_metrics(
        full_runtime=full_result["runtime_seconds"],
        full_converged=full_result["converged"],
        bootstrap_runs=bootstrap_runs,
        bootstrap_summary=bootstrap_summary,
        stable_edges_df=stable_edges,
        bidirectional_edges_df=bidirectional_edges,
        n_original_edges=len(original_edges),
        n_variables=len(variable_names),
        full_data_causal_order=causal_order_names,
        matrix_diagnostics=diagnostics,
    )

    print("\n" + "=" * 80)
    print("HOÀN THÀNH THỰC NGHIỆM ICA-LINGAM")
    print("=" * 80)


if __name__ == "__main__":
    main()