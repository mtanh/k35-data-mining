from __future__ import annotations

from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from causallearn.search.FCMBased import lingam


# ============================================================
# CẤU HÌNH THỰC NGHIỆM
# ============================================================

RANDOM_STATE = 42

# Một hệ số được xem là cạnh trong từng lần chạy bootstrap
# khi giá trị tuyệt đối của hệ số lớn hơn hoặc bằng ngưỡng này.
EDGE_THRESHOLD = 0.05

# Số lần bootstrap.
N_BOOTSTRAP = 500

# Một cạnh được xem là ổn định nếu xuất hiện trong ít nhất
# 80% số lần bootstrap.
STABILITY_THRESHOLD = 0.80

OUTPUT_DIR = Path("outputs_direct_lingam")
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
# 1. TẢI DATASET ENERGY EFFICIENCY
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
    rows = []

    for column in df.columns:
        values = df[column].dropna().unique()
        sorted_values = np.sort(values)

        rows.append(
            {
                "variable": column,
                "dtype": str(df[column].dtype),
                "n_unique": int(df[column].nunique()),
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "unique_values": (
                    sorted_values.tolist()
                    if len(sorted_values) <= 20
                    else "More than 20 values"
                ),
            }
        )

    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(
        OUTPUT_DIR / "variable_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary_df


# ============================================================
# 3. LOẠI BA BIẾN VÀ KIỂM TRA DỮ LIỆU
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

    df_filtered = df.drop(columns=REMOVED_COLUMNS).copy()

    missing_analysis_columns = [
        column
        for column in ANALYSIS_COLUMNS
        if column not in df_filtered.columns
    ]

    if missing_analysis_columns:
        raise ValueError(
            "Không tìm thấy các biến phân tích: "
            f"{missing_analysis_columns}"
        )

    df_filtered = df_filtered[ANALYSIS_COLUMNS]
    df_filtered = df_filtered.apply(
        pd.to_numeric,
        errors="raise",
    )

    if df_filtered.isna().any().any():
        missing_counts = df_filtered.isna().sum()
        raise ValueError(
            "Dữ liệu có giá trị thiếu:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    if np.isinf(df_filtered.to_numpy(dtype=float)).any():
        raise ValueError("Dữ liệu chứa giá trị vô cùng.")

    if df_filtered.empty:
        raise ValueError("Dữ liệu sau tiền xử lý bị rỗng.")

    return df_filtered


# ============================================================
# 4. CHUẨN HÓA DỮ LIỆU
# ============================================================

def standardize_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()

    X_standardized = scaler.fit_transform(
        df.to_numpy(dtype=float)
    )

    standardized_df = pd.DataFrame(
        X_standardized,
        columns=df.columns,
    )

    return X_standardized, standardized_df, scaler


# ============================================================
# 5. CHẠY DIRECTLINGAM
# ============================================================

def fit_direct_lingam(
    X: np.ndarray,
    random_state: int,
) -> tuple[lingam.DirectLiNGAM, float]:
    model = lingam.DirectLiNGAM(
        random_state=random_state,
        measure="pwling",
    )

    start_time = time.perf_counter()
    model.fit(X)
    runtime_seconds = time.perf_counter() - start_time

    return model, runtime_seconds


# ============================================================
# 6. CHUYỂN MA TRẬN KỀ THÀNH DANH SÁCH CẠNH
# ============================================================

def adjacency_to_edges(
    adjacency_matrix: np.ndarray,
    variable_names: list[str],
    threshold: float,
) -> pd.DataFrame:
    edges = []
    n_variables = len(variable_names)

    for effect_index in range(n_variables):
        for cause_index in range(n_variables):
            if cause_index == effect_index:
                continue

            coefficient = float(
                adjacency_matrix[effect_index, cause_index]
            )

            if abs(coefficient) < threshold:
                continue

            edges.append(
                {
                    "Cause": variable_names[cause_index],
                    "Effect": variable_names[effect_index],
                    "Coefficient": coefficient,
                    "Absolute coefficient": abs(coefficient),
                    "Sign": (
                        "Positive"
                        if coefficient > 0
                        else "Negative"
                    ),
                }
            )

    columns = [
        "Cause",
        "Effect",
        "Coefficient",
        "Absolute coefficient",
        "Sign",
    ]

    edges_df = pd.DataFrame(edges, columns=columns)

    if not edges_df.empty:
        edges_df = edges_df.sort_values(
            by="Absolute coefficient",
            ascending=False,
        ).reset_index(drop=True)

    return edges_df


# ============================================================
# 7. VẼ ĐỒ THỊ KẾT QUẢ GỐC
# ============================================================

def plot_causal_graph(
    edges_df: pd.DataFrame,
    variable_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    for row in edges_df.itertuples(index=False):
        graph.add_edge(
            row.Cause,
            row.Effect,
            coefficient=float(row.Coefficient),
        )

    plt.figure(figsize=(16, 10))

    positions = nx.spring_layout(
        graph,
        seed=RANDOM_STATE,
        k=2.1,
        iterations=250,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=5000,
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

    positive_edges = [
        (cause, effect)
        for cause, effect, data in graph.edges(data=True)
        if data["coefficient"] > 0
    ]

    negative_edges = [
        (cause, effect)
        for cause, effect, data in graph.edges(data=True)
        if data["coefficient"] < 0
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=positive_edges,
        edge_color="green",
        width=2.2,
        arrows=True,
        arrowsize=24,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        min_source_margin=20,
        min_target_margin=20,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=negative_edges,
        edge_color="red",
        style="dashed",
        width=2.2,
        arrows=True,
        arrowsize=24,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        min_source_margin=20,
        min_target_margin=20,
    )

    edge_labels = {
        (cause, effect): f"{data['coefficient']:.3f}"
        for cause, effect, data in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
    )

    plt.plot(
        [],
        [],
        color="green",
        linewidth=2.2,
        label="Positive effect",
    )

    plt.plot(
        [],
        [],
        color="red",
        linestyle="dashed",
        linewidth=2.2,
        label="Negative effect",
    )

    plt.title(
        title,
        fontsize=14,
        fontweight="bold",
    )

    plt.legend(loc="upper left")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# ============================================================
# 8. BOOTSTRAP DIRECTLINGAM
# ============================================================

def bootstrap_direct_lingam(
    X: np.ndarray,
    variable_names: list[str],
    n_bootstrap: int,
    edge_threshold: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    n_samples = X.shape[0]
    n_variables = len(variable_names)

    edge_records = []
    run_records = []

    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU BOOTSTRAP: {n_bootstrap} LẦN")
    print("=" * 80)

    for bootstrap_index in range(n_bootstrap):
        sampled_indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        X_bootstrap = X[sampled_indices, :]

        model, runtime_seconds = fit_direct_lingam(
            X_bootstrap,
            random_state=random_state + bootstrap_index,
        )

        adjacency_matrix = np.asarray(
            model.adjacency_matrix_,
            dtype=float,
        )

        causal_order_indices = [
            int(index)
            for index in model.causal_order_
        ]

        causal_order_names = [
            variable_names[index]
            for index in causal_order_indices
        ]

        run_records.append(
            {
                "Bootstrap ID": bootstrap_index + 1,
                "Runtime seconds": runtime_seconds,
                "Causal order": " -> ".join(causal_order_names),
            }
        )

        for effect_index in range(n_variables):
            for cause_index in range(n_variables):
                if cause_index == effect_index:
                    continue

                coefficient = float(
                    adjacency_matrix[effect_index, cause_index]
                )

                selected = abs(coefficient) >= edge_threshold

                edge_records.append(
                    {
                        "Bootstrap ID": bootstrap_index + 1,
                        "Cause": variable_names[cause_index],
                        "Effect": variable_names[effect_index],
                        "Coefficient": coefficient,
                        "Absolute coefficient": abs(coefficient),
                        "Selected": selected,
                        "Positive selected": (
                            coefficient >= edge_threshold
                        ),
                        "Negative selected": (
                            coefficient <= -edge_threshold
                        ),
                    }
                )

        completed = bootstrap_index + 1

        if (
            completed == 1
            or completed % 10 == 0
            or completed == n_bootstrap
        ):
            print(
                f"Đã hoàn thành "
                f"{completed}/{n_bootstrap} lần bootstrap."
            )

    raw_results = pd.DataFrame(edge_records)
    run_summary = pd.DataFrame(run_records)

    edge_summary = (
        raw_results
        .groupby(["Cause", "Effect"], as_index=False)
        .agg(
            Selection_frequency=("Selected", "mean"),
            Selection_count=("Selected", "sum"),
            Mean_coefficient=("Coefficient", "mean"),
            Median_coefficient=("Coefficient", "median"),
            Coefficient_std=("Coefficient", "std"),
            Mean_absolute_coefficient=(
                "Absolute coefficient",
                "mean",
            ),
            Positive_frequency=("Positive selected", "mean"),
            Negative_frequency=("Negative selected", "mean"),
        )
    )

    edge_summary["Sign_consistency"] = (
        edge_summary[
            [
                "Positive_frequency",
                "Negative_frequency",
            ]
        ]
        .max(axis=1)
    )

    edge_summary["Dominant_sign"] = np.where(
        edge_summary["Positive_frequency"]
        > edge_summary["Negative_frequency"],
        "Positive",
        np.where(
            edge_summary["Negative_frequency"]
            > edge_summary["Positive_frequency"],
            "Negative",
            "Mixed/None",
        ),
    )

    edge_summary = edge_summary.sort_values(
        by=[
            "Selection_frequency",
            "Mean_absolute_coefficient",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return raw_results, edge_summary, run_summary


# ============================================================
# 9. LỌC CẠNH ỔN ĐỊNH VÀ CẠNH 2 CHIỀU
# ============================================================

def select_stable_edges(
    bootstrap_summary: pd.DataFrame,
    stability_threshold: float,
) -> pd.DataFrame:
    stable_edges = bootstrap_summary[
        bootstrap_summary["Selection_frequency"]
        >= stability_threshold
    ].copy()

    if stable_edges.empty:
        return stable_edges

    stable_edges["Representative_coefficient"] = (
        stable_edges["Median_coefficient"]
    )

    stable_edges["Absolute representative coefficient"] = (
        stable_edges["Representative_coefficient"].abs()
    )

    stable_edges = stable_edges.sort_values(
        by=[
            "Selection_frequency",
            "Absolute representative coefficient",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return stable_edges


def find_bidirectional_stable_edges(
    stable_edges: pd.DataFrame,
) -> pd.DataFrame:
    if stable_edges.empty:
        return pd.DataFrame()

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
# 10. VẼ ĐỒ THỊ BOOTSTRAP ỔN ĐỊNH
# ============================================================

def plot_bootstrap_stable_graph(
    stable_edges_df: pd.DataFrame,
    variable_names: list[str],
    output_path: Path,
    stability_threshold: float,
) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    for row in stable_edges_df.itertuples(index=False):
        graph.add_edge(
            row.Cause,
            row.Effect,
            coefficient=float(row.Representative_coefficient),
            frequency=float(row.Selection_frequency),
        )

    plt.figure(figsize=(17, 11))

    positions = nx.spring_layout(
        graph,
        seed=RANDOM_STATE,
        k=2.2,
        iterations=300,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=5200,
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

    positive_edges = [
        (cause, effect)
        for cause, effect, data in graph.edges(data=True)
        if data["coefficient"] > 0
    ]

    negative_edges = [
        (cause, effect)
        for cause, effect, data in graph.edges(data=True)
        if data["coefficient"] < 0
    ]

    positive_widths = [
        1.5 + 3.0 * graph.edges[edge]["frequency"]
        for edge in positive_edges
    ]

    negative_widths = [
        1.5 + 3.0 * graph.edges[edge]["frequency"]
        for edge in negative_edges
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=positive_edges,
        edge_color="green",
        width=positive_widths,
        arrows=True,
        arrowsize=25,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        min_source_margin=20,
        min_target_margin=20,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=negative_edges,
        edge_color="red",
        style="dashed",
        width=negative_widths,
        arrows=True,
        arrowsize=25,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        min_source_margin=20,
        min_target_margin=20,
    )

    edge_labels = {
        (cause, effect): (
            f"β={data['coefficient']:.3f}\n"
            f"freq={data['frequency']:.2f}"
        )
        for cause, effect, data in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
    )

    plt.plot(
        [],
        [],
        color="green",
        linewidth=2.5,
        label="Stable positive effect",
    )

    plt.plot(
        [],
        [],
        color="red",
        linestyle="dashed",
        linewidth=2.5,
        label="Stable negative effect",
    )

    plt.title(
        "Bootstrap-stable DirectLiNGAM graph\n"
        f"Selection frequency >= {stability_threshold:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.legend(loc="upper left")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# ============================================================
# 11. BỔ SUNG: HÀM ĐÁNH GIÁ ĐỘ ÔN ĐỊNH CAUSAL ORDER (KENDALL TAU)
# ============================================================

def compute_kendall_tau(order1: list[str], order2: list[str]) -> float:
    """Tính Kendall Rank Correlation giữa 2 chuỗi order."""
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
# 12. BỔ SUNG: XUẤT TỔNG HỢP 5 NHÓM CHỈ SỐ HIỆU NĂNG
# ============================================================

def export_comprehensive_metrics(
    full_runtime: float,
    bootstrap_runs: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    stable_edges_df: pd.DataFrame,
    bidirectional_edges_df: pd.DataFrame,
    n_original_edges: int,
    n_variables: int,
    full_data_causal_order: list[str],
) -> dict:
    max_possible_edges = n_variables * (n_variables - 1)

    # 1. Hiệu năng & Hội tụ
    comp_metrics = {
        "algorithm": "DirectLiNGAM",
        "full_data_runtime_seconds": float(full_runtime),
        "bootstrap_mean_runtime_seconds": float(bootstrap_runs["Runtime seconds"].mean()),
        "bootstrap_median_runtime_seconds": float(bootstrap_runs["Runtime seconds"].median()),
        "bootstrap_std_runtime_seconds": float(bootstrap_runs["Runtime seconds"].std()),
        "convergence_rate": 1.0,  # DirectLiNGAM luôn hội tụ
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

    # 4. Đặc trưng Ma trận Kề (Sparsity)
    adj_metrics = {
        "n_variables": n_variables,
        "max_possible_directed_edges": max_possible_edges,
        "n_original_edges": n_original_edges,
        "original_graph_sparsity": float(1.0 - (n_original_edges / max_possible_edges)),
        "stable_graph_sparsity": float(1.0 - (len(stable_edges_df) / max_possible_edges)),
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
# 13. HÀM MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("DIRECTLINGAM + BOOTSTRAP TRÊN ENERGY EFFICIENCY")
    print("=" * 80)

    print("\n[1] Đang tải bộ dữ liệu Energy Efficiency...")
    raw_df = load_energy_efficiency()

    print("\n[2] Đang kiểm tra số giá trị khác nhau...")
    variable_summary = summarize_variables(raw_df)

    print("\n[3] Đang loại ba biến theo yêu cầu...")
    analysis_df = prepare_analysis_data(raw_df)
    assert analysis_df.shape == (768, 7)

    analysis_df.to_csv(
        OUTPUT_DIR / "energy_efficiency_filtered.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[4] Đang chuẩn hóa dữ liệu...")
    X_standardized, standardized_df, _ = standardize_data(analysis_df)
    variable_names = analysis_df.columns.tolist()

    print("\n[5] Đang chạy DirectLiNGAM trên toàn bộ dữ liệu...")
    model, full_runtime = fit_direct_lingam(
        X_standardized, random_state=RANDOM_STATE
    )

    causal_order_indices = [int(index) for index in model.causal_order_]
    causal_order_names = [variable_names[index] for index in causal_order_indices]
    adjacency_matrix = np.asarray(model.adjacency_matrix_, dtype=float)

    print("\n[6] Đang lưu ma trận kề...")
    adjacency_df = pd.DataFrame(
        adjacency_matrix, index=variable_names, columns=variable_names
    )
    adjacency_df.to_csv(
        OUTPUT_DIR / "direct_lingam_adjacency_matrix.csv", encoding="utf-8-sig"
    )

    print("\n[7] Đang tạo danh sách cạnh...")
    original_edges_df = adjacency_to_edges(
        adjacency_matrix=adjacency_matrix,
        variable_names=variable_names,
        threshold=EDGE_THRESHOLD,
    )
    original_edges_df.to_csv(
        OUTPUT_DIR / "direct_lingam_edges.csv", index=False, encoding="utf-8-sig"
    )

    print("\n[8] Đang vẽ đồ thị DirectLiNGAM gốc...")
    plot_causal_graph(
        edges_df=original_edges_df,
        variable_names=variable_names,
        output_path=OUTPUT_DIR / "direct_lingam_causal_graph.png",
        title=f"DirectLiNGAM causal graph\n|coefficient| >= {EDGE_THRESHOLD}",
    )

    print("\n[9] Đang chạy bootstrap...")
    (
        bootstrap_raw,
        bootstrap_summary,
        bootstrap_runs,
    ) = bootstrap_direct_lingam(
        X=X_standardized,
        variable_names=variable_names,
        n_bootstrap=N_BOOTSTRAP,
        edge_threshold=EDGE_THRESHOLD,
        random_state=RANDOM_STATE,
    )

    bootstrap_raw.to_csv(
        OUTPUT_DIR / "direct_lingam_bootstrap_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        OUTPUT_DIR / "direct_lingam_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_runs.to_csv(
        OUTPUT_DIR / "direct_lingam_bootstrap_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[10] Đang lọc các cạnh ổn định...")
    stable_edges_df = select_stable_edges(
        bootstrap_summary, stability_threshold=STABILITY_THRESHOLD
    )
    stable_edges_df.to_csv(
        OUTPUT_DIR / "direct_lingam_stable_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bidirectional_edges_df = find_bidirectional_stable_edges(stable_edges_df)
    bidirectional_edges_df.to_csv(
        OUTPUT_DIR / "direct_lingam_bidirectional_stable_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[11] Đang vẽ đồ thị các cạnh ổn định...")
    plot_bootstrap_stable_graph(
        stable_edges_df=stable_edges_df,
        variable_names=variable_names,
        output_path=OUTPUT_DIR / "direct_lingam_bootstrap_stable_graph.png",
        stability_threshold=STABILITY_THRESHOLD,
    )

    print("\n[12] Đang tính toán và xuất tổng hợp 5 nhóm chỉ số hiệu năng...")
    export_comprehensive_metrics(
        full_runtime=full_runtime,
        bootstrap_runs=bootstrap_runs,
        bootstrap_summary=bootstrap_summary,
        stable_edges_df=stable_edges_df,
        bidirectional_edges_df=bidirectional_edges_df,
        n_original_edges=len(original_edges_df),
        n_variables=len(variable_names),
        full_data_causal_order=causal_order_names,
    )

    print("\n" + "=" * 80)
    print("HOÀN THÀNH THỰC NGHIỆM DIRECTLINGAM")
    print("=" * 80)


if __name__ == "__main__":
    main()