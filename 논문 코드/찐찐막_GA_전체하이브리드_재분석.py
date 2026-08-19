from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "HOPE_측정자료_상황별정리.xlsx"
REFERENCE_PDF = BASE_DIR / "구현가능한 AP 알고리즘 논문_2015년.pdf"

RSSI_VARIATION_SHEETS = [
    "NATURAL_1H_R1",
    "OPEN2_1H_R1",
    "OPEN2_20M_R1",
    "FREE1_1H_R1",
    "FREE1_1H_R2",
    "MOVE_OPEN2_1H_R1",
    "MOVE_OPEN2_1H_R2",
]
CONGESTION_SHEETS = ["CONGESTED_OPEN2_R1", "CONGESTED_OPEN2_R2"]
ALL_MEASUREMENT_SHEETS = RSSI_VARIATION_SHEETS + CONGESTION_SHEETS

REQUIRED_COLUMNS = [
    "RSSI_dBm",
    "Link_Mbps",
    "UTIL_percent",
    "RTT_ms",
    "PacketLoss_percent",
    "Upload_Mbps",
    "Download_Mbps",
]

# 정상/혼잡 상태 판정 기준은 기존 HOPE 방식과 동일하게 고정한다.
RTT_THRESHOLD_MS = 22.0
UTIL_THRESHOLD_PERCENT = 56.0

# 혼잡 점수에만 사용되는 GA 입력. RSSI 의존도가 큰 Link와 실제 전송률은 포함하지 않는다.
GA_FEATURES = ["RTT_ms", "UTIL_percent"]
GA_SCORE_LABELS = ["RTT", "UTIL"]
GA_DIRECTIONS = np.array([-1.0, -1.0])

# 측정 오류로 확인되어 분석에서 제외하는 원본 행.
# Source_Excel_Row는 헤더를 포함한 실제 Excel 행 번호이다.
MEASUREMENT_ERROR_ROWS = {
    ("NATURAL_1H_R1", 2),   # PacketLoss 26.6667%
    ("FREE1_1H_R2", 2),     # PacketLoss 20%
}

RANDOM_SEED = 42
POPULATION_SIZE = 320
GENERATIONS = 500
ELITE_COUNT = 32
PARENT_POOL = 100
MUTATION_RATE = 0.35
INITIAL_MUTATION_SIGMA = 0.08

# APQI 논문 Table 1 및 식 (3), (4)의 설정값.
APQI_ALPHA = 0.5
APQI_W_RSS = 0.4
APQI_W_LOAD = 0.6
APQI_RSS_MIN_DBM = -82.0
APQI_MAX_BSS_LOAD_RATIO = 0.8
APQI_MAX_BSS_LOAD_255 = int(round(255 * APQI_MAX_BSS_LOAD_RATIO))


@dataclass(frozen=True)
class RangeInfo:
    low: float
    high: float


def configure_plot() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    if np.std(x[valid]) <= 1e-12 or np.std(y[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    xr = pd.Series(x[valid]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[valid]).rank(method="average").to_numpy(float)
    return pearson(xr, yr)


def bootstrap_pearson_ci(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    repetitions: int = 2000,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        index = rng.integers(0, len(x), size=len(x))
        current = pearson(x[index], y[index])
        if np.isfinite(current):
            values.append(current)
    if len(values) < repetitions * 0.9:
        return float("nan"), float("nan")
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def bootstrap_correlation_difference_ci(
    actual: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed: int,
    repetitions: int = 2000,
) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=float)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(score_a) & np.isfinite(score_b)
    actual = actual[valid]
    score_a = score_a[valid]
    score_b = score_b[valid]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        index = rng.integers(0, len(actual), size=len(actual))
        r_a = pearson(actual[index], score_a[index])
        r_b = pearson(actual[index], score_b[index])
        if np.isfinite(r_a) and np.isfinite(r_b):
            differences.append(r_a - r_b)
    if len(differences) < repetitions * 0.9:
        return float("nan"), float("nan")
    return tuple(float(v) for v in np.percentile(differences, [2.5, 97.5]))


def minmax(values: np.ndarray, low: float, high: float, invert: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        score = np.full(values.shape, 0.5, dtype=float)
    else:
        score = np.clip((values - low) / (high - low), 0.0, 1.0)
    return 1.0 - score if invert else score


def load_sheet(sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(DATA_FILE, sheet_name=sheet_name)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{sheet_name} 시트에 필수 열이 없습니다: {missing}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame[REQUIRED_COLUMNS] = frame[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame.insert(0, "Source_Excel_Row", np.arange(2, len(frame) + 2))
    frame.insert(0, "Source_Sheet", sheet_name)
    return frame


def find_integrated_sheet() -> tuple[str, pd.DataFrame]:
    excel = pd.ExcelFile(DATA_FILE)
    candidates: list[tuple[str, pd.DataFrame]] = []
    for sheet in excel.sheet_names:
        frame = pd.read_excel(DATA_FILE, sheet_name=sheet)
        if len(frame) == 400 and set(REQUIRED_COLUMNS).issubset(frame.columns):
            candidates.append((sheet, frame[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")))
    if len(candidates) != 1:
        raise ValueError(f"400개 통합원자료 시트를 하나로 특정하지 못했습니다: {[x[0] for x in candidates]}")
    return candidates[0]


def measurement_error_mask(frame: pd.DataFrame) -> pd.Series:
    keys = list(zip(frame["Source_Sheet"], frame["Source_Excel_Row"]))
    return pd.Series([key in MEASUREMENT_ERROR_ROWS for key in keys], index=frame.index, dtype=bool)


def build_datasets() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, object]]:
    raw_by_sheet = {sheet: load_sheet(sheet) for sheet in ALL_MEASUREMENT_SHEETS}
    rssi_raw = pd.concat([raw_by_sheet[s] for s in RSSI_VARIATION_SHEETS], ignore_index=True)
    congestion_raw = pd.concat([raw_by_sheet[s] for s in CONGESTION_SHEETS], ignore_index=True)
    mixed_from_sheets = pd.concat([raw_by_sheet[s] for s in ALL_MEASUREMENT_SHEETS], ignore_index=True)

    integrated_name, integrated = find_integrated_sheet()
    integrated_equal = np.isclose(
        integrated[REQUIRED_COLUMNS].to_numpy(float),
        mixed_from_sheets[REQUIRED_COLUMNS].to_numpy(float),
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    ).all()
    if not integrated_equal:
        raise ValueError("통합원자료 400개와 9개 상황별 시트를 결합한 400개가 일치하지 않습니다.")

    raw_sets = {
        "RSSI 변화 환경": rssi_raw,
        "RSSI 양호·혼잡 환경": congestion_raw,
        "혼합환경": mixed_from_sheets,
    }
    valid_sets: dict[str, pd.DataFrame] = {}
    qc_rows: list[dict[str, object]] = []

    for label, raw in raw_sets.items():
        missing_mask = raw[REQUIRED_COLUMNS].isna().any(axis=1)
        error_mask = measurement_error_mask(raw)
        valid = raw.loc[~missing_mask & ~error_mask].copy().reset_index(drop=True)
        valid["Actual_Mbps"] = valid["Upload_Mbps"] + valid["Download_Mbps"]
        valid["Is_Congested"] = (
            (valid["RTT_ms"] >= RTT_THRESHOLD_MS)
            | (valid["UTIL_percent"] >= UTIL_THRESHOLD_PERCENT)
        )
        valid["State"] = np.where(valid["Is_Congested"], "혼잡", "정상")
        valid_sets[label] = valid
        qc_rows.append(
            {
                "환경": label,
                "원표본수": len(raw),
                "NaN_공란제외수": int((missing_mask & ~error_mask).sum()),
                "측정오류제외수": int(error_mask.sum()),
                "유효표본수": len(valid),
                "정상상태수": int((~valid["Is_Congested"]).sum()),
                "혼잡상태수": int(valid["Is_Congested"].sum()),
                "중복제거수": 0,
                "처리": "NaN/공란 및 지정 측정오류 2행만 제외, 중복 유지",
            }
        )

    for sheet, raw in raw_by_sheet.items():
        missing_mask = raw[REQUIRED_COLUMNS].isna().any(axis=1)
        error_mask = measurement_error_mask(raw)
        valid = raw.loc[~missing_mask & ~error_mask]
        state = (valid["RTT_ms"] >= RTT_THRESHOLD_MS) | (valid["UTIL_percent"] >= UTIL_THRESHOLD_PERCENT)
        qc_rows.append(
            {
                "환경": sheet,
                "원표본수": len(raw),
                "NaN_공란제외수": int((missing_mask & ~error_mask).sum()),
                "측정오류제외수": int(error_mask.sum()),
                "유효표본수": len(valid),
                "정상상태수": int((~state).sum()),
                "혼잡상태수": int(state.sum()),
                "중복제거수": 0,
                "처리": "NaN/공란 및 지정 측정오류만 제외, 중복 유지",
            }
        )

    verification = {
        "통합원자료_실제시트명": integrated_name,
        "통합원자료_표본수": len(integrated),
        "상황별시트합계": len(mixed_from_sheets),
        "수치일치": bool(integrated_equal),
        "측정오류제외행": [
            {"시트": sheet, "Excel행": row}
            for sheet, row in sorted(MEASUREMENT_ERROR_ROWS)
        ],
    }
    return valid_sets, pd.DataFrame(qc_rows), verification


def build_feature_ranges(mixed: pd.DataFrame) -> dict[str, RangeInfo]:
    return {
        column: RangeInfo(float(mixed[column].min()), float(mixed[column].max()))
        for column in GA_FEATURES
    }


def ga_feature_matrix(frame: pd.DataFrame, ranges: dict[str, RangeInfo]) -> np.ndarray:
    columns = []
    for column, direction in zip(GA_FEATURES, GA_DIRECTIONS):
        info = ranges[column]
        columns.append(minmax(frame[column].to_numpy(float), info.low, info.high, invert=direction < 0))
    return np.column_stack(columns)


def normalize_population(population: np.ndarray) -> np.ndarray:
    population = np.clip(np.asarray(population, dtype=float), 0.0, None)
    sums = population.sum(axis=1, keepdims=True)
    zero = sums[:, 0] <= 1e-12
    if zero.any():
        population[zero] = 1.0
        sums = population.sum(axis=1, keepdims=True)
    return population / sums


def population_fitness(
    population: np.ndarray,
    features: np.ndarray,
    rssi_scores: np.ndarray,
    congested_mask: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:
    congestion_scores = population @ features.T
    scores = np.where(
        np.asarray(congested_mask, dtype=bool)[None, :],
        congestion_scores,
        np.asarray(rssi_scores, dtype=float)[None, :],
    )
    score_centered = scores - scores.mean(axis=1, keepdims=True)
    actual_centered = actual - actual.mean()
    denominator = np.sqrt(np.sum(score_centered**2, axis=1) * np.sum(actual_centered**2))
    return np.divide(
        score_centered @ actual_centered,
        denominator,
        out=np.full(len(population), -1.0),
        where=denominator > 1e-12,
    )


def run_ga(
    features: np.ndarray,
    rssi_scores: np.ndarray,
    congested_mask: np.ndarray,
    actual: np.ndarray,
    seed: int = RANDOM_SEED,
    generations: int = GENERATIONS,
    population_size: int = POPULATION_SIZE,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_features = features.shape[1]
    population = rng.dirichlet(np.ones(n_features), size=population_size)
    population[0] = np.full(n_features, 1.0 / n_features)
    best_weights = population[0].copy()
    best_fitness = -1.0
    history: list[dict[str, float]] = []

    for generation in range(1, generations + 1):
        fitness = population_fitness(population, features, rssi_scores, congested_mask, actual)
        order = np.argsort(fitness)[::-1]
        population = population[order]
        fitness = fitness[order]
        if float(fitness[0]) > best_fitness:
            best_fitness = float(fitness[0])
            best_weights = population[0].copy()

        row = {
            "Generation": generation,
            "Best_Pearson_r": best_fitness,
            "Population_Mean_r": float(np.mean(fitness)),
        }
        row.update({f"Weight_{label}": float(weight) for label, weight in zip(GA_SCORE_LABELS, best_weights)})
        history.append(row)

        elite_count = min(ELITE_COUNT, max(4, population_size // 10))
        parent_pool = min(PARENT_POOL, max(elite_count + 2, population_size // 3))
        elites = population[:elite_count].copy()
        parents = population[:parent_pool]
        sigma = INITIAL_MUTATION_SIGMA * (1.0 - 0.85 * generation / generations)
        children = []
        while len(children) < population_size - elite_count:
            a, b = parents[rng.integers(0, len(parents), size=2)]
            blend = rng.random(n_features)
            child = blend * a + (1.0 - blend) * b
            if rng.random() < MUTATION_RATE:
                child += rng.normal(0.0, sigma, size=n_features)
            children.append(child)
        population = np.vstack([elites, normalize_population(np.asarray(children))])

    return best_weights, pd.DataFrame(history)


def evaluate_cross_validation(
    frame: pd.DataFrame,
    folds: int = 5,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(20260819)
    normal_indices = np.flatnonzero(~frame["Is_Congested"].to_numpy(bool))
    congested_indices = np.flatnonzero(frame["Is_Congested"].to_numpy(bool))
    rng.shuffle(normal_indices)
    rng.shuffle(congested_indices)
    normal_folds = np.array_split(normal_indices, folds)
    congested_folds = np.array_split(congested_indices, folds)
    fold_indices = [np.concatenate([normal_folds[i], congested_folds[i]]) for i in range(folds)]
    all_indices = np.arange(len(frame))
    oof_scores = np.full(len(frame), np.nan)
    rows: list[dict[str, float]] = []

    for fold_no, test_idx in enumerate(fold_indices, start=1):
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        fold_ranges = build_feature_ranges(train)
        fold_rssi_range = RangeInfo(float(train["RSSI_dBm"].min()), float(train["RSSI_dBm"].max()))
        train_features = ga_feature_matrix(train, fold_ranges)
        test_features = ga_feature_matrix(test, fold_ranges)
        train_rssi = baseline_rssi_score(train, fold_rssi_range) / 100.0
        test_rssi = baseline_rssi_score(test, fold_rssi_range) / 100.0
        train_mask = train["Is_Congested"].to_numpy(bool)
        test_mask = test["Is_Congested"].to_numpy(bool)
        weights, _ = run_ga(
            train_features,
            train_rssi,
            train_mask,
            train["Actual_Mbps"].to_numpy(float),
            seed=RANDOM_SEED + fold_no,
            generations=220,
            population_size=180,
        )
        train_scores = 100.0 * np.where(train_mask, train_features @ weights, train_rssi)
        test_scores = 100.0 * np.where(test_mask, test_features @ weights, test_rssi)
        oof_scores[test_idx] = test_scores
        row = {
            "Fold": fold_no,
            "Train_N": len(train),
            "Test_N": len(test),
            "Train_r": pearson(train["Actual_Mbps"].to_numpy(float), train_scores),
            "Test_r": pearson(test["Actual_Mbps"].to_numpy(float), test_scores),
        }
        row.update({f"Weight_{label}": float(weight) for label, weight in zip(GA_SCORE_LABELS, weights)})
        rows.append(row)
    return pd.DataFrame(rows), pearson(frame["Actual_Mbps"].to_numpy(float), oof_scores)


def evaluate_session_cross_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    oof_scores = np.full(len(frame), np.nan)
    rows: list[dict[str, object]] = []
    for fold_no, sheet in enumerate(frame["Source_Sheet"].drop_duplicates(), start=1):
        test_mask_rows = frame["Source_Sheet"].eq(sheet).to_numpy(bool)
        train = frame.loc[~test_mask_rows]
        test = frame.loc[test_mask_rows]
        fold_ranges = build_feature_ranges(train)
        fold_rssi_range = RangeInfo(float(train["RSSI_dBm"].min()), float(train["RSSI_dBm"].max()))
        train_features = ga_feature_matrix(train, fold_ranges)
        test_features = ga_feature_matrix(test, fold_ranges)
        train_rssi = baseline_rssi_score(train, fold_rssi_range) / 100.0
        test_rssi = baseline_rssi_score(test, fold_rssi_range) / 100.0
        train_state = train["Is_Congested"].to_numpy(bool)
        test_state = test["Is_Congested"].to_numpy(bool)
        weights, _ = run_ga(
            train_features,
            train_rssi,
            train_state,
            train["Actual_Mbps"].to_numpy(float),
            seed=RANDOM_SEED + 100 + fold_no,
            generations=220,
            population_size=180,
        )
        train_scores = 100.0 * np.where(train_state, train_features @ weights, train_rssi)
        test_scores = 100.0 * np.where(test_state, test_features @ weights, test_rssi)
        oof_scores[test_mask_rows] = test_scores
        rows.append(
            {
                "Fold": fold_no,
                "Test_Sheet": sheet,
                "Train_N": len(train),
                "Test_N": len(test),
                "Test_Normal_N": int((~test["Is_Congested"]).sum()),
                "Test_Congested_N": int(test["Is_Congested"].sum()),
                "Train_r": pearson(train["Actual_Mbps"].to_numpy(float), train_scores),
                "Test_r": pearson(test["Actual_Mbps"].to_numpy(float), test_scores),
                "Weight_RTT": float(weights[0]),
                "Weight_UTIL": float(weights[1]),
            }
        )
    return pd.DataFrame(rows), pearson(frame["Actual_Mbps"].to_numpy(float), oof_scores)


def baseline_rssi_score(frame: pd.DataFrame, rssi_range: RangeInfo) -> np.ndarray:
    return 100.0 * minmax(frame["RSSI_dBm"].to_numpy(float), rssi_range.low, rssi_range.high)


def hybrid_ga_score(
    frame: pd.DataFrame,
    ranges: dict[str, RangeInfo],
    weights: np.ndarray,
    rssi_range: RangeInfo,
) -> np.ndarray:
    rssi_score = baseline_rssi_score(frame, rssi_range)
    congestion_score = 100.0 * ga_feature_matrix(frame, ranges) @ weights
    return np.where(frame["Is_Congested"].to_numpy(bool), congestion_score, rssi_score)


def smooth_rssi_by_session(frame: pd.DataFrame) -> np.ndarray:
    smoothed = np.empty(len(frame), dtype=float)
    for _, indices in frame.groupby("Source_Sheet", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        raw = frame.loc[idx, "RSSI_dBm"].to_numpy(float)
        current = np.empty(len(raw), dtype=float)
        current[0] = raw[0]
        for i in range(1, len(raw)):
            current[i] = APQI_ALPHA * raw[i] + (1.0 - APQI_ALPHA) * current[i - 1]
        smoothed[idx] = current
    return smoothed


def dbm_to_mw(dbm: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(dbm, dtype=float) / 10.0)


def calculate_apqi_raw(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rss_bar = smooth_rssi_by_session(out)
    channel_load = np.clip(np.rint(out["UTIL_percent"].to_numpy(float) / 100.0 * 255.0), 1, 255)
    rss_ratio = dbm_to_mw(rss_bar) / dbm_to_mw(APQI_RSS_MIN_DBM)
    rss_term = np.log2(np.maximum(rss_ratio, np.finfo(float).tiny))
    load_term = np.log2(255.0 / channel_load)
    apqi_raw = APQI_W_RSS * rss_term + APQI_W_LOAD * load_term
    qualified = (rss_bar > APQI_RSS_MIN_DBM) & (channel_load < APQI_MAX_BSS_LOAD_255)
    out["APQI_RSS_bar_dBm"] = rss_bar
    out["APQI_Channel_Load_1_255"] = channel_load
    out["APQI_RSS_term"] = rss_term
    out["APQI_Load_term"] = load_term
    out["APQI_raw"] = apqi_raw
    out["APQI_Qualified"] = qualified
    return out


def apply_apqi_score(frame: pd.DataFrame, apqi_range: RangeInfo) -> pd.DataFrame:
    out = frame.copy()
    raw_score = 100.0 * minmax(out["APQI_raw"].to_numpy(float), apqi_range.low, apqi_range.high)
    out["APQI_Score_0_100"] = np.where(out["APQI_Qualified"].to_numpy(bool), raw_score, 0.0)
    return out


def regression_line(ax, x: np.ndarray, y: np.ndarray, color: str, style: str) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) <= 1e-12:
        return
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(float(x.min()), float(x.max()), 300)
    ax.plot(xx, slope * xx + intercept, color=color, linestyle=style, linewidth=2.4, label="선형 회귀선")


def format_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("실제 속도 (Upload + Download Mbps)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=8.5)


PANEL_TITLES = [
    [
        "RSSI 변동시 AP 산출 점수-기존 RSSI 방식",
        "RSSI 변동시 AP 산출 점수-래퍼런스 논문 방식",
        "RSSI 변동시 AP 산출 점수-GA 방식",
    ],
    [
        "RSSI양호 및 혼잡환경 多-기존 RSSI방식",
        "RSSI양호 및 혼잡환경 多-래퍼런스 논문 방식",
        "RSSI양호 및 혼잡환경 多-GA 방식",
    ],
    [
        "혼합환경-기존 RSSI방식",
        "혼합환경-래퍼런스 논문 방식",
        "혼합환경-GA 방식",
    ],
]


def draw_ga_training(
    weights: np.ndarray,
    history: pd.DataFrame,
    frame: pd.DataFrame,
    ranges: dict[str, RangeInfo],
    rssi_range: RangeInfo,
    oof_r: float,
    session_oof_r: float,
) -> pd.DataFrame:
    features = ga_feature_matrix(frame, ranges)
    actual = frame["Actual_Mbps"].to_numpy(float)
    rssi_scores = baseline_rssi_score(frame, rssi_range) / 100.0
    state = frame["Is_Congested"].to_numpy(bool)
    ga_scores = 100.0 * np.where(state, features @ weights, rssi_scores)
    equal_weights = np.full(len(GA_FEATURES), 1.0 / len(GA_FEATURES))
    equal_scores = 100.0 * np.where(state, features @ equal_weights, rssi_scores)
    ga_r = pearson(actual, ga_scores)
    equal_r = pearson(actual, equal_scores)
    single = [
        pearson(actual, 100.0 * np.where(state, features[:, i], rssi_scores))
        for i in range(features.shape[1])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    bars = axes[0].bar(GA_SCORE_LABELS, weights * 100.0, color=["#4E79A7", "#E07A5F"])
    axes[0].bar_label(bars, fmt="%.1f%%")
    axes[0].set_ylim(0, max(100.0, float(weights.max() * 115.0)))
    axes[0].set_ylabel("가중치 (%)")
    axes[0].set_title("혼잡 점수 GA 가중치")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].plot(history["Generation"], history["Best_Pearson_r"], label="최고 적합도", color="#2A6FBB")
    axes[1].plot(history["Generation"], history["Population_Mean_r"], label="집단 평균", color="#999999", alpha=0.75)
    axes[1].set_xlabel("세대")
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("GA 수렴 과정")
    axes[1].grid(alpha=0.22)
    axes[1].legend()

    names = ["동일가중치", "GA 학습", "5-fold OOF", "세션 OOF"]
    values = [equal_r, ga_r, oof_r, session_oof_r]
    bars = axes[2].bar(names, values, color=["#9C9C9C", "#3A78B8", "#59A14F", "#AF7AA1"])
    axes[2].bar_label(bars, fmt="%.3f")
    axes[2].set_ylim(min(-0.1, min(values) - 0.1), 1.0)
    axes[2].set_ylabel("Pearson r")
    axes[2].set_title("전체 하이브리드 상관성")
    axes[2].grid(axis="y", alpha=0.22)

    fig.suptitle(
        "393개 전체 상태 전환형 하이브리드 점수를 목적함수로 사용한 RTT·UTIL GA\n"
        "정상: RSSI | 혼잡: RTT·UTIL | 실제 전송률은 적합도·평가 기준으로만 사용",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(BASE_DIR / "01_GA_학습결과.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    rows = [
        {"비교": "동일가중치", "Pearson_r": equal_r},
        {"비교": "GA 전체학습", "Pearson_r": ga_r},
        {"비교": "GA 5-fold OOF", "Pearson_r": oof_r},
        {"비교": "GA 세션단위 OOF", "Pearson_r": session_oof_r},
    ]
    rows.extend({"비교": f"혼잡_{label}단일", "Pearson_r": value} for label, value in zip(GA_SCORE_LABELS, single))
    return pd.DataFrame(rows)


def draw_nine_panel(
    datasets: dict[str, pd.DataFrame],
    apqi_datasets: dict[str, pd.DataFrame],
    ranges: dict[str, RangeInfo],
    weights: np.ndarray,
    rssi_range: RangeInfo,
) -> pd.DataFrame:
    fig, axes = plt.subplots(3, 3, figsize=(22, 19), sharey=True)
    metrics: list[dict[str, object]] = []

    for row, environment in enumerate(["RSSI 변화 환경", "RSSI 양호·혼잡 환경", "혼합환경"]):
        frame = datasets[environment]
        apqi_frame = apqi_datasets[environment]
        actual = frame["Actual_Mbps"].to_numpy(float)
        rssi_score = baseline_rssi_score(frame, rssi_range)
        apqi_score = apqi_frame["APQI_Score_0_100"].to_numpy(float)
        ga_score = hybrid_ga_score(frame, ranges, weights, rssi_range)
        congested = frame["Is_Congested"].to_numpy(bool)
        normal = ~congested
        qualified = apqi_frame["APQI_Qualified"].to_numpy(bool)

        correlations = [
            pearson(actual, rssi_score),
            pearson(actual, apqi_score),
            pearson(actual, ga_score),
        ]
        rank_correlations = [
            spearman(actual, rssi_score),
            spearman(actual, apqi_score),
            spearman(actual, ga_score),
        ]
        score_arrays = [rssi_score, apqi_score, ga_score]

        ax = axes[row, 0]
        ax.scatter(actual, rssi_score, s=38, alpha=0.70, color="#EF6A5B", edgecolors="white", linewidths=0.3,
                   label=f"유효 표본: {len(frame)}개")
        regression_line(ax, actual, rssi_score, "#E64532", "-")
        ax.set_title(f"{PANEL_TITLES[row][0]}\nr = {correlations[0]:.3f}", fontsize=12, fontweight="bold")
        format_axis(ax, "AP 점수")

        ax = axes[row, 1]
        ax.scatter(actual[qualified], apqi_score[qualified], s=39, alpha=0.74, color="#7B61A8",
                   edgecolors="white", linewidths=0.3, label=f"후보 통과: {int(qualified.sum())}개")
        ax.scatter(actual[~qualified], apqi_score[~qualified], s=43, alpha=0.78, color="#F28E2B",
                   marker="x", linewidths=1.2, label=f"임계값 탈락: {int((~qualified).sum())}개")
        regression_line(ax, actual, apqi_score, "#6A4C93", "--")
        ax.set_title(f"{PANEL_TITLES[row][1]}\nr = {correlations[1]:.3f}", fontsize=12, fontweight="bold")
        format_axis(ax, "APQI 환산 점수")

        ax = axes[row, 2]
        ax.scatter(actual[normal], ga_score[normal], s=38, alpha=0.72, color="#42A5E5",
                   edgecolors="white", linewidths=0.3, label=f"정상 모드(RSSI): {int(normal.sum())}개")
        ax.scatter(actual[congested], ga_score[congested], s=43, alpha=0.76, color="#F28E2B",
                   marker="D", edgecolors="white", linewidths=0.3,
                   label=f"혼잡 모드(GA RTT+UTIL): {int(congested.sum())}개")
        regression_line(ax, actual, ga_score, "#1EBC73", "--")
        ax.set_title(f"{PANEL_TITLES[row][2]}\nr = {correlations[2]:.3f}", fontsize=12, fontweight="bold")
        format_axis(ax, "AP 점수")

        methods = ["기존 RSSI", "래퍼런스 APQI", "GA HOPE"]
        for method_index, (method, correlation, rank_correlation, score_array) in enumerate(
            zip(methods, correlations, rank_correlations, score_arrays)
        ):
            ci_low, ci_high = bootstrap_pearson_ci(
                actual,
                score_array,
                seed=20260819 + row * 10 + method_index,
            )
            delta_rssi_low, delta_rssi_high = bootstrap_correlation_difference_ci(
                actual,
                score_array,
                rssi_score,
                seed=20261819 + row * 10 + method_index,
            )
            delta_apqi_low, delta_apqi_high = bootstrap_correlation_difference_ci(
                actual,
                score_array,
                apqi_score,
                seed=20262819 + row * 10 + method_index,
            )
            metrics.append(
                {
                    "환경": environment,
                    "방식": method,
                    "원표본수": {"RSSI 변화 환경": 319, "RSSI 양호·혼잡 환경": 81, "혼합환경": 400}[environment],
                    "유효표본수": len(frame),
                    "정상상태수": int(normal.sum()),
                    "혼잡상태수": int(congested.sum()),
                    "Pearson_r": correlation,
                    "Pearson_95CI_Low": ci_low,
                    "Pearson_95CI_High": ci_high,
                    "Spearman_rho": rank_correlation,
                    "Delta_vs_RSSI_95CI_Low": delta_rssi_low,
                    "Delta_vs_RSSI_95CI_High": delta_rssi_high,
                    "Delta_vs_APQI_95CI_Low": delta_apqi_low,
                    "Delta_vs_APQI_95CI_High": delta_apqi_high,
                }
            )

    weight_text = " + ".join(f"{label} {weight:.3f}" for label, weight in zip(GA_SCORE_LABELS, weights))
    fig.suptitle(
        "정상·혼잡·혼합 환경의 기존 RSSI·래퍼런스 APQI·RTT·UTIL GA HOPE 방식 비교\n"
        f"혼잡 판정: RTT ≥ {RTT_THRESHOLD_MS:g} ms 또는 UTIL ≥ {UTIL_THRESHOLD_PERCENT:g}%  |  "
        f"393개 전체 하이브리드 적합도 GA: {weight_text}",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=3.0, w_pad=1.8)
    fig.savefig(BASE_DIR / "02_기존_래퍼런스_GA_전체환경_9개비교.png", dpi=230, bbox_inches="tight")
    plt.close(fig)
    metric_frame = pd.DataFrame(metrics)
    baseline_map = metric_frame.loc[metric_frame["방식"] == "기존 RSSI"].set_index("환경")["Pearson_r"]
    apqi_map = metric_frame.loc[metric_frame["방식"] == "래퍼런스 APQI"].set_index("환경")["Pearson_r"]
    metric_frame["Delta_vs_RSSI_r"] = metric_frame.apply(
        lambda row: row["Pearson_r"] - baseline_map[row["환경"]], axis=1
    )
    metric_frame["Delta_vs_APQI_r"] = metric_frame.apply(
        lambda row: row["Pearson_r"] - apqi_map[row["환경"]], axis=1
    )
    return metric_frame


def build_row_results(
    datasets: dict[str, pd.DataFrame],
    apqi_datasets: dict[str, pd.DataFrame],
    ranges: dict[str, RangeInfo],
    weights: np.ndarray,
    rssi_range: RangeInfo,
) -> pd.DataFrame:
    rows = []
    for environment, frame in datasets.items():
        apqi_frame = apqi_datasets[environment]
        out = frame.copy()
        out["Environment"] = environment
        out["RSSI_Score_0_100"] = baseline_rssi_score(frame, rssi_range)
        feature_matrix = ga_feature_matrix(frame, ranges)
        for label, values in zip(GA_SCORE_LABELS, feature_matrix.T):
            out[f"GA_Normalized_{label}"] = values
        out["GA_Congestion_Score_0_100"] = 100.0 * feature_matrix @ weights
        out["GA_Hybrid_Score_0_100"] = hybrid_ga_score(frame, ranges, weights, rssi_range)
        out["APQI_raw"] = apqi_frame["APQI_raw"].to_numpy(float)
        out["APQI_Qualified"] = apqi_frame["APQI_Qualified"].to_numpy(bool)
        out["APQI_Score_0_100"] = apqi_frame["APQI_Score_0_100"].to_numpy(float)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    configure_plot()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {DATA_FILE}")
    if not REFERENCE_PDF.exists():
        raise FileNotFoundError(f"래퍼런스 PDF가 없습니다: {REFERENCE_PDF}")

    datasets, qc, verification = build_datasets()
    mixed = datasets["혼합환경"]
    ranges = build_feature_ranges(mixed)
    rssi_range = RangeInfo(float(mixed["RSSI_dBm"].min()), float(mixed["RSSI_dBm"].max()))

    excluded_rows = []
    for sheet, excel_row in sorted(MEASUREMENT_ERROR_ROWS):
        raw = load_sheet(sheet)
        selected = raw.loc[raw["Source_Excel_Row"] == excel_row].copy()
        if len(selected) != 1:
            raise ValueError(f"측정오류 제외행을 하나로 찾지 못했습니다: {sheet} {excel_row}행")
        selected["제외사유"] = "패킷손실률 측정 오류"
        excluded_rows.append(selected)
    excluded_frame = pd.concat(excluded_rows, ignore_index=True)

    # GA 적합도는 393개 전체 상태 전환형 하이브리드 점수와 실제 전송률의 Pearson r이다.
    # 정상 표본은 RSSI 점수, 혼잡 표본은 RTT·UTIL 가중 점수를 사용한다.
    train_features = ga_feature_matrix(mixed, ranges)
    train_rssi = baseline_rssi_score(mixed, rssi_range) / 100.0
    train_state = mixed["Is_Congested"].to_numpy(bool)
    train_actual = mixed["Actual_Mbps"].to_numpy(float)
    weights, history = run_ga(train_features, train_rssi, train_state, train_actual)
    cv_results, oof_r = evaluate_cross_validation(mixed)
    session_cv_results, session_oof_r = evaluate_session_cross_validation(mixed)

    # 2지표 문제의 전역 최적점을 독립적으로 확인하기 위한 0.001 간격 전수 검증.
    grid_rtt = np.linspace(0.0, 1.0, 1001)
    grid_population = np.column_stack([grid_rtt, 1.0 - grid_rtt])
    grid_fitness = population_fitness(grid_population, train_features, train_rssi, train_state, train_actual)
    grid_results = pd.DataFrame(
        {
            "Weight_RTT": grid_rtt,
            "Weight_UTIL": 1.0 - grid_rtt,
            "Whole_Hybrid_Pearson_r": grid_fitness,
        }
    )
    grid_best = grid_results.loc[grid_results["Whole_Hybrid_Pearson_r"].idxmax()]

    # APQI 원식을 세 환경에 적용하고, 혼합환경의 후보 통과 APQI 범위로 0~100 환산한다.
    apqi_raw_sets = {name: calculate_apqi_raw(frame) for name, frame in datasets.items()}
    mixed_apqi = apqi_raw_sets["혼합환경"]
    qualified_raw = mixed_apqi.loc[mixed_apqi["APQI_Qualified"], "APQI_raw"]
    apqi_range = RangeInfo(float(qualified_raw.min()), float(qualified_raw.max()))
    apqi_sets = {name: apply_apqi_score(frame, apqi_range) for name, frame in apqi_raw_sets.items()}

    ga_comparison = draw_ga_training(
        weights,
        history,
        mixed,
        ranges,
        rssi_range,
        oof_r,
        session_oof_r,
    )
    metrics = draw_nine_panel(datasets, apqi_sets, ranges, weights, rssi_range)
    row_results = build_row_results(datasets, apqi_sets, ranges, weights, rssi_range)
    mixed_ga_metric = metrics.loc[
        (metrics["환경"] == "혼합환경") & (metrics["방식"] == "GA HOPE")
    ].iloc[0]
    congestion_ga_metric = metrics.loc[
        (metrics["환경"] == "RSSI 양호·혼잡 환경") & (metrics["방식"] == "GA HOPE")
    ].iloc[0]
    paper_metrics = pd.DataFrame(
        [
            {"분류": "GA 가중치", "지표": "RTT", "값": float(weights[0]), "비고": "전체 하이브리드 Pearson r 최적화"},
            {"분류": "GA 가중치", "지표": "UTIL", "값": float(weights[1]), "비고": "전체 하이브리드 Pearson r 최적화"},
            {"분류": "혼합환경", "지표": "GA Pearson r", "값": float(mixed_ga_metric["Pearson_r"]), "비고": "유효표본 393개"},
            {"분류": "혼합환경", "지표": "GA Spearman rho", "값": float(mixed_ga_metric["Spearman_rho"]), "비고": "순위 일치도"},
            {"분류": "혼합환경", "지표": "GA-기존 RSSI Delta r", "값": float(mixed_ga_metric["Delta_vs_RSSI_r"]), "비고": "양수이면 GA 우세"},
            {"분류": "혼합환경", "지표": "GA-APQI Delta r", "값": float(mixed_ga_metric["Delta_vs_APQI_r"]), "비고": "양수이면 GA 우세"},
            {"분류": "혼잡다수환경", "지표": "GA Pearson r", "값": float(congestion_ga_metric["Pearson_r"]), "비고": "유효표본 81개"},
            {"분류": "혼잡다수환경", "지표": "GA-기존 RSSI Delta r", "값": float(congestion_ga_metric["Delta_vs_RSSI_r"]), "비고": "양수이면 GA 우세"},
            {"분류": "혼잡다수환경", "지표": "GA-APQI Delta r", "값": float(congestion_ga_metric["Delta_vs_APQI_r"]), "비고": "양수이면 GA 우세"},
            {"분류": "교차검증", "지표": "5-fold OOF Pearson r", "값": float(oof_r), "비고": "상태 비율 층화"},
            {"분류": "교차검증", "지표": "세션단위 OOF Pearson r", "값": float(session_oof_r), "비고": "측정 세션 누출 방지"},
        ]
    )

    history.to_csv(BASE_DIR / "04_GA_세대별_수렴.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(BASE_DIR / "05_상태분류_및_표본수.csv", index=False, encoding="utf-8-sig")
    row_results.to_csv(BASE_DIR / "06_행별_산출결과.csv", index=False, encoding="utf-8-sig")
    cv_results.to_csv(BASE_DIR / "07_GA_5fold_검증결과.csv", index=False, encoding="utf-8-sig")
    ga_comparison.to_csv(BASE_DIR / "08_GA_가중치비교.csv", index=False, encoding="utf-8-sig")
    excluded_frame.to_csv(BASE_DIR / "09_측정오류_제외행.csv", index=False, encoding="utf-8-sig")
    session_cv_results.to_csv(BASE_DIR / "10_GA_세션단위_검증결과.csv", index=False, encoding="utf-8-sig")
    grid_results.to_csv(BASE_DIR / "11_GA_가중치_전수검증.csv", index=False, encoding="utf-8-sig")
    paper_metrics.to_csv(BASE_DIR / "12_논문용_핵심지표.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(BASE_DIR / "03_전체환경_성능결과.csv", index=False, encoding="utf-8-sig")

    summary = {
        "입력파일": DATA_FILE.name,
        "원자료검증": verification,
        "결측처리": "7개 필수열 중 NaN/공란이 하나라도 있는 행을 제외",
        "측정오류처리": "NATURAL_1H_R1 2행(26.6667%)과 FREE1_1H_R2 2행(20%)을 제외",
        "중복처리": "실측 중복 가능성을 보존하기 위해 중복 제거를 수행하지 않음",
        "상태판정": f"RTT >= {RTT_THRESHOLD_MS:g} ms 또는 UTIL >= {UTIL_THRESHOLD_PERCENT:g}%",
        "환경별표본": qc.iloc[:3].to_dict(orient="records"),
        "GA": {
            "점수입력": GA_FEATURES,
            "점수입력에서제외": [
                "RSSI_dBm",
                "Link_Mbps",
                "PacketLoss_percent",
                "Upload_Mbps",
                "Download_Mbps",
            ],
            "실제전송률의역할": "Upload+Download는 GA 적합도 및 성능 평가 기준으로만 사용",
            "목적함수": "393개 전체 상태 전환형 하이브리드 점수와 Actual_Mbps의 Pearson r 최대화",
            "정상표본점수": "RSSI",
            "혼잡표본점수": "RTT·UTIL GA 가중합",
            "학습표본": len(mixed),
            "정상학습표본": int((~mixed["Is_Congested"]).sum()),
            "혼잡학습표본": int(mixed["Is_Congested"].sum()),
            "가중치": {label: float(weight) for label, weight in zip(GA_SCORE_LABELS, weights)},
            "학습_r": float(history.iloc[-1]["Best_Pearson_r"]),
            "5fold_OOF_r": float(oof_r),
            "세션단위_OOF_r": float(session_oof_r),
            "전수검증_최적가중치": {
                "RTT": float(grid_best["Weight_RTT"]),
                "UTIL": float(grid_best["Weight_UTIL"]),
                "Pearson_r": float(grid_best["Whole_Hybrid_Pearson_r"]),
            },
        },
        "APQI": {
            "원문": "A Novel WLAN Roaming Decision and Selection Scheme for Mobile Data Offloading (2015)",
            "식": "APQI = 0.4*log2(P(RSS_bar)/P(RSS_MIN)) + 0.6*log2(255/Channel_Load)",
            "파라미터": {
                "alpha": APQI_ALPHA,
                "w_r": APQI_W_RSS,
                "w_l": APQI_W_LOAD,
                "RSS_MIN_dBm": APQI_RSS_MIN_DBM,
                "MaximumBSSLoadValue_ratio": APQI_MAX_BSS_LOAD_RATIO,
            },
            "환산": "채널 사용률을 1~255 범위로 변환하고, APQI를 혼합환경 후보 통과 표본의 최소-최대로 0~100 환산",
            "재현범위": "실측 RSSI와 UTIL을 이용한 APQI 산식·평활화·후보 필터의 부분 재현",
        },
        "성능결과": metrics.to_dict(orient="records"),
    }
    (BASE_DIR / "실행결과_설명.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_lines = [
        f"- {row['환경']} / {row['방식']}: r={row['Pearson_r']:.3f}"
        for row in metrics.to_dict(orient="records")
    ]
    readable = "\n".join(
        [
            "찐찐막_GA 전체 상태 전환형 적합도 재분석 핵심 결과",
            "",
            f"상태 판정: RTT >= {RTT_THRESHOLD_MS:g} ms 또는 UTIL >= {UTIL_THRESHOLD_PERCENT:g}%",
            "GA 입력: RTT, UTIL",
            "GA 입력 제외: RSSI, Link, PacketLoss, Upload, Download",
            "실제 전송률(Upload+Download): GA 적합도 및 성능 평가 기준으로만 사용",
            "GA 목적함수: 정상은 RSSI, 혼잡은 RTT·UTIL 점수를 적용한 393개 전체 하이브리드 Pearson r",
            "",
            "제외 처리:",
            "- NaN/공란 포함 행 5개 제외",
            "- NATURAL_1H_R1 2행(PacketLoss 26.6667%) 제외",
            "- FREE1_1H_R2 2행(PacketLoss 20%) 제외",
            "- 중복 측정값은 제거하지 않음",
            "",
            f"혼합환경 유효표본: {len(mixed)}개 (정상 {int((~mixed['Is_Congested']).sum())}개, 혼잡 {int(mixed['Is_Congested'].sum())}개)",
            f"GA 가중치: RTT={weights[0]:.6f}, UTIL={weights[1]:.6f}",
            f"GA 전체 하이브리드 학습 r={history.iloc[-1]['Best_Pearson_r']:.6f}",
            f"GA 5-fold OOF r={oof_r:.6f}",
            f"GA 세션단위 OOF r={session_oof_r:.6f}",
            f"0.001 간격 전수검증 최적: RTT={grid_best['Weight_RTT']:.3f}, UTIL={grid_best['Weight_UTIL']:.3f}, r={grid_best['Whole_Hybrid_Pearson_r']:.6f}",
            "",
            "환경별 3방식 Pearson r:",
            *metric_lines,
            "",
            "해석 주의:",
            "- GA는 393개 전체 상태 전환형 점수의 상관성을 최적화하며, 혼잡 표본 내부 상관만을 최적화한 이전 코드와 목적함수가 다름.",
            "- Pearson r, Spearman rho, 95% bootstrap CI, 5-fold 및 세션단위 OOF 결과를 함께 해석해야 함.",
            "- 결과를 임의 조정하지 않았으며, APQI는 원문 산식·가중치·후보필터를 실측 RSSI/UTIL 범위에서 부분 재현함.",
        ]
    )
    (BASE_DIR / "재분석_핵심결과.txt").write_text(readable, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
