from __future__ import annotations
"""比较多种权重方法并输出轨迹排序 Spearman 相关。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.entropy_weight import entropy_weight


INDICATOR_COLS = ['time_rate', 'integrity_rate', 'speed_rate', 'angle_rate']


def normalize_cost_to_benefit(df: pd.DataFrame, cols: list[str], eps: float = 1e-12) -> pd.DataFrame:
    """将成本型指标(越小越好)归一化为效益型(越大越好)。"""
    x = df[cols].astype(float)
    z = pd.DataFrame(index=x.index, columns=cols, dtype=float)
    for col in cols:
        col_min = float(x[col].min())
        col_max = float(x[col].max())
        span = col_max - col_min
        if span <= eps:
            z[col] = 1.0
        else:
            z[col] = (col_max - x[col]) / (span + eps)
    return z


def equal_weight(cols: list[str]) -> pd.Series:
    """等权重。"""
    return pd.Series(np.full(len(cols), 1.0 / len(cols)), index=cols)


def ahp_weight(pairwise: np.ndarray, cols: list[str]) -> tuple[pd.Series, dict]:
    """根据 AHP 判断矩阵计算权重并返回一致性指标。"""
    eigvals, eigvecs = np.linalg.eig(pairwise)
    max_idx = int(np.argmax(eigvals.real))
    principal = np.abs(eigvecs[:, max_idx].real)
    weights = principal / principal.sum()

    n = pairwise.shape[0]
    lambda_max = float(eigvals[max_idx].real)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    return pd.Series(weights, index=cols), {'lambda_max': lambda_max, 'CI': ci, 'CR': cr}


def critic_weight(df: pd.DataFrame, cols: list[str], eps: float = 1e-12) -> pd.Series:
    """CRITIC 权重：标准差 × 冲突性。"""
    z = normalize_cost_to_benefit(df, cols, eps=eps)
    std = z.std(axis=0, ddof=0)
    corr = z.corr(method='pearson').fillna(0.0)
    conflict = (1 - corr).sum(axis=0)
    information = std * conflict

    if float(information.sum()) <= eps:
        return equal_weight(cols)
    return information / information.sum()


def score_by_weight(df: pd.DataFrame, cols: list[str], w: pd.Series) -> pd.Series:
    """基于成本型指标计算质量分(越大越好)。"""
    risk = (df[cols] * w[cols]).sum(axis=1)
    return 1 - risk


def run_compare(
    indicator_path: Path,
    output_dir: Path,
    ahp_matrix: np.ndarray | None = None,
) -> dict:
    """执行权重计算、排序和 Spearman 相关分析。"""
    df = pd.read_csv(indicator_path)
    cols = [c for c in INDICATOR_COLS if c in df.columns]
    if len(cols) != 4:
        raise ValueError(f'指标列不完整，期望 {INDICATOR_COLS}，实际 {cols}')

    _, entropy_w = entropy_weight(df, cols)
    equal_w = equal_weight(cols)

    if ahp_matrix is None:
        ahp_matrix = np.array(
            [
                [1, 2, 2, 1 / 2],
                [1 / 2, 1, 1, 1 / 3],
                [1 / 2, 1, 1, 1 / 3],
                [2, 3, 3, 1],
            ],
            dtype=float,
        )

    ahp_w, ahp_consistency = ahp_weight(ahp_matrix, cols)
    critic_w = critic_weight(df, cols)

    scores = pd.DataFrame(
        {
            'file': df['file'],
            'entropy': score_by_weight(df, cols, entropy_w),
            'equal': score_by_weight(df, cols, equal_w),
            'ahp': score_by_weight(df, cols, ahp_w),
            'critic': score_by_weight(df, cols, critic_w),
        }
    )

    ranks = scores.copy()
    for m in ['entropy', 'equal', 'ahp', 'critic']:
        ranks[f'{m}_rank'] = scores[m].rank(ascending=False, method='average')

    rank_cols = ['entropy_rank', 'equal_rank', 'ahp_rank', 'critic_rank']
    spearman = ranks[rank_cols].corr(method='spearman')

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_df = pd.DataFrame(
        {'entropy': entropy_w, 'equal': equal_w, 'ahp': ahp_w, 'critic': critic_w}
    )
    weights_path = output_dir / 'weight_compare_weights.csv'
    scores_path = output_dir / 'weight_compare_scores.csv'
    spearman_path = output_dir / 'weight_compare_spearman.csv'
    summary_path = output_dir / 'weight_compare_summary.json'

    weights_df.to_csv(weights_path, encoding='utf-8-sig')
    ranks.to_csv(scores_path, index=False, encoding='utf-8-sig')
    spearman.to_csv(spearman_path, encoding='utf-8-sig')

    result = {
        'sample_count': int(len(df)),
        'weights': {
            'entropy': {k: float(v) for k, v in entropy_w.to_dict().items()},
            'equal': {k: float(v) for k, v in equal_w.to_dict().items()},
            'ahp': {k: float(v) for k, v in ahp_w.to_dict().items()},
            'critic': {k: float(v) for k, v in critic_w.to_dict().items()},
        },
        'ahp_consistency': ahp_consistency,
        'outputs': {
            'weights': str(weights_path),
            'scores': str(scores_path),
            'spearman': str(spearman_path),
        },
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def parse_ahp_matrix(ahp_json: str | None) -> np.ndarray | None:
    """解析命令行 JSON 判断矩阵。"""
    if not ahp_json:
        return None
    matrix = np.array(json.loads(ahp_json), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError('AHP 判断矩阵必须是 4x4')
    return matrix


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description='比较等权/AHP/CRITIC/熵权并计算Spearman相关')
    parser.add_argument(
        '--indicator',
        type=str,
        default='output/entropy_indicator_table.csv',
        help='指标表路径',
    )
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--ahp-matrix-json', type=str, default=None, help='AHP 4x4 判断矩阵(JSON字符串)')
    args = parser.parse_args()

    result = run_compare(
        indicator_path=Path(args.indicator),
        output_dir=Path(args.output_dir),
        ahp_matrix=parse_ahp_matrix(args.ahp_matrix_json),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
