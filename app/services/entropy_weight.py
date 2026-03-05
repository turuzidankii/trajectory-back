from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.other import parse_upload_points


def build_indicator_table(test_dir: Path) -> pd.DataFrame:
    rows = []
    csv_files = sorted(test_dir.glob('*.csv'))
    for file_path in csv_files:
        content = file_path.read_bytes()
        _, summary, details = parse_upload_points(content)

        counts = summary.get('counts', {})
        total = int(len(details))

        if total == 0:
            # parse_upload_points 未返回总点数时，使用最保守兜底，避免除零
            total = 1

        rows.append(
            {
                'file': file_path.name,
                'total_points': total,
                'time_rate': counts.get('time', 0) / total,
                'integrity_rate': counts.get('integrity', 0) / total,
                'speed_rate': counts.get('speed', 0) / total,
                'angle_rate': counts.get('angle', 0) / total,
            }
        )

    if not rows:
        raise ValueError(f'目录下未找到 CSV 文件: {test_dir}')

    return pd.DataFrame(rows)


def entropy_weight(df: pd.DataFrame, cols: list[str], eps: float = 1e-12) -> tuple[pd.DataFrame, pd.Series]:
    x = df[cols].astype(float).copy()

    # 成本型指标正向化：值越小越好
    z = pd.DataFrame(index=x.index, columns=cols, dtype=float)
    for col in cols:
        col_min = float(x[col].min())
        col_max = float(x[col].max())
        span = col_max - col_min
        if span <= eps:
            z[col] = 1.0
        else:
            z[col] = (col_max - x[col]) / (span + eps)

    p = (z + eps).div((z + eps).sum(axis=0), axis=1)

    n = len(df)
    k = 1.0 / np.log(n)
    e = -k * (p * np.log(p)).sum(axis=0)
    d = 1 - e

    if float(d.sum()) <= eps:
        w = pd.Series(np.full(len(cols), 1.0 / len(cols)), index=cols)
    else:
        w = d / d.sum()

    detail = pd.DataFrame({'entropy': e, 'diversity': d, 'weight': w})
    return detail, w


def run_entropy(test_dir: Path, output_dir: Path) -> dict:
    indicator_df = build_indicator_table(test_dir)
    cols = ['time_rate', 'integrity_rate', 'speed_rate', 'angle_rate']
    detail, weight = entropy_weight(indicator_df, cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    indicator_path = output_dir / 'entropy_indicator_table.csv'
    detail_path = output_dir / 'entropy_weight_detail.csv'
    summary_path = output_dir / 'entropy_weight_summary.json'

    indicator_df.to_csv(indicator_path, index=False, encoding='utf-8-sig')
    detail.to_csv(detail_path, index=True, encoding='utf-8-sig')

    result = {
        'sample_count': int(len(indicator_df)),
        'weights': {
            'time': float(weight['time_rate']),
            'integrity': float(weight['integrity_rate']),
            'speed': float(weight['speed_rate']),
            'angle': float(weight['angle_rate']),
        },
        'indicator_table': str(indicator_path),
        'weight_detail': str(detail_path),
    }

    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='基于 4 个质量维度异常率计算熵权')
    parser.add_argument('--test-dir', type=str, default='test', help='样本 CSV 目录')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    args = parser.parse_args()

    result = run_entropy(Path(args.test_dir), Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
