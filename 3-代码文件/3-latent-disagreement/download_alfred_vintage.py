# -*- coding: utf-8 -*-
"""
ALFRED vintage 数据下载（run_bw6_vintage.py 的前置数据步骤）
=============================================================
机制（2026-08 实测打通，无需 API key）：
1. GET https://alfred.stlouisfed.org/series/downloaddata?seid={SERIES} 取回 HTML，
   <select name="form[selected_vintage_dates][]"> 的 option values = 全部 vintage 日期；
2. POST 同 URL（form[file_type]=2 = Observations by Vintage Date），
   每批 40–60 个 vintage 日期，返回 zip，内含 CSV：observation_date + 每 vintage 一列。
   注意：传入无效 vintage 日期时服务器静默回退为 HTML 表单页（已实测），
   因此选中的日期必须逐一对照页面列表。

选点规则：1983-01 ~ 2024-12 每月最后一个 vintage 日期（当月末信息集代表），
外加 2025 年起每月最后一个 vintage 日期（对照用，含最终 vintage 供锚点 a）。

输出：1-数据文件/ALFRED_vintage/{SERIES}_vintages.csv
     （宽表：observation_date + 每 vintage 一列 SERIES_YYYYMMDD）
礼貌下载：每 POST 间隔 2.5s，失败重试 3 次（指数退避）。
"""
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent.parent / '1-数据文件' / 'ALFRED_vintage'
OUT_DIR.mkdir(exist_ok=True)

BASE = 'https://alfred.stlouisfed.org/series/downloaddata'
# 系列与观测起点（按系列实际范围；INDPRO 1919 起、PAYEMS 1939 起、PCE 三系列 1959 起）
# 注意：obs_end_date 超过该系列最新观测会被静默拒绝（回退 HTML 表单页，
# 实测 INDPRO "Data is not available after 2026-07-01"），故端点一律取页面表单的默认值。
SERIES = {
    'INDPRO': '1919-01-01',
    'PAYEMS': '1939-01-01',
    'PCEDG': '1959-01-01',
    'PCEND': '1959-01-01',
    'PCES': '1959-01-01',
}
BATCH = 50
SLEEP = 2.5

SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'curl/8.0'})  # 实测 Mozilla UA 易被 WAF 重置连接


def get_vintage_dates(seid):
    """GET 下载页，解析 selected_vintage_dates 的全部 option values。"""
    html = None
    for att in range(5):
        try:
            r = SESSION.get(BASE, params={'seid': seid}, timeout=60)
            r.raise_for_status()
            html = r.text
            break
        except Exception as e:
            print(f'    GET 失败（第 {att + 1} 次）: {type(e).__name__}: {e}')
            if att == 4:
                raise
            time.sleep(5 * (att + 1))
    m = re.search(r'<select[^>]*name="form\[selected_vintage_dates\]\[\]"[^>]*>(.*?)</select>',
                  html, re.S)
    assert m, f'{seid}: 未找到 vintage 日期 select 块'
    dates = re.findall(r'value="(\d{4}-\d{2}-\d{2})"', m.group(1))
    assert len(dates) > 100, f'{seid}: vintage 日期数异常（{len(dates)}）'
    # 观测端点取表单默认值（超出系列实际范围会被静默拒绝）
    me = re.search(r'name="form\[obs_end_date\]"[^>]*value="(\d{4}-\d{2}-\d{2})"', html)
    obs_end = me.group(1) if me else '2026-07-01'
    return sorted(dates), obs_end


def pick_vintages(all_dates):
    """1983-01 起每月最后一个 vintage 日期。"""
    by_month = {}
    for d in all_dates:
        if d >= '1983-01-01':
            by_month[d[:7]] = max(by_month.get(d[:7], ''), d)
    picked = sorted(by_month.values())
    return picked


def post_batch(seid, obs_start, obs_end, dates, attempt_max=3):
    """POST 一批 vintage 日期，返回 {列名: Series}。失败重试（指数退避）。"""
    fields = [('form[units]', 'lin'),
              ('form[obs_start_date]', obs_start),
              ('form[obs_end_date]', obs_end),
              ('form[file_type]', '2'),
              ('form[file_format]', 'csv')]
    fields += [('form[selected_vintage_dates][]', d) for d in dates]
    fields.append(('form[download_data]', ''))
    for att in range(attempt_max):
        try:
            r = SESSION.post(BASE, params={'seid': seid}, data=fields, timeout=300)
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))  # 非 zip（HTML 回退）会抛 BadZipFile
            cols = {}
            for nm in zf.namelist():
                if not nm.endswith('.csv'):
                    continue
                with zf.open(nm) as f:
                    df = pd.read_csv(f, dtype=str)
                df = df.set_index('observation_date')
                for c in df.columns:
                    cols[c] = pd.to_numeric(df[c], errors='coerce')
            got = {c.split('_')[-1] for c in cols}
            want = {d.replace('-', '') for d in dates}
            missing = want - got
            if missing:
                raise RuntimeError(f'返回缺 vintage 列: {sorted(missing)[:5]}...')
            return cols
        except Exception as e:
            hint = ''
            if isinstance(e, zipfile.BadZipFile):
                lis = re.findall(r'<li>([^<]{5,200})</li>', r.text)
                hint = f' | 页面提示: {lis[:2]}' if lis else ''
            print(f'    POST 失败（第 {att + 1} 次）: {type(e).__name__}: {e}{hint}')
            if att == attempt_max - 1:
                raise
            time.sleep(5 * (att + 1))


def download_series(seid, obs_start):
    print(f'===== {seid} =====')
    all_dates, obs_end = get_vintage_dates(seid)
    print(f'  页面 vintage 日期总数: {len(all_dates)}（{all_dates[0]} ~ {all_dates[-1]}），obs_end={obs_end}')
    picked = pick_vintages(all_dates)
    print(f'  选中月末 vintage: {len(picked)}（{picked[0]} ~ {picked[-1]}）')
    out_path = OUT_DIR / f'{seid}_vintages.csv'
    if out_path.exists():
        old = pd.read_csv(out_path, index_col=0, nrows=1)
        have = {c.split('_')[-1] for c in old.columns}
        picked = [d for d in picked if d.replace('-', '') not in have]
        print(f'  断点续传：已有 {len(have)} 列，待下 {len(picked)} 个 vintage')
    cols = {}
    for b0 in range(0, len(picked), BATCH):
        batch = picked[b0:b0 + BATCH]
        cols.update(post_batch(seid, obs_start, obs_end, batch))
        print(f'  批次 {b0 // BATCH + 1}: {batch[0]} ~ {batch[-1]} 完成（累计 {len(cols)} 列）')
        time.sleep(SLEEP)
    if not cols:
        print('  无新增，跳过合并')
        return
    new = pd.DataFrame(cols)
    new.index.name = 'observation_date'
    if out_path.exists():
        old = pd.read_csv(out_path, index_col=0)
        new = old.join(new, how='outer')
    new = new.sort_index()[sorted(new.columns)]
    new.to_csv(out_path)
    print(f'  落盘 {out_path.name}: {new.shape[0]} 行 × {new.shape[1]} 列')


if __name__ == '__main__':
    seids = sys.argv[1:] or list(SERIES)
    for seid in seids:
        download_series(seid, SERIES[seid])
    print('\n全部完成。')
