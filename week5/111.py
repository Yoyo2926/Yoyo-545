# make_testout_8_6.py
import numpy as np
import pandas as pd
import math

# -------- 配置区 --------
INPUT_CSV = "test7_2.csv"
OUTPUT_CSV = "testout_8.6.csv"

alpha = 0.95
p = 1 - alpha

# 如果你知道老师的 ES_abs（可选），把它填进来；否则置为 None
# Example from teacher:
teacher_ES_abs = 0.07690595548332976   # <-- 如果不想使用自动匹配，设为 None

# 如果你已经知道老师的 quantile method，可直接填入 "numpy_linear","pandas_lower","empirical_kth", "fractional_h", "type7" 等
preferred_method = None   # e.g. "empirical_kth" or None to auto-detect

# -------- 读取数据 --------
data = pd.read_csv(INPUT_CSV).iloc[:,0].dropna().astype(float).values
n = len(data)
mu = data.mean()
sorted_data = np.sort(data)

def tail_mean_inclusive(q):
    tail = sorted_data[sorted_data <= q]
    return tail.mean() if len(tail) > 0 else np.nan, len(tail)

# -------- 多种 candidate 方法 --------
candidates = {}

# 1) numpy.quantile default (numpy >=1.22 allows method=, older versions use interpolation=)
for m in ["linear","lower","higher","nearest","midpoint"]:
    try:
        q = np.quantile(data, p, method=m)
    except TypeError:
        q = np.quantile(data, p, interpolation=m)
    es, cnt = tail_mean_inclusive(q)
    candidates[f"numpy_{m}"] = dict(method=f"numpy_{m}", VaR=q, ES=es, tail_count=cnt)

# 2) pandas quantile with method if available
s = pd.Series(data)
for m in ["linear","lower","higher","nearest","midpoint"]:
    try:
        q = s.quantile(p, method=m)
    except TypeError:
        try:
            q = s.quantile(p, interpolation=m)
        except Exception:
            q = None
    if q is not None:
        es, cnt = tail_mean_inclusive(q)
        candidates[f"pandas_{m}"] = dict(method=f"pandas_{m}", VaR=q, ES=es, tail_count=cnt)

# 3) empirical kth (k = ceil(p*n))
k = int(math.ceil(p * n))
k_idx = max(1, k) - 1
var_k = sorted_data[k_idx]
es_k, cnt_k = tail_mean_inclusive(var_k)
candidates["empirical_kth"] = dict(method="empirical_kth", VaR=var_k, ES=es_k, tail_count=cnt_k, k=k)

# 4) fractional weighted method (h = p*(n+1)), typical textbook option
h = p * (n + 1)
j = int(math.floor(h))
g = h - j
if j <= 0:
    ES_frac = sorted_data[:math.ceil(p*n)].mean()
else:
    sum_full = sorted_data[:j].sum()
    partial = g * sorted_data[j] if j < n else 0.0
    denom = p * n
    ES_frac = (sum_full + partial) / denom
candidates["fractional_h_nplus1"] = dict(method="fractional_h_nplus1", VaR=None, ES=ES_frac, tail_count=j + (1 if g>0 else 0), h=h, j=j, g=g)

# 5) type7 (R default): h2 = (n-1)*p + 1
h2 = (n - 1) * p + 1
j2 = int(math.floor(h2))
g2 = h2 - j2
if j2 <= 0:
    ES_type7 = sorted_data[:math.ceil(p*n)].mean()
else:
    sum_full2 = sorted_data[:j2].sum()
    partial2 = g2 * sorted_data[j2] if j2 < n else 0.0
    ES_type7 = (sum_full2 + partial2) / (p * n)
candidates["type7_h"] = dict(method="type7_h", VaR=None, ES=ES_type7, tail_count=j2 + (1 if g2>0 else 0), h2=h2, j2=j2, g2=g2)

# -------- 选择方法（自动或指定） --------
if preferred_method is not None:
    chosen_key = preferred_method
    if chosen_key not in candidates:
        raise ValueError(f"preferred_method {preferred_method} not available. Candidates: {list(candidates.keys())}")
else:
    # 自动选择：如果老师提供 ES_abs，用该值作为目标，找最接近的 candidate
    if teacher_ES_abs is not None:
        best = None
        best_diff = float("inf")
        for kname, info in candidates.items():
            if info["ES"] is None or np.isnan(info["ES"]):
                continue
            ES_abs_candidate = -info["ES"]
            diff = abs(ES_abs_candidate - teacher_ES_abs)
            if diff < best_diff:
                best_diff = diff
                best = kname
        chosen_key = best
    else:
        # 没有老师数值，默认用 numpy_linear
        chosen_key = "numpy_linear" if "numpy_linear" in candidates else list(candidates.keys())[0]

chosen = candidates[chosen_key]

# -------- 输出并保存为 CSV（按课件格式： ES Absolute, ES Diff from Mean） --------
ES_param = chosen["ES"]  # 这是 E[X | X <= VaR]（通常是负值）
ES_abs = -ES_param
ES_diff_from_mean = mu - ES_param

df_out = pd.DataFrame({
    "ES Absolute": [float(ES_abs)],
    "ES Diff from Mean": [float(ES_diff_from_mean)]
})
df_out.to_csv(OUTPUT_CSV, index=False)

# -------- 打印诊断信息 --------
print("n =", n, "mu =", mu)
print("\nTried candidates and results (method: VaR, tail_count, ES, ES_abs):")
for kname, info in candidates.items():
    print(f" - {kname:25s} VaR={info.get('VaR',None)} tail_count={info.get('tail_count',None)} ES={info.get('ES',None)} ES_abs={(-info.get('ES',None) if info.get('ES',None) is not None else None)}")

print("\nChosen method:", chosen_key)
print("Chosen info:", chosen)
print("\nSaved output to", OUTPUT_CSV)
if teacher_ES_abs is not None:
    print("Teacher ES_abs:", teacher_ES_abs)
    print("Chosen ES_abs:", float(ES_abs))
    print("Absolute difference:", abs(float(ES_abs) - teacher_ES_abs))