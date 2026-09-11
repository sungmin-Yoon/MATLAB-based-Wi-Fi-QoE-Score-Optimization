import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HOPE_DIR = r'C:\Users\kimse\OneDrive\Desktop\hope'

# ── 데이터 로드 ──
df1_all = pd.read_excel(f'{HOPE_DIR}/wifi_data.xlsx').dropna().reset_index(drop=True)
df2_all = pd.read_excel(f'{HOPE_DIR}/wifi_data2.xlsx').dropna().reset_index(drop=True)
df2_all.columns = df1_all.columns

# ── 전역 정규화 범위 (wifi_data1 전체 기준 고정) ──
G = {
    'rtt':  (df1_all.iloc[:,3].min(), df1_all.iloc[:,3].max()),
    'util': (df1_all.iloc[:,2].min(), df1_all.iloc[:,2].max()),
}

def norm(s, lo, hi, invert=False):
    v = ((s - lo) / (hi - lo)).clip(0, 1)
    return 1 - v if invert else v

# ── 임계값: 혼잡 샘플만 추출 ──
TH_RTT, TH_UTIL = 22, 56

df2r = df2_all.copy(); df2r.columns = range(df2_all.shape[1])
is_cong = (df2r.iloc[:,3] >= TH_RTT) | (df2r.iloc[:,2] >= TH_UTIL)
df_cong_only = df2r[is_cong].reset_index(drop=True)

print(f"혼잡 환경 전체: {len(df2r)}개")
print(f"혼잡 조건 해당(RTT>={TH_RTT}ms OR Util>={TH_UTIL}%): {len(df_cong_only)}개")
print()

# ── 혼잡 샘플 특징 ──
P_RTT  = norm(df_cong_only.iloc[:,3], *G['rtt'],  invert=True)
P_UTIL = norm(df_cong_only.iloc[:,2], *G['util'],  invert=True)
y_true = df_cong_only.iloc[:,6] + df_cong_only.iloc[:,5]  # 실제전송률

# ══════════════════════════════════════════════════════
# Genetic Algorithm (Differential Evolution)
# 목적: 혼잡 샘플에서 RTT+Util 점수가
#        실제 속도와 가장 높은 상관관계를 갖는 가중치 탐색
# ══════════════════════════════════════════════════════
def objective(weights):
    w_rtt, w_util = weights
    t = w_rtt + w_util
    if t == 0: return 1.0
    w_rtt, w_util = w_rtt / t, w_util / t

    score = w_rtt * P_RTT + w_util * P_UTIL
    if np.std(score) == 0 or np.std(y_true) == 0:
        return 1.0
    corr = np.corrcoef(score, y_true)[0, 1]
    return 1 - corr if not np.isnan(corr) else 1.0

print("=" * 50)
print("GA 실행 중 (Differential Evolution)...")
print("  탐색 범위: RTT [0~1], 채널사용률 [0~1]")
print("  목적함수: 실제속도와의 상관계수 최대화")
print("=" * 50)

result = differential_evolution(
    objective,
    bounds=[(0, 1), (0, 1)],
    strategy='best1bin',
    maxiter=3000,
    popsize=30,
    tol=1e-7,
    seed=42,
    polish=True      # 최종 결과 정밀 보정
)

w_rtt_raw, w_util_raw = result.x
t = w_rtt_raw + w_util_raw
W_RTT_GA  = w_rtt_raw / t
W_UTIL_GA = w_util_raw / t

print()
print("=" * 50)
print("GA 결과")
print("=" * 50)
print(f"  RTT 가중치:       {W_RTT_GA:.4f}  ({W_RTT_GA*100:.1f}%)")
print(f"  채널사용률 가중치: {W_UTIL_GA:.4f}  ({W_UTIL_GA*100:.1f}%)")
print(f"  수렴 성공 여부:    {result.success}")
print(f"  목적함수 최솟값:   {result.fun:.6f}  (상관계수: {1-result.fun:.6f})")
print()

# ── 기존 수동 60:40과 성능 비교 ──
score_ga     = (W_RTT_GA  * P_RTT + W_UTIL_GA  * P_UTIL)
score_manual = (0.60      * P_RTT + 0.40        * P_UTIL)

corr_ga     = np.corrcoef(score_ga,     y_true)[0,1]
corr_manual = np.corrcoef(score_manual, y_true)[0,1]

print("=" * 50)
print("GA 도출 vs 수동 60:40 성능 비교 (혼잡 샘플)")
print("=" * 50)
print(f"  GA 도출  ({W_RTT_GA*100:.1f}%:{W_UTIL_GA*100:.1f}%): 상관계수 = {corr_ga:.4f}")
print(f"  수동 60:40:                    상관계수 = {corr_manual:.4f}")
print(f"  차이: {(corr_ga - corr_manual)*100:.2f}%p")
