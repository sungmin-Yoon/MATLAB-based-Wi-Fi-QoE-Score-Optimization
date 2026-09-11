# 📡 HOPE: Hybrid Wi-Fi QoE Optimization Algorithm

데이터 기반 혼잡 감지 및 체감 품질(QoE) 향상을 위한 하이브리드 AP 선택 알고리즘 프로젝트입니다.

---

## 📌 주요 특징
1. **RSSI 종속성 탈피:** 단순 신호 세기(RSSI) 대신 RTT와 채널사용률(Airtime Utilization)을 결합하여 실제 서비스 품질 평가.
2. **동적 하이브리드 모드:**
   - **정상 모드:** RSSI 100% (신호 세기 기반 AP 탐색)
   - **혼잡 모드:** RTT 60% + 채널사용률 40% (체감 품질 기반 AP 선택)
3. **데이터 기반 임계값 도출:** 전수 데이터 분류 정확도 탐색을 통해 `RTT ≥ 22ms` 또는 `채널사용률 ≥ 56%`를 혼잡 모드 전환 임계값으로 결정.
4. **유전 알고리즘 (Differential Evolution):** 혼잡 환경에서의 최적 가중치(RTT vs Util)를 알고리즘으로 산출.

---

## 📁 프로젝트 구조

```text
HOPE_WiFi_QoE/
├── README.md                 # 프로젝트 설명서
├── data/                     # 측정 데이터셋
│   ├── wifi_data.xlsx        # 일반/기본 환경 데이터 (300개)
│   └── wifi_data2.xlsx       # 혼잡/과부화 환경 데이터 (65개)
├── src/                      # 핵심 소스코드
│   ├── hybrid_mode.py        # [메인] HOPE 하이브리드 알고리즘 및 성능 평가
│   ├── ga_weight_finder.py   # Differential Evolution 기반 가중치 탐색
│   ├── threshold_analysis.py # 데이터 기반 임계값 전수 탐색
│   └── rssi_influence.py     # RSSI와 주요 지표 간 상관관계 분석
└── results/                  # 시각화 그래픽 결과물
    ├── hybrid_scatter.png    # 하이브리드 산점도 (정상 vs 혼잡)
    ├── threshold_analysis.png# 임계값 분류 정확도 분석
    ├── rssi_influence.png    # RSSI 상관관계 분석 그래프
    └── hope_120_60_40.png    # 수동 vs 하이브리드 비교 그래프
```

---

## 🚀 실행 방법

```bash
# 필요 라이브러리 설치
pip install pandas numpy matplotlib seaborn scipy openpyxl

# 하이브리드 알고리즘 실행
python src/hybrid_mode.py

# 가중치 탐색 (GA) 실행
python src/ga_weight_finder.py

# 임계값 분석 실행
python src/threshold_analysis.py
```
