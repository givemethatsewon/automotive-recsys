# Automotive 추천 본실험 최종 결과

2026-11-10에 서버/Drive 결과를 재확인했다. **2026-11-07 06:58 KST에 32/32개 작업과 보고서 생성 완료**, 약 07:00 KST에 제어기가 결과를 회수하고 정상 종료했다. 과거 31/32 스냅샷과 구분한다.

과제 전체 맥락과 다음 작업은 [PROJECT_BRIEF](../PROJECT_BRIEF.md), 원문 출처는 [sources](sources/README.md)를 참고한다. 32개 추천 벤치마크의 완료가 제안서의 Spark 원본 분석·텍스트/차종 재정렬 완료를 뜻하지는 않는다.

## 주 비교

동일 warm-start test 158,805명·333,797개 정답·248,186개 상품 후보. train만 학습, valid만으로 설정/best 선택 후 test 평가. 학습 모델은 seed 42/43/44 평균·sample SD, Popularity는 결정적 1회다.

|모델|NDCG@10 평균 ± seed SD|Recall@10|APLT@10 (%)|Coverage@10 (%)|
|---|---:|---:|---:|---:|
|Popularity|0.00204150 ± 해당 없음|0.00369687|0.00000000|0.0056|
|BPR-MF|0.00117841 ± 0.00004268|0.00202953|0.15524280|12.9130|
|LightGCN|0.00369642 ± 0.00004705|0.00620172|0.73011975|6.4841|
|XSimGCL|0.00441927 ± 0.00011744|0.00724502|0.00004198|1.5942|

XSimGCL의 정확도가 가장 높지만 비인기 노출은 거의 0이다. APLT는 추천 10개 중 비인기 상품 비율이며, ARP와 다른 인기 편향 지표다. 대조학습이 tail 노출을 개선했다는 주장은 지지되지 않는다.

## CL 제거: 세 seed 모두 완료

|조건|seed|NDCG 평균|APLT (%)|
|---|---|---:|---:|
|XSimGCL CL=0|42/43/44|0.00414158|0.06876358|
|XSimGCL CL=0.1|42/43/44|0.00441927|0.00004198|

동일 encoder/noise의 이 recipe에서는 CL 사용 시 NDCG가 약 6.70% 높고 APLT는 낮았다. CL=0 세 실행은 모두 100 epoch에 도달했으며 best epoch는 75/60/75다. 자동 보고서의 “상한 부근 best 0개”는 best≥95 조건의 집계일 뿐 충분한 수렴을 증명하지 않는다. 제한된 탐색 범위·공통 raw L2 recipe variant라는 점도 함께 보고한다.

## 사용자별 paired CI

최종 paired_differences.csv는 모델 4개 간 6쌍 × 3 seeds × 2지표(NDCG/APLT) = 36행이다. seed마다 같은 158,805명을 짝지어 사용자 500회 bootstrap했다. 학습 seed 3개를 새로운 독립 사용자처럼 합치지 않았다. 아래는 각 seed의 XSimGCL−LightGCN NDCG 차이다.

|seed|평균 차이|사용자 paired 95% CI|
|---|---:|---|
|42|0.00062278|[0.00044112, 0.00081518]|
|43|0.00077575|[0.00057514, 0.00095998]|
|44|0.00077002|[0.00058407, 0.00096625]|

세 고정 seed의 사용자 bootstrap CI가 모두 양수다. 이것은 학습 seed까지 통합한 95% CI나 모든 학습 recipe에 대한 일반적 우월성 증명은 아니다. CL=0과의 paired CI는 이 자동 보고서의 4모델 비교에 포함되지 않는다.

## 저장 결과와 운영 종료

- [원래 자동 보고서 Markdown](results/final/RESULTS.md), [HTML](results/final/RESULTS.html). 자동 Markdown은 6자리 반올림으로 아주 작은 APLT를 0처럼 표시하므로 위 정밀 표 또는 CSV를 사용한다. 원본은 변형하지 않았다.
- [모델 평균/SD](results/final/model_comparison.csv), [모든 seed·그룹](results/final/all_seed_results.csv), [활동 그룹](results/final/group_results.csv), [층별](results/final/depth_results.csv), [보조 실험](results/final/ablation_results.csv), [paired CI](results/final/paired_differences.csv), [시간·메모리](results/final/timing_cost.csv).
- [학습 곡선](results/final/figures/learning_curves.png), [정확도·tail](results/final/figures/accuracy_tail.png), [층별 그림](results/final/figures/depth_ablation.png).
- [32개 완료 marker](results/final/completion.json), [서버와 로컬 사본 해시·할당 종료 검증](results/final/verification.json). 13개 원본 보고서 파일의 SHA-256을 서버 사본과 대조했다.
- 종료 후 계정 CU 감소 74.79, 잔액 100.14, 활성 할당 0개, 소모율 0.0 CU/h. 계정 관측값이므로 모델별 정확한 비용으로 쓰지 않는다.
- timing_cost.csv의 training_seconds는 기록된 epoch 학습 시간의 합이다. 재개 시 elapsed_seconds가 구간 기준일 수 있으므로 이를 캠페인 전체 시간으로 단순 합산하지 않는다.
- 텍스트/차종 특징이 없어 재정렬·Mismatch는 미실시다. warm-start test는 원래 test 상호작용의 약 55.2%이며 사용자 유지율이 아니다. 절대 정확도·tail 정답 회수는 낮아 실제 서비스 성능을 입증한 결과로 일반화하지 않는다.
