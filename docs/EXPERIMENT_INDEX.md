# 차종 후속 실험 결과 색인

2026-11-13 로컬 보관 산출물을 대조했다. 실행 완료와 해석 범위를 구분한다. 전체 상태의 기준 문서는 [PROJECT_BRIEF](../PROJECT_BRIEF.md)다.

|실험|run|후보|차종 출처|test NDCG 변화 (%)|사용자 paired 95% CI|원본 요약|
|---|---|---:|---|---:|---|---|
|vehicle-pilot-2026-11-11|final_bpr_mf_seed42|100|both / model|+1.93|[-0.00001201, 0.00005845]|[JSON](results/vehicle-pilot-2026-11-11/summary.json)|
|vehicle-pilot-models-2026-11-11|final_lightgcn_seed42|100|both / model|+1.21|[-0.00001062, 0.00010122]|[JSON](results/vehicle-pilot-models-2026-11-11/final_lightgcn_seed42/summary.json)|
|vehicle-pilot-models-2026-11-11|final_xsimgcl_seed42|100|both / model|+0.22|[-0.00003009, 0.00004683]|[JSON](results/vehicle-pilot-models-2026-11-11/final_xsimgcl_seed42/summary.json)|
|vehicle-pilot-seeds-2026-11-11|final_bpr_mf_seed43|100|items / model|+2.23|[0.00000021, 0.00005663]|[JSON](results/vehicle-pilot-seeds-2026-11-11/final_bpr_mf_seed43/summary.json)|
|vehicle-pilot-seeds-2026-11-11|final_bpr_mf_seed44|100|items / model|+7.91|[0.00005242, 0.00013167]|[JSON](results/vehicle-pilot-seeds-2026-11-11/final_bpr_mf_seed44/summary.json)|
|vehicle-pilot-seeds-2026-11-11|final_lightgcn_seed43|100|items / model|+1.70|[0.00001559, 0.00010889]|[JSON](results/vehicle-pilot-seeds-2026-11-11/final_lightgcn_seed43/summary.json)|
|vehicle-pilot-seeds-2026-11-11|final_lightgcn_seed44|100|items / model|+1.34|[0.00000278, 0.00009507]|[JSON](results/vehicle-pilot-seeds-2026-11-11/final_lightgcn_seed44/summary.json)|
|vehicle-pilot-seeds-2026-11-11|final_xsimgcl_seed43|100|items / model|+0.16|[-0.00000710, 0.00002294]|[JSON](results/vehicle-pilot-seeds-2026-11-11/final_xsimgcl_seed43/summary.json)|
|vehicle-pilot-seeds-2026-11-11|final_xsimgcl_seed44|100|items / model|+0.53|[0.00000291, 0.00004511]|[JSON](results/vehicle-pilot-seeds-2026-11-11/final_xsimgcl_seed44/summary.json)|
|vehicle-pilot-pool500-2026-11-11|final_bpr_mf_seed42|500|items / model|+5.82|[0.00002128, 0.00011790]|[JSON](results/vehicle-pilot-pool500-2026-11-11/final_bpr_mf_seed42/summary.json)|
|vehicle-pilot-pool500-2026-11-11|final_lightgcn_seed42|500|items / model|+1.96|[0.00001662, 0.00012844]|[JSON](results/vehicle-pilot-pool500-2026-11-11/final_lightgcn_seed42/summary.json)|
|vehicle-pilot-pool500-2026-11-11|final_xsimgcl_seed42|500|items / model|+0.47|[-0.00000241, 0.00004351]|[JSON](results/vehicle-pilot-pool500-2026-11-11/final_xsimgcl_seed42/summary.json)|

- `both`는 A1 리뷰+A2 상품 추론 합집합, `items`는 A2(n_items≥2)다. 실험별 주 조건이 달라 위 행들을 같은 조건의 seed 평균으로 합치지 않는다. 원본 test_results.json에는 다른 family도 포함된다.
- 43/44 seed의 design.seed=42와 single-seed 설명은 생성기의 고정 문자열 오류다. 실제 run ID, baseline 학습 settings.seed, 입력 추천 artifact SHA로 seed를 대조했다. 원본은 보존하고 CSV의 seed_from_baseline을 사용한다.
- pool500의 BPR-MF는 attempt2 결과다. 원본 summary 사본 일치는 확인했지만 attempt2 전용 manifest가 보관돼 있지 않아 입력 추천 hash lineage는 다른 실행과 같은 수준으로 재검증하지 못했다. attempt3 manifest는 그래프 모델 두 개만 포함한다.
- seed43/44의 여섯 주 조건 점추정은 모두 양수다. 이는 서로 다른 seed/사용자를 합친 통합 CI나 그래프 모델이 차종을 이미 학습했다는 증거가 아니다.
- pool500은 후보별 점수 범위를 사용했고 후보 수·가점 크기가 함께 바뀐다. 후보 수 자체의 비교는 [고정 단위 후보 진단](CANDIDATE_DIAGNOSTIC_2026-11-11.md)을 사용한다.
- 후보 진단은 세 모델 seed42, valid 220,516명, 후보100/500/1000이다. 추가 test나 학습은 없다. [receipt](results/candidate-diagnostic-2026-11-11/receipts.json), [CSV](results/candidate-diagnostic-2026-11-11/candidate_availability.csv).
- 최초 파일럿의 test/특징 개발 탐색·메타데이터 시점·동일 규칙 mismatch 한계는 [파일럿 해설](VEHICLE_PILOT_2026-11-11.md)에 있다.
