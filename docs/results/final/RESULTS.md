# Automotive 추천 본실험 결과

test NDCG@10 평균이 가장 높은 모델은 xsimgcl (0.004419)이다. 설정·epoch 선택에는 valid만 사용했다.

## 평가 범위

train만 학습한 warm-start 전체 상품 평가다. test 대상은 158,805명, 정답 333,797개이며 원래 test 상호작용의 약 55.2%다.

학습 모델 seed는 [42, 43, 44]이고 Popularity는 결정적 1회 기준선이다. seed SD와 사용자별 paired bootstrap CI는 별도로 제공한다.

|모델|NDCG@10 평균|seed SD|Recall@10|APLT@10|
|---|---:|---:|---:|---:|
|xsimgcl|0.004419|0.000117|0.007245|0.000000|
|lightgcn|0.003696|0.000047|0.006202|0.007301|
|popularity|0.002041|해당 없음|0.003697|0.000000|
|bpr_mf|0.001178|0.000043|0.002030|0.001552|

## 저장 파일

model_comparison.csv: 평균/seed SD, all_seed_results.csv: seed별 점수, group_results.csv: 활동 그룹, depth_results.csv/ablation_results.csv: 보조 실험, paired_differences.csv: 동일 사용자 차이의 CI, timing_cost.csv: 시간/메모리.

## 해석과 한계

예산 상한 부근에서 best epoch를 기록한 실행은 0개다. 수렴 부족 가능성을 결과와 함께 고려한다. XSimGCL은 공통 squared raw L2를 적용한 recipe variant다.

텍스트·차종 특징이 제공되지 않아 재정렬과 차종 불일치는 미실시다. 해당 지표를 0으로 대체하지 않았다.

## 재현

experiment_plan.json, frozen_selection.json, 입력 manifest, 소스 snapshot, 실행별 manifest/epoch CSV/사용자별 Parquet/추천 점수와 best 모델을 같은 Drive 폴더에 보관한다.
