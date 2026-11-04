# 본실험에서 수집할 결과

2026-11-05 기준 수집 계약. 이 문서는 **필요한 산출물**과 **현재 코드가 실제 저장하는 항목**을 구분한다.
현재 기준 데이터는 전처리된 5-core이고, 모델 학습·평가는 이 데이터에서 시작한다.

## 1. 연구 질문과 최종 표

| 질문 | 비교 | 필요한 결과 |
|---|---|---|
| 개인화/그래프/대조학습으로 정확도와 편향이 어떻게 달라지는가? | Popularity, BPR-MF, LightGCN, XSimGCL | NDCG@10, Recall@10, APLT@10, ARP@10, Catalog Coverage@10 |
| 그래프를 깊게 만들면 정확도와 인기 편향이 어떻게 달라지는가? | LightGCN 1·2·3층 | 같은 지표, 학습 시간·메모리, 층별 변화 |
| 대조 손실의 기여는 무엇인가? | XSimGCL CL 가중치 0 / 선택한 가중치 | encoder·noise·다른 설정을 유지한 보조 실험 |
| 이력이 적은 사용자에게도 효과가 있는가? | train 이력 1–5개 / 6개 이상 | 그룹별 지표·사용자 수·모델 간 차이 |
| 인기 감점·텍스트·차종 재정렬은 어떤 효과가 있는가? | 원래 순위 / +인기 감점 / +텍스트 / +차종 | 정확도 유지율, APLT, Mismatch, unknown/coverage, 후보 Recall@100 |

주 정확도 지표는 NDCG@10, 주 비인기 노출 지표는 APLT@10이다.
모델이 모두 같은 방향의 결과를 낼 것이라고 가정하지 않는다.

최종 표의 각 수치는 실행 ID와 사용자별 결과로 추적 가능해야 한다.
학습 모델은 seed 42·43·44 각각의 결과와 평균·sample SD를 함께 보인다.
Popularity는 결정적 1회 결과로 표기한다. 그 사용자 평균의 불확실성과 학습 seed SD는 서로 다른 개념이다.

## 2. 실행 하나마다 보관할 데이터

### A. 실행·데이터 manifest

- `run_id`, `purpose`(tuning/final/ablation/pilot), 모델 이름·variant, seed, 시작/종료 시각.
- 코드 commit 또는 실행 시 소스 hash·dirty diff. 새 프로젝트의 미커밋 소스도 재현할 수 있게 보존한다.
- 입력 파일별 SHA-256, user/item map 버전, split, 상호작용/사용자/상품/평가 후보 수.
- 실제 설정: dim, batch, LR, 정규화 계수·식, 층 수, CL 계수·temperature·noise, 초기화, optimizer와 precision.
- GPU 이름·메모리, Python/PyTorch/CUDA/driver와 패키지 버전, graph layout, symmetric-backward 사용 여부.
- train만 학습했는지, 후보·seen mask·동점·집계 규칙, early stopping 조건, best epoch와 checkpoint hash.
- 소요 시간(초기화/학습/평가/저장/전송), peak allocated, CU 잔액·소모율 관측과 측정 시각.
- 계정 잔액 차이는 다른 작업의 사용량을 포함할 수 있으므로 `account_balance_delta`와 실행별 추정 비용을 구분한다.

### B. epoch별 곡선

`epoch, train_total_loss, train_bpr_loss, train_cl_loss_raw, cl_weight, train_reg_term,
valid_ndcg10, valid_recall10, valid_aplt10, valid_arp10, valid_coverage10,
training_seconds, validation_seconds, wall_seconds, is_best`.

검증을 하지 않은 epoch의 valid 값은 null로 둔다. 0이나 이전 값을 새 관측치처럼 쓰지 않는다.
분해 loss는 raw/가중 후 값을 구분하고 batch/interaction 가중 집계 규칙을 기록한다.
매 batch의 모든 음성 ID를 저장하는 대신 seed·RNG state·설정과 필요한 추출 집계를 보관한다.

이 곡선으로 미수렴·과적합·early stopping을 판단한다. 1 epoch pilot은 성능 순위의 근거로 쓰지 않는다.

### C. 전체·그룹별 지표

`run_id, split, group, n_users, n_ground_truth, n_candidate_items,
ndcg10, recall10, hit_rate10, aplt10, arp10, catalog_coverage10`.

- `group`: `all`, `1-5`, `6+`. 차종 실험에서는 미리 정한 동일한 vehicle-known cohort도 추가한다.
- NDCG/Recall/APLT/ARP는 사용자별 값의 macro 평균이다. NDCG의 ideal list는 `min(10, 정답 수)`를 사용한다.
- Catalog Coverage는 추천된 고유 상품 수 / 전체 학습 상품 수다.
- 보조적으로 비인기 정답이 있는 사용자에 대한 Tail Recall@10을 추가할 수 있다. 정답이 없는 사용자는 null로 두고 집계 대상 수를 적는다.
- test는 warm-start 대상 158,805명, 333,797개 상호작용이라는 범위를 함께 보고한다. 원래 test 상호작용의 55.2%가 남았다는 점을 명시한다.

### D. 사용자별 점수

`run_id, split, uid, group, n_train, n_gt, ndcg10, recall10, hit_rate10, aplt10, arp10`.

이 데이터로 같은 사용자의 모델 간 차이, 활동 그룹 차이, paired bootstrap을 계산한다.
집계표만 남기면 나중에 신뢰구간·그룹 분석을 다시 만들기 어렵다.
세 seed의 점수를 독립적인 새 사용자처럼 합치지 않는다.
우선 seed별 paired 사용자 차이의 95% CI와 seed별 평균/SD를 구분해 보고한다.
둘을 합친 CI가 필요하면 seed와 사용자의 반복 구조를 반영한 방법을 별도로 명시한다.

### E. 추천 후보와 인기 노출

top-100을 효율적인 배열로 보관한다.

```text
uid[n_users]
iid[n_users, 100]
score[n_users, 100]   # 원래 모델 점수, float32
metadata: run_id, split, checkpoint_hash, candidate/mask/tie 규칙
```

재정렬 결과는 base score와 인기/텍스트/차종 각 항, 최종 점수·순위를 연결해서 보관한다.
원래 ID가 필요하면 user_map/item_map으로 연결한다. 이 추천 데이터는 Git에 넣지 않는다.

추가 item exposure 표:
`iid, train_n, popular, recommendation_count_top10, recommendation_share_top10`.
전체 학습 catalog의 0회 노출 상품도 포함한다.
이 표로 head/tail 노출, coverage, 필요 시 exposure Gini와 집중도를 다시 계산할 수 있다.

## 3. 재정렬·차종 실험의 추가 수집

먼저 같은 checkpoint·후보 목록에서 다음 네 조건을 비교한다.

1. base ranking.
2. base + popularity penalty.
3. base + popularity penalty + text similarity.
4. base + popularity penalty + text similarity + vehicle compatibility.

각 항의 변환/정규화 방식과 λ·γ·β를 기록한다. 계수는 valid로만 선택한다.
NDCG 유지율은 **동일 사용자 집합의 base NDCG 대비 비율**이다.
valid에서 base의 약 95% 이상을 유지하는 설정 중 다양성/적합성을 비교한 뒤 test 설정을 고정한다.
후보 Recall@100을 기록해 재정렬 이전의 후보 누락을 확인한다. 후보 수를 늘릴지는 valid에서 결정한다.

필요한 특징 입력:

- 사용자: 원래 `user_id`, 제조사·모델·연식, 추출 근거·시점·출처, unknown/복수 차량 정보.
- 상품: `parent_asin`, 상품명/설명, 적합 제조사·모델·연식 범위, universal-fit/unknown 상태.
- 추출 품질: 사람이 검수한 표본 수, 오류 유형, 정확도/coverage와 표본 추출 방법.

차종 평가 cohort는 ranking을 보기 전에 vehicle feature의 유무로 정하고 모든 비교에 공통 사용한다.
전체 사용자 지표와 vehicle-known cohort 지표를 함께 남긴다.
Mismatch는 `판정 가능한 top-10 추천 중 불일치 비율`로 보고하며 분자/분모를 저장한다.
unknown은 compatible/incompatible 어느 쪽으로도 세지 않는다.
각 방법의 판정 가능 비율과 unknown 추천 비율을 함께 표시해 분모 변화로 인한 착시를 방지한다.
같은 compatibility 정보를 가점과 평가에 사용하므로 mismatch 감소만으로 실제 효과를 입증하지 않는다.
NDCG 변화와 추출 품질을 함께 판단한다.

현재 제공 폴더에는 이 특징 테이블들이 없다. 따라서 첫 본실험은 네 추천기의 기본 비교·층 수·사용자 그룹을 먼저 진행할 수 있다.
제안서의 raw 0-core vs 5-core 분포 비교는 별도 데이터 분석이다. 지금 '기본 모델 입력'인 전처리 5-core와 혼동하지 않는다.
raw 파일/집계가 필요하면 전처리 담당의 Spark 집계를 받아 연결하고, 이를 기다리느라 기본 모델 학습을 멈추지 않는다.

## 4. 산출물 구조

아래는 본실험을 위한 목표 구조다. 현재 학습기 출력 파일명을 이 구조로 옮길 때 매핑을 남긴다.

```text
results/experiments/
  experiment_plan.json              # 사전에 정한 비교·설정 탐색·선택 규칙
  dataset_manifest.json
  runs/<run_id>/
    manifest.json
    epochs.jsonl
    best.pt / last.pt
    valid/{metrics.json,per_user.parquet,recommendations.npz,item_exposure.parquet}
    test/{metrics.json,per_user.parquet,recommendations.npz,item_exposure.parquet}
    execution-log.ipynb
  aggregates/
    model_comparison.csv
    group_comparison.csv
    paired_differences.csv
    ablation_results.csv
    reranking_results.csv
    timing_cost.csv
```

보고서에 우선 사용할 그림은 학습/검증 곡선, NDCG–APLT 관계, 층 수에 따른 정확도/인기 편향, 활동 그룹 비교다.
계산값은 CSV/Parquet로 남겨 그림을 다시 만들 수 있도록 한다.

## 5. 구현된 항목과 본실험 전 보완할 항목

| 항목 | 2026-11-05 상태 |
|---|---|
| train/valid 학습·best/last·early stopping·optimizer/RNG 복원 | 구현됨. 세 모델 모두 한 epoch pilot 검증 완료 |
| 입력 hash·실제 설정·seed·GPU/PyTorch·epoch total loss/시간 | 저장됨 |
| valid 전체/그룹별 지표 | best/latest JSON에 저장됨 |
| valid 사용자별 지표 | best 시점 Parquet 저장됨. n_train/n_gt 추가 필요 |
| valid top-100 | ID 저장 가능. score 추가 필요. pilot에서는 저장 생략 |
| 모든 검증 시점의 전체 지표·분해 loss | 기록 확장 필요. 현재 history의 검증 지표는 NDCG 중심 |
| 독립적인 최종 test evaluator와 사용자/후보 집합 대조 | 구현·검증 필요 |
| run manifest의 코드/전체 환경/CU·집계표·seed 통계/paired CI | 기록·집계 구현 필요 |
| item exposure·Tail Recall·재정렬·차종 평가 | 추가 구현 필요. 특징 의존 항목은 테이블 도착 후 진행 |
| 장기 실행 중 checkpoint의 VM 밖 회수 | 본실험 전에 운용 보완 필요. VM 내부 복원만으로 충분하지 않음 |

본실험의 완료 조건은 선택 규칙을 고정한 test 결과를 입력·코드·checkpoint·사용자별 출력으로 추적할 수 있고,
네 모델 비교, seed별 결과, 그룹 비교, 실행 시간/비용, 실제 수행한 ablation이 갖춰지는 것이다.
차종/텍스트가 미제공인 실험은 미실시로 구분하고 기본 모델 결과부터 완성한다.
