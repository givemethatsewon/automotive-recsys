# 과제 설명·세원 담당 범위·이어서 작업할 때

갱신: **2026-11-10**. 다른 채팅에서는 이 문서와 루트 [AGENTS.md](AGENTS.md)를 먼저 읽는다.

**추천 모델 본실험은 32/32개 완료됐다. 그러나 제안서의 Spark 원본 분석·인기 감점/텍스트/차종 재정렬을 포함한 팀 과제 전체가 끝난 것은 아니다.** 다음 작업은 완료된 결과를 재학습하는 것이 아니라, 아직 전달되지 않은 자료와 후속 실험 범위를 확인하는 것이다.

## 1. 무슨 과제인가

- 수업: 중앙대학교 2026년 2학기 빅 데이터 01분반의 팀 프로젝트.
- 제안서 제목: **리뷰 속 차종 정보를 활용한 비인기 자동차 용품 추천**.
- 팀: 민세원·박신빈. 역할은 [Notion 개요 사본](docs/sources/project-overview.notion.md)에 명시돼 있다.
- 데이터: Amazon Reviews 2023의 Automotive. 원본 전체는 인기 분포 분석·차종 추출에, 5-core는 추천 모델 실험에 사용하는 계획이다.
- 목적: 인기 상품에 대한 추천 쏠림을 측정하고, 그래프 추천·인기 감점·텍스트·차종 정보가 정확도와 비인기 상품 노출·적합성에 미치는 영향을 비교한다.
- 예측 목표: 사용자–상품의 암묵적 상호작용 추천. 리뷰를 구매의 대용 지표로 삼되 실제 구매 전체를 관측한 데이터라고 설명하지 않는다. 평점 예측 과제가 아니다.
- 교수님 공식 상세 과제 공지·루브릭·제출 양식·확정 마감일은 아직 확보한 자료에 없다. 제안서에는 실험 코드와 결과표 제출 계획이 있다. 회의의 ‘수요일/금요일/26일’ 언급으로 공식 마감을 확정하지 않는다.

전체 계획의 원문은 [제안서 사본](docs/sources/proposal.notion.md), 결정 과정은 [회의 정리](docs/MEETING_NOTES.md), 자료의 출처·스냅샷 한계는 [sources 안내](docs/sources/README.md)에 있다. 원문 Notion을 열지 않아도 이 폴더에서 읽을 수 있다.

## 2. 연구 질문과 여기서 확인할 수 있는 진행 범위

| 질문 | 원래 계획 | 이 폴더의 확인 상태 |
|---|---|---|
| RQ1 인기 편향 규모 | raw/0-core와 5-core의 인기 분포 비교, 모델별 편향 측정 | 모델의 APLT/ARP/coverage·노출 결과는 완료. 원본-vs-5-core Spark 분석의 재현 코드/설정별 처리 시간은 전처리 담당 자료 확인이 필요. 제안서의 사전 집계값을 이번 모델 실험에서 새로 측정한 값으로 쓰지 않는다. |
| RQ2 그래프 모델 득실 | BPR-MF 대비 LightGCN의 정확도·편향, LightGCN 1/2/3층 | 동일 warm-start test와 3 seeds 비교 완료. 추가로 XSimGCL과 CL=0 보조 실험도 완료. |
| RQ3 편향 완화 | 인기 감점 vs 텍스트 유사도 재정렬, 이력 1–5/6+ 그룹 비교 | 기본 모델의 활동 그룹 비교 완료. 인기 감점/텍스트 재정렬 비교는 미실시. **인기 감점만의 실험은 텍스트 특징 없이도 후속 수행 가능**하나 별도 실험 설정이 필요. |
| RQ4 차종 정보 효과 | 다양성만 높인 경우와 차종 일치 가점을 추가한 경우 비교, Mismatch/NDCG | 사용자 차종·상품 fitment·텍스트 특징이 전달된 processed 폴더에 없어 미실시. ‘차종 추출 완료’ 여부와 ‘모델 담당에게 파일 전달 완료’는 구분해 확인해야 한다. |

추가 ALS 비교는 제안서의 여유 시간 옵션이다. 현재 필수로 실행한 모델이나 완료된 결과로 표기하지 않는다.

## 3. 내 담당 범위: 민세원

사용자와 Notion 프로젝트 개요가 명시한 역할:

> 모델 학습 및 평가 담당. baseline 및 특성이 다른 모델 약 3개를 고르고 GNN을 포함한다. 기본 데이터와 데이터 전처리·차종 추출 결과가 포함된 조건을 학습·평가하며, 5-core를 기본 입력으로 사용한다.

| 담당 | 확인된 범위 |
|---|---|
| 민세원 | 모델 선정, 학습·평가 구현/실행, 지표·seed·보조 실험 비교, 결과표·그림·설명. 특징을 전달받은 뒤 차종/텍스트를 반영한 조건의 효과 평가. |
| 박신빈 | Notion 개요에 명시된 **Spark 데이터 전처리 담당**. 회의에서는 전처리·추출 결과를 모델 담당에게 순차 전달하기로 논의했다. 필요한 파일/집계의 실제 전달 여부는 확인 필요. |

세원은 전처리 원본을 임의로 다시 만들거나 Spark 담당의 완료 여부를 추정해서는 안 된다. 먼저 전달된 학습용 데이터로 기본 비교를 진행하고, 차종·텍스트 파일이 오면 동일 분할/평가 조건에서 후속 비교를 연결한다.

‘원본 데이터로 학습’의 **원본은 이 모델 작업에서 기본 입력으로 합의한 전처리 5-core**다. RQ1에서 필터링 전 전체(raw/0-core)를 분석하는 일과 다르다. 최종 수치에서는 5-core 전체 규모와 날짜 분할·warm-start 필터 후 train/test 규모도 구분한다.

## 4. 제안서 이후 확정한 모델·운영 방식

원래 제안서는 Popularity + BPR-MF + LightGCN을 제시했다. 사용자가 모델 선정을 위임하고 Popularity 별도 기준선 및 강의의 대조학습 주제를 요청해 다음과 같이 확장했다.

| 구분 | 모델 | 비교 축 |
|---|---|---|
| 결정적 기준선 | Popularity | 사용자 과거 상품 제외 후 학습 인기도순 추천. 가짜 seed 반복을 만들지 않음. |
| 학습 기준선 | BPR-MF | 사용자·상품 임베딩 내적, 그래프 전파 없음. |
| GNN | LightGCN | 사용자–상품 그래프 전파, 1/2/3층 비교. |
| 그래프 대조학습 | XSimGCL | 노이즈·층 간 대조학습. CL=0 보조 실험으로 대조 손실의 기여를 별도 비교. |

즉 **학습 모델 3개 + 비학습 기준선 1개**다. XSimGCL이 원래 제안서에 있었다고 고쳐 쓰지 않는다. LightGCN과 XSimGCL은 encoder 평균 범위 등도 달라 둘의 차이를 CL 하나의 인과 효과로 단정하지 않는다. 강의계획서에는 그래프 대조학습(10주차)·GNN 추천(11주차)이 있으나 XSimGCL 자체를 수업에서 가르쳤다고 확인한 것은 아니다. [강의계획서 사본](docs/sources/course-syllabus.pdf).

운영 요구: 프로젝트·가상환경은 `~/git/automotive-recsys`, 무거운 학습/전체 상품 평가는 Colab CUDA. Mac 부하는 피하고 기존 크레딧을 사용했다. Drive 자동 저장·JSONL/CSV 진행 로그·checkpoint 복구·서버 자동 제어기를 구현했다. [사용자 후속 요구](docs/sources/conversation-decisions.md), [GPU 검증](docs/GPU_REVIEW.md), [자동 복구](docs/CONTROLLER_RELIABILITY.md).

## 5. 바꾸면 안 되는 평가 계약

1. 제공된 날짜 분할 그대로 **train만 학습 → valid로 설정·best epoch 선택 → frozen 설정으로 test 평가**. train+valid 재학습은 하지 않는다. 회의에서 제안됐다가 채택하지 않은 방식이다([18:58–19:09 정리](docs/MEETING_NOTES.md)).
2. train에 5-core를 다시 적용하지 않고, 입력 파일을 수정하지 않는다. 그래프·인기도·그룹·음성 제외 목록은 train만 사용한다.
3. 전체 학습 상품 후보에서 valid는 seen_for_valid, test는 seen_for_test를 제외한다. test 마스크에 valid 이력이 들어가는 것은 valid로 모델을 학습했다는 뜻이 아니다.
4. 리뷰 존재를 binary interaction으로 사용한다. 미관측 상품을 ‘싫어한 상품’으로 단정하지 않는다.
5. 같은 사용자·후보·정답으로 비교한다. test는 warm-start 158,805명/333,797개 정답이고 후보 상품은 248,186개다. 원래 test 상호작용의 약 55.2%가 남았으며 사용자 유지율이나 모든 cold-start 사용자 성능이 아니다.
6. 비인기는 제공 item_map.popular=False. 학습 리뷰 11개 이상을 인기군으로 묶으며 동점 포함 인기군 리뷰 비중은 약 81.48%다. test로 경계를 다시 고르지 않는다.
7. NDCG@10·Recall@10·APLT@10·ARP@10은 사용자 macro 평균. seed 42/43/44의 sample SD와 seed별 사용자 paired bootstrap CI는 별개다.
8. 특징의 출처/시점·unknown/coverage를 기록한다. 사용자 차종 추출에 valid/test 리뷰를 읽어 넣지 않는다. unknown 차종을 일치/불일치로 간주하지 않고 Mismatch 미실시를 0으로 쓰지 않는다.
9. 새 재정렬 λ/γ/β는 valid로만 선택한다. 제안서의 ‘정확도 약 95% 유지’는 재정렬 조건 선택 기준이지 모든 모델의 통과 점수가 아니다.

정확한 계약은 [EXPERIMENT_PLAN](docs/EXPERIMENT_PLAN.md), [EXPERIMENT_OUTPUTS](docs/EXPERIMENT_OUTPUTS.md), 실행 시 보관한 frozen plan/selection을 참고한다.

## 6. 현재 완료 상태와 결과

- 캠페인: `automotive-20261104T204146Z`.
- **32/32개 완료**: coarse 탐색 12 + 층 선택 2 + 주 모델 3×3 seeds 9 + 층별 보조 6 + CL=0 보조 3. Popularity valid/test와 복구 pilot은 이 32개 학습 작업 밖이다.
- 결과 저장 완료: **2026-11-07 06:58 KST**. 서버 제어기의 보고서 다운로드·정상 종료 확인은 약 07:00 KST다.
- 2026-11-10 재확인: Ubuntu 제어기 inactive/정상 완료, 활성 Colab 할당 0개, CU 소모율 0.0. 새 GPU를 할당하거나 완료 캠페인을 resume할 필요가 없다.
- 계정 관측 기준 소모 74.79 CU, 잔액 약 100.14 CU. 정확한 모델별 비용으로 배분하지 않는다.
- [최종 결과 해설](docs/RESULTS_FINAL.md), [최종 CSV/그림/HTML](docs/results/final/), [완료 marker](docs/results/final/completion.json), [아티팩트 해시·종료 검증](docs/results/final/verification.json).
- `docs/RESULTS_2026-11-07.md`와 `docs/results/2026-11-07/`은 31/32개 시점의 **과거 스냅샷**이다. 최신 상태나 CL=0 3-seed 결과로 혼동하지 않는다.

| 모델 | test NDCG@10 평균 | APLT@10 (%) |
|---|---:|---:|
| Popularity | 0.00204150 | 0 |
| BPR-MF | 0.00117841 | 0.15524280 |
| LightGCN | 0.00369642 | 0.73011975 |
| XSimGCL | 0.00441927 | 0.00004198 |

XSimGCL은 정확도가 가장 높지만 비인기 노출은 거의 0이다. 3-seed CL=0 비교에서도 CL 사용 시 NDCG는 약 6.70% 높고 APLT는 낮다. 가설과 다른 이 결과를 보존한다. ARP 감소와 tail 노출 증가를 같은 의미로 설명하지 않는다. CL=0 세 실행은 모두 100 epoch 상한에 도달했으므로 충분한 수렴을 보장하지 않는다.

## 7. 다른 채팅에서 바로 이어갈 작업

1. AGENTS.md → 이 문서 → [sources 안내](docs/sources/README.md)·[회의 정리](docs/MEETING_NOTES.md) → [최종 결과](docs/RESULTS_FINAL.md) 순으로 읽는다. 교수님 과제 공지와 원래 제안서/회의록, 이후 사용자 요구를 구분한다.
2. 보고서·발표를 만드는 요청이면 이미 완료된 CSV와 그림을 사용한다. 모델을 다시 학습하지 않는다. 통계 설명에는 최종 paired_differences.csv와 seed SD를 구분한다.
3. RQ1을 완성하려면 전처리 담당에게 원본/5-core 분포 집계, Spark 처리 코드·버전/설정·처리 시간 자료를 받아 연결한다. 제안서의 수치를 이번 실행에서 재현했다고 꾸미지 않는다.
4. 재정렬을 이어갈 요청이면 먼저 범위를 확인한다. 인기 감점만은 기존 item_map와 저장된 후보/score로 시작 가능하다. 텍스트·차종 조건은 아래 파일을 받아야 한다. 완료한 baseline과 다른 후속 실험으로 설정·선택·지표를 별도 기록한다.
5. 팀 전체 제출물에는 아직 미실시인 RQ3/RQ4와 자료 부재를 밝힌다. 공식 마감·평가표·제출 양식은 별도 공지로 확인해야 한다. 메시지 전송·과제 제출은 이 인수인계 요청에 포함되지 않았다.

필요한 추가 입력([세부 계약](docs/EXPERIMENT_PLAN.md#6-추가-데이터가-필요할-때의-계약)):

- 사용자 차종: 원래 user_id, 제조사/모델/연식, 추출 근거·시점, unknown·복수 차량 처리.
- 상품 fitment/텍스트: parent_asin, 허용 차종/연식·범용/unknown, 상품명/설명, 추출·원자료 출처.
- 추출 품질: 사람이 확인한 표본 수, 정확도·coverage, 오류 사례/규칙 버전.
- Spark 분석: raw/5-core의 집계 기준과 코드·입력 버전·설정별 실행 시간. 모델 평가용 train 규모와 원본 전체 규모를 혼동하지 않는다.

## 8. 파일·경로·실행 의존성

| 위치 | 내용 |
|---|---|
| `docs/sources/` | 회의 전사 원본, Notion 제안서/팀 개요 사본·metadata, 강의계획서 PDF, 사용자 후속 요구. 원본 밖에 해설을 쓴다. |
| `docs/MEETING_NOTES.md` | 녹음 시각별 결정·토론 중 제안·후속 적용의 대응. |
| `docs/RESULTS_FINAL.md`, `docs/results/final/` | 32/32 완료 결과와 3-seed/paired CI/그림/HTML. Git으로 함께 이동 가능. |
| `docs/explainers/experiment-results-eli5.html` | 사용자가 실험을 이해하고 설명하기 위한 ELI5 시각 설명서. 모델·지표·결과·불확실성 조작과 발표 원고를 포함한 단일 오프라인 HTML. |
| `src/automotive_recsys/` | 입력 검사·GPU 수식·학습·평가·복구·집계 구현. |
| `data` | `/Users/givemethatsewon/Downloads/processed`로 가는 읽기용 심볼릭 링크. 원본 데이터를 복제해 Git에 넣지 않는다. |
| `results/campaigns/automotive-20261104T204146Z/` | 로컬 실행/수집 기록. Git 제외. |
| `ubuntu-wan:~/git/automotive-recsys` | 기존 Ubuntu 제어기·수집 결과. GPU 학습 호스트가 아니다. |
| `~/.config/automotive-recsys/` | task 전용 Colab/Drive ADC·세션 상태. Git·문서·출력에 내용이나 token을 넣지 않는다. |

이 폴더만으로 **과제 맥락·원문·확정 결과를 읽고 인수인계**할 수 있다. 다른 컴퓨터에서 실행하려면 외부 데이터와 개인 인증을 따로 준비해야 한다. 대형 추천 배열·사용자별 Parquet·best checkpoint는 [캠페인 Drive](https://drive.google.com/drive/folders/1IBQzFgM2ZDR-zz4mgwk6BjIf6c9pRwmx)에 있다. [팀 원자료 Drive](https://drive.google.com/drive/folders/1WaZBTxYqEdEozh2FIZQQIGz8O1IVCkKt)는 Notion 개요에 있는 별도 폴더이며, 파일 전달 현황까지 여기서 확인한 것은 아니다.

읽기 전용 상태 확인:

```bash
cd ~/git/automotive-recsys
ssh ubuntu-wan 'cd ~/git/automotive-recsys && .venv/bin/python scripts/experiment.py status'
```

원문과 결과만 읽는 작업에는 uv 환경 재설치·GPU 할당·제어기 재시작이 필요 없다. 새 채팅에서는 아래처럼 요청하면 된다.

> 이 프로젝트의 AGENTS.md와 PROJECT_BRIEF.md, docs/sources/README.md, docs/RESULTS_FINAL.md부터 읽어줘. 추천 벤치마크 32개는 완료됐고 팀 과제 전체에는 아직 후속 항목이 있어. 원문·확정 결정·실행 결과·미확인 사항을 구분해서 내가 요청하는 다음 작업을 이어가줘.
