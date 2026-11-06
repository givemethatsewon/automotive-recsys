# Automotive 추천 실험

Amazon Reviews 2023 Automotive의 전처리된 5-core 데이터를 사용한다.
목표는 **추천 정확도와 인기 편향을 같은 평가 조건에서 비교하고, 이후 차종 정보를 추가했을 때의 효과를 확인하는 것**이다.

프로젝트 위치: `/Users/givemethatsewon/git/automotive-recsys`

## 결과 확인: 2026-11-07 스냅샷

**주 모델 3개 × seed 42·43·44의 학습·test 평가와 Popularity test 평가는 완료됐다.**
전체 캠페인은 04:57 KST 기록 기준 **31/32개 완료**이며, CL=0 보조 실험 seed 44는 진행 중이다.
완료된 주 비교 결과는 [한국어 결과 정리](docs/RESULTS_2026-11-07.md)와 [CSV](docs/results/2026-11-07/main_comparison.csv)에 있다.

| 모델 | test NDCG@10 평균 | seed SD | 비인기 노출 APLT@10 (%) |
|---|---:|---:|---:|
| Popularity | 0.00204150 | 해당 없음(결정적 1회) | 0 |
| BPR-MF | 0.00117841 | 0.00004268 | 0.15524280 |
| LightGCN | 0.00369642 | 0.00004705 | 0.73011975 |
| XSimGCL | 0.00441927 | 0.00011744 | 0.00004198 |

XSimGCL의 정확도가 가장 높지만 비인기 노출은 거의 0이다. 이번 조건에서 대조학습이 비인기 노출도 개선했다는 결론은 지지되지 않는다.
사용자 158,805명, 전체 학습 상품 후보 248,186개의 동일 warm-start test 평가이며 valid 점수나 pilot 점수를 섞지 않았다.
seed SD와 사용자별 CI는 다르다. CL=0 3-seed 집계·paired CI·최종 자동 보고서는 마지막 보조 실험 완료 후 생성된다.

실제 GPU 학습은 Colab L4, 자동 제어기는 기존 Ubuntu 서버에서 운영한다. Mac이 꺼져도 Drive 저장·복구를 수행한다.
재현용 입력 hash, frozen valid 선택과 완료된 test/그룹 집계를 대조했으며 모델 수식과 선택 plan은 변경하지 않았다.
운용 절차는 [Drive 본실험 안내](docs/DURABLE_EXPERIMENTS.md), 연결 장애·재개 검증은 [제어기 기록](docs/CONTROLLER_RELIABILITY.md)에 있다.

환경·입력 계약(39개 검사), 전체 train 한 epoch GPU pilot와 optimizer/RNG 재현 검증도 보관했다.
CUDA 프로파일링 기반 최적화 근거는 [GPU 점검 결과](docs/GPU_REVIEW.md)를 참고한다.

## 실행

```bash
cd ~/git/automotive-recsys
uv sync --locked
uv run automotive-recsys audit
uv run automotive-recsys popularity --split valid
uv run --locked --extra training --group colab python -m unittest discover -s tests -v
```

`data`는 `/Users/givemethatsewon/Downloads/processed`를 가리키는 로컬 심볼릭 링크다.
다른 컴퓨터에서는 전처리 파일들을 원하는 폴더에 두고 `--data-dir`로 지정한다.

```bash
uv run automotive-recsys --data-dir /path/to/processed audit
uv run automotive-recsys --data-dir /path/to/processed popularity --split valid
```

입력 데이터, 가상환경, 추천 목록, 체크포인트는 `.gitignore`로 제외한다.
공유할 코드는 이 폴더 안에서 관리하며 SecondBrain에는 쓰지 않는다.

## 모델 구성

| 구분 | 모델 | 비교하려는 차이 |
|---|---|---|
| 비개인화 기준선 | Popularity | 많이 리뷰된 상품만 추천하면 어느 정도인가? |
| 학습 기준선 | BPR-MF | 사용자와 상품의 잠재 취향만 학습하면 개선되는가? |
| 그래프 추천 | LightGCN | 사용자-상품 그래프 전파가 정확도와 인기 편향에 어떤 영향을 주는가? |
| 그래프 대조학습 | XSimGCL | 그래프에 대조학습을 더하면 정확도와 비인기 상품 노출이 달라지는가? |

**학습 모델 3개 + 인기순 기준선 1개**로 진행한다.
모델 선택 근거, 강의와의 연결, 실험 순서는 [실험 계획](docs/EXPERIMENT_PLAN.md)에 기록했다.
로컬 속도 실측과 Colab 전환 기준은 [학습 환경 점검](docs/HARDWARE.md)에 기록했다.
현재 기본 실행 환경은 [Colab GPU](docs/COLAB.md)다.
이후 작업 규칙과 최적화 적용 조건은 [AGENTS.md](AGENTS.md), 본실험의 수집 항목과 미구현 항목은 [결과 수집 계약](docs/EXPERIMENT_OUTPUTS.md)을 따른다.

## 파일

- `src/automotive_recsys/audit.py`: 결측·중복·ID·시간 분할·평가 제외 목록 점검, 입력 SHA-256 기록.
- `src/automotive_recsys/evaluation.py`: 공통 사용자별 NDCG·Recall·Hit Rate·APLT·ARP 및 집계.
- `src/automotive_recsys/popularity.py`: 전체 상품 인기도 순위에서 이미 본 상품을 제외하고 추천.
- `src/automotive_recsys/gpu_kernels.py`: CSR·대칭 그래프 역전파·BPR/InfoNCE·GPU 음성 추출·전체 상품 top-k.
- `src/automotive_recsys/train_gpu.py`: CUDA 전용 train/valid 학습기, best/last checkpoint, early stopping, 재시작.
- `results/colab/train_pilot/summary.json`: 실제 모델별 한 epoch와 전체 검증·재시작 검사 결과.
- `results/audit.json`: 원본 크기·날짜·검사 결과·입력 해시.
- `results/popularity_valid/metrics.json`: 검증 지표와 정확한 평가 조건.
- `results/popularity_valid/per_user.parquet`: 사용자별 지표. 이후 동일 사용자의 모델 간 차이 비교에 사용.
- `results/popularity_valid/recommendations.npz`: `uid` 배열과 상위 100개 `iid` 배열.

## 해석할 때의 범위

이 실험에서 리뷰는 상호작용의 대용 지표이며 실제 구매 전체를 뜻하지 않는다.
학습에 없는 사용자·상품이 제거된 warm-start 평가다. 전달받은 통계 기준으로
원래 테스트 상호작용의 약 55.2%가 남아 있으므로 모든 사용자에 대한 성능으로 일반화하지 않는다.
차종·상품 설명 파일은 아직 전달받은 폴더에 없으며, Mismatch@10과 텍스트/차종 재정렬은 아직 계산하지 않는다.

인기도만으로 추천한 결과 APLT가 0인 것은 기준선의 예상 가능한 특성이다.
NDCG가 작다는 이유만으로 전처리 오류나 학습 실패라고 판단하지 않는다.
