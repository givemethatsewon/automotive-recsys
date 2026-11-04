# Automotive 추천 실험

Amazon Reviews 2023 Automotive의 전처리된 5-core 데이터를 사용한다.
목표는 **추천 정확도와 인기 편향을 같은 평가 조건에서 비교하고, 이후 차종 정보를 추가했을 때의 효과를 확인하는 것**이다.

프로젝트 위치: `/Users/givemethatsewon/git/automotive-recsys`

## 현재 완료된 것

- Python 3.12 전용 `.venv`와 `uv.lock`.
- 입력 데이터 계약 점검 39개 통과. 원본 파일은 수정하지 않았다.
- 인기순 추천의 검증 집합 전체 평가: 사용자 220,516명, 상품 후보 248,186개.
- NDCG@10 **0.00197863**, Recall@10 **0.00349243**, APLT@10 **0.0**.
- 사용자별 지표, 그룹별 집계, 사용자별 상위 100개 추천 저장.
- 지표·GPU 수식·이미 본 상품 제외·경계 동점·음성 추출 테스트 11개 통과.
- M3 Max 로컬 GPU 연산 점검 기록을 보관했다. 사용자의 선택에 따라 실제 학습은 Colab GPU에서 진행한다.
- Colab CLI 0.7.5 설치 및 전용 브라우저 인증 완료. [Colab 실행 방법](docs/COLAB.md).
- 세 모델 모두 전체 train 1 epoch·전체 valid 평가·optimizer/RNG 체크포인트 재시작 검사 통과.
- CUDA 프로파일링으로 LightGCN 약 3.48배, XSimGCL 약 2.19배 속도 개선. [GPU 점검 결과](docs/GPU_REVIEW.md).

현재 점수는 **검증(valid)** 결과다. 테스트(test) 최종 평가는 아직 실행하지 않았다.
BPR-MF·LightGCN·XSimGCL의 CUDA 학습기와 한 epoch pilot은 구현·검증했다.
설정 탐색과 50 epoch/seed 반복은 아직 수행하지 않았으며 pilot 점수를 최종 모델 비교에 사용하지 않는다.

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
