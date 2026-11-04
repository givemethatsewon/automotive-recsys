# GPU 코드 점검과 시간 예산

2026-11-05. NVIDIA Skills의 live catalog를 확인했고, 일반 PyTorch 협업 필터링에 직접 맞는
전용 skill은 찾지 못했다. NVIDIA CUDA Best Practices의 실제 workload 프로파일링,
전송 최소화, CUDA event 타이밍 원칙에 따라 Colab에서 직접 점검했다.

## 현재 상태

초기 코드는 Colab 연결 시험이었다. 이번 작업에서 세 모델의 공통 CUDA 학습 커널,
전체 상품 평가 경로, CUDA 전용 학습 루프를 구현하고 **세 모델 모두 전체 train 한 epoch를 실제 실행했다.**
본격적인 설정 탐색·50 epoch 학습·seed 반복·최종 test 평가는 아직 수행하지 않았다.

코드: `src/automotive_recsys/gpu_kernels.py`, `src/automotive_recsys/train_gpu.py`.
실측: `results/colab/gpu_performance/summary.json`.
CUDA trace: 같은 디렉터리의 `cuda-trace.json`.

## 측정 조건

- NVIDIA L4, PyTorch 2.11.0+cu130, 전체 train 4,818,321개.
- 사용자+상품 노드 847,928개, 임베딩 64차원, batch size 8,192.
- 그래프 모델은 3층. FP32를 유지하며 TF32/AMP는 사용하지 않았다.
- 같은 목표함수에서 COO+기존 Adam, COO+fused Adam, CSR+fused Adam,
  CSR+fused Adam+대칭 그래프 역전파 재사용을 비교했다.
- warm-up 2회 후 업데이트 7회. CUDA event와 동기화한 wall time을 함께 기록했다.
- 학습 이력에 없는 상품을 GPU에서 음성 추출하는 과정도 측정에 포함했다.

## 실측 결과

| 모델 | 기본 업데이트 | 선택한 CUDA 경로 | 속도 개선 | epoch 학습 연산 환산 |
|---|---:|---:|---:|---:|
| BPR-MF | 37.7ms | 27.4ms | 1.38배 | 약 16초 |
| LightGCN 3층 | 321.1ms | 92.3ms | 3.48배 | 약 54초 |
| XSimGCL 3층 | 421.8ms | 193.0ms | 2.19배 | 약 114초 |

선택 경로의 peak allocated는 BPR-MF 약 1.54GiB, LightGCN 약 2.14GiB,
XSimGCL 약 2.95GiB였다. PyTorch에서 집계한 값이며 전체 GPU driver/runtime 점유량과 다르다.
비교를 위해 두 그래프 형식을 함께 올려둔 측정이므로 실제 학습은 선택 형식 하나만 사용한다.

전체 valid 사용자 220,516명 × 상품 248,186개의 점수 계산·기존 상품 제외·상위 100개 추출·CPU 결과 회수는
사용자 batch 1,024에서 **5.91초**였다. 이 값은 random embedding에서의 평가 경로 측정으로,
최종 학습 임베딩 생성·지표 집계·파일 저장 시간은 포함하지 않는다.

## 수정한 부분

1. 그래프·학습 ID·음성 추출용 train key를 GPU에 유지한다.
2. 희소 행렬을 CSR로 바꾸고 CUDA fused Adam을 사용한다.
3. 레이어 결과를 거대한 stack으로 추가 복제하지 않고 누적 합산한다.
4. 고정된 대칭 그래프의 역전파에서 같은 행렬을 재사용한다.
   `dL/dE = Aᵀ dL/d(AE) = A dL/d(AE)`이므로 CSR 전치·정렬을 반복하지 않아도 된다.
   이 경로는 `normalized_graph`가 만든 고정 대칭 그래프에만 적용한다.
5. 학습에서 본 상품은 음성 추출에서 제외한다. valid/test 정답은 추출에 사용하지 않는다.
6. 평가 점수는 GPU 사용자 배치별로 계산하고 상위 100개만 CPU로 가져온다.
7. 경계의 동점까지 score 내림차순·iid 오름차순으로 처리한다.
   XSimGCL에서 이웃이 동일한 상품은 표현이 같을 수 있으므로 이 처리가 필요하다.
8. 학습 루프의 data RNG를 대조학습 noise RNG와 분리했다.
   같은 seed의 모델들은 같은 train 순서와 음성 샘플을 사용한다.

## 정확성 확인

- 작은 수식·평가·마스킹·동점·음성 추출 테스트 11개 통과.
- CPU reference와 CUDA의 세 모델 손실·기울기 대조 통과.
- fused Adam과 기존 Adam의 업데이트 대조 통과.
- 실제 전체 train 그래프에서 기본 CSR backward와 대칭 재사용 backward의 손실 차이는 0.
  기울기 최대 절대 차이는 LightGCN `4.55e-13`, XSimGCL `3.87e-12`였다.
- 전체 상품 후보에 대한 GPU 상위 10개가 CPU 계산과 일치하는지 8명의 사용자로 대조했다.
- 별도로 GPU 동점 입력을 만들어 경계 동점 처리를 확인했다.
- PyTorch CUDA profiler에서 실제 GPU 이벤트 637개를 기록했다.
  이전 프로파일의 큰 CSR 정렬 비용은 변경 후 상위 비용 연산에서 사라졌다.

전체 그래프 기울기 대조는 고정한 한 minibatch의 검사다.
모든 학습 상태에서의 비트 단위 일치나 최종 모델의 정확도 동등성을 주장하지 않는다.

## 전체 데이터 한 epoch 실행 결과

전체 train을 GPU에서 shuffle하고, 각 상호작용을 한 번씩 학습한 실제 pilot 결과다.

| 모델 | 실제 학습 | 전체 valid 지표 집계 | 검증·저장 포함 epoch wall time |
|---|---:|---:|---:|
| BPR-MF | 16.52초 | 6.93초 | 25.55초 |
| LightGCN 3층 | 54.47초 | 6.45초 | 63.04초 |
| XSimGCL 3층 | 114.09초 | 6.56초 | 126.51초 |

원격에서 세 모델의 checkpoint를 저장·복원한 뒤 동일한 다음 업데이트를 실행했다.
optimizer·CPU/CUDA/data RNG 복원 후 파라미터 차이는 BPR-MF 0,
LightGCN `5.96e-8`, XSimGCL `2.98e-8`로 검사 허용오차 안이었다.
pilot은 valid만 평가했고 test는 평가하지 않았다.
모든 요약·사용자별 지표·실행 로그를 다운로드한 뒤 VM을 종료했다.
이 실행들은 수렴한 모델의 성능 비교가 아니다.

## 시간·CU 계획

짧은 업데이트 측정치를 전체 상호작용 수로 환산하면, 각 모델 50 epoch 한 번씩은
학습 연산만 약 2.6시간이다. 실제 실행·검증·저장·결과 회수를 포함한 첫 비교는 **3–4시간**으로 계획한다.

세 모델을 seed 42·43·44로 50 epoch씩 반복하는 9개 본 실행은 약 7.8시간,
짧은 설정 탐색까지 더한 단순 환산은 약 10시간이다.
공통 설정 탐색은 모델별 4개 조합으로 시작하고 LightGCN의 층 수 1·2에 대한 추가 실험을 둔다.
저장·초기화·지표 집계·가용성·재시작 여유까지 포함해 **12–16시간**을 잡는다.
이는 현재 50 epoch 예산의 추정이며 수렴까지 걸리는 시간의 보장은 아니다.
100 epoch가 필요하면 학습 연산 부분은 대략 두 배로 늘어난다.

당시 L4 소모율 1.94 CU/시간으로 12–16시간은 약 23–31 CU다.
운영 여유를 포함해 **25–35 CU**를 초기 예산으로 잡을 수 있다.
단가와 GPU 가용성은 실행할 때 다시 확인한다. A100은 할당 지연으로 취소했고 비용/속도 비교는 하지 못했다.

## 학습기 운용 범위

학습기는 CUDA가 없으면 종료한다. 로컬 CPU/MPS로 자동 전환하지 않는다.
train만 학습하고 valid만으로 best checkpoint/early stopping을 선택한다.
`last.pt`에는 파라미터·optimizer·CPU/CUDA/data RNG·입력 해시·epoch를 저장한다.
재시작하면 설정과 입력 해시가 일치하는지 확인한다.

Colab에서 모듈을 설치하거나 `PYTHONPATH=/content/automotive-recsys/src`를 설정한 뒤:

```bash
python -m automotive_recsys.train_gpu \
  --data-dir /content/automotive-recsys/data \
  --model lightgcn --epochs 50 --seed 42 \
  --output /content/automotive-recsys/results/lightgcn_seed42
```

재시작은 동일한 설정에 `--resume /content/.../last.pt`를 추가한다.
VM이 종료되면 `/content`는 사라지므로, 본 실행은 주기적으로 체크포인트를 회수해야 한다.
한 epoch pilot 결과는 `results/colab/train_pilot`에 보관하며, 수렴한 성능표로 사용하지 않는다.
pilot의 대형 가중치는 원격 저장·복원 시험에만 쓰고 최종 제출용 모델로 보관하지 않는다.

## 방법과 출처

XSimGCL의 cross-layer noise와 InfoNCE 구조를 저자 코드에 대조했다.
모델 간 통제 비교를 위해 raw embedding squared L2를 공통 적용하므로
저자 저장소의 정규화 방식과 다른 recipe variant임을 기록한다.
이 구현으로 논문 성능을 그대로 재현했다고 표현하지 않는다.

- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch sparse.mm](https://docs.pytorch.org/docs/2.11/generated/torch.sparse.mm.html)
- [PyTorch Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html)
- [XSimGCL 저자 구현](https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/XSimGCL.py)
