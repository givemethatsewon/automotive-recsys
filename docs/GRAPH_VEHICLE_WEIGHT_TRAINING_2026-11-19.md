# LightGCN·XSimGCL 차종 신뢰도 가중 학습

2026-11-19. 사용자가 별도 그래프 실험의 즉시 실행을 요청했다. 완료된 [BPR-MF6runs](VEHICLE_WEIGHT_TRAINING_2026-11-17.md)의 같은 차종 가중치를 LightGCN·XSimGCL로 확장한다. 현재 로컬 검사를 통과한 준비 단계이며, CUDA 모델별 검증·실제 실행 증거는 아래에 추가한다.

## 동결할 비교

- 두 모델 × 비가중/불일치0.5 조건 × seed42/43/44 =12runs. 각 조건을 처음부터 학습한다. 먼저 seed42의 각 모델 대조/가중 쌍을 실행한 뒤43/44로 이어가며, 결과로 계획을 바꾸지 않는다.
- LightGCN: dim64, 3층, batch8192, lr0.003, raw-ID L2=1e-5. XSimGCL: dim64, 3층, batch8192, lr0.001, raw-ID L2=1e-4, CL=.1, eps=.1, temperature=.2. 이전 valid로 선택한 설정을 그대로 고정한다.
- 데이터·날짜 분할·5-core·평가 사용자·후보·seen 마스킹·동점·float32/TF32 off/fused Adam은 기존 계약과 같다. 최대100epoch, valid 첫 epoch/매5epoch, patience8. train+valid 재학습은 없다.
- BPR-MF와 동일한 v2 A1+A2(n_items≥2)·제공 no-year matcher로 train만 판정한다. weights SHA는 `477f562f0c656b35fd207522ada9571e3f5978cb6a1d0733239509b56bc9a833`로 사전 고정한다. 원래 processed 및 차량 입력 hash도 BPR 계보와 대조한다.
- mismatch217,168/4,818,321개(4.507%)에0.5, unknown/match는1. BPR 손실만 전체 train 평균 가중치0.977464349095878로 정규화한다. 그래프·고정 대칭 CSR·모델별 encoder 평균 범위·대조 손실과 L2는 바꾸지 않는다.
- 차종 노드/학습 가능한 edge 가중치를 추가하지 않는다. XSimGCL의 CL·noise RNG와 shuffle/negative RNG를 분리해 두 조건의 입력 순서·음성과 noise 진행을 맞춘다. 서로 다른 두 모델의 차이를 CL만의 효과로 해석하지 않는다.
- 각 모델·seed의 대조/가중을 따로 비교하고 모델별3seed 평균/SD·seed별 사용자 paired CI를 낸다. 두 모델이나 seed×사용자를 한 집단으로 합치지 않는다. 설정/epoch는 valid로만 선택하고 각 고정 조건을 test에서 평가한다.

## 실행·복구 계약

한 L4를 기존 Ubuntu controller로 제어한다. 할당 전 관측 잔액85.50CU·활성0개다. 새 캠페인의 계정 관측 비용 상한40CU, 잔액30CU 하한이며 host와 worker가 각각 확인한다. 계정 차이는 다른 작업 비용을 포함할 수 있고, GPU 가용성·예산 한도로 중단되면 저장본을 보존한다.

각 모델의 본 학습 전에 실제 전체 그래프·동일 파라미터/고정 XSimGCL noise로 weighted loss/gradient를 native CSR autograd와 대조한다. 전체 weighted train1epoch+전체 valid 후 Drive의 optimizer/RNG checkpoint를 별도 출력 폴더로 SHA 복원해 다음 epoch와 next-update replay를 확인한다. 두 모델 모두 통과한 receipt가 있어야12runs에 진입한다. 완료 receipt는 plan hash·가중치 SHA를 대조한다.

기존 JSONL/CSV·progress·불변 checkpoint 두 세대·600초 I/O adapter와 first/best/final 강제 저장을 재사용한다. 모델별 preflight marker를 분리하고, 완료 직후 receipt 게시 전에 끊긴 경우도 완료 checkpoint 기록에서 복구한다. 이전 완료 BPR/32개 캠페인·다른 사용자 세션을 재개/종료하지 않는다. 성공/실패/중단 시 자기 VM을 반납한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_vehicle_training.py \
  --directory results/graph-vehicle-weight-training-20261119 \
  --models lightgcn xsimgcl --cu-budget 40
```

불일치는 추정 정보이며 실제 비선호/정답이 아니다. A2 자기 일관성, 상품 메타데이터 시점·v2 규칙의 기존 test 개발 사용 등 탐색적 한계를 BPR 실험과 함께 유지한다. 그래프 확장 결과로 과제 전체 RQ가 완료됐다고 표시하지 않는다.
