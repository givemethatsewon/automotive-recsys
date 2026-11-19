# LightGCN·XSimGCL 차종 신뢰도 가중 학습

2026-11-19. 사용자가 별도 그래프 실험의 즉시 실행을 요청했다. 완료된 [BPR-MF6runs](VEHICLE_WEIGHT_TRAINING_2026-11-17.md)의 같은 차종 가중치를 LightGCN·XSimGCL로 확장한다. 로컬43개 검사와 두 모델의 실제 CUDA·외부 checkpoint 재개 검증이 통과했고, 본12개 학습이 진행 중이다. 15:09 KST 조회에서1/12개 완료·LightGCN 가중 seed42의10epoch 저장을 확인했다.

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

## 실제 실행과 검증 (11/19 15:09 KST)

- campaign: `vehicle-weight-20261119T035520Z`, plan hash `ceb79ce5643b5216c35e26e45c2ab91241d47cbbfe430ec053391f56f0d8c7f3`.
- [Drive checkpoints·events·epochs·progress](https://drive.google.com/drive/folders/1alraUeKPD0QMoM3777vNX-FJ-S_duEX5).
- 서버: `ubuntu-wan:~/git/automotive-recsys-graph-vehicle-weight-20261119`, user service `automotive-graph-vehicle-weight-1119.service`, config `results/graph-vehicle-weight-training-20261119/config.json`.
- 동결 모델/worker 소스 [696dec1](https://github.com/givemethatsewon/automotive-recsys/commit/696dec147b21f875f97a0b257a91dea578effb3c). bundle의 src/scripts55개 파일을 로컬 SHA와 대조했다. 아래 실행 기록의 후속 커밋과 동결 소스 revision은 구분한다.
- LightGCN native CSR 대조 loss 차이0, gradient 최대 절대 차이1.82e-12. XSimGCL은 loss 차이0, gradient 최대 절대 차이2.36e-11였다. 전체 train 한 epoch와 전체 valid를 거친 optimizer/RNG checkpoint를 Drive에서 별도 폴더로 복원한 뒤 각각2epoch까지 이어 학습했다.
- next-update replay의 파라미터 최대 절대 차이는 LightGCN1.19e-7, XSimGCL5.96e-8로 통과했다. 해당 환경의 허용 오차 검증이며 bitwise 동일성이나 수렴 보장이 아니다.
- LightGCN 비가중 seed42는55epoch에서 완료됐다. best checkpoint217,075,732bytes와 집계 manifest/epochs/valid 파일의 Drive hash를 확인했다. 현재 가중 seed42가10epoch를 마쳤으며 나머지는 같은 고정 계획으로 이어간다. 아직12runs 전체 최종 비교 결과는 없다.
- 초기 프런트엔드 연결이 끊긴 상태에서 원격 worker 미시작을 확인하고, 같은 VM의 새 kernel 연결로 준비를 복구했다. worker 시작 receipt 뒤 제어기가 같은 endpoint를 채택한 것을 확인했다. 복구 과정에서 GPU를 추가 할당하지 않았고 모델·입력 source bundle은 유지했다.

검증 자료: [로컬43개·소스 SHA](results/graph-vehicle-weight-training-2026-11-19/local_verification.json), [train 가중치 감사](results/graph-vehicle-weight-training-2026-11-19/vehicle_weight_audit.json), [LightGCN CUDA](results/graph-vehicle-weight-training-2026-11-19/lightgcn_weighted_objective.json)/[외부 복원](results/graph-vehicle-weight-training-2026-11-19/lightgcn_verified.json), [XSimGCL CUDA](results/graph-vehicle-weight-training-2026-11-19/xsimgcl_weighted_objective.json)/[외부 복원](results/graph-vehicle-weight-training-2026-11-19/xsimgcl_verified.json), [통합 preflight](results/graph-vehicle-weight-training-2026-11-19/preflight_verified.json), [본 학습 진척·checkpoint](results/graph-vehicle-weight-training-2026-11-19/startup_verification.json).

```bash
ssh ubuntu-wan 'systemctl --user status automotive-graph-vehicle-weight-1119.service --no-pager'
```

Drive `progress.txt/json`, `events.jsonl`, `runs/<run>/restore.json`, `runs/<run>/completed.json`에서 확인한다. 완성 보고서는 `report/`로 저장·회수하고 해당 VM을 반납한다. 현재는 한 L4에서 자동 진행 중이다.
