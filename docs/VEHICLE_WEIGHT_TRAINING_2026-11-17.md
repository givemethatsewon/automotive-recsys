# 차종 신뢰도 가중 BPR-MF 학습

2026-11-17. 사용자 요청에 따라 v2 재정렬 완료 다음의 학습 단계 비교를 준비한다. 로컬41개 검사와 실제 CUDA weighted loss/gradient·Drive checkpoint 외부 복원/2epoch 재개 검증이 통과했다. 현재6개 본 학습이 진행 중이다. 기존32개 학습 캠페인·v2 재정렬 결과를 재개하거나 덮어쓰지 않는다.

## 질문과 사전 고정 조건

차종 불일치로 추정된 train 상호작용을 덜 반영하면, 같은 BPR-MF의 정확도·비인기 노출이 달라지는가? 모델 구조를 고정하고 학습 손실만 바꾸는 별도 실험이다. 차종 노드나 추론 시 재정렬은 없다.

- v2 A1+A2(n_items≥2), train-only 프로필, 제공 matcher의 연식 제외 규칙을 사용한다. 세 사용자/상품 파일의 SHA와 모든 last_seen < min(valid.timestamp)를 검사한다. `_test` 프로필은 bundle에 넣지 않는다.
- 일치/unknown의 가중치는1, 판정 가능한 불일치는0.5다. 불일치도 관측 양성 상호작용으로 유지한다. inferred mismatch는 실제 비선호나 실제 차종 정답이 아니다. A2가 같은 train 상품에서 추론됐다는 자기 일관성 한계를 기록한다.
- BPR 손실은 `mean_batch(w * softplus(s_negative - s_positive)) / mean_train(w)`다. 분모는 전체 train에서 한 번 고정한다. raw ID embedding L2는 기존 식 그대로다. 대조군은 기존 비가중 손실을 사용한다.
- 두 조건 모두 처음부터 학습한다. seed42·43·44마다 초기화·shuffle·음성 RNG를 맞추고 train 양성만 음성에서 제외한다. 같은 seed의 배치 순서/음성은 손실 변화와 독립적이다.
- BPR-MF dim64, batch8192, lr0.003, L2=0.00001: 기존 valid 선택 설정을 고정한다. float32, TF32/AMP off, fused Adam. 최대100epoch, valid는 첫 epoch와 매5epoch, patience8회다. 상한은 수렴 보장이 아니다.
- 2조건×3seeds=6runs. mismatch 계수0.5는 탐색하지 않는다. train만 학습하고 valid로 best checkpoint/early stopping만 결정한 뒤, 고정한 두 조건 모두 같은 test158,805명을 평가한다. train+valid 재학습은 하지 않는다.
- 평가: 전체 상품 후보·기존 seen 마스킹·동점 순서. NDCG/Recall/APLT/ARP/Tail Recall, 그룹별 지표·추천 ID/score·item exposure를 남긴다. seed 평균/표준편차와 seed별 사용자 paired CI를 분리한다. 이전 test/규칙 개발 이력이 있는 탐색적 후속 연구다.

## 실행과 검증

1. 로컬 수식/gradient 대조, all-one 가중치와 원래 경로의 loss·gradient 일치, unknown 보존, train 파일 순서/hash·가중치 변조 거부, 6run 보고서의 seed SD/paired 분리까지41개 검사가 통과했다.
2. 전체 train 가중치는 Colab에서만 만들고 제공 pandas matcher의 실제 쌍과 독립 대조한다. known/mismatch/unknown 수와 고정 정규화 분모, source SHA를 Drive audit에 저장한다. mismatch가 없으면 본 학습을 중단한다.
3. 실제 CUDA batch에서 weighted loss/gradient를 독립 수식과 비교하고 가중치 파일의 Drive 업로드·다운로드 SHA를 검증한다.
4. weighted 전체 train1epoch + 전체 valid + optimizer/양쪽 RNG next-update replay를 수행한다. checkpoint를 Drive에서 별도 폴더로 SHA 복원해2epoch까지 이어 학습하고 replay를 다시 확인해야 본6runs에 진입한다.
5. 기존 `CampaignJournal`/`DurableRun`과600초 I/O adapter, 최초/best/마지막 강제 저장, 마지막 두 불변 세대와 hash 복원을 사용한다. events.jsonl·epochs.csv·progress.txt와 per-worker events/cost를 Drive에 올린다.
6. 기존 Ubuntu controller·활성 notebook launcher를 사용한다. 한 L4, 새 campaign ID, owner 확인, 무진척 재할당6회 상한이다. host와 worker가 각각 CU를 확인하며 운영 상한12CU·잔액30CU 하한으로 시작한다. 계정 관측 비용은 다른 작업 비용을 포함할 수 있다.
7. 성공/중단/실패 시 해당 worker는 자기 VM을 반납한다. Mac이 꺼져도 Ubuntu가 제어한다. completed marker와 plan hash가 맞는 run만 건너뛴다.

준비 명령:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_vehicle_training.py --directory results/vehicle-weight-training-20261117
```

credential은 기존 owner-only 로컬 경로에서 별도 전달하고 source/input bundle·Drive·log에 포함하지 않는다. 학습 진행 확인 후 실제 campaign/Drive/서버 경로와 검증 receipt를 이 문서에 추가한다.

## 실제 실행 확인

- campaign: `vehicle-weight-20261117T071534Z`, frozen plan hash `10736612793ffb22940b2cf7f1f95fa30a45388f068a9737d33c73624e311849`.
- [새 Drive: checkpoints·epochs·progress·보고서](https://drive.google.com/drive/folders/1F_2uIHAA4zePcD04PFe_j6haQb4M6-xH).
- 서버: `ubuntu-wan:~/git/automotive-recsys-vehicle-weight-20261117`, user service `automotive-vehicle-weight-1117.service`. host config: `results/vehicle-weight-training-20261117/config.json`.
- 실행 소스 [c0cefef](https://github.com/givemethatsewon/automotive-recsys/commit/c0cefeffcf8398e8abe1b4329be099d8cabd0dcb). bundle의 소스55개 SHA가 로컬과 일치했다. 현재 문서의 후속 갱신과 실제 동결 소스 revision을 구분한다.
- train4,818,321개 중 unknown3,617,538 / mismatch217,168(4.507%) / match983,615. 제공 구현과 실제223쌍을 대조했고, 가중치 파일의 실제 Drive 왕복 SHA를 확인했다. 고정 train 평균 가중치는0.977464349095878이다.
- 실제 CUDA8,192 batch loss 차이0, gradient 최대 절대 차이5.82e-11. 전체 가중 train1epoch + 전체 valid를 마친 checkpoint를 Drive에서 별도 폴더로 복원해2epoch까지 이어 학습했다. optimizer/RNG next-update replay의 관측 파라미터 차이는0이었다. 관측값을 다른 환경의 bitwise 재현 보장으로 일반화하지 않는다.
- 16:36 KST에 본 대조군 seed42가5epoch 학습 구간으로 전진했고, epoch1의217,072,980-byte best checkpoint와 집계 로그/manifest를 Drive에서 hash 확인했다. 이것은6runs 전체 완료나 최종 성능 개선의 증거는 아니다.

검증 근거: [로컬41개·소스 해시](results/vehicle-weight-training-2026-11-17/local_verification.json), [가중치 감사](results/vehicle-weight-training-2026-11-17/vehicle_weight_audit.json), [CUDA 대조](results/vehicle-weight-training-2026-11-17/weighted_objective.json), [외부 복원/재개](results/vehicle-weight-training-2026-11-17/preflight_verified.json), [본 학습 초기 진척·체크포인트](results/vehicle-weight-training-2026-11-17/startup_verification.json).

```bash
ssh ubuntu-wan 'systemctl --user status automotive-vehicle-weight-1117.service --no-pager'
```

Drive의 `progress.txt/json`, `events.jsonl`, `runs/<run>/restore.json`, `runs/<run>/completed.json`, `workers/<worker>/events.jsonl`과 cost observations를 확인한다. 완료 보고서는 `report/`에 seed별 결과·평균/SD·paired CI로 저장되고 Ubuntu controller가 회수한 뒤 자기 GPU를 반납한다. 사용자별 추천·개인 입력·credential은 Git 제외다.
