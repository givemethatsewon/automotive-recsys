# 차종 v2 비교 실행 기록

이 문서는 첫 실행의 실패 이력이다. 이후 저장·복구 검증을 통과한 재시작 상태는 [새 실행 기록](VEHICLES_V2_DURABLE_2026-11-17.md)을 따른다.

**최신 확인: 실패 종료.** 2026-11-17 12:02 KST에 서버를 조회했다. 세션이 사라진 뒤 제어기가 04:10 KST에 시간 초과로 실패했고 최종 결과 archive는 없다. 초기 진행만 보존됐으므로 v2 성능 개선 여부를 보고할 수 없다. 현재 잔액96.33 CU·소모율0·활성 할당0개를 확인했다. [실패 검증 JSON](results/vehicle-v2-2026-11-17/failure_verification.json). 아래 시작·초기 검증 기록은 당시 관측으로 보존한다.

2026-11-17 KST. 사용자 요청에 따라 [v2 계획](VEHICLES_V2_PLAN.md)의 첫 단계인 **기존 checkpoint에서 v1↔v2 재정렬 비교**를 시작했다. 완료를 기다리지 않고 초기 실행을 검증한 뒤 백그라운드 제어기에 맡기는 요청이다. 학습 단계 실험은 이 실행에 포함하지 않는다.

## 실행 범위

- 모델: 기존 BPR-MF·LightGCN·XSimGCL seed42 best checkpoint. 재학습 없음.
- 같은 사용자·정답·checkpoint·전체 상품 후보에서 top100/500/1000을 재계산하고 비교한다. 가점 단위는 사용자별 top100 점수 범위로 고정한다.
- 조건: v1+기존 exact matcher / v2+같은 matcher / v2+제공 matcher(연식 제외) / v2+제공 matcher(연식 포함).
- A1+A2(n_items≥2), valid/test 모두 train-only 프로필. β는 valid 전체 NDCG로만 선택하고 test 전 frozen_selection을 저장한다. 주 비교는 후보500에서 v1 valid로 고른 β를 양쪽에 동일하게 적용한 v2_exact−v1_exact다. 조건별로 β를 다시 고른 결과와 다른 후보·규칙/연식 효과는 별도 보고한다.
- 입력 hash·시점·ID join, 원래 checkpoint 대비 후보 재계산 차이, 제공 pandas matcher와 JIT adapter의 실제 쌍 비교, CPU/CUDA 정렬을 검사한다.
- test의 이전 연구·규칙 개발 사용 이력이 있으므로 탐색적 후속 실험이다. 단일 학습 seed의 사용자 paired CI를 3-seed 불확실성으로 표현하지 않는다.

## 실행 위치와 확인 방법

전용 snapshot: `ubuntu-wan:~/git/automotive-recsys-v2-20261117`. 기존 완료 캠페인 제어기는 재개하지 않았다.

```bash
ssh ubuntu-wan 'systemctl --user status automotive-v2-1117.service --no-pager'
ssh ubuntu-wan 'cd ~/git/automotive-recsys-v2-20261117 && cat results/vehicle-v2-20261117/controller_status.json'
ssh ubuntu-wan 'cd ~/git/automotive-recsys-v2-20261117 && tail -40 results/vehicle-v2-20261117/controller.log'
```

Colab 작업 세션은 `automotive-v2-1117` 하나다. 제어기는 완료·실패 시 결과 archive를 가능한 범위에서 회수하고 이 세션을 반납한다. Mac이 잠들어도 Ubuntu 제어기와 Colab worker는 계속 진행한다. 다른 세션은 종료하지 않는다. 최초 조회 잔액98.14 CU·활성 할당0개는 할당 전 관측값이다.

회수 경로는 전용 snapshot의 `results/vehicle-v2-20261117/`이다. `controller_status.json`의 completed/failed, `output/<run>/summary.json`, `valid_sweep.csv`, `frozen_selection.json`, `test_results.json`, `paired_comparisons.json`, 사용자별 Parquet·추천 ID/score·로그를 확인한다. 완료 후 `RESULTS.md`와 `selected_test_results.csv`를 자동 생성한다. `usage_after.txt`·`sessions_after.txt`로 자원 반납 결과를 확인한다. 개인별 결과·입력·credential은 Git에 넣지 않는다.

실행 소스는 서버의 독립 Git snapshot revision과 실행 manifest의 파일 SHA-256으로 추적한다. 로컬 테스트30개가 통과했다. GPU에서의 초기 진행 확인은 별도 기록을 따른다. 이 문서 자체는 완료 결과표가 아니다.

## 할당 복구

첫 할당 응답은 시간 초과됐지만, 시작 전0개였던 서버 할당이 이 요청 후 L4 한 개로 바뀐 것을 확인했다. 해당 endpoint를 CLI StateStore API로 같은 작업명에 연결했고 새 VM을 중복 요청하지 않았다. 실패 시도 로그는 서버 `attempt1/`에 보존했다. 현재 제어기는 같은 할당으로 환경·입력을 준비 중이다.

## 초기 진행 검증 후 인계

[시작 검증 JSON](results/vehicle-v2-2026-11-17/startup_verification.json)에 입력 hash·시점 검사와 실제 worker 이벤트를 보관했다. PyTorch2.11.0+cu130/NVIDIA L4에서 BPR-MF의 valid220,516명·test158,805명 후보 재계산이 완료됐고, 두 split의 보관 top100 ID·점수와 정확히 일치했다. 이후 v1_exact의 valid 전체 sweep 완료를 확인했다. test 후보 계산은 점수/ID 생성이며 test 정답을 이용한 β 선택이 아니다. 전체 test 비교·나머지 모델의 완료를 기다리지는 않았다.

시작 WebSocket이 끊긴 문제는 같은 VM에서 새 Jupyter kernel로 한 번 재연결해 해결했다. PID 기반 시작 receipt 확인 후 worker가 실제 평가 중인 것을 확인했으며 중복 worker나 VM을 만들지 않았다. 실행/학습 소스와 별개로 제어용 colab_safe.py의 복구 변경을 기록한다. 최종 상태는 서버 controller_status.json을 확인한다.

## 실패와 보존 범위

- 제어기 상태는 failed이며 최종 오류는 Pilot did not finish within the time limit이다. 그보다 앞서 CLI가 Session not found를 반환하기 시작했다. VM 소실 이유 자체는 보존 로그로 확정할 수 없다.
- 서버에 결과 archive·output 디렉터리·최종 test 결과표가 없다. 마지막으로 별도 저장한 worker.log는 BPR-MF v1_exact valid 전체 sweep까지다. 이후 실행 여부·부분 수치를 추정해 채우지 않는다.
- 초기 입력 감사에서 train 사용자 차종 프로필은 v1 379,912명→v2 389,951명으로 늘었다. 이것은 입력 coverage이며 추천 정확도 개선 결과가 아니다.
- 계정 잔액 관측 차이는98.14−96.33=1.81 CU다. 준비·할당 복구·전송을 포함한 계정 차이이며 모델별 정확 비용이 아니다.
- 결과 회수를 전체 worker 마지막에만 둬 중간 산출물을 VM 밖으로 보존하지 못했다. 다음 실행 전 조건별·모델별 외부 저장, 세션 소실 즉시 감지, 원격 진단 로그 회수, 재개 검증을 보완해야 한다. 이번 상태 조회에서는 GPU를 새로 할당하거나 실험을 재시작하지 않았다.
