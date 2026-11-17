# v2 재정렬의 Drive 저장·재개 검증

2026-11-17. 이전 실행의 실패 기록은 [원래 실행 문서](VEHICLES_V2_RUN_2026-11-17.md)에 보존한다. 사용자가 기존 중간 저장·복구 경로를 전부 검증한 뒤 재시작하도록 요청했다. **최종 검증을 모두 통과했고 14:07 KST에 본 데이터 비교를 재시작했다. 최종 모델 비교는 진행 중이다.** 최종 통과·실행 증거는 아래 검증 기록을 따른다.

## 기존 코드 재사용

- `DriveStore`: 원래 인증 경로, content hash·MD5/bytes 검증 업로드, SHA-256 검증 다운로드.
- `CampaignJournal`: progress.json/txt, events.jsonl과 주기적 Drive 갱신.
- `DurableRun.upload_outputs(immutable=True)`: 해시가 붙은 불변 결과 파일 업로드. 새 `DurableStages`는 optimizer checkpoint 대신 평가 조건·완료 단계·배치 cursor와 마지막 두 세대의 artifact 목록을 commit한다.
- 기존 launcher 방식: 노트북 셀이 worker가 끝날 때까지 활성 실행을 유지한다. 호스트는 WebSocket 연결 상태와 별개로 Drive heartbeat·결과 pointer를 읽는다.
- 기존 controller 정책: 소유자 확인, 자기 endpoint만 반납, 진척 없는 할당 최대6회, 계정 잔액30 CU 하한. 이 실행의 추가 계정 관측 예산은8 CU다.

## 재개 단위와 보존 자료

각 모델의 입력 감사, 후보 재계산, valid/test 조건, 최종 비교를 단계로 기록한다. 긴 재정렬은 약120초마다 완료한 사용자 배치까지 저장한다. 이전 단계·계수를 검증해 완료 조건은 건너뛰고 부분 합계·추천 ID/score와 분류 결과를 복원한다. 중간 저장에는 초기화되지 않은 배열 영역을 포함하지 않는다.

`restore.json`은 파일 업로드 검증이 끝난 뒤 갱신한다. 복원은 빈 임시 디렉터리에서 해시 검증을 마친 세대만 공개하므로, 로컬의 미완료 파일을 완료 결과로 잘못 읽지 않는다. 새 입력·코드 context가 다르면 복원을 거부한다. 이 실행은 재학습이 아니므로 가짜 optimizer/epoch checkpoint를 만들지 않는다.

## 검증 순서

1. 로컬36개 단위 시험: 기존 수식·평가 검사에 저장 실패/파일 손상/입력 변경 거부와 제어기 장애·다른 세션 보호 시험 추가.
2. 실제 Drive: 업로드·다운로드 SHA 대조, 잘못된 hash 거부, 새 pointer commit 실패 후 이전 세대 복원. 서버 host_verified.json과 Drive host_verification.json에 보관.
3. 실제 Colab CUDA: 8명·1,300상품 fixture의 전체 비교를 사용한다. valid 첫 배치 저장 직후 exit86으로 중단 → 출력 삭제 → Drive 복원 → test 첫 배치의 추천 ID/score·fit 분류를 저장한 뒤 다시 중단 → 출력 삭제·복원 → 완료 조건 건너뛰기 → test 전체 결과가 무중단 실행과 동일한지 대조. 모든 committed artifact를 다시 내려받아 SHA를 검사한다.
4. 실제 Ubuntu 제어기: 본 데이터 게이트를 닫은 CUDA preflight 실행 중 systemd main만 종료한다. 자동 재시작 뒤 같은 VM·worker를 채택하고 중복 할당/시작이 없는지 확인한다.
5. 검증 증거를 저장한 뒤에만 control.json의 release_real을 켠다. 첫 본 데이터의 결과가 Drive pointer에 commit됐는지 확인한 뒤 백그라운드로 인계한다.

## 실행·로그 위치

[이번 전용 Drive](https://drive.google.com/drive/folders/1GdQ89Q7e7lWfu8gbtS-vp3ajUDS8Fvh8)는 과거 완료 캠페인과 분리돼 있다. `progress.json/txt`, `events.jsonl`, `worker_heartbeat.json`, `workers/<worker>/worker.log`, `runs/<run>/restore.json`, `runs/<run>/completed.json`과 해시 이름의 snapshots를 읽는다. 입력·사용자별 결과·credential은 Git에 넣지 않는다.

호스트: `ubuntu-wan:~/git/automotive-recsys-v2-durable-20261117`.

```bash
ssh ubuntu-wan 'systemctl --user status automotive-v2-durable-r4-1117.service --no-pager'
ssh ubuntu-wan 'cd ~/git/automotive-recsys-v2-durable-20261117 && cat results/vehicle-v2-durable-r4-20261117/controller_status.json'
```

Drive heartbeat가 살아 있으면 WebSocket 단절만으로 worker를 재실행하지 않는다. 세션이 소실되면 검증된 Drive 세대에서 새 작업 VM으로 복원한다. 완료 시 Drive 결과를 호스트로 SHA 대조 회수하고 자기 VM을 반납한다. 영구 코드/인증 오류·무진척/예산 한도는 failed 상태로 멈추며 자동 무한 재시작하지 않는다. 예상하지 못한 제어기 프로세스 종료는 systemd가 재시작한다.

## 검증 중 발견한 결함

첫 CUDA preflight는 fixture의 후보 계산 스크립트 경로 누락으로 실패했다. 본 데이터 실행 게이트는 닫힌 채 유지됐고 해당 VM을 반납했다. fixture 스크립트 경로, restore pointer의 stage/cursor 필드, 실패 시 최종 worker 로그 업로드를 수정한 뒤 새 source context와 별도 Drive root로 다시 검증한다. 이전 실패를 통과한 시험으로 계산하지 않는다.

완료 restore pointer 저장 직후 completed marker 게시가 끊기는 경우도 추가 시험했다. 외부 복원에서 artifact SHA를 모두 검증한 후 완료 marker를 복구한다. 두 번째 준비 실행은 이 보완과 test 배열 부분 재개 검증을 포함하기 위해 본 데이터 게이트를 열지 않고 종료·VM 반납했다.

세 번째 CUDA 시험은 valid/test 두 강제 중단과 Drive 복원을 수행했지만, 기준 NDCG=0인 fixture에서 상대 변화율이 NaN이 되어 strict JSON 저장이 거부됐다. 본 데이터는 실행하지 않았다. 상대 변화율이 정의되지 않는 경우 null로 보고하고 절대 차이는 유지하도록 수정했으며, 로컬 회귀 시험을 추가해36개가 통과했다. 새 source context에서 같은 CUDA 대조를 다시 수행한다.

## 시도별 로그 보존

| 시도 | 상태 | Drive |
|---|---|---|
| 1 | fixture 경로 누락으로 실패, VM 반납 | [로그](https://drive.google.com/drive/folders/17H66JmhIwGliz9qsL2YKbHPdJmDs5UB0) |
| 2 | 완료 marker 복구·test 배열 검증 보완을 위해 본 실행 전 교체, VM 반납 | [로그](https://drive.google.com/drive/folders/1aX-Tn7767p8sXsLoMpmr2p1PyVh1L9sH) |
| 3 | 두 중단/복원 후 NDCG=0 상대율의 NaN 저장 오류 발견, VM 반납 | [로그](https://drive.google.com/drive/folders/17SqzX1fFxVhXKpTAE9xVM63K7cISuzFj) |
| 4 | 수정본 최종 검증, 이후 본 실행 | [로그·결과](https://drive.google.com/drive/folders/1GdQ89Q7e7lWfu8gbtS-vp3ajUDS8Fvh8) |

각 시도는 별도 source context와 Drive root를 사용한다. 이전 시도의 통과 receipt를 새 context의 통과 근거로 재사용하지 않는다.

## 최종 통과·재시작 근거

- source context: `198cff1bafb8628114b425ae50136c064fd7ae3be10e2ea92c90b3b6f736a5fe`. 서버 frozen source52개 파일과 로컬 원본의 SHA가 모두 일치했다.
- [로컬36개 검사](results/vehicle-v2-durable-2026-11-17/local_tests.json), [실제 Drive 실패/복원](results/vehicle-v2-durable-2026-11-17/host_verification.json), [격리 환경 준비](results/vehicle-v2-durable-2026-11-17/environment_ready.json).
- [CUDA 강제 중단/복원 대조](results/vehicle-v2-durable-2026-11-17/cuda_restore_verification.json): valid 배치·test 추천 배열에서 각각 exit86, 출력·로컬 journal 삭제 후 외부 복원, 완료 단계 건너뛰기, 최종 test 지표12행이 무중단 실행과 동일, committed artifact 전부 SHA 다운로드 검증. 실제 VM 강제 소실 시험과는 구분한다.
- [제어기 실제 SIGKILL/재시작](results/vehicle-v2-durable-2026-11-17/controller_restart_verification.json): 같은 VM·worker 시작 receipt, 중복 할당/시작 없음, heartbeat 전진. 세션 소실/다른 세션 보호는 로컬 제어기 시험으로 확인했다.
- [소스 해시 대조](results/vehicle-v2-durable-2026-11-17/frozen_source_verification.json), [14:07 KST 실행 게이트 해제](results/vehicle-v2-durable-2026-11-17/release.json).

시작 시 Colab WebSocket이 끊겨 원격 worker 시작 receipt가 없음을 조회한 뒤 같은 VM의 새 kernel 연결로 복구했다. 실제 실행 중인 notebook launcher의 활성 출력과 Drive heartbeat를 확인했고, 끊어진 이전 호스트 CLI는 종료했다. 이후 학습 중 진단 exec를 추가하지 않고 Drive/log만 읽는다.

## 실제 데이터 초기 결과 회수 확인

14:11 KST에 BPR-MF valid220,516명의 v1_exact 전체 sweep이 완료된 Drive restore pointer와 valid_sweep.csv를 조회했다. CSV·입력 감사·pool receipt를 실제 다운로드해 SHA를 확인했다. valid220,516명/test158,805명의 기존 top100 ID와 점수는 모두 정확히 재현됐다. [초기 결과 회수 검증](results/vehicle-v2-durable-2026-11-17/real_startup_verification.json).

이는 시작 PID만 확인한 기록이 아니라 실제 전체 valid 평가 결과의 외부 보존 증거다. 나머지 조건과 세 모델의 최종 비교는 계속 진행하며, 이 초기 관측으로 v2 성능 개선을 결론내리지 않는다. 제어기는 완료 시 결과 전체를 SHA 대조해 호스트로 내려받고 자기 VM을 반납한다.
