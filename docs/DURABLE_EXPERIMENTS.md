# Drive에 보관하는 본실험

2026-11-10 재확인: 캠페인 `automotive-20261104T204146Z`는 **2026-11-07 06:58 KST에 32/32개 완료**했고 제어기도 보고서를 회수한 뒤 정상 종료했다. 현재 활성 Colab 할당 0개, 소모율 0이다. [최종 결과](RESULTS_FINAL.md)와 [인수인계](../PROJECT_BRIEF.md)를 먼저 읽는다. 아래 시작/재개 명령은 새 실험 또는 미완료 캠페인의 운영 절차이며, 완료 캠페인을 다시 시작하는 지시가 아니다.

코드·가상환경·제어 프로그램은 이 Git 프로젝트에서 관리한다. 학습·전체 평가는 Colab CUDA에서 실행한다.
Drive는 결과와 복구 상태를 보관하는 영구 저장소다. 인증 파일은 Drive와 Git에 넣지 않는다.

## 기록 형식

- `events.jsonl`: 실행 시작/종료·epoch 완료·체크포인트 검증 이벤트. 한 줄이 독립된 JSON이다.
- `progress.txt`, `progress.json`: 현재 작업·모델·seed·epoch/상한·진행 step·best valid·최근 Drive 저장 시각. 대략 30초 간격으로 갱신한다. 실제 업로드가 오래 걸리면 저장 단계가 표시된다.
- `runs/<id>/epochs.csv`, `epochs.jsonl`: loss와 BPR/CL raw/정규화 항, 관측된 valid 지표, 시간, best 여부. 검증하지 않은 epoch의 valid CSV 값은 빈칸이다.
- `valid/`, `test/`: 지표 JSON, 사용자별 Parquet(`n_train`, `n_gt`, tail 정답 수 포함), 전체 상품의 노출 Parquet. 본 모델의 top-100은 NPZ의 uid/iid/score 배열로 저장한다.
- `report/`: seed별/평균·표준편차/그룹/ablation/paired 차이/시간 CSV, PNG 그림, `RESULTS.html`, `RESULTS.md`.
- `account_cost.json`, `workers/*/cost_observations.json`: 시각별 계정 CU 잔액·소모율. 다른 Colab 작업이 있으면 포함될 수 있는 **계정 단위 관측**이다. 실행별 정확한 CU로 꾸며 쓰지 않는다.

사용자별 paired bootstrap 95% CI는 seed마다 계산한다(사용자 500회 복원추출, 고정 RNG). 3개의 seed를 새로운 사용자처럼 합치지 않는다. Popularity의 seed SD는 null이다.
CL=0 보조 실험은 encoder/noise를 유지하고 InfoNCE 계산을 생략한다. 계산하지 않은 raw CL은 null, `cl_raw_evaluated=false`로 표시한다.

## 사전에 정한 실험 예산

`experiment_plan.json`과 `frozen_selection.json`을 보존한다. train만 학습하고, valid로 설정/epoch를 고른 뒤 test를 평가한다.

1. coarse 탐색 12회: 모델당 4설정, seed 42, 각 15 epoch.
2. 선택한 LightGCN LR/L2로 1·2층을 각 15 epoch 비교한다. 3층 탐색 결과와 함께 valid가 가장 좋은 층을 main으로 선택한다.
3. 본 모델 3개 × seed 42·43·44 = 9회.
4. 나머지 LightGCN 층 2개와 XSimGCL CL=0 × 3 seeds = 보조 실험 9회.

총 32개 학습 작업이며 Popularity valid/test는 별도 결정적 평가다. final은 최대 100 epoch, epoch 1과 매 5 epoch에 valid 평가, 개선 없는 검증 8회 후 종료한다. 100 epoch 도달을 수렴으로 단정하지 않고 상한 부근의 best epoch를 보고서에 표시한다. coarse 15 epoch의 설정 선택에는 탐색 예산의 한계가 있다.

L4 전체 25–40시간은 초기 추정이었다. frozen plan의 초기 CU 상한 100은 보존하고, 사용자 요청에 따라 운영 control에서 130 CU/최소 잔액 30 CU를 적용했다. 이번 캠페인의 종료 시 계정 감소는 약 74.79 CU다. 전송/재할당/early stopping과 다른 동시 세션이 시간·계정 관측 비용에 영향을 줄 수 있다.

## 복구가 가능한 저장 순서

각 epoch 종료 때 VM의 `last.pt`를 원자적으로 저장한다. 최초 epoch, best 갱신, 마지막 epoch 강제 저장은 유지한다. 일반 Drive 회수 간격은 초기 120초에서 운영 I/O adapter 적용 후 600초로 바꾸었고 64 MiB chunk를 사용했다. 모델 수식과 frozen 선택 plan은 유지했다.

1. `best-e####-<SHA>.pt`, `state-e####-<SHA>.pt`와 내용 hash가 붙은 불변 artifact를 업로드한다.
2. Drive metadata의 size/MD5를 읽어 업로드 내용과 대조한다. SHA-256을 manifest에 기록한다.
3. 모든 파일 확인 후에만 `restore.json`의 현재 세대를 갱신한다. 새 파일 전송 실패가 이전 세대의 파일을 덮어쓰지 않는다.
4. 두 세대의 참조를 남긴 뒤, 어느 세대도 참조하지 않는 이 프로젝트의 이전 체크포인트/artifact만 삭제한다. 휴지통에 계속 쌓아 quota를 쓰지 않는다.
5. 새 VM은 committed manifest의 SHA-256을 대조하며 다운로드하고 optimizer와 model/data RNG를 복원한다. 완료 marker가 있는 작업은 건너뛴다.

완료된 작업은 best inference 모델과 평가 출력만 보관하고 큰 optimizer snapshot은 정리한다. 따라서 완료된 작업은 재평가할 수 있지만 epoch 연장은 새 실행으로 해야 한다. 업로드가 중단된 미참조 파일은 남을 수 있으며 복구 위치로 사용하지 않는다. Drive 파일의 외부 삭제/변조나 모든 복사본 소실까지 복구한다는 보장은 아니다.

## 인증·시작·확인

2026-11-05 실제 Drive 로그인 후 ADC의 quota project가 없어 HTTP 403을 확인했다. Drive 전용 ADC에 기존 `thatzfit-curation`을 quota project로 설정하고 Cloud Service Usage API로 `drive.googleapis.com`을 활성화했다. 기존 Colab 인증과 다른 gcloud 설정은 변경하지 않았다. 공식 근거: [ADC quota project](https://docs.cloud.google.com/docs/quotas/set-quota-project), [서비스 활성화 REST API](https://docs.cloud.google.com/service-usage/docs/reference/rest/v1/services/enable). Drive와 Colab의 계정 일치 및 실제 Drive quota 응답도 확인했다.

전역 `colab-operator` 스킬을 적용한다. 대화형 `colab drivemount` 대신 별도 Drive OAuth의 `drive.file` 권한으로 이 앱이 만든 파일을 다룬다. gcloud에 필수인 기본 scope도 함께 요청한다. 기존 Colab ADC는 덮어쓰지 않는다.

```bash
cd ~/git/automotive-recsys
uv sync --locked --group colab
.venv/bin/python scripts/colab_login.py --drive
.venv/bin/python scripts/experiment.py prepare
.venv/bin/python scripts/colab_campaign.py --background
.venv/bin/python scripts/experiment.py status
```

Drive에 새 비공개 프로젝트 폴더를 만들고 URL을 `results/campaigns/current.json`에 기록한다. 12 GiB 이상의 여유 공간을 먼저 확인한다. 새 VM에 인증 파일 두 개를 별도 전송해 0700 디렉터리/0600 파일로 보관한다. 데이터·소스 bundle, Drive 결과, notebook 출력에는 인증 내용을 넣지 않는다. worker 종료 시 파일을 지우고 자기 endpoint만 반납한다.

가벼운 controller는 기존 `ubuntu-wan` 서버의 user systemd `automotive-colab.service`에서 120초마다 상태를 확인했다. 학습 kernel은 LLM/채팅과 독립적으로 작동한다. worker는 결과를 올리고 자기 VM을 반납하며, controller는 완료 보고서를 서버 프로젝트에 내려받은 뒤 정상 종료한다. 로컬 원문/결과 검토를 위해 서버 제어기를 다시 켤 필요가 없다.

초기에는 Mac 제어기의 절전/연결 문제에 영향을 받았지만, Ubuntu로 이전한 이후에는 Mac 연결 상태가 자동 재할당·결과 수집의 조건이 아니었다. Drive owner ID로 제어 호스트를 하나로 제한한다. WebSocket 단절만으로 살아 있는 학습을 중복 실행하지 않는다. 자동 재할당은 진척 없는 연속 worker 최대 6개로 제한하고, 실제 완료 실행/저장 epoch가 전진하면 이 연속 실패 수를 다시 계산한다. 통신 오류는 backoff 후 재시도하고 중단 사유를 기록한다. [원인·장애 시험](CONTROLLER_RELIABILITY.md)에 실제 근거가 있다.

11월 5일 18:20 한국 시각 상태 점검에서 설정 탐색 8개 완료, XSimGCL 6 epoch 도중 중단/5 epoch 외부 저장, 활성 VM 0개를 확인했다. 이전 제어기가 누적 worker 6개 한도에 걸려 종료됐는데 progress에는 running이 남아 있었다. 위 연속 무진척 기준과 네트워크 재연결 정책은 이 실측 문제를 보완한 것이다. 중간 seed 42 valid 최고 NDCG는 BPR-MF 0.00136339, LightGCN 0.00420011, 진행 중인 XSimGCL 0.00504616이다. 최종 test/3-seed 결과가 아니다.

```bash
.venv/bin/python scripts/experiment.py pause
# Drive의 마지막 검증 checkpoint를 보존하고 다음 control 확인에서 중단한다.
.venv/bin/python scripts/experiment.py resume
.venv/bin/python scripts/colab_campaign.py --background
```

`results/campaigns/<id>/controller.log`와 worker별 log도 남는다. 중단 후 현재 epoch의 아직 업로드되지 않은 계산은 다시 수행할 수 있다. 저장·전송은 실제 실행 시간에 영향을 주며 timing CSV의 training 시간에는 진행 로그 callback 지연이 포함될 수 있다. 순수 커널 속도 비교는 별도 GPU benchmark를 사용한다.

## 검증 상태

실험 파이프라인의 작은 테스트는 잘못된 history mask/평가 cohort, 추천 ID/score 대응, loss gradient 유지, 저장 세대 정리/복원, 새 pointer commit 실패 후 이전 복원, seed 집계와 단일 Popularity 기준선, 보고서 생성까지 검사한다.

2026-11-05 작은 테스트 19개 통과. 새 loss/평가 기록 경로로 실제 L4에서 세 모델의 전체 train 한 epoch·전체 valid·optimizer/RNG replay를 다시 통과했다. train 시간은 BPR 16.52초 / LightGCN 54.48초 / XSimGCL 114.12초, 확장된 전체 valid 집계는 약 9.7–10.2초였다. 시험 결과/로그를 회수하고 VM을 종료했으며 잔액 174.93 CU·활성 0개를 확인했다. 이 pilot은 최종 test 결과가 아니다. 이후 진행 이벤트/환경 metadata 필드의 변경은 수학·optimizer를 바꾸지 않았다.

2026-11-05 실제 Drive upload/download의 MD5·SHA-256 검증, 두 세대 보관/이전 세대 정리, 별도 출력 폴더 복원을 통과했다. L4에서 실제 BPR checkpoint를 외부 저장 후 내려받아 epoch 2를 이어 학습했고, optimizer/RNG replay의 최대 파라미터 차이는 이번 검사에서 0이었다. 이는 이 실행의 관측값이며 일반적인 bitwise 재현 보장은 아니다. 차종·텍스트 특징은 미제공이므로 해당 실험은 미실시로 남긴다.

worker 시작 시 별도 preflight로 BPR 1 epoch의 실제 optimizer/RNG checkpoint를 Drive에 보관하고 다른 출력 폴더로 내려받아 epoch 2를 이어 학습한다. 복원 후 optimizer/RNG update replay까지 통과해야 본실험을 실행한다. 이 preflight는 32개 본 작업/최종 성능표에 포함하지 않고 `preflight/verified.json`으로 구분한다.

## 2026-11-05 본실험 시작 기록

- ID: `automotive-20261104T204146Z` (ID의 날짜는 UTC, 한국 날짜는 11월 5일).
- 고정 실행 소스 commit: `060f5a312a15f9a4469452bb6637b09ec792a4b6`, 입력 51개 파일의 SHA 확인.
- worker: `auto-20261104t204146z-1`, NVIDIA L4. 시작 시각 한국 05:49.
- Drive: [실험 폴더](https://drive.google.com/drive/folders/1IBQzFgM2ZDR-zz4mgwk6BjIf6c9pRwmx), [현재 진행 로그](https://drive.google.com/file/d/1THXOtfc99Ps6u-YS5pURLUb2GhnE73vo/view).
- 외부 checkpoint 재개 preflight 통과 후 `popularity_valid` 본실험 단계 진입을 확인했다. 이 기록은 시작 사실이며 32개 작업/최종 test 평가 완료를 뜻하지 않는다. 최신 상태는 Drive progress를 읽는다.
- local controller PID/log는 `results/campaigns/automotive-20261104T204146Z/`에서 관리한다. 결과 완료 시 같은 경로의 `report/`에 CSV/PNG/HTML/Markdown을 내려받는다.
