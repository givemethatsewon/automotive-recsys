# 장기 실행 원인 점검과 서버 제어기

2026-11-05 점검. 확인된 사실과 추정을 구분한다.

- Mac 전원 기록에 06:02 및 15:18 등의 Idle Sleep, 반복된 Maintenance Sleep이 있다. 같은 실행 기간 CLI 기록에는 SSL/연결 오류 및 runtime proxy의 HTTP 10초 read timeout이 있다. Mac에서는 네트워크 변경으로 사설망 SSH도 도중에 접근 불가가 됐고 외부 SSH 경로는 동작했다.
- 완료된 모델 결과/optimizer checkpoint는 Drive에서 계속 읽을 수 있었다. 이전 6개 runtime에는 정상 종료 marker가 없고 마지막 진행 이후 할당이 없어졌다. 공개 API/로컬 로그에는 Google 내부의 VM 삭제 사유가 없어 모든 삭제를 절전 하나로 확정하지 않는다.
- 기존 제어기는 총 runtime 6개 한도와 연속 통신 오류 5회 종료 조건 때문에 복구가 멈췄다. 상태에는 오래된 running이 남았다. 이 두 제어 정책은 진척 없는 연속 6개/일시적 네트워크 backoff로 수정했다.
- `google-colab-cli 0.7.5`, `jupyter-kernel-client 0.9.0`을 사용한다. [관련 timeout 버그](https://github.com/googlecolab/google-colab-cli/issues/82)는 설정한 전체 timeout 후 local CPU spin에 관한 것이므로, 현재의 72시간 timeout 전에 발생한 모든 VM 삭제의 원인으로 단정하지 않는다.
- [공식 Colab FAQ](https://research.google.com/colaboratory/faq.html#resource-limits)는 VM 수명과 자원 가용성이 가변적임을 명시한다. 영구적으로 같은 VM을 유지하려 하지 않고 정상 저장/재할당/재개로 전체 실험을 진행한다.

## 적용한 구조

제어기를 기존 Ubuntu 서버의 `~/git/automotive-recsys`와 user systemd로 옮긴다. 원격 학습·실제 데이터 연산은 계속 Colab CUDA에서 수행한다. 서버의 linger가 활성화되어 SSH를 닫아도 제어기가 실행된다. 서비스는 예기치 않은 프로세스 종료 후 다시 시작하고 같은 VM을 확인해 채택한다. Drive owner ID로 Mac과 서버가 동시에 새 GPU를 할당하지 못하게 한다.

자격 증명은 SSH로만 별도 전송하고 서버의 `~/.config/automotive-recsys`에 owner-only로 보관한다. 코드/가상환경/managed Python/cache는 서버의 Git 프로젝트 안에 둔다. 초기 frozen 모델/입력/선택 plan의 hash는 유지하며, 제어기 및 운영 I/O source hash는 별도 기록한다.

완료 후 CSV/PNG/HTML/Markdown은 Drive에 저장되고 Ubuntu의 캠페인 report 폴더에도 내려받는다. Mac의 연결 상태는 이 완료 과정의 조건이 아니다. 새 결제/GCP 학습 VM 대신 기존 Colab 잔액과 기존 제어 서버를 사용한다.

## 저장과 예산

실측에서 checkpoint 파일은 optimizer 포함 약 621 MiB, best weights는 약 207 MiB다. 일반 snapshot을 2분 간격으로 전송하면 XSimGCL에서는 거의 매 epoch에 큰 전송이 발생한다. 버전별 adapter로 일반 간격을 600초로 늘리되 최초/best/마지막 강제 저장과 2세대 보관은 유지한다. 강제 저장이 없는 구간에서 중단되면 마지막 저장 이후 계산은 다시 수행할 수 있다. Drive resumable transfer는 64 MiB chunk로 처리한다.

실행별 manifest의 `operational_io`와 worker의 source SHA를 기록한다. 모델의 loss·GPU 수식·seed/data RNG·split·valid 선택 기준은 바꾸지 않는다. 일반/강제 저장과 복원을 포함한 테스트를 추가했다.

기존 frozen plan은 초기 예산 100 CU를 그대로 보존한다. 긴 실행이 단순 예산 상한 때문에 중단되지 않도록 `control.json.cu_budget=130`을 별도 운영 설정으로 적용하고, 잔액 30 CU 이하는 계속 중단한다. 계정 CU delta는 다른 동시 작업도 포함할 수 있다. 연구 설정과 운영 예산을 혼동하지 않는다.

## 실행 경로 추가 점검

- worker 8/9/12 등은 GPU 할당 후 실제 작업 시작 전에 CLI의 `os.chdir` 실행 요청에서 `Connection was lost`가 발생했다. GPU 할당과 학습 시작은 별개다. CLI의 `status`에 표시되는 BUSY/IDLE도 로컬 상태이므로 실제 생존 증거로 단독 사용하지 않는다. Drive의 worker 시작 receipt와 새 학습 이벤트/검증된 checkpoint를 확인한다.
- 원격 Python의 예외에도 `colab exec`가 exit 0을 반환한 사례가 있었다. 환경 준비는 exit code에 더해 CUDA 확인/커널 패키지 변경 없음 receipt를 내려받아 검증하며, 실행 후 180초 안에 worker 시작 receipt가 없으면 해당 작업 VM만 반납하고 제한된 재시도를 한다.
- Colab 기본 Python에 `ensurepip`가 없어 일반 venv 생성이 실패했다. `with_pip=False` venv와 uv로 해결했다. Torch/CUDA는 시스템 패키지를 상속하고 실험 의존성만 venv에 설치한다. 기본 커널의 ipykernel/jupyter-client/tornado/google-auth/numpy/torch 버전은 설치 전후 대조한다. 이 격리는 연결 환경 변경을 피하는 조치이며 과거 모든 VM 삭제의 원인 증명은 아니다.
- worker 13에서 위 환경 receipt를 확인했고, 저장된 XSimGCL trial 09 epoch 1을 복원해 epoch 2–5 계산 및 epoch 5 Drive 저장을 확인했다. 준비 실패/연결 실패 기록과 실제 학습 기록을 구분한다.
- Jupyter transport 1.0.2 override는 후보로만 검토하고 제거했다. 실제 재개가 확인된 CLI 0.7.5의 기본 transport 0.9.0을 lockfile에 유지한다. 미검증 dependency override를 복구 성공 원인으로 보고하지 않는다.

## 제어기 장애 시험과 현재 시간 추정

2026-11-05 실제 학습 중 systemd main process에만 SIGKILL을 보내 자동 재시작을 시험했다. `KillMode=process`이므로 실행 중인 CLI 자식/Colab 학습은 유지한다. 검증 결과는 Drive `operations/controller_restart_verified.json`에 저장하며, PID 변경/서비스 active/worker 수와 GPU assignment 불변/새 worker progress를 함께 판정한다. 일반 VM 상실 후 optimizer/RNG 복구는 기존 `preflight/verified.json` 및 worker 13의 epoch 연속 기록으로 별도 확인한다.

21:30 KST 부근 기준 완료된 탐색 9/32, XSimGCL trial 09를 진행 중이다. 남은 탐색, 주 모델 3개×3 seed, 보조 실험, test 및 보고서 생성을 포함한다. L4 실측 학습 시간은 epoch당 MF 약 19초, LightGCN 64초(3층), XSimGCL 138–146초다. 큰 checkpoint의 Drive 전송과 early stopping 여부가 총 시간을 크게 바꾼다.

- 주 비교 3모델×3seed test 결과까지: 지금부터 약 18–26시간.
- 나머지 깊이/대조학습 제거 실험과 최종 보고서까지: 총 약 28–44시간.
- 이는 안정적으로 L4를 할당받는 경우의 운영 추정이다. Colab 공급 대기, 재시도, 실제 early stopping으로 변하며 완료 시각 보장이 아니다. 최대 100 epoch에 도달한 모델은 미수렴 가능성을 보고한다.

완료 시 GPU를 반납하고 CSV/그림/HTML/Markdown 및 완료 상태를 Drive에 저장한다. 서버 제어기가 결과를 내려받은 뒤 정상 종료한다. 사용자 Mac과 LLM 대화가 계속 연결될 필요가 없다.
