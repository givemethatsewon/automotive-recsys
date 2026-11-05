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
