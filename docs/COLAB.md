# Colab CLI 실행 환경

장기 본실험의 Drive 인증·자동 저장·재개·진행 로그·GPU 종료는 [DURABLE_EXPERIMENTS](DURABLE_EXPERIMENTS.md)를 따른다.

2026-11-05 사용자의 선택에 따라 **학습과 무거운 연산은 Colab에서 실행한다.**
로컬은 편집·CLI 제어·파일 전송·결과 보관에 사용한다. 로컬 학습 가능성에 대한
기존 측정은 참고 기록으로 남기며 기본 실행 환경은 Colab으로 바꾼다.

## 설치

- 프로젝트 `.venv`: `google-colab-cli==0.7.5` (`colab` dependency group, uv.lock에 고정).
- 전역 Codex 스킬: `~/.codex/skills/colab-operator/SKILL.md`.
- 스킬 출처: [Google 공식 GitHub](https://github.com/googlecolab/google-colab-cli/tree/main/skills/colab-operator).
- 전역 스킬은 다음 턴부터 자동 발견할 수 있다.

```bash
cd ~/git/automotive-recsys
uv sync --locked --group colab
scripts/colab.sh version
scripts/colab.sh usage
scripts/colab.sh sessions
```

## 인증과 세션 상태

`scripts/colab.sh`는 `--auth adc`를 명시하고 Colab 전용 gcloud 설정을 사용한다.
다른 서비스의 gcloud 로그인·MCP 설정을 덮어쓰지 않는다.

- 인증: `~/.config/automotive-recsys/colab/gcloud/`.
- 세션 상태: `~/.config/automotive-recsys/colab/sessions.json`.
- 디렉터리는 0700, 인증·세션 파일은 owner-only로 생성한다.
- 인증 파일은 Git 폴더 밖에 둔다. 내용이나 토큰을 출력하지 않는다.
- CLI 오류 메시지의 인증 매개변수는 `scripts/colab_safe.py`에서 가린다.

최초 인증은 이미 브라우저에서 완료했다. 다시 인증해야 할 때:

```bash
.venv/bin/python scripts/colab_login.py
```

브라우저에서 CU가 있는 계정으로 승인하면 localhost callback으로 완료된다.
채팅에 비밀번호·인증 코드·토큰을 전달하지 않는다.

## 세션 운영

항상 이름을 명시하며 한 번에 작업용 VM 하나만 사용한다.

```bash
scripts/colab.sh new -s automotive-train --gpu L4
scripts/colab.sh usage
scripts/colab.sh exec -s automotive-train --timeout 600 -f /path/to/script.py
scripts/colab.sh download -s automotive-train /content/output.tar.gz results/output.tar.gz
scripts/colab.sh log -s automotive-train --output results/training-log.ipynb
scripts/colab.sh stop -s automotive-train
```

실행 스크립트는 로컬에서 읽어 원격 커널에 전송된다. 기본 경로는 `/content`이고,
로컬 Mac 경로가 원격 VM에 자동으로 존재하지 않으므로 데이터/모듈 파일은 따로 업로드한다.
실제 원격 경로는 `/content/automotive-recsys`를 사용한다.
`--timeout`은 작업에 맞게 지정한다. 대화형 Drive mount나 VM-side GCP 인증은 필요할 때만 수행한다.

현재 계정에서 잔액 **175.32 CU**를 확인했고, L4 시험 세션 할당 직후 소모율은 **1.94 CU/시간**이었다.
이 값은 당시 표시값이며, 이후 GPU/세션마다 `usage`로 다시 확인한다.
Google Cloud 별도 유료 VM이나 새 결제를 만들지 않고 기존 Colab 잔액을 사용한다.
GPU 가용성이 다르면 명시적으로 T4 등 지원되는 장치로 바꾸며 임의의 이름은 사용하지 않는다.

## 재현 가능한 연결 시험

```bash
scripts/colab.sh new -s automotive-setup --gpu L4
.venv/bin/python scripts/colab_smoke.py --session automotive-setup
```

시험은 다음을 수행한다.

1. 소스·테스트·전처리 파일을 `results/colab/staging`에 묶어 업로드한다. `.venv`·Git·인증 파일은 제외한다.
2. 원격에서 각 파일의 SHA-256을 검증하고 데이터 계약 검사 39개를 실행한다.
3. 전체 valid 사용자에 대한 인기순 추천 점수를 로컬 기준값과 대조한다.
4. 평가 코드 테스트 5개와 전체 train 그래프의 CUDA forward/backward를 실행한다.
5. 작은 GPU 시험 체크포인트와 지표를 다운로드하고 체크포인트 해시를 다시 확인한다.
6. 실행 기록을 notebook으로 회수하고 `finally`에서 해당 시험 VM을 종료한다.

이 시험의 GPU 업데이트는 연결·연산 점검용이다. 수렴한 모델 학습이나 최종 test 평가로 기록하지 않는다.
결과는 `results/colab/colab_smoke/summary.json`, 기록은 `results/colab/setup-log.ipynb`에 보관한다.
파일 전송은 CLI의 base64 처리에 맞춰 16MiB 단위로 나누며, 로컬에서 압축 학습 작업을 실행하지 않는다.

2026-11-05 시험 결과: **통과**.
NVIDIA L4 / Python 3.13.16 / PyTorch 2.11.0+cu130에서 파일 17개 해시,
데이터 계약 검사 39개, 평가 코드 테스트 5개, 전체 검증 사용자 220,516명의 기준선 재현,
전체 train 그래프 CUDA 역전파, 체크포인트 저장·복원·다운로드를 확인했다.
3층 LightGCN 형태의 측정 업데이트는 약 0.320초, CUDA peak allocated는 약 2,889MiB였다.
두 번의 시험 업데이트뿐이므로 전체 epoch 시간이나 추천 모델 성능으로 간주하지 않는다.
시험 종료 후 잔액 175.28 CU, 소모율 0 CU/시간, 활성 세션 0개를 확인했다.

Colab VM은 임시 저장소다. 장기 학습은 중간 체크포인트를 주기적으로 회수하고,
완료·실패 시 결과를 확보한 뒤 이 작업에서 생성한 세션만 종료한다.

## GPU 학습기 검증

세 모델의 CUDA 학습 루프·전체 상품 평가·저장/재시작은 실제 전체 train 한 epoch로 검증했다.
실행 시간과 커널 최적화 근거는 [GPU 점검 결과](GPU_REVIEW.md)를 참고한다.

```bash
scripts/colab.sh new -s automotive-pilot --gpu L4
.venv/bin/python scripts/colab_perf.py --session automotive-pilot --pilot
```

pilot은 세 모델 한 epoch씩 실행하고 결과를 회수한 뒤 VM을 종료한다.
수렴한 최종 모델을 만들거나 test 점수를 선택에 사용하지 않는다.
장기 본 학습에서는 `colab exec --timeout`도 모델별 시간 예산보다 길게 잡는다.
최종 test는 valid로 설정을 고정한 뒤 `best.pt`의 임베딩으로 별도 평가한다.

[공식 CLI 안내](https://developers.googleblog.com/introducing-the-google-colab-cli/)
· [공식 저장소](https://github.com/googlecolab/google-colab-cli)
