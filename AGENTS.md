# Automotive recommendation experiments: agent instructions

## 작업 위치와 실행 환경

- 프로젝트 루트는 `/Users/givemethatsewon/git/automotive-recsys`다. 코드·문서·가상환경·결과는 이 프로젝트에서 관리한다. SecondBrain에는 쓰지 않는다.
- 사용자는 로컬 부하를 원하지 않는다. 전체 데이터 학습·GPU benchmark·긴 평가는 **Colab CUDA GPU에서 실행**한다. 로컬에서는 편집, CLI 제어, 파일 전송, 작은 수식 테스트만 수행한다.
- `train_gpu.py`의 CUDA 필수 검사를 유지한다. CPU/MPS 자동 fallback을 추가하지 않는다.
- 모델은 **Popularity 기준선 + BPR-MF + LightGCN + XSimGCL**이다. Popularity는 비학습·결정적 기준선이다.
- 최신 작업 상태는 [README](README.md), 연구 계획은 [EXPERIMENT_PLAN](docs/EXPERIMENT_PLAN.md), 결과 수집 계약은 [EXPERIMENT_OUTPUTS](docs/EXPERIMENT_OUTPUTS.md)를 읽는다.

## Colab 운영·인증

- `scripts/colab.sh`를 사용한다. CLI 0.7.5와 프로젝트 dependency group `colab`은 `uv.lock`에 기록돼 있다. 새 CLI 버전의 동작은 `--help`로 확인한다.
- 이 wrapper는 `--auth adc`와 Colab 전용 `CLOUDSDK_CONFIG`를 명시한다. 다른 서비스의 gcloud/MCP 설정을 덮어쓰지 않는다.
- 인증·세션 상태는 `~/.config/automotive-recsys/colab/`에 두고 디렉터리 0700, 파일 0600을 유지한다. 토큰·인증 URL의 비밀값·환경 전체를 출력하거나 Git에 넣지 않는다.
- 재인증이 필요하면 `.venv/bin/python scripts/colab_login.py`로 브라우저 승인을 요청한다. 비밀번호·인증 코드를 채팅으로 받지 않는다. `colab auth`는 VM-side GCP 인증이므로 CLI 로그인 해결책으로 사용하지 않는다.
- 전역 공식 스킬은 `~/.codex/skills/colab-operator/SKILL.md`에 설치돼 있다. 원격 런타임 운영 시 적용한다.
- 현재 기본 장치는 L4다. 실제 GPU 가용성·잔액·CU/시간은 할당 전후 `usage`로 다시 확인한다. 문서의 단가는 과거 관측값이다.
- 항상 명시적인 세션 이름을 사용한다. 작업용 VM은 기본 한 개만 운영하고, 다른 작업/사용자의 세션은 종료하지 않는다.
- 할당 요청이 timeout/중단되면 `sessions`로 서버 상태를 확인한 뒤 재시도한다. 불확실한 할당을 중복 요청하지 않는다.
- 성공·실패 시 결과와 로그를 회수하고 **해당 작업에서 생성한 VM을 종료**한다. 종료 후 활성 세션과 CU 소모율을 확인한다.
- `/content`는 임시 저장소다. 장기 본실험은 체크포인트를 주기적으로 VM 밖으로 회수하고 해시를 검증하는 절차를 마련한 뒤 실행한다. VM 내부 저장/복원 검사만으로 장애 후 복구가 보장된다고 표현하지 않는다.

```bash
cd /Users/givemethatsewon/git/automotive-recsys
uv sync --locked --group colab
scripts/colab.sh usage
scripts/colab.sh sessions
```

## 입력 데이터와 평가 조건

- `data`는 `/Users/givemethatsewon/Downloads/processed`를 가리키는 로컬 링크다. 전달받은 파일을 수정·덮어쓰지 않는다. 실행별 SHA-256을 보관한다.
- 모델 학습 기준 데이터는 전달된 **전처리 5-core**다. 전체 train 4,818,321개, 사용자 599,742명, 상품 248,186개였다. 실제 실행에서 manifest/audit로 재확인한다.
- 제공된 날짜 분할을 그대로 사용한다. README의 일 단위 날짜로 자정 경계를 새로 만들지 않는다. 분할 후 train에 다시 5-core를 적용하지 않는다.
- 리뷰 유무를 binary interaction으로 사용한다. 평점 예측이나 실제 구매 전체에 대한 관측으로 해석하지 않는다.
- **train만 학습**, **valid만으로 설정·best checkpoint·early stopping 선택**, 설정 고정 후 test 평가한다. 현재 합의는 train+valid 재학습을 사용하지 않는 것이다.
- 그래프·음성 제외 목록·인기도·사용자 활동 그룹은 train으로만 만든다. 음성 추출에 valid/test 정답을 읽어 넣지 않는다.
- 검증 마스킹은 `seen_for_valid`, 테스트 마스킹은 `seen_for_test`를 사용한다. 후자는 train+valid의 과거 상품 제외이며, valid로 모델을 학습한다는 뜻이 아니다.
- 모든 모델을 같은 사용자·정답·전체 학습 상품 후보로 평가한다. sampled-negative 평가 점수와 섞지 않는다.
- warm-start 평가임을 밝힌다. 제공 통계에서 test 상호작용 604,665개 중 333,797개(55.2%)가 남았다. 이 비율은 사용자 유지율이 아니다.
- `item_map.popular`를 공통 사용한다. 당시 train 리뷰 11개 이상을 인기군으로 묶었고, 동점 포함으로 train 상호작용의 81.48%를 차지했다. test로 경계를 다시 정하지 않는다.
- 사용자 그룹은 `user_map.group`의 train 이력 1–5개 / 6개 이상이다. 이력이 0인 cold-start 집단에 대한 결과로 설명하지 않는다.

## 검증된 GPU 최적화: 보존할 조건

코드: `src/automotive_recsys/gpu_kernels.py`, `src/automotive_recsys/train_gpu.py`.
근거·프로파일·실측 수치는 [GPU_REVIEW](docs/GPU_REVIEW.md)에 있다.

1. **데이터 상주:** 그래프·train ID·음성 제외용 정렬 key는 실행 시작 때 GPU에 올려 재사용한다. 매 update마다 큰 데이터/그래프를 CPU↔GPU로 왕복시키지 않는다.
2. **CSR:** 고정된 사용자–상품 정규화 그래프는 CSR로 보관한다. 밀집 사용자×상품/노드×노드 행렬을 만들지 않는다.
3. **대칭 역전파 재사용:** `_FixedSymmetricMM`은 `normalized_graph`가 만든 **고정·대칭** `D^-1/2 A D^-1/2`에만 적용한다. `Aᵀ=A`이므로 backward에 같은 CSR을 사용해 전치·정렬을 생략한다. directed graph, trainable edge weights, 그래프 변경을 도입하면 이 가정을 재검토하고 native autograd와 손실·기울기를 다시 대조한다.
4. **fused Adam:** CUDA의 `torch.optim.Adam(..., fused=True)`와 `zero_grad(set_to_none=True)`를 사용한다. optimizer 종류를 바꾸면 단순한 구현 최적화가 아니라 학습 recipe 변경으로 기록한다.
5. **레이어 누적:** LightGCN은 초기 embedding을 포함한 0..L층 평균, XSimGCL은 1..L층 평균을 유지한다. 누적 합산으로 별도의 큰 layer stack 복제를 피한다. 평균 포함 범위를 바꿔 두 모델을 같게 만들지 않는다.
6. **대조학습:** XSimGCL은 embedding noise·cross-layer InfoNCE를 사용한다. `cross_entropy`와 diagonal log-softmax의 출력/gradient 대조를 유지한다. 경계·정규화·temperature를 임의로 바꾸지 않는다.
7. **음성 추출:** GPU rejection sampling으로 해당 사용자의 train 양성을 제외한다. 미관측 상품을 '싫어한 상품'이라는 사실로 표현하지 않는다.
8. **RNG 분리:** data shuffle/negative RNG와 model noise RNG를 분리한다. 같은 입력·batch·seed의 모델들이 동일한 train 순서와 음성을 보도록 유지한다.
9. **FP32:** 검증한 기본값은 float32, TF32 비활성, AMP 비활성이다. precision 변경은 손실·기울기·추천 순위/valid 정확도와 속도를 비교한 뒤 별도 설정으로 기록한다.
10. **배치 평가:** 임베딩은 GPU에 유지하고 전체 상품 점수를 사용자 배치별로 계산한다. 초기 검증 batch 1,024, 후보 top-100이며 결과 ID만/필요한 점수만 CPU에 가져온다.
11. **동점:** score 내림차순·iid 오름차순을 사용한다. k번째 경계에 걸친 동점까지 처리한다. 동일한 이웃을 가진 XSimGCL 상품은 같은 표현을 가질 수 있다.

현재 XSimGCL은 통제 비교를 위해 다른 두 모델과 같은 raw-embedding squared L2를 사용한다.
저자 저장소와 regularizer가 다른 **recipe variant**임을 문서·manifest에 남긴다.
LightGCN↔XSimGCL 차이를 대조 손실만의 인과 효과로 단정하지 않는다. 이를 분리하려면 같은 XSimGCL encoder/noise에서 CL 가중치 0인 보조 실험을 사용한다.

## 성능·정확성 변경 검증

- 기준 조건은 2026-11-05 L4 / PyTorch 2.11.0+cu130 / dim 64 / batch 8,192 / GNN 3층이다. 이때 LightGCN은 약 3.48배, XSimGCL은 약 2.19배 빨라졌다. 다른 GPU·설정에 그대로 보장하지 않는다.
- 전체 데이터 한 epoch 실측은 BPR-MF 16.52초, LightGCN 54.47초, XSimGCL 114.09초였다. 전체 valid 집계는 약 6–7초였다. 50 epoch 수렴 성능의 증거는 아니다.
- 커널/gradient 변경 시 작은 dense CPU 기준값, CUDA native 경로와 대조하고 실제 전체 그래프에서도 고정한 batch·noise·파라미터로 비교한다.
- 기존 전체 그래프 대조에서 loss 차이는 0, gradient 최대 절대 차이는 LightGCN 4.55e-13, XSimGCL 3.87e-12였다. 이는 해당 검사에 대한 관측값이다.
- 타이밍은 같은 장치·dtype·차원·batch·데이터·warm-up 조건에서 수행한다. CUDA synchronize/event를 사용하며 profiler overhead를 본 타이밍과 분리한다.
- CPU fallback 없이 실제 CUDA kernel이 실행됐는지 profiler로 확인한다. warm-up·측정 횟수·peak allocated·GPU/driver/PyTorch/CUDA 버전을 기록한다.
- 학습 루프/마스킹/checkpoint 변경 시 전체 train 한 epoch + 전체 valid + 저장/복원 후 다음 update 재현을 확인한다. 기존 pilot의 파라미터 차이는 최대 약 6e-8이었다. bitwise 동일성으로 과장하지 않는다.
- 수치 변경과 무관한 문서 수정 때문에 GPU 학습을 다시 돌리지 않는다.

작은 로컬 테스트:

```bash
uv run --locked --extra training --group colab python -m unittest discover -s tests -v
```

원격 성능 점검 또는 실제 한 epoch pilot(각 wrapper는 결과 회수 후 VM 종료):

```bash
scripts/colab.sh new -s automotive-perf --gpu L4
.venv/bin/python scripts/colab_perf.py --session automotive-perf

scripts/colab.sh new -s automotive-pilot --gpu L4
.venv/bin/python scripts/colab_perf.py --session automotive-pilot --pilot
```

## 본실험 기록·해석

- 결과 계약은 [EXPERIMENT_OUTPUTS](docs/EXPERIMENT_OUTPUTS.md)를 따른다. 거기에 있는 미구현 필드는 본실험 전에 수집을 보완한다.
- 실행별 입력 hash, 코드 revision/dirty diff, 실제 config, seed, 환경, best epoch, epoch 곡선, 사용자별 지표, top-100 ID/score, 시간·메모리·CU, checkpoint/log 경로를 남긴다.
- 설정 탐색과 seed 반복을 구분한다. seed 42·43·44를 각각 보고하고 평균·표준편차를 계산한다. 결정적인 Popularity를 가짜 seed 반복 관측치로 늘리지 않는다.
- 사용자별 paired 비교와 seed 간 변동을 구분한다. seed×사용자 행을 모두 독립 관측치로 취급해 신뢰구간을 좁히지 않는다.
- `best.pt`는 valid로 고른 checkpoint, `last.pt`는 재시작용이다. test 점수로 epoch·seed·모델·재정렬 계수를 선택하지 않는다.
- 작은 pilot·성능 benchmark·실패/중단 실행은 최종 모델 성능표에서 분리한다. 가설과 반대인 결과도 보존한다.
- 50 epoch는 시작 예산이다. valid 수렴 곡선에 근거해 조정하며, 수렴했다고 미리 단정하지 않는다.
- `evaluate_run.py`의 독립 test evaluator, seed 집계/paired bootstrap, 추천 score·item exposure·Tail Recall, 불변 세대 기반 Drive 회수를 구현했다. 실제 운용 상태는 [DURABLE_EXPERIMENTS](docs/DURABLE_EXPERIMENTS.md)와 Drive `progress.json`에서 확인한다. 구현 완료와 실제 본실험 완료를 구분한다.
- 차종·상품 텍스트는 아직 제공 폴더에 없다. 이 정보가 필요 없는 모델 비교를 먼저 수행할 수 있다. 특징 파일이 오면 원래 ID로 join하고 출처·시점·unknown 처리를 기록한다.
- 사용자 차종 추출에 valid/test 리뷰를 사용하지 않는다. unknown을 호환/불일치로 단정하지 않고 coverage와 함께 보고한다.
- 차종 가점에 쓴 정보로 측정한 mismatch 감소는 독립적인 효과 검증이 아니다. 같은 사용자 집합에서 NDCG 변화와 추출 품질을 함께 확인한다.
- 데이터·사용자별 추천·checkpoint·실행 token·개인 자료는 Git에 넣지 않는다. 버전 관리에는 코드·설정·문서와 필요한 집계표만 포함한다.

## 자동 본실험과 진행 로그

- [DURABLE_EXPERIMENTS](docs/DURABLE_EXPERIMENTS.md)의 인증·예산·32개 작업·저장 계약을 따른다. 본 모델은 3 seeds, Popularity는 1회다. 최대 100 epoch와 valid early stopping은 수렴 보장이 아니다.
- Drive OAuth는 `~/.config/automotive-recsys/drive/gcloud/`에 별도로 보관한다. 기존 Colab 로그인과 분리하며 `drive.file`을 사용한다. credential은 source/data bundle·Drive·log·notebook에 넣지 않는다.
- `scripts/experiment.py prepare`로 입력/source snapshot과 plan을 확인하고 `scripts/colab_campaign.py --background`로 제어한다. 실제 Drive upload/download/hash·checkpoint 재개 확인 후 본실험을 시작한다.
- 학습 이벤트는 JSONL, epoch/최종 표는 CSV, 진행 상황은 `progress.txt`/JSON으로 남긴다. progress는 모델/variant/seed/epoch/step/best valid/마지막 검증된 Drive 저장 시각과 완료 작업 수를 반드시 포함한다.
- 체크포인트 이름과 resume artifacts는 내용 hash가 붙은 불변 파일이다. 전송 확인 후 `restore.json`을 갱신하고 두 세대가 참조하는 파일은 삭제하지 않는다. 삭제는 이 캠페인의 manifest가 가리키고 hash가 일치하는 생성 파일에만 수행한다.
- CU 잔액/소모율을 worker도 확인하고 완료/실패/중단 시 자기 VM을 반납한다. 계정 감소 100 CU 또는 잔액 30 CU가 초기 중단 기준이다. 다른 세션의 비용과 합산될 수 있음을 명시한다.
- 단일 학습 작업 감시용 Codex 자동화를 만들지 않는다. LLM이 수시간 로그를 반복해서 읽지 않도록 worker와 가벼운 controller가 저장·복구·보고서를 담당한다. 맥이 꺼져도 원격 실행은 계속되지만 VM 재할당에는 켜진 controller가 필요하다.
- 살아 있는 worker를 WebSocket 단절만으로 중복 실행하지 않는다. 자동 재할당은 **저장된 진척이 없는 연속 worker 최대 6회**로 제한하며 다른 세션은 종료하지 않는다. 완료 실행 수나 검증된 checkpoint epoch가 전진하면 연속 실패 한도를 다시 계산한다. 계산 예산 100 CU/잔액 30 CU는 그대로 유지한다. completed marker는 plan hash를 대조한 뒤 재사용한다.
- 2026-11-05 총 VM 6개 한도에 도달해 유효한 진척이 있어도 멈춘 문제를 확인했다. controller의 일시적 연결 오류는 backoff 후 재시도하며, 영구 오류/진척 없는 재할당 한도에 도달하면 `controller_status.json`과 가능한 Drive progress에 실제 중단 상태를 기록한다. 기존 모델 소스/input/선택 plan은 바꾸지 않고 제어기 revision을 별도로 기록한다.
