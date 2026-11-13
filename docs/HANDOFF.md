# 팀원 인수인계: 결과에서 코드와 로그까지

점검일: **2026-11-13**. 로컬 보관 파일을 대조했다. 원격 Git·Drive 권한, 서버와 Colab의 오늘 실시간 상태는 조회하지 않았다. 문서를 읽거나 보고서를 쓰기 위해 GPU를 할당할 필요는 없다.

## 1. 문서의 역할과 읽는 순서

**현재 연구 상태·확정 결정·다음 작업의 SSOT는 [PROJECT_BRIEF](../PROJECT_BRIEF.md)다.** 이 문서는 기술 인수인계와 근거 위치를 안내한다. README는 진입점이며 새로운 수치를 따로 관리하지 않는다.

|순서|읽을 문서|얻는 내용|
|---|---|---|
|1|[PROJECT_BRIEF](../PROJECT_BRIEF.md)|과제 목표, RQ1–4의 완료 범위, 역할·평가 계약·다음 작업|
|2|[본실험 최종 결과](RESULTS_FINAL.md)|32개 완료, 모델/3-seed/CL 제거/paired CI|
|3|[차종 후속 결과 색인](EXPERIMENT_INDEX.md)|첫 파일럿·다른 모델·seed43/44·후보500의 실제 결과와 서로 다른 조건|
|4|[후보 병목 진단](CANDIDATE_DIAGNOSTIC_2026-11-11.md)|같은 가점 단위로 비교한 valid 후보100/500/1000|
|5|[GPU 검증](GPU_REVIEW.md), [운영 복구](CONTROLLER_RELIABILITY.md)|최적화 수식·실측과 운영 변경·복구 증거|
|6|[원문 출처](sources/README.md), [회의 결정](MEETING_NOTES.md)|제안서·회의·후속 지시를 결과와 구분|

개념이 낯설면 [오프라인 설명서](explainers/README.md)를 먼저 본다. 설명서의 실측 범위는 기본 모델 본실험이다. 차종 후속 결과까지 포함한 최신 상태표로 사용하지 않는다.

## 2. 실험을 어떻게 연결해 읽나

```text
제공 processed 5-core → train만 학습 → valid로 설정/best 선택 → test
                                             ↓
                        기존 checkpoint의 후보 ID·점수 저장
                                             ↓
차종 A1/A2 + 상품 fitment → 원래 ID로 join → 일치 가점으로 재정렬
                                             ↓
                      후보 크기100/500/1000 병목 진단(valid만)
```

차종을 넣었다는 것은 재학습이나 차종 노드를 추가했다는 뜻이 아니다. 사용자 노드와 상품 노드 사이에 train 리뷰 링크를 둔 기본 그래프를 학습한 뒤, 별도 차종 표로 순위를 조정했다. BPR-MF도 사용자·상품 벡터를 쓰며 그래프 모델 역시 벡터를 쓴다. 차이는 그래프 전파 여부다.

A1은 train 리뷰 차종, A2는 train 리뷰 상품에서 동일 차량을 지지하는 서로 다른 상품 2개 이상으로 추론한 차종이다. `user_id→uid`, `parent_asin→iid`로 연결한다. train-only timestamp를 검사하며 연식·새 별칭 규칙은 적용하지 않았다. unknown/범용에 가점·감점은 없다. 여러 차량 중 하나라도 일치하면 가점이다.

원래 점수 + β × 사용자별 후보 점수 범위 × 차종 일치가 초기 파일럿 식이다. 후보가 커지면 범위도 달라져 β의 효과가 함께 바뀐다. 후보 진단에서는 **top100 점수 범위를 공통 단위로 고정**했다. 이 두 종류의 숫자를 섞어 비교하지 않는다.

## 3. 한 수치를 로그까지 추적하는 예

기본 모델 `final_lightgcn_seed43`을 예로 든다.

1. [all_seed_results.csv](results/final/all_seed_results.csv)에서 run과 split/group을 확인한다. 주 표는 3-seed 평균이며 이 행은 한 seed다.
2. 프로젝트 소유자의 로컬 `results/campaigns/automotive-20261104T204146Z/review/final_lightgcn_seed43.json`에서 `summary.settings`, `best`, `artifacts`를 확인한다. best는 checkpoint ID/SHA, artifacts는 원래 상대 경로·Drive ID/SHA를 가리킨다. 파일 전체에 개인 경로나 provider 식별자가 있으므로 문서에 그대로 붙여 넣지 않는다.
3. 원격 캠페인 Drive에서 해당 artifact를 받아 SHA-256을 대조한다. `manifest.json`은 코드·입력·실제 설정, `epochs.csv/jsonl`은 loss·valid·best 변화, `test/metrics.json`은 집계, `test/per_user.parquet`는 사용자별 값이다.
4. `test/recommendations.npz`의 uid/iid/score는 후보 재정렬 입력이다. 전체 train 상품을 채점한 top100이며 sampled-negative 평가가 아니다.
5. 동일 seed paired CI는 [paired_differences.csv](results/final/paired_differences.csv)에서 찾는다. 사용자 bootstrap CI와 학습 seed SD를 구분한다.

차종 seed43은 `docs/results/vehicle-pilot-seeds-2026-11-11/final_lightgcn_seed43/`의 design → valid_sweep.csv → frozen_selection → test_results → summary 순서로 읽는다. 원본 valid sweep은 Git 제외 로컬 output에 있고, 집계 CSV는 해당 docs 폴더의 보관 파일을 확인한다. [색인 CSV](handoff/vehicle_results_index.csv)는 run별 split·family·β·CI·원본 위치를 연결한다.

**메타데이터 정정:** 43/44 차종 결과의 `design.seed=42`, 일부 단일 seed 설명은 생성기의 고정 문자열 오류다. 실제 baseline `settings.seed`와 추천 입력 SHA로 구분한다. 원본 JSON을 수정하지 않았으며 정정값은 색인 CSV에 있다. 최초 A1+A2 주 조건과 후속 A2 주 조건을 섞어 3-seed 평균을 내지 않는다. 생성기의 고정 seed 문자열은 아직 코드에 남아 있으므로 다음 실행 전에 실제 baseline seed를 기록하도록 보완해야 한다. 이번 작업에서는 과거 실행 소스/결과를 수정하지 않았다.

## 4. 저장소에 있는 것과 별도 전달이 필요한 것

|위치|내용|새 팀원의 접근 방법|
|---|---|---|
|`docs/results/final/`|본실험 집계 CSV·그림·완료/검증 JSON|해당 파일이 포함된 checkout만으로 읽기 가능|
|`docs/results/vehicle-pilot*/`, `docs/results/candidate-diagnostic*/`|차종·후보 집계와 선택/검증 기록|색인에서 정확한 family·pool·split 확인|
|`src/`, `scripts/`, `tests/`, `uv.lock`|구현·제어기·검사·의존성|새 환경에서 uv로 준비|
|로컬 `data` 링크|제공 processed 입력|개인 Downloads 링크이므로 팀원은 별도 데이터를 받아 `--data-dir` 지정|
|로컬 `results/vehicle-pilot-20261111/inputs/`|차종 입력 parquet 3개|팀 차량 데이터 Drive에서 같은 버전 확보, manifest SHA 대조|
|로컬 `results/campaigns/.../review/`|완료 run manifest와 artifact 위치|실험 소유자가 별도 안전한 경로로 제공하거나 Drive 완료 기록으로 확인|
|로컬 `results/vehicle-pilot*-20261111/`|controller.log, worker.py, session.ipynb, output, 회수 archive|Git 제외. 원본 진단/재현이 필요할 때 접근 권한을 별도 마련|
|로컬 `results/candidate-diagnostic-20261111/`|후보 사용자별 개수·로그·archive|개별 uid가 포함돼 Git에 넣지 않음|
|[캠페인 Drive](https://drive.google.com/drive/folders/1IBQzFgM2ZDR-zz4mgwk6BjIf6c9pRwmx)|본실험 대형 checkpoint·원본 지표·후보·이벤트·복구 세대|공유 권한 필요. 이번 점검에서 팀원 권한은 확인하지 않음|
|[팀 자료 Drive](https://drive.google.com/drive/folders/1WaZBTxYqEdEozh2FIZQQIGz8O1IVCkKt), [차종 폴더](https://drive.google.com/drive/folders/1mYYDiornIreYYatRbICYNKmlPFhjPYRn)|전처리 담당 자료·차종 입력|최신 파일명·hash·버전을 manifest와 비교|

차종 폴더의 원문과 provenance는 [입력 출처 metadata](sources/vehicles-readme-2026-11-11.metadata.json)와 실행 manifest를 우선한다. 원본 로그는 모두 Drive에 업로드됐다고 보장할 수 없다. 특히 차종 후속 로그는 로컬 회수 경로까지만 확인했다.

## 5. 로그 종류와 읽는 방법

|파일/기록|뜻|주의|
|---|---|---|
|`events.jsonl`, `progress.json/txt`|실행·epoch·저장 이벤트와 최신 진행|완료는 plan hash에 대응하는 completion marker와 함께 확인|
|`epochs.csv`, `epochs.jsonl`|epoch loss·valid·시간·best 여부|valid 미평가 epoch는 null/빈칸이며 0이 아님|
|`controller.log`, `auto-*.log`|할당·전송·연결·복구|CLI 오류와 실제 worker 실패를 구분|
|`workers/*`, `operational_io`, readiness/start receipt|제어기·I/O adapter·환경·시작 증거|CLI BUSY/IDLE나 exit0만으로 시작 성공을 판단하지 않음|
|`frozen_selection.json`|valid로 선택한 설정|test 점수로 선택하지 않았는지 확인|
|`best.pt`, `last.pt`, `restore.json`|선택 모델·재시작 모델·불변 세대 참조|파일명만으로 같다고 판단하지 않고 SHA 대조|
|`usage_before/after`, `sessions_after`|당시 계정 비용·할당 종료|오늘 실시간 상태나 모델별 정확 비용이 아님|

필요한 이벤트만 추려 보고 credential·사용자별 추천·환경 전체를 출력하지 않는다. 인증은 팀원 본인의 계정으로 준비하며 소유자의 `~/.config/automotive-recsys`를 저장소/공유 archive에 넣지 않는다.

## 6. 최적화와 코드 변경의 구분

|구분|주요 코드|변경·보존 조건|증거|
|---|---|---|---|
|기본 학습/평가|[gpu_kernels.py](../src/automotive_recsys/gpu_kernels.py), [train_gpu.py](../src/automotive_recsys/train_gpu.py), [evaluate_run.py](../src/automotive_recsys/evaluate_run.py)|BPR, 그래프 전파, 독립 test, train/valid 분리|[GPU 검증](GPU_REVIEW.md), [최종 결과](RESULTS_FINAL.md)|
|CUDA 성능|동일 커널/학습기|GPU 데이터 상주, CSR, 고정 대칭 backward 재사용, fused Adam, 층 누적, FP32/TF32 off, 분리 RNG, 동점 처리|L4/torch2.11에서 LightGCN3층 3.48배·XSimGCL3층 2.19배. 전체 그래프 loss차이0, gradient최대차이 4.55e-13/3.87e-12|
|Recipe 차이|동일 커널|XSimGCL은 1..L 평균, LightGCN은 0..L 평균. 공통 raw embedding L2는 저자 구현과 다른 recipe variant|[AGENTS](../AGENTS.md), CL=0 보조. 두 모델 차이를 CL 하나의 효과로 해석하지 않음|
|영구 저장·평가 기록|[campaign_io.py](../src/automotive_recsys/campaign_io.py), [drive_store.py](../src/automotive_recsys/drive_store.py)|hash 기반 불변 세대, 두 세대 보존, 외부 복구, loss 분해·추천 score·exposure|[저장 계약](DURABLE_EXPERIMENTS.md)|
|운영 안정화|[colab_campaign.py](../scripts/colab_campaign.py), [operational_io.py](../scripts/operational_io.py), [colab_campaign_env.py](../scripts/colab_campaign_env.py)|일시 오류 backoff, 무진척 재할당 상한, Ubuntu systemd, 환경 격리, 시작 receipt. I/O 600초·64MiB|[복구 기록](CONTROLLER_RELIABILITY.md). 학습 수식 최적화와 구분|
|차종 재정렬|[vehicle_rerank.py](../src/automotive_recsys/vehicle_rerank.py), [vehicle_pilot.py](../scripts/vehicle_pilot.py), [wrapper](../scripts/colab_vehicle_pilot.py)|train-only join, unknown, β valid선택·test동결, paired CI|[파일럿](VEHICLE_PILOT_2026-11-11.md), [후속 색인](EXPERIMENT_INDEX.md)|
|후보 진단|[candidate_pool.py](../scripts/candidate_pool.py), [candidate_diagnostic.py](../scripts/candidate_diagnostic.py)|같은 best로 full-catalog 후보 재계산. 진단은 valid만·top100 단위고정|[진단](CANDIDATE_DIAGNOSTIC_2026-11-11.md). float32 근접 동점 차이는 receipt에 기록|

속도 수치는 특정 장치·설정의 관측이며 최종 정확도 개선 배수가 아니다. directed/trainable 그래프로 바꾸면 대칭 backward 가정을 재검증한다. 수식/precision 변경은 CPU·native CUDA 손실/gradient 및 전체 그래프를 대조해야 한다.

### Git과 실행 소스 연결

본실험 frozen source는 `060f5a312a15f9a4469452bb6637b09ec792a4b6`이며 이후 제어기/I/O revision은 worker manifest로 별도 추적한다. 오늘 checkout HEAD가 당시 학습 소스라는 뜻은 아니다.

|commit|의미|
|---|---|
|`ff3954a`|초기 추천 실험·GPU 경로|
|`05a46f7`, `dedd7f7`|영구 본실험·평가 산출물·Drive 외부 복구 확인|
|`08e5efa`, `8057dba`, `20fd7fc`|진척 기반 복구·독립 제어기·일시 오류 재시도|
|`aad7ad3`, `0b9c5e2`, `64b547e`|Colab 환경 격리·준비 receipt·제어기 실제 장애 복구 검증|
|`534a726`|당시 결과 요약. 제목의 partial은 과거 상태이며 최종32개 상태는 completion을 확인|
|`dcea9b3`, `d8a9e5c`, `96b1337`|원문 인수인계·오프라인 결과 설명서|
|차종·후보·이번 인수인계 변경|이번 점검 시작 시 미커밋/미추적. 실행 source SHA와 현재 source SHA를 구분|

`git log --oneline -- <파일>`로 범위를 좁혀 읽는다. [이번 점검 JSON](handoff/audit.json)은 현재 소스 SHA와 원본 사본 대조 결과를 보관한다. SHA 일치는 수학적 정확성이나 원격 게시를 증명하지 않는다.

## 7. 새 팀원의 최소 실행 절차

읽기만 할 때는 README/Markdown/CSV/오프라인 HTML이면 충분하다. Python 검사는 데이터·CUDA 없이 작은 fixture로 실행한다.

```bash
uv sync --locked --extra training --group colab
uv run --locked --extra training --group colab python -m unittest discover -s tests -v
```

입력과 개인 원본 로그까지 받은 소유자 환경에서만 인수인계 대조를 다시 생성한다. audit 스크립트는 provider 호출·학습 없이 작은 JSON/CSV·hash를 확인하며 사용자별 Parquet는 읽지 않는다.

```bash
.venv/bin/python scripts/audit_handoff.py
```

다른 컴퓨터에서 audit의 로컬 원본 경로가 없으면 실패할 수 있다. 저장된 audit.json과 Git에 포함할 집계표를 읽는 데 원본 경로는 필요하지 않다. 실제 학습·전체 평가를 다시 수행해야 할 때만 [Colab 운영](COLAB.md)·[저장 절차](DURABLE_EXPERIMENTS.md)를 따르고 새 이름의 task VM을 사용한다.

## 8. 남은 일과 갱신 규칙

다음 실험은 후보500/1000에서 기본 / 인기 감점 / 차종 가점 / 인기 감점+차종 가점을 동일 사용자로 비교하는 것이다. valid NDCG≥기본95% 조건에서 APLT·Tail Recall과 coverage를 보고 후보·λ/β를 동결한다. 독립 추출 검수·Spark 원본 분석·텍스트 특징·공식 과제 공지 확보는 별도 의존성이다. 재학습을 수행했다는 표현은 쓰지 않는다.

새 결과가 나오면 (1) 원본·hash·split·run·family 보존 → (2) 실험별 결과 문서/색인 추가 → (3) PROJECT_BRIEF의 RQ 상태·다음 일 갱신 → (4) README 링크와 인수인계 점검 갱신 순서다. 과거 스냅샷·원문 JSON은 덮어쓰지 않는다. 현재 상태 수치를 여러 안내 문서에 독립적으로 복제하지 않는다.

문서·코드가 미커밋이면 팀원의 원격 clone에 보이지 않는다. 데이터·로그·credential을 제외하고 scoped diff를 검토한 후 팀의 Git 공유 절차로 전달해야 한다. 사용자의 후속 원격 반영 요청에 따라 코드·문서·집계를 커밋/푸시 대상으로 준비했다. 실제 게시 revision은 이 문서가 포함된 Git 이력을 따른다. 팀원 메시지 전송이나 Drive 권한 변경은 수행하지 않는다.
