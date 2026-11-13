# Automotive 추천 실험

Amazon Reviews 2023 Automotive의 processed 5-core에서 **정확도를 유지하면서 사용자 차종에 맞는 비인기 상품을 더 추천할 수 있는지** 연구한다. 모델은 Popularity + BPR-MF + LightGCN + XSimGCL이다.

## 처음 보는 팀원은 여기부터

1. **[PROJECT_BRIEF — 현재 상태 SSOT](PROJECT_BRIEF.md)**: RQ1–4, 역할, 확정 평가 조건, 완료 범위와 다음 작업.
2. **[HANDOFF — 팀원 인수인계](docs/HANDOFF.md)**: 결과 한 행에서 실행 ID·manifest·epoch 로그·checkpoint까지 추적하는 방법, 코드/최적화 변경, 외부 파일 준비와 재현 절차.
3. [기본 모델 최종 결과](docs/RESULTS_FINAL.md) → [차종 후속 실행 색인](docs/EXPERIMENT_INDEX.md) → [후보 병목 진단](docs/CANDIDATE_DIAGNOSTIC_2026-11-11.md).
4. [쉽게 읽는 오프라인 설명서](docs/explainers/README.md), [원문 출처](docs/sources/README.md), [회의 결정](docs/MEETING_NOTES.md).

## 현재 상태를 읽는 법

11/13 로컬 보관 파일 점검 기준으로 기본 모델 32개 학습 작업, 차종 재정렬의 다른 모델/seed 확장, 후보100/500/1000 valid 진단 산출물이 있다. **재정렬은 기존 checkpoint의 후보 순서를 조정한 것이며 차종을 넣어 재학습한 결과가 아니다.** 인기 감점·텍스트 조건을 포함한 RQ3/RQ4 전체는 완료되지 않았다. 상세 상태와 다음 실험은 PROJECT_BRIEF 한 곳에서 관리한다.

차종 후속 결과에는 주 조건 차이와 seed 메타데이터 오류가 있으므로 색인의 정정/제약을 먼저 읽는다. `docs/RESULTS_2026-11-07.md`는 31/32 시점의 과거 기록이고, `docs/results/final/`이 기본 모델 최종 집계다.

## 결과·구현 찾아보기

|찾는 내용|위치|
|---|---|
|기본 모델 평균/SD·그룹·paired CI·곡선|[최종 집계 폴더](docs/results/final/)|
|차종별 run·family·pool·β·CI|[색인 CSV](docs/handoff/vehicle_results_index.csv)|
|후보 내 정답·비인기·차종 일치 및 valid sweep|[후보 진단 집계](docs/results/candidate-diagnostic-2026-11-11/)|
|입력 검사·모델·평가|[src/automotive_recsys](src/automotive_recsys/)|
|GPU 최적화 수식·실측·정확성 검증|[GPU_REVIEW](docs/GPU_REVIEW.md)|
|Drive 저장·로그 계약·운영 복구|[DURABLE_EXPERIMENTS](docs/DURABLE_EXPERIMENTS.md), [CONTROLLER_RELIABILITY](docs/CONTROLLER_RELIABILITY.md)|
|이번 문서·산출물 점검|[audit.json](docs/handoff/audit.json)|

## 읽기와 재현

문서·CSV·오프라인 HTML을 읽는 데 GPU나 실행 환경은 필요하지 않다. 작은 fixture 검사는 아래처럼 실행한다.

```bash
uv sync --locked --extra training --group colab
uv run --locked --extra training --group colab python -m unittest discover -s tests -v
```

실제 입력은 Git 제외다. 소유자 환경의 `data`는 개인 Downloads의 processed를 가리키는 링크이므로 팀원은 입력을 따로 받고 `--data-dir /path/to/processed`를 지정한다. checkpoint·추천 NPZ·사용자별 Parquet·원본 로그의 접근 방법과 hash 검증은 HANDOFF에 있다. 무거운 학습/전체 평가는 Colab CUDA에서 수행한다.

[AGENTS.md](AGENTS.md)는 에이전트 실행 규칙이다. 문서 확인을 위해 완료 캠페인을 재개하거나 새 GPU를 할당하지 않는다. 새 결과가 나오면 실험별 근거를 보존한 뒤 PROJECT_BRIEF와 README 링크를 갱신한다.

**공유 범위:** 코드·문서·집계 결과를 Git으로 전달한다. 원본 데이터·개인 결과·credential은 별도 접근이 필요하며 Git에 넣지 않는다. 원격 반영 revision은 이 문서가 포함된 Git 이력으로 확인한다. 팀원 Drive 권한은 별도 확인이 필요하다.
