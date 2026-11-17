# 차종 v2 비교 최종 결과

2026-11-17 KST. 세 모델 seed42의 네 차종 조건·후보100/500/1000 비교를 완료했다. 모델 계산 완료는15:28:58, Ubuntu 전체 결과 회수·제어기 정상 종료는15:36:35다. 15:41 실제 provider 재조회에서 활성 할당0개·0 CU/hr를 확인했다. 새 학습이나 기존32개 캠페인 재개는 하지 않았다.

## 주 비교: 데이터 클리닝만 바꾸기

후보500, 동일 checkpoint·전체 사용자·매칭 규칙, v1 valid에서 고른 β를 v1/v2에 동일하게 적용했다. test158,805명 전체 NDCG@10의 상대 변화이며 기본 모델 대비 개선율과 구분한다.

| 모델 | β | v1 NDCG | v2 NDCG | v2/v1 상대 변화 | 차이의 사용자 paired 95% CI |
|---|---:|---:|---:|---:|---|
| bpr_mf | 2 | 0.001221538 | 0.001229506 | +0.652% | [-0.00001454, 0.00003289] |
| lightgcn | 0.5 | 0.003708566 | 0.003711260 | +0.073% | [-0.00001947, 0.00002522] |
| xsimgcl | 0.2 | 0.004303323 | 0.004305938 | +0.061% | [-0.00001031, 0.00001521] |

세 모델 모두 점추정은 양수지만 신뢰구간이0을 포함한다. 이 결과로 클리닝에 따른 정확도 향상을 확정하지 않는다. seed42 한 번의 학습 checkpoint에서 얻은 사용자 paired CI이며 seed 간 변동의 검증이 아니다. test는 이전 연구·규칙 개발에서도 사용한 탐색적 후속 평가다.

평가 사용자 중 차종 프로필 coverage는58.51%→59.96%다. 양쪽 모두 같은 전체158,805명을 평가했고, 프로필이 없는 사용자는 그 조건의 기준 추천을 유지한다. 이 수치를 독립적인 차종 추출 정확도로 해석하지 않는다.

## 저장·회수 검증

- 모델별 valid96행·test30행, 총 test90행. 제공 matcher의 연식 제외/포함 및 후보별 조건도 완료 산출물에 포함한다. 주 표는 사전 고정한 비교만 발췌했다.
- Drive의 세 completed marker와 루트 completion, controller의 results_recovered=true를 확인했다. 호스트로 회수된 집계 JSON/CSV51개를 Drive artifact SHA와 다시 대조했다. 사용자별 추천·대형 배열은 Git에 넣지 않았다.
- 작업 worker는 총2개를 사용했고 최종 제어기는 completed로 정상 종료됐다. 이번 상태 확인에서는 GPU를 새로 할당하지 않았다.
- [완료·회수·자원 검증](results/vehicle-v2-durable-2026-11-17/completion_verification.json), [주 비교 CSV](results/vehicle-v2-durable-2026-11-17/primary_comparison.csv), [저장·복구 검증 이력](VEHICLES_V2_DURABLE_2026-11-17.md).
- [원본 Drive 로그·전체 결과](https://drive.google.com/drive/folders/1GdQ89Q7e7lWfu8gbtS-vp3ajUDS8Fvh8). 서버 회수 경로: `ubuntu-wan:~/git/automotive-recsys-v2-durable-20261117/results/vehicle-v2-durable-r4-20261117/output/`.

차종을 학습 단계에 연결하는 비교와 RQ3의 인기 감점·텍스트 조건은 별도 미완료 범위다. 이번 완료를 팀 과제 전체 완료로 표시하지 않는다.
