# 차종 v2 후속 실행 계획

2026-11-17 반영. **실행 상태:** 재정렬 비교는 시작했고 [초기 진행 검증](VEHICLES_V2_RUN_2026-11-17.md)을 보관했다. 아래는 실행 전 계획이며 학습 단계 실험은 아직 시작하지 않았다. 근거는 [11/13 회의 결정](MEETING_2026-11-13.md)과 [v2 인수인계 사본](sources/vehicles-v2-handoff-2026-11-17.md)이다. **설계·준비 단계이며 v2 평가·재학습을 실행한 기록은 아니다.** 기본 캠페인 32개와 v1 결과는 완료 상태로 보존한다.

## 비교 순서

1. **수령 검증:** v2의 valid용 A1/A2·fitment와 `vehicle_fit.py`를 별도 실행 폴더에 보관한다. 원본을 덮어쓰지 않고 Drive ID·수정 시각·bytes·로컬 SHA-256, 입력 schema/행 수·ID join coverage·null/중복·`last_seen < min(valid.timestamp)`를 검사한다. v1과 같은 processed hash를 사용한다. 데이터 전체 검사는 Colab에서 한다.
2. **클리닝 효과 분리:** BPR-MF·LightGCN·XSimGCL seed42의 같은 best checkpoint·사용자·정답·후보를 유지한다. v1/v2 입력을 **같은 기존 make/model matcher, 연식 제외, train-only 프로필**로 비교한다. v2의 연식·별칭·segment matcher로 바꾸는 비교는 별도 행으로 둔다. 입력과 코드 규칙을 동시에 바꾼 결과를 클리닝 하나의 인과 효과로 설명하지 않는다.
3. **재정렬 진단:** 후보100/500/1000은 모델당 같은 전체 상품 점수의 상위 목록에서 얻는다. β 단위는 사용자별 top100 점수 범위로 고정한다. 기본 / β / λ / λ+β를 같은 사용자에서 비교한다. λ는 train 인기도만 사용하며 정의·grid를 실행 전에 manifest에 고정한다. valid NDCG 유지율≥95%인 설정 중 APLT·Tail Recall과 정확도의 trade-off를 보고 후보·계수를 동결한다. γ는 텍스트 특징을 받은 뒤 별도 비교한다.
4. **원인 분석:** 전체·차종 known/unknown·활동1–5/6+에서 NDCG, Recall, APLT, ARP, Tail Recall, coverage를 낸다. 후보 내 차종 일치/비인기+일치/정답의 개수, 실제 top10 변경 사용자 비율과 가점 분포를 함께 확인한다. Mismatch는 분모(전체 top10 / 판정 가능 쌍), 사용자 macro/쌍 가중 집계, unknown 비율을 명시한다. 같은 추출 규칙으로 계산한 Mismatch 감소를 독립적인 적합성 증거로 쓰지 않는다.
5. **학습 단계 확장:** 회의에서 합의한 별도 작업이다. 현재 학습기는 v2 특징을 사용하지 않으므로 기존 명령을 그대로 재실행하는 것으로 대체하지 않는다. 모델 구조를 유지하는 특징 loss인지, 차종 노드·엣지인지, train 데이터 정제인지 먼저 수식·입력·대조군을 설계하고 작은 fixture로 검증한다. 모델 구조 변경/recipe variant를 기록하고 **새 campaign/run**으로 CUDA pilot → valid 확인 → seed 반복을 진행한다. directed/trainable graph를 도입하면 기존 고정 대칭 CSR backward 가정도 재검증한다. 원래 32개 캠페인을 resume하지 않는다.
6. **동결 후 test:** test 정답·test용 차종 파일은 선택에 사용하지 않는다. v1↔v2 주 비교는 train-only 프로필을 유지한다. `_test` 파일은 valid 기간 정보까지 포함하는 별도 시나리오로만 사전 선언하고, `last_seen < min(test.timestamp)`를 검증한다. 학습에는 train만 사용한다. 같은 조건으로 seed42/43/44를 맞춘 뒤 평균·SD와 seed별 사용자 paired CI를 구분한다. 이미 test를 본 뒤 추가한 v2 분석이라는 탐색적 성격도 보고한다.

위 순서는 후속 설계안이다. 후보 수·λ/β grid·차종 학습 방식이 회의에서 모두 확정됐다는 뜻이 아니다. 신빈의 top100·두 그래프 모델 중심 제안과 기존 후보 확대 진단을 함께 고려하되 valid만으로 결정한다.

## 전달 자료 해석에서 지킬 조건

- `VehicleFit`은 v1의 exact make/model matcher와 같지 않다. Dodge/Ram·BMW·E-Series 정규화, 모델 prefix, segment 및 연식을 사용한다. 양쪽 중 모델/연식이 없으면 해당 비교를 생략하는 규칙도 manifest에 남긴다. 범용 상품이 matching이면 +1을 받을 수 있고 matching이 아니면 unknown인 동작과 기존 범용 zero-bonus 정책의 차이를 코드 검사로 확인해야 한다.
- README의 `515,201` valid 구매·`134,809` known 사용자 예시를 warm-start valid `220,516`명의 직접 기준선으로 복사하지 않는다. 같은 warm-start 정답·사용자·상위K 계약으로 실제 관측 리뷰 적합성 기준선을 다시 계산한다. 리뷰를 실제 구매 전체로 부르지 않는다.
- v3 검수는 Claude + 일부 사람 판정이다. 상품 완전 정답과 차종 정밀도는 서로 다른 정의다. 사용자 v2와 v3는 판정자·표본이 다르므로 78→89%를 통제된 개선량으로 쓰지 않는다. ‘남은 오류가 모델 결과에 영향 없음’은 제공자의 판단이며 이번 모델 실험에서 입증된 사실이 아니다.
- 입력 시점 통제와 별개로 v2 규칙 개발에는 test 적합성 분석이 포함됐다. 새로운 test 평가를 완전히 독립적인 holdout 검증으로 과장하지 않는다. 상품 메타데이터의 과거 관측 시점도 미확인 상태다.
- 후보를 전처리 담당에게 전달할 때는 사용자별 결과를 Git에 넣지 않는다. 텍스트 특징은 후보 uid/iid·split·원래 ID·입력 hash를 연결하는 별도 전달 계약이 필요하다. 팀원 메시지는 별도 요청 없이 보내지 않는다.

## 실행·회수 계약

무거운 실행은 프로젝트 `scripts/colab.sh`와 Colab CUDA를 사용한다. 할당 전 현재 `usage`/`sessions`, 예산·잔액을 확인하고 명시적인 새 task session을 하나만 사용한다. 기존 문서의 CU 단가·종료 상태는 과거 기록이다. 장기 학습을 시작하기 전에 외부 checkpoint 회수·hash·재개를 확인하고, 결과/로그/manifest/추천 score·사용자별 지표를 회수한 뒤 task VM을 반납한다. 새 수치가 생기면 실험별 문서·색인 → PROJECT_BRIEF → README 순으로 갱신한다.
