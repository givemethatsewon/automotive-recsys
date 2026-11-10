# 우리 실험을 이해하고 설명하기

**[experiment-results-eli5.html](experiment-results-eli5.html)을 브라우저에서 연다.** HTML 하나에 CSS·JavaScript·그래프·실측 수치·MathML을 넣었다. 다른 폴더로 복사해도 학습 내용과 조작은 인터넷 없이 작동한다. 선택적인 프로젝트 원문 링크만 원래 폴더 구조를 사용한다.

## 권장 순서

1. 자동차용품 추천 10칸과 train→valid→설정 동결→test의 역할을 이해한다.
2. 모델 버튼을 바꿔 인기순·취향 벡터·그래프 전파·대조학습의 차이를 본다.
3. 정답 순위와 비인기 개수를 바꿔 NDCG/Recall/APLT 계산을 따라간다. 정답을 놓친 경우와 두 개 정답인 경우도 있다.
4. 실제 지표·사용자 그룹을 바꾸고 정확도와 비인기 노출이 다른 결론을 낼 수 있음을 확인한다.
5. 층 수·CL 켜기/끄기, seed SD와 사용자 paired CI를 구분한다.
6. 퀴즈를 풀고 마지막의 30초/90초 원고·예상 질문으로 발표를 연습한다.

## 근거와 재생성

원본은 [32개 완료 결과](../RESULTS_FINAL.md)와 `docs/results/final/`의 CSV다. 실측은 그대로 쓰고, 상품·그래프·NDCG 연습 예제는 가상임을 표시했다. 학습이나 GPU 작업을 새로 실행하지 않는다.

빌드할 때만 LaTeX→MathML 변환 패키지가 필요하다. 생성한 HTML 실행에는 패키지 설치나 서버가 필요 없다.

```bash
cd ~/git/automotive-recsys
UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-project --python .venv/bin/python \
  --with latex2mathml==3.78.1 scripts/build_results_eli5.py
```

템플릿은 `scripts/templates/results_eli5.html`, `results_eli5.js`, `eli5_reading.css`다. 읽기 스타일은 사용자의 ELI5 스킬 기준을 프로젝트에 보관했다.

## 실제 수행한 확인

- 원본 CSV와 모델 3-seed 평균, 인기순의 고유 상품 14종, XSimGCL의 총 비인기 추천 슬롯 2회 계산을 대조했다.
- NDCG 예제: 정답 3위의 DCG→IDCG→정규화→사용자 평균 다섯 단계, 정답 1위·10위 밖·정답 2개, APLT 0개/10개, 이전·초기화를 확인했다.
- 실제 결과 지표 5개×그룹 3개, 양수·음수·같은 모델의 상대 변화, 층 1/2/3·CL on/off·seed 3개·CI 3개·발표 원고 3종을 확인했다.
- 퀴즈 오답→정답·5문제 완성·초기화와 Enter 키 동작/포커스 유지를 확인했다.
- Aside 브라우저의 1280px·1000px 데스크톱에서 화면과 긴 수식을 확인했다. 가로 넘침, MathML 오류, 수식 명령어의 노출, 관측 구간의 JS 오류가 없었다. 모바일 전용 QA는 하지 않았다.
- 외부 런타임 리소스 태그·미해결 템플릿이 없고 브라우저의 외부 리소스 요청 0개를 확인했다. 문서 CSP도 외부 연결을 금지한다.

검증에 쓴 로컬 서버는 임시 미리보기용이며, 완성 파일을 사용하기 위해 켜 둘 필요가 없다.

## 두 번째 설명서: 10칸 진열대

**[experiment-results-explainer.html](experiment-results-explainer.html)** 은 같은 결과를 다른 순서로 설명한다. 첫 화면의 정확도(NDCG@10)–비인기 노출(APLT@10) 산점도에서 시작해 시험 규칙, 10칸 채점 계산기, Hit rate 점 그림(10,000명), 모델 소개, CL on/off, LightGCN 층 수, seed별 paired CI, 해석 범위 순서로 진행한다. 2026-11-10 claude.ai Artifact(비공개)로도 게시했다.

```bash
cd ~/git/automotive-recsys
UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-project --python .venv/bin/python \
  --with latex2mathml==3.78.1 scripts/build_results_explainer.py            # 오프라인 단일 HTML
# Artifact 게시용 본문만: ... scripts/build_results_explainer.py --fragment <경로>
```

템플릿은 `scripts/templates/results_explainer.html`, `results_explainer.js`다. 빌드는 `all_seed_results.csv`에서 평균/SD를 다시 계산하고 `model_comparison.csv`의 NDCG·APLT 평균과 1e-12 이내로 일치하는지 검사한다. 진열대 예시와 점 배치는 설명용이고 비율만 실측이다.

실제 수행한 확인(2026-11-10): JS 구문 검사, 1280px 데스크톱 headless Chromium 렌더에서 가로 넘침 없음·노출된 LaTeX 없음·JS 오류 없음, "정답 2개 중 1위만" 예시 NDCG 0.6131(=1/1.6309), 그룹 전환과 CL=0 토글 동작. 모바일 폭과 다크 모드는 별도로 확인하지 않았다.
