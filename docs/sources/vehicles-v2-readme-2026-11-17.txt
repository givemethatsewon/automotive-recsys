차종 데이터 (사용자 차종 · 상품 적합 차종)
================================================================

리뷰와 상품명에서 뽑은 "사용자가 타는 차"와 "상품이 맞는 차".
re-ranking의 β(차종 일치 가점) 항과 Mismatch@10(차에 안 맞는 부품 비율) 계산에 사용.

** 2026-11-15 최종본은 vehicles/v2/ 폴더에 있음 (이 README도 v2/ 안의 것이 최신). **
   code/vehicle_fit.py는 vehicles/v2/가 있으면 자동으로 그 안의 파일을 읽음 (VEHICLES_VERSION 환경변수로 변경 가능).
   v2/ 밖 vehicles/에 있는 같은 이름 파일은 예전 버전이니 쓰지 말 것.
   v2/ 구성
     (바로 아래)   parquet 5개(B가 쓰는 것), README.txt, handoff_to_B.md, *_stats.json, vehicle_source_comparison.csv
     review/       정확도 검수 샘플과 판정 (v3, v3_claude, v2 오류 분류)
     dictionary/   차종 사전 (aliases.csv, vocab_draft.csv, makes_draft.csv). 추출 재현용
     evidence/     사용자 차종 근거 리뷰 표현 (user_vehicle_mentions*.parquet, 검토용, 큼)
     code/         이 버전을 만든 코드. vehicle_fit.py는 여기서 import (sys.path에 vehicles/v2/code 추가)

B에게 필요한 파일은 두 개:
  user_vehicles.parquet     사용자 차종
  product_fitment.parquet   상품 적합 차종
둘 다 원래 ID(user_id, parent_asin)로 되어 있으므로 processed/의 user_map, item_map으로
번호(uid, iid)와 연결해서 쓸 것.


1. 만드는 순서 (코드는 모두 code/ 폴더, Colab은 code/vehicles_colab.ipynb)
----------------------------------------------------------------
 1) fetch_vehicle_list.py       NHTSA(미국 교통안전청) 공개 차종 목록 받기 -> vpic_*.csv
 2) explore_vehicle_mentions.py 리뷰·상품명에서 연식 바로 뒤에 나오는 단어 빈도 (Spark) -> explore/
 3) (사람) aliases.csv           별칭 표 직접 작성 (chevy, vw, jk, miata ...)
 4) build_vehicle_vocab.py      공식 목록 + 별칭을 실제 빈도로 다듬어 차종 사전 -> vocab_draft.csv, makes_draft.csv
 5) extract_product_fitment.py  상품명 -> 상품 적합 차종 (Spark) -> product_fitment.parquet
 6) extract_user_vehicles.py    학습 기간 리뷰 -> 사용자 차종 (Spark) -> user_vehicles.parquet
 공통 함수: vehicle_match.py (5, 6에서 사용. 같은 폴더에 있어야 함)
 처리 시간 (로컬 Spark 14코어): 2) 약 6분, 5) 3.8분, 6) 6분


2. 결과 파일
----------------------------------------------------------------
 user_vehicles.parquet  (500,473줄, 2026-11-13 기준)  사용자 한 명이 차 여러 대면 여러 줄
    user_id       str    원래 사용자 ID
    make          str    제조사 (예: Honda, Ford, Harley-Davidson)
    model         str    모델 (예: Civic, F-150). 비어 있으면 제조사만 아는 경우 (11.4%)
    year          float  연식. 2026-11-13부터 언급된 연식마다 한 줄 (같은 모델 여러 대). 비어 있을 수 있음
    year_mentions float  그 연식이 언급된 횟수 (2026-11-13 추가)
    n_mentions    int    리뷰에서 언급된 횟수
    best_pattern  str    가장 확실한 근거 표현 종류 (아래 3절 표)
    segment       str    car / powersports / car|powersports

 product_fitment.parquet  (1,667,868줄, 2026-11-13 기준)  상품 하나가 여러 차에 맞으면 여러 줄
    parent_asin   str    원래 상품 ID
    make          str    제조사
    model         str    모델. 비어 있으면 "for Honda"처럼 제조사만 적힌 상품 (19.6%)
    segment       str    car / powersports / car|powersports
    matched       str    상품명에서 실제로 잡힌 표현 (검토용)
    n_makes       int    상품명에 나온 제조사 수
    universal     bool   범용 상품 ("universal", "most cars" 같은 표현이 있거나 서로 다른 제조사 계열 3개 이상 나열.
                         GM 계열 Tahoe/Yukon/Escalade처럼 한 계열 안의 나열은 범용 아님, 10절).
                         차종 일치 판단에서 사용자 차와 안 맞아도 "안 맞음"이 아니라 "모름"으로 처리
    year_from     float  적합 연식 시작. 비어 있을 수 있음
    year_to       float  적합 연식 끝. 비어 있으면 연식 정보가 없거나 "2013+"처럼 끝이 열린 범위
    open_end      bool   "2013+", "2018-Current"처럼 끝이 열린 범위면 True
    year_scope    str    "vehicle": 차종별로 붙인 연식 / "item": 상품명 전체의 연식 범위를 공유 (9절)

 user_vehicle_mentions.parquet  (989,479줄)  사용자 차종의 근거가 된 리뷰 표현 (검토용)
    user_id, timestamp, span(잡힌 표현), context(앞뒤 60자), make, model, segment, year, pattern

 user_vehicles_stats.json, product_fitment_stats.json   크기·비율 요약

 정확도 확인용 샘플 (사람이 Y/N 기록)
    review_product_fitment_v3.csv    상품 100개   <- 정확도 측정은 v3만 사용 (2026-11-13, 10절 수정 후 결과)
    review_user_vehicles_v3.csv      사용자 차종 근거 100개   (code/make_review_samples.py로 생성,
                                     v1·v2 샘플의 상품·사용자는 제외)
    review_product_fitment_v3_claude.csv, review_user_vehicles_v3_claude.csv
                                     위 v3의 판정 결과 (11절). 최종 판정은 final_vehicle / final_year,
                                     claude_correct 열
    review_product_fitment_v2.csv    (tuning) 10절 수정에 씀. 사람 판정 19개
    review_product_fitment_v2_claude_check.csv  위 v2의 Claude 1차 확인 (나머지 81개 포함)
    review_user_vehicles_v2.csv      (tuning) 10절 수정에 씀. 100개 판정 완료
    review_user_vehicles_v2_with_error_type.csv  위 v2 + 줄별 오류 분류(error_type)
    review_product_fitment.csv, review_user_vehicles.csv  (v1, tuning) 11-10 수정 전 샘플
    tuning_sample_product_fitment.csv  규칙 다듬는 데 쓴 샘플 (정확도 계산에 쓰면 안 됨)
    규칙을 다시 고치면 v3도 tuning 샘플이 되므로 새 샘플(v4)을 뽑아서 다시 잴 것.

 검수 결과 기록
  사용자 차종 v2 (2026-11-13 판정, 10절 수정 '전' 버전의 정확도. 이 판정을 보고 규칙을 고쳤으므로
                  v2는 이제 tuning 샘플이고, 수정 후 정확도는 review_user_vehicles_v3.csv로 잴 것)
    판정 그대로 (정보가 덜 채워져도 N)         Y 78 / 100 = 78%  (95% CI 69~85%)
    차종 정밀도 (엉뚱한 차를 넣었나)           98 / 100 = 98%  (93~99%)
        틀림 2: Yamaha WaveRunner(수상스키), 상품 설명의 "deville eldorado seville" 나열을 옮겨 적은 리뷰
    연식 정확도 (연식이 붙은 69줄)              약 66 / 69 = 96%  (88~99%)
    누락 (리뷰에 있는데 못 뽑음)                약 14 / 100
        모델 뒤 연식 6 ("silverado 2001"), 두 자리 연식 3 ("'04 gmc canyon"),
        모델 못 잡음 5 (is250, econoline은 사전에 없음 / exedition, cienna 오타 / nissan fuga 미국 미판매)
    줄마다 분류: review_user_vehicles_v2_with_error_type.csv의 error_type 열


3. 추출 규칙
----------------------------------------------------------------
 공통
  - 띄어쓰기·하이픈 차이는 무시 (F-150 = F150 = F 150, CR-V = CRV). 긴 표현 우선
    ("grand cherokee"가 "cherokee"보다 먼저).
  - 일반 단어와 겹치는 모델(fit, focus, escape, volt, bolt ...)과 2글자 이하·숫자 모델(is, a4, 1500)은
    제조사가 함께 있어야 인정 (vocab_draft.csv의 flags=needs_context).
  - 여러 제조사에 있는 모델(Ranger: Ford/Polaris, Outlander: Mitsubishi/Can-Am ...)은
    제조사가 함께 있어야 확정 (flags=ambiguous).

 상품 적합 차종 (상품명)
  - 상품명에 그 모델의 제조사가 있으면 확정.
  - 제조사 없이 모델만 있으면: 일반 단어·애매한 모델이 아니고, 상품 카테고리와 차 종류가 맞고
    (Motorcycle & Powersports 상품 <-> 오토바이 모델), 그 모델명이 상품명들에서 자기 제조사와
    함께 나오는 비율이 30% 이상일 때만 인정. 상품명 맨 앞 단어(대개 상품 브랜드)는 제외.
    -> Continental 타이어, Yukon Gear, Spectra Premium 같은 브랜드명 오탐 방지
  - ram, mini, smart 같은 일반 단어 제조사는 제조사만으로는 인정하지 않음 ("RAM Mounts", "mini fuse").
  - 자동차가 아닌 장비의 부품은 적합 차종에서 뺌: 선박 엔진(outboard, mercruiser), 발전기·소형 엔진
    (Honda GX160, EU2000i, Kohler, Briggs), 잔디깎이·정원 장비, 농기계(트랙터, John Deere, Kubota).
    단 Lexus GX460/470, 클래식카 "generator"(옛날 차의 발전기), "tractor trailer"(대형 트럭)는 남김.
  - 연식: "2012-2015", "2011-13", "2013+", "2018-Current", 두 자리 범위 "99-04", 나열 "2004 05 06",
    "2015/ON" 등 처리 (10절).
    한 상품명에 차가 여러 대이고 연식 범위도 여러 개면 차종마다 가까운 범위를 붙임 (9절, 2026-11-13).
    범위가 하나뿐이거나 제조사만 있는 줄은 상품 단위 범위를 공유.

 사용자 차종 (리뷰)
  - **학습 기간 리뷰만 사용** (첫 검증 리뷰 시각 2021-08-11 00:57 이전).
    검증·테스트 기간 리뷰를 쓰면 정답 상품의 리뷰("내 Civic에 잘 맞음")가 새어 들어가
    β의 효과가 부풀려지므로.
  - 대상: 5-core 학습 사용자(599,742명). 단 원본 리뷰 전체(5-core에 없는 상품 리뷰 포함)에서 찾음.
  - 차종 표현 바로 옆에 근거가 있어야 인정:
        pattern           예시                  비율   조건
        year_make_model   "2015 honda civic"    36%   모델이 그 제조사 것
        year_model        "2015 civic"          26%   애매한 모델·2글자 이하·숫자 모델 제외
                                                      (Jeep jk/tj/jl 같은 세대 코드는 예외)
        make_model        "honda civic"         17%   모델이 그 제조사 것
        my_model          "my civic"             8%   위 조건 + 일반 단어 모델 제외
        my_make           "my honda"             7%   일반 단어 제조사(ram, mini ...) 제외
        year_make         "2015 honda"           6%
        (비율은 검증 시점 사용자 차종 줄 기준, 2026-11-13)
    "the fit is great"처럼 모델명만 단독으로 나온 것은 버림.
  - 표현 바로 뒤에 outboard, generator, mower, tractor 등이 오면 자동차가 아니라 버림
    ("my honda eu2200i generator", "my 2013 yamaha 70 hp outboard").
  - 같은 사용자·차는 한 줄로 합침. 같은 제조사의 모델을 아는 사용자는 '제조사만' 줄을 뺌.


4. 크기
----------------------------------------------------------------
 상품 (전체 2,003,129개 / 5-core 248,186개)
 (2026-11-13 10절 수정 후 기준)
                       전체      5-core
   차종 정보 있음       45.6%     39.4%
   모델까지 있음        34.4%     28.9%
   연식까지 있음        27.3%     21.2%  (52,682개)
   범용 상품으로 표시    23,175개

 사용자 (학습 사용자 599,742명, 검증 시점 user_vehicles.parquet 기준)
   차종 정보 있음       49.4%  (296,156명, 1인당 평균 1.64대)
   모델까지 있음        44.8%
   연식까지 있음        37.9%
   평가 대상 테스트 사용자 158,805명 중 47.9% (기록 1~5개 그룹 34.2%, 6개 이상 64.7%)
   평가 대상 검증 사용자 220,516명 중 48.1% (1~5개 34.8%, 6개 이상 63.3%)
   (테스트 시점 파일 user_vehicles_test.parquet은 학습 사용자의 53.2%. 테스트 평가에는 이쪽을 씀, 7절)

 차 종류: 사용자 자동차 88% / 오토바이·ATV 11%, 상품 자동차 84% / 오토바이·ATV 13%

 실제 테스트 구매와의 일치율 (차종 품질 지표, code/vehicle_fit.py, vehicle_source_comparison.csv 첫 줄)
   테스트 기간에 실제로 산 상품 중 판정 가능한 47,567건에서 사용자 차와 맞는 비율 (A1, 검증 시점 정보)
     제조사 + 모델 + 연식   32.0%
     제조사만               44.7%
     (비교) 랜덤 상품        4.7%
   -> 차종 정보는 랜덤보다 약 7배 잘 맞지만, 실제 구매의 2/3는 "안 맞음"으로 나옴.
      대부분 사용자가 차를 바꿨거나, 차가 여러 대이거나, 다른 사람 차의 부품을 산 경우
      (학습 기간 리뷰로 뽑은 차종을 테스트 기간에 그대로 쓰는 한계). Mismatch@10을 해석할 때
      실제 구매의 Mismatch를 기준선으로 함께 볼 것.


5. 쓸 때 주의할 점
----------------------------------------------------------------
 - 모델 이름이 공식 목록 기준이라 같은 차가 다른 이름일 수 있음:
     "Dodge Ram"(옛 표기) vs "Ram 1500"(2010년 이후 Ram 브랜드)
     "Wrangler" vs "Wrangler JK" (jk/tj/jl 별칭은 Wrangler로 합쳐져 있음)
     BMW는 "328i"처럼 세부 모델과 "3 Series"가 섞여 있음
   -> 차종 일치를 판단할 때 제조사 단위 일치와 모델 단위 일치를 나눠서 보는 것을 추천.
      모델 이름을 더 합칠지는 A와 B가 함께 정할 것.
 - model이 비어 있는 줄(제조사만 아는 경우)은 제조사 단위로만 비교 가능.
 - Chevrolet/GMC가 함께 쓰는 모델(C/K 픽업)은 두 제조사로 각각 한 줄씩 들어 있음.
 - 리뷰에 언급된 차가 꼭 본인 차는 아닐 수 있음 ("아내의 Tacoma", "아들 차").
   사용자당 여러 대를 모두 남겼으므로 "그중 하나라도 맞으면 일치"로 보는 것을 추천.
 - 엄밀히는 자동차가 아닌 것이 일부 섞여 있음 (Yamaha 선박 엔진, Dodge 모터홈, Honda 스쿠터).
   대부분 제조사 단위로만 잡혀서 영향은 작음.
   사용자 차종 규모 확인 (2026-11-10, user_vehicle_mentions_test.parquet):
     앞뒤 문맥에 비자동차 단어(mower, tractor, outboard, hp, generator, golf cart, boat ...)가 있는 언급 0.2%
     근거 언급이 전부 그런 문맥인 차종 줄 1,266 / 574,635줄 (0.22%, 상한값), 그중 55%는 제조사만 있는 줄
     표본을 보면 상당수는 진짜 자동차 ("generator into the back of my 2004 tundra", "1969 corvette 350/300hp")
     실제 비자동차는 Mercury·Yamaha 선외기, Husqvarna 예초기, E-Z-GO·Club Car 골프 카트 등 약 0.1% 수준
   -> β·Mismatch@10에 영향을 줄 규모가 아니고, 문맥 규칙을 넓히면 진짜 차까지 빠질 위험이 더 커서
      사용자 쪽 비자동차 필터는 더 고치지 않음. 보고서에는 "비자동차 언급 혼입 0.2% 이하"로 적을 것.
      (상품 쪽은 extract_product_fitment.py의 NON_VEHICLE_RE로 이미 제외하고 있음)
 - Mismatch@10은 이 차종 정보로 계산하므로, β를 키우면 당연히 낮아짐 (제안서 4장).
   차종 정보의 실제 효과는 정확도(NDCG@10)가 오르는지로 판단할 것.


6. 불러오기 예시
----------------------------------------------------------------
 import pandas as pd
 P = "/content/drive/MyDrive/빅데이터 팀플"
 uv = pd.read_parquet(f"{P}/vehicles/user_vehicles.parquet")
 pf = pd.read_parquet(f"{P}/vehicles/product_fitment.parquet")
 um = pd.read_parquet(f"{P}/processed/user_map.parquet")
 im = pd.read_parquet(f"{P}/processed/item_map.parquet")
 uv = uv.merge(um[["user_id", "uid"]], on="user_id")          # 학습 사용자만 남음
 pf = pf.merge(im[["parent_asin", "iid"]], on="parent_asin")  # 5-core 상품만 남음

7. 평가 시점별 사용자 차종과 A2(구매 상품 기반), 출처 비교  (2026-11-09 추가)
----------------------------------------------------------------
 차종은 "평가하는 시점 이전 정보"만 써야 함 (정답 누출 방지). 검증 기간은 테스트보다 과거라서
 테스트 평가에는 검증 기간 정보까지 써도 됨. 그래서 파일이 두 벌:
     검증 평가·튜닝용 (2021-08-11 이전 정보)     테스트 평가용 (2022-07-16 이전 정보)
   A1 user_vehicles.parquet                     user_vehicles_test.parquet
   A2 user_vehicles_items.parquet               user_vehicles_items_test.parquet
   (A1 근거: user_vehicle_mentions.parquet / user_vehicle_mentions_test.parquet)
 ** 하이퍼파라미터(β 등)를 고를 때는 반드시 검증용 파일을 쓸 것 **

 A1 (리뷰 본문, extract_user_vehicles.py --until valid|test)
   열 추가: first_seen, last_seen (그 차를 처음/마지막으로 언급한 리뷰 시각, 밀리초)
 A2 (리뷰를 남긴 상품의 적합 차종, infer_vehicles_from_items.py --until valid|test)
   user_id, make, model, year(비움), n_items(그 차에 맞는 서로 다른 상품 수), segment, last_seen
   범용 상품은 쓰지 않음. n_items >= 2 처럼 기준을 걸어서 씀.

 불러오기: code/vehicle_fit.py
   from vehicle_fit import VehicleFit, load_vehicles
   uv = load_vehicles("both", until="test", min_items=2)     # "text"(A1) / "items"(A2) / "both"
   vf = VehicleFit(user_vehicles=uv)                          # level="make"면 제조사 단위로만 비교

 출처 비교 (compare_vehicle_sources.py -> vehicle_source_comparison.csv)
   실제 테스트 구매 중 판단 가능한 쌍에서 사용자 차와 맞는 비율. lift = 맞는 비율 / 랜덤 상품의 맞는 비율
   (2026-11-13 10절 수정 후 다시 계산한 값. 처음(11-09) 값은 backup_20261113/README_before_numfix.txt)
                                      테스트 사용자   맞는 비율          랜덤   lift
                                      커버리지       모델    제조사
   A1 학습 기간 정보 (예전 방식)          47.9%      32.0%   44.7%     4.7%    6.8
   A1 테스트 시점 정보                    54.1%      36.7%   49.3%     5.1%    7.2
   A1 테스트 시점, 최근 2년 언급만        31.0%      37.3%   48.8%     4.0%    9.3
   A1 테스트 시점, 가장 최근 차만         54.1%      28.1%   38.3%     3.1%    9.1
   A2 상품 2개 이상                       41.3%      47.7%   52.7%     9.2%    5.2
   A1 + A2(2개 이상)                      67.5%      46.2%   53.8%     8.3%    5.6
   -> 평가 시점 기준 정보로 바꾸면 커버리지와 일치율이 모두 오름.
      최근 언급만 남기는 건 일치율은 거의 그대로인데 커버리지만 크게 줄어서 쓰지 않는 것을 추천.
      가장 최근 차 한 대만 남기면 오히려 나빠짐 (차를 여러 대 가진 경우가 많음).
      A1 + A2를 합치면 테스트 사용자의 67.5%가 차종을 갖고 일치율도 높음 -> β 실험의 기본 설정으로 추천.


8. 상품 적합 차종 오탐 수정  (2026-11-10)
----------------------------------------------------------------
 품질 점검에서 일반 단어와 겹치는 모델명의 오탐이 확인되어 extract_product_fitment.py를 고치고
 product_fitment.parquet, user_vehicles_items(_test).parquet, vehicle_source_comparison.csv를 다시 만듦.
 이전 파일은 backup_20261110/에 있음. A1(user_vehicles*.parquet)은 바뀌지 않음.

 고친 것
  - Fit, Spark, Bolt와 Dodge Ram: 상품명에 제조사가 '어딘가' 있는 것으로는 부족하고, 제조사가 모델 바로
    앞쪽(8단어 이내)에 있어야 인정. "custom fit", "fit for", "bolt kit", "spark plug" 같은 일반 표현은 제외.
    ("Fit for 2008-2011 Honda Accord"가 Honda Fit으로 잡히던 문제. 원본 상품명 대조로 Fit 약 80%,
     Bolt 약 88%, Spark 약 65%가 오탐이었음)
  - 매칭 전에 지우는 표현: British Columbia, Sierra International(선박 부품 브랜드), Ram Air, Spark Plug,
    Bolt Kit/On, "x4 clips" 같은 수량 표현, Volvo Truck/VNL(대형 트럭)
  - 범용 상품 표현 추가: "for Jeep, Trailer, RV, SUV, Camper"처럼 차 종류를 3개 이상 나열

 결과 (이전 -> 이후)
   전체 줄 수            1,677,591 -> 1,668,927
   Honda Fit                 8,850 -> 1,821
   Chevrolet Bolt            1,101 -> 242
   Chevrolet Spark           1,211 -> 561
   International(제조사만)    2,771 -> 1,821
   Volvo(제조사만)            3,577 -> 2,997
   범용 상품                57,656 -> 58,912개
   5-core 상품 차종 정보 있음 39.5% (변화 없음), 모델까지 29.0% -> 28.9%
   실제 테스트 구매와 맞는 비율(A1+A2): 47.2% -> 47.4%, lift 6.3 -> 6.4
 모델이 빠진 상품도 상품명에 제조사가 있으면 '제조사만' 줄로 남음 (예: "Spark Plug for Chevrolet ...")

 로컬 실행 메모 (Windows): JAVA_HOME을 Temurin JDK로, 호스트 이름에 '_'가 있으면
   SPARK_LOCAL_HOSTNAME=localhost 를 지정해야 Spark가 뜸.


9. 상품 적합 차종의 차종별 연식  (2026-11-13)
----------------------------------------------------------------
 문제: 상품명에 차가 여러 대면 모든 연식 범위를 합친 최솟값~최댓값을 모든 차가 공유했음.
   "2006-2008 Lincoln Mark LT 2004-2008 Ford F-150" -> 둘 다 2004-2008 (Mark LT가 2년 넓음)
   합친 범위는 실제보다 넓기만 하므로 "안 맞는 연식이 맞음으로" 판정되는 쪽으로만 틀림.
 고친 것 (extract_product_fitment.py assign_vehicle_years):
   연식 범위가 2개 이상이고 모델도 2개 이상인 상품은 차종마다 위치상 가까운 범위를 붙임.
     연식이 앞에 오는 상품명 ("2007-2011 Dodge Nitro, 2008-2012 Jeep Liberty") -> 바로 앞 범위
     연식이 뒤에 오는 상품명 ("Buick Enclave 2008-2012, Traverse 2009-12")     -> 바로 뒤 범위
     범위가 안 붙은 차는 읽는 방향으로 가장 가까운 범위 ("2004-14 Forester/Impreza" -> 둘 다 2004-2014)
   제조사만 있는 줄, 범위가 하나뿐인 상품은 그대로. 새 열 year_scope로 구분.
 결과: 줄 수·차종은 그대로 (1,668,927줄), 연식만 바뀜
   차종별 연식이 붙은 상품 46,855개 / 147,804줄 (5-core 상품의 줄 14,309개)
     범위가 좁아짐 110,448줄, 그대로 37,278줄, 넓어짐 78줄
     넓어진 78줄은 이전 코드가 "2010+"의 시작 연식을 무시하던 것을 바로잡은 경우가 대부분
   실제 테스트 구매와 맞는 비율: A1 39.0% -> 38.9%, A1+A2 47.4% -> 47.4% (사실상 변화 없음)
     A2는 사용자 연식이 없어서 영향 없음, A1도 연식까지 알고 여러 차·여러 범위 상품을 산 경우가 적음
     -> β·Mismatch@10 결과도 거의 안 바뀔 것. 다만 연식 판정이 더 정확해짐
 남은 한계: 앞·뒤 방식이 섞인 상품명 ("2011-2018 Commander 1000 2010+ / Maverick 1000 2013+")과
   같은 모델명이 여러 번 나오는 상품명 ("2014+ RZR XP 1000 / 2016-2020 RZR S 1000")은 틀릴 수 있음.
 user_vehicles_items(_test).parquet(A2)는 연식을 쓰지 않아서 다시 만들지 않음.
 이전 파일은 backup_20261113/에 있음.


10. 검수(v2) 결과를 반영한 정제  (2026-11-13)
----------------------------------------------------------------
 사용자 v2(100개 판정)·상품 v2(사람 19개 + Claude 1차 확인 81개, review_product_fitment_v2_claude_check.csv)를
 보고 고침. 9절의 "A2는 다시 만들지 않음"은 이 절로 대체됨: 사전이 바뀌어 모든 파일을 다시 만듦.
 이전 파일은 backup_20261113/에 있음 (product_fitment는 9절 전 11-10 버전).

 사전 (aliases.csv, build_vehicle_vocab.py)
  - 추가: Lexus 숫자 표기(IS250, ES350, GX470 ...), Econoline(Ford E-Series), Epica, Verona, MX-6, MX-3, CR-X,
          FX45, FX50, EX37, GLK, NV, CJ-8, RM125/250/85, V Star
  - aliases.csv의 별칭은 빈도 기준(MIN_COUNT)과 상관없이 남김 (드문 모델이 빠지던 문제)
  - 영문·숫자만 남긴 표기 기준으로 중복 제거 ("town & country" = "town country")
  - vehicle_fit.py: "E-Series" = E-150/250/350/450 (Super Duty와 같은 방식)

 상품 적합 차종 (extract_product_fitment.py)
  - 연식 계산을 pandas로 옮기고 표기 추가: 두 자리 범위 "05-11"·"99-04", 나열 "2004 05 06"·"2007 2008 ... 2014",
    열린 범위 "2015/ON"·"2006-later", 거꾸로 "2022-14"·"2016-2011"
  - 두 자리 범위는 모델 찾기 전에 지움 ("86-98" -> Toyota 86 오탐)
  - 제조사 없이 나온 모델은, 상품명의 제조사가 모두 다른 한 계열(MAKE_GROUP)이면 버림
    ("Mercedes Sprinter ... E250" -> Ford E-250, "Freightliner ... Century" -> Buick Century, "BMW ... G20" -> Infiniti G20)
  - 범용 판정을 제조사 수 -> 계열 수로 ("Tahoe/Yukon/Escalade 전용 부품"이 제조사 3개라 범용이 되던 문제)
  - Onan 발전기 부품 제외
 사용자 차종 (extract_user_vehicles.py)
  - 모델 뒤 연식 ("chevy silverado 2001", "honda civic lx 2010"). 차종 인정 근거로는 안 쓰고 연식 값만 채움
  - 두 자리 연식 ("'04 gmc canyon", "98 dodge ram", 제조사 바로 앞일 때만)
  - 연식 나열·범위 뒤 연식은 비움 ("1998 1999 2000 cadillac", "fits 2015-2020 f150")
    -> 제조사·my 근거 없이 연식만으로 인정되던 상품 설명 인용("fits 2015-2020 f150")은 빠짐
  - 연식은 표현 맨 앞([my] 다음)에서만 찾음: "son's 2000 f-150"이 Honda S2000으로, "s2000"의 2000이 연식으로
    잡히던 문제 해결
  - 같은 모델의 연식을 모두 남김 (year_mentions 열), "waver runner" 비자동차 처리

 결과
                                     이전        이후
   상품: 차종 있음 (5-core)           39.5%       39.4%
         연식 있음 (전체 상품)        472,575     546,175개
         연식 있음 (5-core 상품)      46,741      52,682개
         범용 상품                    58,912      23,175개
         차종별 연식이 붙은 상품          0       84,776개
   사용자(학습, 검증 시점): 차종 있음   49.07%      49.38%
                          모델 있음   44.66%      44.83%
                          연식 있음   34.97%      37.93%
                          언급 중 연식 있음  63.2% -> 69.5%
   A2 (테스트 시점, 상품 2개 이상) 커버리지  37.9% -> 41.3%
   A1+A2 테스트 사용자 커버리지              65.9% -> 67.5%

 실제 테스트 구매와 맞는 비율 (vehicle_source_comparison.csv): A1 39.0% -> 36.7%, A1+A2 47.4% -> 46.2%,
   lift 6.4 -> 5.6.  나빠진 것이 아니라 판정 대상이 늘어난 것 (A1, 테스트 시점으로 요인 분해):
     사용자 차종만 새것                       39.0 -> 38.9   (영향 없음)
     상품도 새것, 범용 기준은 이전대로         -> 38.6   (연식 판정이 엄격해짐. 연식 빼면 42.3 그대로)
     범용 기준까지 새것                        -> 36.7   (범용에서 빠진 상품 약 2,700쌍이 '모름'에서 판정 대상으로)
   무작위 상품의 일치율도 7.4 -> 8.3%로 올라 lift가 내려감. Mismatch@10의 실제 구매 기준선도 다시 계산할 것.

 정확도: 수정 후 버전은 v3로 잴 것 (review_product_fitment_v3.csv, review_user_vehicles_v3.csv, 각 100개).
   v2는 수정에 쓴 tuning 샘플. 사용자 v2의 줄별 오류 분류는 review_user_vehicles_v2_with_error_type.csv.
 고치지 않은 것: 오타(exedition, cienna), 제조사 없는 일반 단어 모델(Canyon, Suburban, Avenger),
   "for Jeep for Grand for Cherokee" 같은 특이 표기, 상품 브랜드가 제조사로 잡히는 경우(Yamalube, BMW Motorrad 보조배터리)


11. 최종 검수 (v3) 결과와 알려진 한계  (2026-11-15)
----------------------------------------------------------------
 10절 수정 후 버전의 정확도. v3 샘플은 규칙 수정에 쓰지 않았으므로 이 숫자를 보고서에 씀.
 판정: Claude(Sonnet)가 100개씩 판정. 상품명만으로 차종이 불분명한 상품 21개는 웹 검색으로 실제 적합 차종 확인.
       사람(A)이 상품 9개를 직접 판정했고, 그 줄은 사람 판정을 우선함. 보고서에는 "Claude 판정 + 일부 사람 확인"으로 적을 것.
 판정 기준: 추출기는 상품명·리뷰 문장만 보므로 그 안의 정보 기준으로 판단.
   상품명에 모델이 없으면 제조사만 뽑아도 정답 (예: "Genuine Honda 71116-SNE-A50 Spoiler" -> Honda).
   모델 단위 병합은 정답 (Accord Coupe/Sedan -> Accord, Sierra 1500/2500HD -> Sierra).

 상품 적합 차종 (review_product_fitment_v3_claude.csv, 100개)
   차종 완전 정답        Y 77 / partial 11 / N 12      Y 77%  (95% CI 68~84%)
   엉뚱한 차를 넣은 상품  7 / 100  -> 정밀도 약 93%
       Arctic White 도료 -> Arctic Cat (2), BMW 의류(바지), RC 드론 "X Type" -> Jaguar,
       LS 엔진 코드 "LR4" -> Land Rover, Yamaha 수상오토바이 커버, 범용 스로틀 컨트롤러(범용 표시돼 있어 판정엔 영향 없음)
   나머지 오류는 누락: 나열된 모델 일부만 잡음 (V-Strom/SV, Mercedes ML/GL/R, Nissan D21, Can-Am Sport 400 ...)
   연식                  Y 92 / N 8                   92%  (85~96%)
       잘린 범위 (Sierra HD의 2014, 15-18 2500HD의 2018, 1955-1971 -> 1971), 나열 "03 04 05" 누락

 사용자 차종 (review_user_vehicles_v3_claude.csv, 100개)
   정답                  89 / 100  (95% CI 81~94%)
   차종 정밀도            93 / 100  (86~97%)
       엉뚱한 차 2: 상품 호환 목록을 옮겨 적은 리뷰(Fiat 500), "2007 fj cruiser엔 안 맞음"(본인 차 아님)
       모델이 문맥에 있는데 제조사만 뽑음 5: vw bug, kia forte/sorento, honda crf100, volvo s60r, honda trx250ex
   연식                  연식이 붙은 69줄 중 68줄 정답 (98.6%)
       샘플이 '언급' 단위라 연식이 빠져 보인 4줄 중 2줄(2016.5 cx5, 2002 crv)은 같은 사용자의 다른 언급에서
       연식을 잡아 user_vehicles.parquet에는 연식이 들어 있음. 진짜 누락은 연식 나열(의도적으로 비움)과 "c10 79".
   (v2 78%와는 판정자가 달라 직접 비교하지 말 것)

 알려진 한계 (전체 데이터에서 센 규모. 재추출하지 않기로 함, 2026-11-15)
   오탐 유형                                 전체 상품   5-core 상품
     색 이름 "Arctic White" 등 -> Arctic Cat      565        76
     LS 엔진 코드 "LR4" -> Land Rover             213        62
     "X Type" -> Jaguar (Jaguar 언급 없음)         32         4
     브랜드 의류·굿즈(티셔츠, 바지, 모자)       약 4,900   약 360
   -> 합쳐도 차종 정보가 있는 5-core 상품 97,887개의 약 0.5%. 11-13 수정(훨씬 큰 변경)도 실제 구매 일치율을
      0.1%p밖에 바꾸지 않았으므로 β·Mismatch@10 결과에 영향은 무시할 수준으로 판단.
   그 밖의 한계
     - 상품명에 모델이 없는 순정 부품(부품번호만 있는 상품)은 제조사 단위로만 앎. 웹 검색으로 보면 실제로는
       특정 차종 전용인 경우가 많음 (예: Chrysler 점화코일 -> Town & Country, Caravan, Wrangler)
     - 오토바이·ATV 모델 코드(crf, trx 등)가 사전에 부족해 제조사만 잡히는 경우가 있음 (커버리지 문제, 정밀도 영향 없음)
     - 나열된 모델 일부 누락, 한 상품 안의 세부 모델별 연식 범위 차이는 합쳐짐
     - 같은 리뷰가 여러 상품에 중복으로 들어 있어 n_mentions가 부풀려진 경우가 있음 ("하나라도 맞으면 일치"
       판단에는 영향 없음)
     - 사용자 쪽 비자동차 혼입 0.2% 이하 (5절)
   다시 고친다면: v3가 tuning 샘플이 되므로 새 샘플(v4)로 정확도를 다시 잴 것.
