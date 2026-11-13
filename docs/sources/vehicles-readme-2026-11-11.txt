차종 데이터 (사용자 차종 · 상품 적합 차종)
================================================================

리뷰와 상품명에서 뽑은 "사용자가 타는 차"와 "상품이 맞는 차".
re-ranking의 β(차종 일치 가점) 항과 Mismatch@10(차에 안 맞는 부품 비율) 계산에 사용.

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
 user_vehicles.parquet  (482,827줄)  사용자 한 명이 차 여러 대면 여러 줄
    user_id       str    원래 사용자 ID
    make          str    제조사 (예: Honda, Ford, Harley-Davidson)
    model         str    모델 (예: Civic, F-150). 비어 있으면 제조사만 아는 경우 (11.3%)
    year          float  연식 (가장 많이 언급된 값). 비어 있을 수 있음
    n_mentions    int    리뷰에서 언급된 횟수
    best_pattern  str    가장 확실한 근거 표현 종류 (아래 3절 표)
    segment       str    car / powersports / car|powersports

 product_fitment.parquet  (1,668,927줄)  상품 하나가 여러 차에 맞으면 여러 줄
    parent_asin   str    원래 상품 ID
    make          str    제조사
    model         str    모델. 비어 있으면 "for Honda"처럼 제조사만 적힌 상품 (20.1%)
    segment       str    car / powersports / car|powersports
    matched       str    상품명에서 실제로 잡힌 표현 (검토용)
    n_makes       int    상품명에 나온 제조사 수
    universal     bool   범용 상품 ("universal", "most cars" 같은 표현이 있거나 제조사 3개 이상 나열).
                         차종 일치 판단에서 사용자 차와 안 맞아도 "안 맞음"이 아니라 "모름"으로 처리
    year_from     float  적합 연식 시작. 비어 있을 수 있음
    year_to       float  적합 연식 끝. 비어 있으면 연식 정보가 없거나 "2013+"처럼 끝이 열린 범위
    open_end      bool   "2013+", "2018-Current"처럼 끝이 열린 범위면 True

 user_vehicle_mentions.parquet  (981,086줄)  사용자 차종의 근거가 된 리뷰 표현 (검토용)
    user_id, timestamp, span(잡힌 표현), context(앞뒤 60자), make, model, segment, year, pattern

 user_vehicles_stats.json, product_fitment_stats.json   크기·비율 요약

 정확도 확인용 샘플 (사람이 Y/N 기록)
    review_product_fitment.csv       상품 100개 (규칙 다듬는 데 쓰지 않은 새 샘플)
    review_user_vehicles.csv         사용자 차종 근거 100개
    tuning_sample_product_fitment.csv  규칙 다듬는 데 쓴 샘플 (정확도 계산에 쓰면 안 됨)


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
  - 연식: "2012-2015", "2011-13", "2013+", "2018-Current" 처리. 한 상품명에 차가 여러 대면
    같은 연식 범위를 공유 (단순화). "Mustang 99-04"처럼 두 자리만 있으면 연식은 비움.

 사용자 차종 (리뷰)
  - **학습 기간 리뷰만 사용** (첫 검증 리뷰 시각 2021-08-11 00:57 이전).
    검증·테스트 기간 리뷰를 쓰면 정답 상품의 리뷰("내 Civic에 잘 맞음")가 새어 들어가
    β의 효과가 부풀려지므로.
  - 대상: 5-core 학습 사용자(599,742명). 단 원본 리뷰 전체(5-core에 없는 상품 리뷰 포함)에서 찾음.
  - 차종 표현 바로 옆에 근거가 있어야 인정:
        pattern           예시                  비율   조건
        year_make_model   "2015 honda civic"    32%   모델이 그 제조사 것
        year_model        "2015 civic"          26%   애매한 모델·2글자 이하·숫자 모델 제외
                                                      (Jeep jk/tj/jl 같은 세대 코드는 예외)
        make_model        "honda civic"         21%   모델이 그 제조사 것
        my_model          "my civic"             8%   위 조건 + 일반 단어 모델 제외
        my_make           "my honda"             8%   일반 단어 제조사(ram, mini ...) 제외
        year_make         "2015 honda"           5%
    "the fit is great"처럼 모델명만 단독으로 나온 것은 버림.
  - 표현 바로 뒤에 outboard, generator, mower, tractor 등이 오면 자동차가 아니라 버림
    ("my honda eu2200i generator", "my 2013 yamaha 70 hp outboard").
  - 같은 사용자·차는 한 줄로 합침. 같은 제조사의 모델을 아는 사용자는 '제조사만' 줄을 뺌.


4. 크기
----------------------------------------------------------------
 상품 (전체 2,003,129개 / 5-core 248,186개)
                       전체      5-core
   차종 정보 있음       45.6%     39.5%
   모델까지 있음        34.3%     28.9%
   연식까지 있음        23.6%       -
   범용 상품으로 표시    58,912개

 사용자 (학습 사용자 599,742명)
   차종 정보 있음       49.1%  (294,275명, 1인당 평균 1.64대)
   모델까지 있음        44.7%
   연식까지 있음        35.0%
   평가 대상 테스트 사용자 158,805명 중 47.6% (기록 1~5개 그룹 33.9%, 6개 이상 64.4%)
   평가 대상 검증 사용자 220,516명 중 47.8%

 차 종류: 사용자 자동차 88% / 오토바이·ATV 11%, 상품 자동차 84% / 오토바이·ATV 12%

 실제 테스트 구매와의 일치율 (차종 품질 지표, code/vehicle_fit.py)
   테스트 기간에 실제로 산 상품 중 적합 차종 정보가 있는 44,948건에서 사용자 차와 맞는 비율
     제조사 + 모델 + 연식   34.3%
     제조사만               46.6%
     (비교) 랜덤 상품        약 6%
   -> 차종 정보는 랜덤보다 약 6배 잘 맞지만, 실제 구매의 2/3는 "안 맞음"으로 나옴.
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
                                      테스트 사용자   맞는 비율          랜덤   lift
                                      커버리지       모델    제조사
   A1 학습 기간 정보 (예전 방식)          47.6%      34.3%   46.6%     4.6%    7.5
   A1 테스트 시점 정보                    53.8%      39.0%   51.2%     4.9%    8.0
   A1 테스트 시점, 최근 2년 언급만        30.8%      39.7%   51.0%     4.0%    9.9
   A1 테스트 시점, 가장 최근 차만         53.8%      30.2%   40.4%     3.0%   10.1
   A2 상품 2개 이상                       37.9%      48.8%   53.7%     8.4%    5.8
   A1 + A2(2개 이상)                      65.9%      47.4%   55.1%     7.4%    6.4
   -> 평가 시점 기준 정보로 바꾸면 커버리지와 일치율이 모두 오름.
      최근 언급만 남기는 건 일치율은 거의 그대로인데 커버리지만 크게 줄어서 쓰지 않는 것을 추천.
      가장 최근 차 한 대만 남기면 오히려 나빠짐 (차를 여러 대 가진 경우가 많음).
      A1 + A2를 합치면 테스트 사용자의 66%가 차종을 갖고 일치율도 높음 -> β 실험의 기본 설정으로 추천.


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

