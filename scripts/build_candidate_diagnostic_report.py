"""Publish only aggregate candidate audit receipts; per-user data stays private."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/candidate-diagnostic-20261111'
DEST=ROOT/'docs/results/candidate-diagnostic-2026-11-11'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    counts=[];sweep=[];receipts=[]
    labels={'bpr_mf':'BPR-MF','lightgcn':'LightGCN','xsimgcl':'XSimGCL'}
    hashes={}
    for name,label in labels.items():
        folder=RUN/'output'/f'final_{name}_seed42'
        receipt=json.loads((folder/'summary.json').read_text())
        assert receipt['status']=='completed' and not receipt['test_answers_read']
        receipts.append(receipt)
        for filename,target in [('candidate_availability.csv',counts),('valid_fixed_scale_sweep.csv',sweep)]:
            for row in csv.DictReader((folder/filename).open()):target.append({'model':label,**row})
        hashes[name]={}
        for f in folder.iterdir():
            if f.is_file():
                with f.open('rb') as stream: hashes[name][f.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    def save(name,rows):
        with (DEST/name).open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    save('candidate_availability.csv',counts);save('valid_fixed_scale_sweep.csv',sweep)
    (DEST/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (DEST/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    shutil.copy2(RUN/'preregistration.json',DEST/'preregistration.json')
    lines=['# 후보 수 병목 진단 — 2026-11-11','','기존 seed 42 best checkpoint로 valid 전체 220,516명·515,201개 정답만 평가했다. 모델을 재학습하지 않았고 test 정답은 읽지 않았다. 100/500/1,000은 같은 top-1,000의 중첩 prefix다.','','## 후보 내 정답과 비인기 상품','','|모델|후보 수|비인기 평균 개수|비인기 0개 사용자 (%)|정답 회수율 macro (%)|후보 내 정답 0개 사용자 (%)|Oracle NDCG@10|','|---|---:|---:|---:|---:|---:|---:|']
    for r in counts:
        if r['group']=='all':lines.append(f"|{r['model']}|{r['pool']}|{float(r['tail_mean']):.2f}|{100*float(r['tail_zero_share']):.2f}|{100*float(r['candidate_recall']):.3f}|{100*float(r['truth_zero_share']):.2f}|{float(r['oracle_ndcg10']):.6f}|")
    lines+=['','정답 회수율은 사용자별 후보 내 정답 수 / 사용자별 전체 valid 정답 수의 평균이다. 후보 내 정답 0개 비율은 사용자 기준이다. Oracle은 정답을 알고 후보를 이상적으로 재정렬했을 때의 진단용 상한이며 실제 추천 성능이 아니다.','','## 차종 정보가 있는 사용자에서 후보 구성','','A1 리뷰 + A2 상품 추론(n_items≥2)의 합집합, 제조사+모델 일치다. train 기간 프로필만 사용하고 연식은 사용하지 않는다. unknown/범용은 일치·불일치로 세지 않는다.','','|모델|후보 수|사용자 수|차종 일치 평균 개수|일치 0개 사용자 (%)|비인기+차종 일치 평균 개수|비인기+일치 0개 사용자 (%)|','|---|---:|---:|---:|---:|---:|---:|']
    for r in counts:
        if r['group']=='has_vehicle':lines.append(f"|{r['model']}|{r['pool']}|{r['users']}|{float(r['match_mean']):.2f}|{100*float(r['match_zero_share']):.2f}|{float(r['tail_match_mean']):.3f}|{100*float(r['tail_match_zero_share']):.2f}|")
    lines+=['','## 후보 확대와 차종 재정렬 효과 분리','','모든 후보 수에서 β=0이면 기존 top-10 정확도가 같다. 차종 가점은 원 점수 + β × **같은 사용자의 top-100 점수 범위** × 일치 여부다. 후보 수마다 점수 범위를 바꾸지 않아 가점 단위의 변화를 통제했다. β grid는 0/0.02/0.05/0.1/0.2/0.5/1/2이며 valid NDCG 최대(동점은 작은 β)를 표시한다.','','|모델|후보 수|선택 β|기본 valid NDCG@10|재정렬 valid NDCG@10|변화 (%)|재정렬 APLT@10 (%)|','|---|---:|---:|---:|---:|---:|---:|']
    for rec,label in zip(receipts,labels.values()):
        for k,row in rec['selected_on_valid'].items():
            base=next(float(r['ndcg']) for r in sweep if r['model']==label and r['pool']==k and float(r['beta'])==0)
            lines.append(f"|{label}|{k}|{row['beta']}|{base:.8f}|{row['ndcg']:.8f}|{100*(row['ndcg']/base-1):+.2f}|{100*row['aplt']:.5f}|")
    lines+=['','동일 β별 비교와 활동 그룹(1–5/6+)·unknown 사용자 통계는 CSV에 있다. 후보 수와 β 모두 valid를 보고 탐색한 결과이며 독립적인 test 개선으로 해석하지 않는다. 기존 pool500 실험은 후보별 점수 범위를 가점 단위로 사용했으므로 이번 통제 비교와 구분한다.','','## 과제 목표에 맞는 다음 행동','','[Notion 제안서](https://app.notion.com/p/40de3a7d940880d5acb2f73f3bb48b1e)의 RQ3은 인기 감점/텍스트 재정렬 비교, RQ4는 다양성만 높인 조건과 차종 정보를 더한 조건의 비교다. [팀 개요](https://app.notion.com/p/40de3a7d9408803eb856e3efecedf8a1)의 액션은 전달된 차종 데이터로 탐색 → 성능이 오르지 않는 원인 분석 → 처리 방향 제안이며, 코드 설명은 학습 후 재정렬·평가라고 명시한다. 조회된 Notion 응답은 unverified 스냅샷이다. 개요의 응답 시각은 11/11, 제안서의 응답 시각은 11/3이다.','','1. 이번 valid 진단으로 후보 부족을 확인하고 후보 수를 동결한다. 재학습은 필요하지 않다.','2. 같은 후보·같은 사용자에서 기본 / 인기 감점 / 차종 가점 / 인기 감점+차종 가점을 비교한다. 인기 감점만으로 다양성을 높인 조건에 차종을 더한 차이가 RQ4의 핵심 비교다.','3. valid NDCG를 기본의 95% 이상 유지하는 조건에서 APLT와 Tail Recall을 보고 설정을 고정한다. Mismatch는 판단 가능한 쌍의 분모와 coverage를 함께 제시한다.','4. 원자료를 검수해 A1/A2 오탐과 fitment 누락을 분리한다. 텍스트 조건은 별도 특징 확보 후 추가하며 이번 결과로 RQ3 전체 완료라 하지 않는다.','5. 차종 정보를 encoder/학습 손실에 넣는 재학습은 별도 확장이다. 후보 확대·재정렬·추출 품질 원인을 확인한 뒤에도 필요할 때 진행한다.','','## 검증·재현','','CUDA L4, PyTorch 2.11.0+cu130, TF32 비활성. 입력 fingerprint·checkpoint SHA-256·train-only 프로필 시점·seen 제외·후보 중복 없음·중첩 후보의 개수 단조성·β=0 후보별 정확도 동일을 검사했다. 원래 top-100과의 근소한 float32 그래프 연산 차이는 receipts.json에 보관하며 기존 실험과 같은 허용 기준을 적용했다.','','집계: [후보 구성 CSV](results/candidate-diagnostic-2026-11-11/candidate_availability.csv), [고정 단위 β sweep](results/candidate-diagnostic-2026-11-11/valid_fixed_scale_sweep.csv), [실행 receipt](results/candidate-diagnostic-2026-11-11/receipts.json). 사용자별 후보 개수는 Git 제외 `results/candidate-diagnostic-20261111/output/<run>/candidate_counts_per_user.parquet`에만 보관한다.']
    (ROOT/'docs/CANDIDATE_DIAGNOSTIC_2026-11-11.md').write_text('\n'.join(lines)+'\n')
    report=ROOT/'docs/CANDIDATE_DIAGNOSTIC_2026-11-11.md'
    text=report.read_text().replace('## 과제 목표에 맞는 다음 행동', '''## 판단

- top-100은 세 모델 모두 비인기 후보가 없는 사용자가 91% 이상이고, 차종 프로필이 있는 사용자 중 비인기+차종 일치 후보가 없는 비율은 97.85~99.90%다. 현재 후보만으로 비인기이면서 차에 맞는 상품을 올릴 여지는 매우 작다.
- top-1,000은 이 제약을 줄이지만 차종 프로필 사용자 중에도 비인기+일치 후보가 없는 비율은 BPR-MF 93.53%, LightGCN 82.43%, XSimGCL 78.57%다. 후보 확대만으로 모든 사용자에게 해결되지는 않는다.
- 차종 가점만 쓴 valid NDCG는 BPR-MF에서 top-500이 가장 높다. LightGCN은 top-100보다 확대 조건이 조금 낮고, XSimGCL은 동일하다. 후보의 정답 회수율 증가를 실제 정확도 증가와 혼동하지 않는다.
- 따라서 후보 부족은 확인됐지만 모든 모델의 차종 개선이 작은 유일 원인은 아니다. 비인기 상품을 끌어올리는 항이 없는 차종 가점만으로는 프로젝트의 비인기 추천 목표를 충분히 검증할 수 없다. 다음은 후보 500/1,000에서 인기 감점 조건을 추가해 valid의 정확도–APLT 균형을 확인하고 후보 수·계수를 동결하는 일이다. 재학습은 이 비교의 선행 조건이 아니다.

## 과제 목표에 맞는 다음 행동''')
    text+='\n운영 종료: 결과 archive와 조각 SHA-256 대조 회수 완료, task VM 반납, 활성 할당 0개·소모율 0.00 CU/hr. 계정 잔액 98.45 → 98.30 CU(관측 차이 0.15 CU, 초기 접속 복구·전송 포함). [종료 검증](results/candidate-diagnostic-2026-11-11/verification.json).\n'
    report.write_text(text)

if __name__=='__main__':main()
