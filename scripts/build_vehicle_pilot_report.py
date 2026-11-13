"""Publish aggregate-only pilot evidence locally; never copy user rankings."""
import argparse
import json
from pathlib import Path
import re
import shutil


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=args.run/'output'
    dest=root/'docs/results/vehicle-pilot-2026-11-11'
    dest.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text())
    results=json.loads((source/'test_results.json').read_text())
    baseline=json.loads((source/'test_baseline.json').read_text())
    primary=summary['primary_result']
    selected=[r for r in results if r['selected_on_valid']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[r['source']+' / '+r['level'] for r in selected]
    figure,axes=plt.subplots(1,2,figsize=(11,4.5))
    for index,r in enumerate(selected):
        mean=r['ndcg_delta']*1e5
        low,high=[v*1e5 for v in r['ndcg_paired_ci95']]
        axes[0].errorbar(mean,index,xerr=[[mean-low],[high-mean]],fmt='o',capsize=3,color='#357cb5')
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].axvline(0,color='#b34838',linestyle='--')
    axes[0].set_title('Change in NDCG@10 (x 100,000)\nPointwise 95% paired CI, one model seed')
    axes[0].invert_yaxis()
    axes[1].barh(labels,[r['aplt']*100 for r in selected],color='#357cb5')
    axes[1].axvline(baseline['aplt']*100,color='#b34838',linestyle='--',label='BPR-MF seed 42 baseline')
    axes[1].set_title('Tail exposure APLT@10 (%)')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    figure.suptitle('Vehicle reranking pilot: validation-selected settings, test cohort')
    figure.tight_layout()
    figure.savefig(dest/'comparison.png',dpi=160)
    plt.close(figure)
    for name in ['summary.json','data_audit.json','design.json','frozen_selection.json',
                 'test_results.json','test_results.csv','valid_sweep.csv','valid_baseline.json','test_baseline.json']:
        shutil.copyfile(source/name,dest/name)
    manifest=json.loads((args.run/'manifest.json').read_text())
    (dest/'input_hashes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    def usage(name):
        text=(args.run/name).read_text()
        return {key:float(re.search(pattern,text).group(1)) for key,pattern in [
            ('balance',r'Current balance: ([\d.]+)'),('rate',r'Usage rate: ([\d.]+)'),
            ('assignments',r'Active assignments: (\d+)')]}
    first='attempt1-usage-before.txt' if (args.run/'attempt1-usage-before.txt').exists() else 'usage_before.txt'
    cost={'before':usage(first),'successful_attempt_before':usage('usage_before.txt'),'after':usage('usage_after.txt'),
          'scope':'account-observed balance difference, not model-exclusive billing',
          'sessions_after':(args.run/'sessions_after.txt').read_text().strip()}
    cost['balance_difference']=round(cost['before']['balance']-cost['after']['balance'],2)
    assert cost['after']['assignments']==0 and cost['after']['rate']==0
    (dest/'release_verification.json').write_text(json.dumps(cost,indent=2)+'\n')
    lines=['# 차종 데이터 재정렬 파일럿 — 2026-11-11','',
        f"BPR-MF seed 42의 기존 추천에 차종 가점을 더한 탐색 실험이다. 사전 지정한 A1+A2·모델 일치 조건의 test NDCG@10은 **{baseline['ndcg']:.8f} → {primary['ndcg']:.8f} ({primary['ndcg_relative_change']:+.2%})**다.",
        '','## 평가 조건','',
        '- 어제 회의의 BPR-MF 우선·단일 seed 방향을 따른다. 기존 best checkpoint의 valid/test top-100과 점수를 재사용했으며 모델을 재학습하지 않았다.',
        '- valid 220,516명에서 3개 출처 × 2개 일치 수준 × 8개 β = 48개 설정을 평가하고, 출처·수준별 NDCG가 가장 높은 β를 고정했다. 동점이면 작은 β다.',
        '- frozen_selection.json을 저장한 뒤 test 158,805명 / 333,797개 정답을 평가했다. 기본 후보는 전체 train 상품 248,186개에서 이미 본 상품을 제외해 얻은 top-100이며, 새 상품 검색 효과는 이 실험 범위에 없다.',
        '- A1은 train 기간 리뷰의 차종, A2는 train 기간 리뷰 상품 중 같은 차에 맞는 서로 다른 상품이 2개 이상인 추론이다. valid/test용 확장 프로필은 사용하지 않았다.',
        '- 여러 차종 중 하나라도 일치하면 가점. 제조사/모델 표기는 앞뒤 공백 정리·소문자 변환만 하고 새 별칭 규칙은 만들지 않았다. 연식은 사용하지 않았다.',
        '- 조정 점수 = 원 점수 + β × 사용자별 top-100 점수 범위 × 일치 여부. 범위가 0이면 1. β는 0/0.02/0.05/0.1/0.2/0.5/1/2다. float64 CUDA 정렬, score 내림차순·iid 오름차순이다.',
        '- 정보 없음과 범용 상품은 unknown이며 가점·감점이 없다. β=0의 순위와 지표가 보관된 baseline과 일치하는지 확인했다.',
        '', '## 동결한 설정의 test 결과','',
        '![차종 재정렬 정확도와 비인기 노출](results/vehicle-pilot-2026-11-11/comparison.png)','',
        '| 차종 출처 | 수준 | valid 선택 β | NDCG@10 | 기본 대비 | APLT@10 (%) | 판정 가능한 추천 (%) | 추론 불일치 (%) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    labels={'text':'A1 리뷰','items':'A2 상품','both':'A1+A2'}
    for r in selected:
        mismatch='미정의' if r['inferred_mismatch_rate'] is None else f"{r['inferred_mismatch_rate']*100:.2f}"
        lines.append(f"| {labels[r['source']]} | {r['level']} | {r['beta']:g} | {r['ndcg']:.8f} | {r['ndcg_relative_change']:+.2%} | {r['aplt']*100:.4f} | {r['judged_pair_coverage']*100:.2f} | {mismatch} |")
    ci=primary['ndcg_paired_ci95']
    lines += ['',f"기본 BPR-MF의 APLT@10은 {baseline['aplt']*100:.4f}%다. 주 조건의 NDCG 차이는 {primary['ndcg_delta']:+.8f}, 사용자 paired bootstrap 95% CI는 [{ci[0]:+.8f}, {ci[1]:+.8f}]다. 학습 seed 간 변동에 대한 CI는 아니다.",
        '', '## 해석과 남은 범위','',
        '- A2·모델 일치는 NDCG +2.84%이고 보정 전 paired CI가 0보다 높았다. 6개 출처/수준을 비교한 탐색 결과이므로 다중 비교와 다른 seed의 재현을 고려해 후속 후보로 해석한다.',
        '- 위 불일치는 추출 규칙상 판단한 mismatch이며 실제 장착 불가의 확정값이 아니다. 분모는 차종을 판단할 수 있는 추천 쌍(일치+불일치)이고 unknown은 제외했다. 분모가 바뀔 수 있어 판정 coverage도 함께 보고한다.',
        '- 가점에 사용한 정보로 불일치도 측정하므로 그 감소 자체는 독립적인 효과 검증이 아니다. NDCG·Recall 변화와 원자료 추출 품질을 함께 확인해야 한다.',
        '- 단일 seed·하나의 후보 생성기·valid 탐색 결과다. 다른 모델·다른 seed의 개선이나 차종 특징을 넣어 재학습한 효과로 일반화하지 않는다.',
        '- 상품 제목의 관측 시점은 제공되지 않아 과거 시점 메타데이터 정합성은 검증하지 못했다. 사용자 프로필의 last_seen이 valid 시작보다 이른지는 실제 데이터로 검사했다.',
        '- 제공 README의 A1/A2 추출 품질 수치와 Spark 처리 시간은 공급자 보고값이다. 이 파일럿에서 독립적으로 재현한 값으로 사용하지 않는다.',
        '- 공급자 README에는 test 상호작용에 대한 출처별 일치율과 정제 전후 비교가 포함돼 있다. 이 파일럿의 β는 valid로만 선택했지만, 특징 개발 과정까지 test와 완전히 독립된 최종 검증으로 보지는 않는다.',
        '', '## 실행·검증·보관','',
        f"- 장치 {summary['gpu']}, PyTorch {summary['torch']}, CUDA {summary['cuda']}. 평가 본체 {summary['elapsed_seconds']:.1f}초. 전송·패키지 설치 시간은 별도다.",
        f"- 본체 peak RSS {summary['peak_rss_MiB']:.0f} MiB, CUDA peak allocated {summary['cuda_peak_allocated_MiB']:.0f} MiB. idle 메모리 수치가 아니다.",
        f"- 계정 잔액 {cost['before']['balance']:.2f} → {cost['after']['balance']:.2f} CU, 관측 차이 {cost['balance_difference']:.2f} CU. 종료 후 활성 할당 0개·CU 소모율 0 확인.",
        '- 입력/source SHA-256, 원래 baseline 지표 재현, 전체 cohort·seen 마스킹, train-only 프로필 timestamp, CPU/CUDA 표본 순위 일치를 검사했다.',
        '- 첫 세션의 연결 오류, 분리 환경 생성 시 pip 부재, 큰 파일 회수 지연을 복구했다. 완료 묶음은 17개 조각과 전체 SHA-256을 대조해 회수했다. 세부 검증은 verification.json에 있다.',
        '- 집계 자료: [results/vehicle-pilot-2026-11-11](results/vehicle-pilot-2026-11-11/). 원본 README 사본은 [sources](sources/vehicles-readme-2026-11-11.txt).',
        '- 원본 Parquet·사용자별 지표·top10 추천·세션 notebook·로그·회수 archive는 Git 제외 `results/vehicle-pilot-20261111/`에 있다. 원본 데이터와 완료된 32개 캠페인은 수정하지 않았다.',
        '- 재실행은 `scripts/colab_vehicle_pilot.py --session <새 이름> --directory <입력 3개가 있는 별도 results 경로>`를 프로젝트 Python으로 실행한다. 새 Colab L4 비용이 발생하며 자동 종료한다.',
        '']
    (root/'docs/VEHICLE_PILOT_2026-11-11.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'cost':cost},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
