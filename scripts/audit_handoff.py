"""Read-only local handoff audit. No data loading, training, or provider calls.

Writes aggregate provenance and a navigable index, never private rows or tokens.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/handoff'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    review=ROOT/'results/campaigns/automotive-20261104T204146Z/review'
    rows=[];lineage=[]
    families=[('vehicle-pilot-2026-11-11','vehicle-pilot-20261111'),('vehicle-pilot-models-2026-11-11','vehicle-pilot-models-20261111'),('vehicle-pilot-seeds-2026-11-11','vehicle-pilot-seeds-20261111'),('vehicle-pilot-pool500-2026-11-11','vehicle-pilot-pool500-20261111')]
    for family,private in families:
        root=ROOT/'docs/results'/family
        for p in sorted(root.rglob('summary.json')):
            s=json.loads(p.read_text());assert s['status']=='completed'
            run=p.parent.name if p.parent!=root else 'final_bpr_mf_seed42'
            model,seed=re.fullmatch(r'final_(.+)_seed(\d+)',run).groups();seed=int(seed)
            record=json.loads((review/f'{run}.json').read_text())
            assert record['summary']['settings']['seed']==seed
            private_root=ROOT/'results'/private
            if family=='vehicle-pilot-pool500-2026-11-11' and model=='bpr_mf':
                raw=private_root/'attempt2-torch2.11-output'/run/'summary.json'
                manifest=private_root/'attempt2-manifest.json'
            else:
                raw=private_root/'output'/run/'summary.json' if p.parent!=root else private_root/'output/summary.json'
                manifest=private_root/'manifest.json'
            assert raw.exists() and sha(raw)==sha(p),str(p)
            # Pool500 BPR was recovered from attempt 2; its manifest was overwritten
            # by attempt 3. Its checkpoint/input lineage is therefore a known gap.
            verified=None
            if manifest.exists():
                m=json.loads(manifest.read_text())
                expected={a['relative']:a['sha256'] for a in record['artifacts'] if a['relative'].endswith(('recommendations.npz','metrics.json'))}
                actual={a['relative'].removeprefix('baseline/'+run+'/'):a['sha256'] for a in m['baseline'] if a['relative'].startswith('baseline/'+run+'/')}
                if family=='vehicle-pilot-2026-11-11':
                    actual={a['relative'].removeprefix('baseline/'):a['sha256'] for a in m['baseline']}
                if actual:assert all(actual.get(k)==v for k,v in expected.items());verified=True
            r=s['primary_result'];pool=s['design'].get('candidate_pool',100)
            rows.append(dict(experiment=family,run_id=run,seed_from_baseline=seed,raw_design_seed=s['design']['seed'],pool=pool,source=r['source'],level=r['level'],split='test',beta=r['beta'],ndcg=r['ndcg'],relative_change=r['ndcg_relative_change'],ci_low=r['ndcg_paired_ci95'][0],ci_high=r['ndcg_paired_ci95'][1],summary=p.relative_to(ROOT).as_posix(),raw_summary=raw.relative_to(ROOT).as_posix(),baseline_manifest_hash_verified=verified))
            lineage.append({'summary':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'raw_copy_identical':True,'seed_from_baseline':seed,'raw_seed_label_correct':seed==s['design']['seed'],'baseline_manifest_hash_verified':verified})
    with (OUT/'vehicle_results_index.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    lines=['# 차종 후속 실험 결과 색인','','2026-11-13 로컬 보관 산출물을 대조했다. 실행 완료와 해석 범위를 구분한다. 전체 상태의 기준 문서는 [PROJECT_BRIEF](../PROJECT_BRIEF.md)다.','','|실험|run|후보|차종 출처|test NDCG 변화 (%)|사용자 paired 95% CI|원본 요약|','|---|---|---:|---|---:|---|---|']
    for r in rows:
        link=r['summary'].removeprefix('docs/')
        lines.append(f"|{r['experiment']}|{r['run_id']}|{r['pool']}|{r['source']} / {r['level']}|{100*r['relative_change']:+.2f}|[{r['ci_low']:.8f}, {r['ci_high']:.8f}]|[JSON]({link})|")
    lines+=['','- `both`는 A1 리뷰+A2 상품 추론 합집합, `items`는 A2(n_items≥2)다. 실험별 주 조건이 달라 위 행들을 같은 조건의 seed 평균으로 합치지 않는다. 원본 test_results.json에는 다른 family도 포함된다.','- 43/44 seed의 design.seed=42와 single-seed 설명은 생성기의 고정 문자열 오류다. 실제 run ID, baseline 학습 settings.seed, 입력 추천 artifact SHA로 seed를 대조했다. 원본은 보존하고 CSV의 seed_from_baseline을 사용한다.','- pool500의 BPR-MF는 attempt2 결과다. 원본 summary 사본 일치는 확인했지만 attempt2 전용 manifest가 보관돼 있지 않아 입력 추천 hash lineage는 다른 실행과 같은 수준으로 재검증하지 못했다. attempt3 manifest는 그래프 모델 두 개만 포함한다.','- seed43/44의 여섯 주 조건 점추정은 모두 양수다. 이는 서로 다른 seed/사용자를 합친 통합 CI나 그래프 모델이 차종을 이미 학습했다는 증거가 아니다.','- pool500은 후보별 점수 범위를 사용했고 후보 수·가점 크기가 함께 바뀐다. 후보 수 자체의 비교는 [고정 단위 후보 진단](CANDIDATE_DIAGNOSTIC_2026-11-11.md)을 사용한다.','- 후보 진단은 세 모델 seed42, valid 220,516명, 후보100/500/1000이다. 추가 test나 학습은 없다. [receipt](results/candidate-diagnostic-2026-11-11/receipts.json), [CSV](results/candidate-diagnostic-2026-11-11/candidate_availability.csv).','- 최초 파일럿의 test/특징 개발 탐색·메타데이터 시점·동일 규칙 mismatch 한계는 [파일럿 해설](VEHICLE_PILOT_2026-11-11.md)에 있다.']
    (ROOT/'docs/EXPERIMENT_INDEX.md').write_text('\n'.join(lines)+'\n')
    v=json.loads((ROOT/'docs/results/final/verification.json').read_text())
    for name,h in v['report_sha256'].items():assert sha(ROOT/'docs/results/final'/name)==h,name
    candidate=json.loads((ROOT/'docs/results/candidate-diagnostic-2026-11-11/artifact_hashes.json').read_text())
    candidate_files=0
    for model,files in candidate.items():
        for name,h in files.items():
            p=ROOT/'results/candidate-diagnostic-20261111/output'/f'final_{model}_seed42'/name
            assert sha(p)==h,str(p)
            candidate_files+=1
    # Verify only new/current handoff entrypoints; historical documents retain their provenance.
    missing=[]
    for name in ['README.md','PROJECT_BRIEF.md','docs/HANDOFF.md','docs/EXPERIMENT_INDEX.md']:
        p=ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#')[0]
            if target and (p.parent/target).resolve()!=(OUT/'audit.json').resolve() and not (p.parent/target).exists():missing.append({'file':name,'target':target})
    assert not missing,missing
    code={p.relative_to(ROOT).as_posix():sha(p) for folder in ['src','scripts'] for p in (ROOT/folder).rglob('*.py')}
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'local stored files; no live server/Drive/Colab query','base_report_hashes_verified':len(v['report_sha256']),'vehicle_summary_copies_verified':len(rows),'candidate_artifacts_verified':candidate_files,'lineage':lineage,'local_markdown_links_missing':missing,'current_source_sha256':code,'publication':'Pre-publication audit: git_head is the parent checkout revision, not the commit containing this report. Consult Git history for publication; this audit does not verify Drive permissions.'}
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'Verified {len(rows)} vehicle summaries, {len(v["report_sha256"])} final artifacts and entrypoint links.')
if __name__=='__main__':main()
