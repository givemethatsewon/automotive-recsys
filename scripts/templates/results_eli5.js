(() => {
'use strict';
const D = JSON.parse(document.getElementById('experiment-data').textContent);
const $ = id => document.getElementById(id);
const names = {popularity:'Popularity',bpr_mf:'BPR-MF',lightgcn:'LightGCN',xsimgcl:'XSimGCL'};
const colors = {popularity:'#64748b',bpr_mf:'#275bdd',lightgcn:'#bb3171',xsimgcl:'#167252'};
const press = (attribute, value) => document.querySelectorAll(`[${attribute}]`).forEach(button => {
  const active = button.getAttribute(attribute) === String(value);
  button.setAttribute('aria-pressed', String(active));button.classList.toggle('active', active);
});
const onButtons = (attribute, action) => document.querySelectorAll(`[${attribute}]`).forEach(button => button.addEventListener('click', () => action(button.getAttribute(attribute))));
const phaseNotes = [
  '<strong>train: 과거 연결만으로 배운다.</strong> 사용자·상품 벡터와 그래프는 train 상호작용에서 만든다. valid나 test의 정답 연결을 학습에 넣지 않는다.',
  '<strong>valid: 연습 시험으로 설정을 고른다.</strong> 학습률·정규화·층 수 등을 비교하고, 어느 epoch의 모델을 쓸지도 고른다. valid 점수는 최종 test 점수가 아니다.',
  '<strong>설정 동결: 선택을 끝내고 기록한다.</strong> test를 보기 전에 모델별 설정을 고정했다. 최종 학습도 train만 사용하고 best checkpoint는 valid로 선택했다. train+valid 재학습은 하지 않았다.',
  '<strong>test: 한 번 정한 조건으로 최종 비교한다.</strong> 같은 사용자·후보·정답으로 평가했다. test 점수를 보고 학습률·best epoch를 다시 고르면 마지막 시험의 답을 보고 연습한 셈이 된다.'
];
function phase(value){press('data-phase',value);document.querySelectorAll('#timeline .time-card').forEach((card,index)=>card.classList.toggle('active',index===Number(value)));$('phase-note').innerHTML=phaseNotes[value];}
onButtons('data-phase',phase);phase(0);

let model='popularity', hop=0;
const graphCopy=[
  ['출발 · 내 과거 연결','나는 와이퍼에 리뷰를 남겼다. 이 연결만으로 차종·연식이 무엇인지는 알 수 없다.'],
  ['1번 연결 · 나와 와이퍼','사용자와 상품이 직접 연결돼 있다. 그래프 전파에서는 각 노드가 이웃 벡터를 섞는 첫 단계를 생각할 수 있다.'],
  ['2번 연결 · 와이퍼와 다른 사용자','같은 와이퍼에 리뷰를 남긴 다른 사용자의 정보가 관계를 통해 닿는다. 개별 사용자를 찾아 직접 복사하는 알고리즘이라는 뜻은 아니다.'],
  ['3번 연결 · 다른 사용자의 세정제','나→와이퍼→다른 사용자→세정제라는 간접 연결이 생긴다. 실제 모델은 모든 노드 벡터를 동시에 섞으며, 이 경로를 직접 검색하는 것이 아니다.']
];
function graph(){
  const points=[[80,140,'나','사용자 A'],[320,100,'와이퍼','상품'],[550,140,'다른 사람','사용자 B'],[810,100,'세정제','상품']];
  return `<svg viewBox="0 0 920 260" role="img" aria-label="나에서 와이퍼, 다른 사용자, 세정제로 이어지는 세 번의 가상 연결"><title>그래프 연결 · ${hop}단계</title>`+
    points.slice(0,3).map((p,i)=>`<line x1="${p[0]+42}" y1="${p[1]}" x2="${points[i+1][0]-42}" y2="${points[i+1][1]}" class="graph-edge ${i<hop?'active':''}"/>`).join('')+
    points.map((p,i)=>`<g><circle cx="${p[0]}" cy="${p[1]}" r="43" class="graph-node ${i<=hop?'active':''}"/><text x="${p[0]}" y="${p[1]+5}" text-anchor="middle" font-size="${i===2?14:18}">${p[2]}</text><text x="${p[0]}" y="${p[1]+76}" text-anchor="middle" font-size="14">${p[3]}</text></g>`).join('')+'</svg>';
}
function renderModel(){
  press('data-model',model);
  const isGraph=model==='lightgcn'||model==='xsimgcl';
  $('model-prev').disabled=!isGraph||hop===0;$('model-next').disabled=!isGraph||hop===3;$('model-reset').disabled=!isGraph||hop===0;
  $('model-step').textContent=isGraph?`${hop}/3번 연결`: '모델을 바꾸면 다른 원리를 볼 수 있어';
  if(model==='popularity'){
    $('model-title').textContent='Popularity: 많이 관측한 상품부터';
    $('model-story').textContent='모든 사용자에게 같은 인기도 기준을 적용한다. 사용자가 이미 본 상품은 빼지만, 사람마다 따로 취향 벡터를 배우지는 않는다.';
    $('model-visual').innerHTML='<div class="rank-mini"><div><b>120회</b>와이퍼</div><div><b>80회</b>세정제</div><div><b>30회</b>매트</div><div><b>5회</b>필터</div></div><p class="scene-caption">가상 train 리뷰 수 · 정답인지 알기 전에 이 횟수로 순서를 정한다</p>';
    $('model-detail').innerHTML='<strong>기준선이 필요한 이유:</strong> 복잡한 모델을 만들었을 때, 단순히 인기순으로 고르는 것보다 나아졌는지 확인해야 한다. 우리 BPR-MF는 이번 조건에서 이 기준선보다 NDCG가 낮았다.';
  }else if(model==='bpr_mf'){
    $('model-title').textContent='BPR-MF: 사용자와 상품에 숫자 카드를 배운다';
    $('model-story').textContent='사용자와 상품을 숫자 벡터로 표현한다. 같은 위치 숫자를 곱해 더한 점수가 높으면 더 앞에 추천한다. 관측한 상품이 뽑아 온 미관측 상품보다 높은 점수를 받도록 연습한다.';
    $('model-visual').innerHTML='<div class="vector-cards"><div class="vector-card"><b>사용자 카드</b><span>가상 벡터 '+D.math.vecUser+'</span></div><div class="bridge">·</div><div class="vector-card"><b>상품 카드</b><span>가상 벡터 '+D.math.vecItem+'</span></div></div><div class="formula medium mathblock">'+D.math.dot+'</div><p class="scene-caption">2차원으로 줄인 설명용 예시 · 실제는 64차원이며 축의 인간적 의미가 정해져 있지 않다</p>';
    $('model-detail').innerHTML='<strong>LightGCN과의 차이:</strong> 숫자 카드를 직접 학습하지만 이웃 노드끼리 벡터를 전파하는 단계는 없다. 미관측을 오답으로 뽑아 연습하는 것과 실제로 그 상품을 싫어했다는 것은 다르다.';
  }else{
    $('model-title').textContent=model==='lightgcn'?'LightGCN: 이웃과 연결된 정보를 섞는다':'XSimGCL: 그래프 전파에 대조 연습을 더한다';
    $('model-story').textContent=model==='lightgcn'?'같은 상품에 연결된 사용자와 그 사용자의 다른 상품 정보가 벡터를 통해 전달된다. 아래는 간접 관계가 생기는 과정을 보여 주는 가상 경로다. 다음 연결 버튼으로 따라가 보자.':'그래프에서 이웃 정보를 섞으면서 학습 중 작은 노이즈를 더한다. 같은 노드의 얕은 층 표현과 여러 층을 모은 표현을 서로 맞추고, 다른 노드의 표현과 상대적으로 구분하는 추가 목표를 쓴다.';
    $('model-visual').innerHTML=graph()+(model==='xsimgcl'?'<div class="cl-pair"><div><strong>얕은 층의 표현</strong><div class="dots"><i class="dot me"></i><i class="dot"></i><i class="dot"></i></div><p>파랑: 노드 A · 회색: 다른 노드</p></div><div class="bridge">↔</div><div><strong>여러 층을 모은 표현</strong><div class="dots"><i class="dot me"></i><i class="dot"></i><i class="dot"></i></div><p>같은 A를 대응시켜 구분을 연습</p></div></div>':'');
    $('model-detail').innerHTML='<strong>'+graphCopy[hop][0]+'</strong><p>'+graphCopy[hop][1]+'</p>'+(model==='xsimgcl'?'<p>노이즈가 있어도 같은 노드의 표현을 대응시키되 다른 노드와 구분하려는 연습이다. 벡터를 전부 똑같이 만드는 것이 목표가 아니다. 상품 인기도를 직접 감점하는 규칙도 아니므로 tail 노출 증가가 보장되지는 않는다.</p><p class="note">위 색 점은 대조 목표를 설명한 기호다. 실제 학습 벡터나 학습 후 거리를 그린 것은 아니다. test 때는 노이즈를 끄고 추천 점수를 계산한다.</p>':'');
  }
}
onButtons('data-model',value=>{model=value;hop=0;renderModel();});
$('model-prev').addEventListener('click',()=>{hop=Math.max(0,hop-1);renderModel();});
$('model-next').addEventListener('click',()=>{hop=Math.min(3,hop+1);renderModel();});
$('model-reset').addEventListener('click',()=>{hop=0;renderModel();});renderModel();

let calcStep=0;
function toyCase(){return D.toy[$('toy-case').value==='two'?'two':'one-'+$('toy-rank').value];}
function renderToy(){
  const c=toyCase(),two=$('toy-case').value==='two',tail=Number($('toy-tail').value),rank=Number($('toy-rank').value);
  $('toy-rank').disabled=two;$('tail-count').textContent=`${tail}개`;
  $('toy-shelf').innerHTML=Array.from({length:10},(_,index)=>{
    const place=index+1,hit=c.hits.includes(place),focus=hit&&(calcStep===1||calcStep===3);
    return `<div class="toy-card ${hit?'hit':''} ${index<tail?'tail':''} ${focus?'focus':''}" data-rank="${place}"><span class="position">${place}위</span><b>${hit?'정답 상품 '+(two?(place===1?'A':'B'):'A'):'다른 상품'}</b></div>`;
  }).join('');
  $('toy-ndcg').innerHTML=c.scoreMath;$('toy-recall').innerHTML=c.recallValueMath;$('toy-aplt').innerHTML=D.tailValue[tail];$('toy-tail-rule').innerHTML=D.tail[tail];
  $('toy-prev').disabled=calcStep===0;$('toy-next').disabled=calcStep===4;$('toy-step').textContent=`계산 ${calcStep+1}/5`;
  let title,text,formula='';
  if(calcStep===0){title='1. 무엇이 정답인지 먼저 정한다';text=two?'이 사용자의 정답은 A와 B, 총 2개다. 현재 추천에서는 1위와 3위에 있다. 비인기 배지는 정답 여부와 독립적인 정보다.':rank<=10?`이 사용자의 정답은 A 한 개다. 지금은 ${rank}위에 있다. 오답 칸에는 순위 보상을 주지 않는다.`:'정답 A는 있지만 추천 10칸 안에는 없다. 정답이 없는 사용자를 평가하는 것이 아니라, 정답을 추천에서 놓친 경우다.';
  }else if(calcStep===1){title='2. 높은 순위의 정답에 큰 보상을 준다 · DCG';text=two?'1위 정답은 1점, 3위 정답은 0.5점이다. 두 보상을 더해 1.5가 된다. 오답의 보상은 0이다.':rank>10?'상위 10개 안에 정답이 없으므로 더할 보상은 0이다. 11위의 보상을 넣지 않는다.':rank===1?'1위는 log₂(2)가 1이므로 1을 1로 나눈 보상 1을 받는다.':rank===3?'3위는 log₂(4)가 2이므로 1을 2로 나눈 보상 0.5를 받는다.':`${rank}위의 분모는 log₂(${rank+1})다. 순위가 내려갈수록 분모가 커져 받는 보상은 작아진다.`;formula=c.dcgMath;
  }else if(calcStep===2){title='3. 가장 잘 배치했을 때를 계산한다 · IDCG';text=two?'정답이 2개이므로 이상적인 목록은 A와 B를 1위·2위에 놓는다. 보상 1과 약 0.63093을 더한 약 1.63093이 비교 기준이다. 현재 목록의 1위·3위 배치를 바꿨다는 뜻은 아니다.':'정답이 1개이므로 이상적인 목록은 A를 1위에 놓는다. 그때 보상은 1이다. 현재 추천에서 놓쳤더라도 정답 개수는 여전히 1개이므로 분모가 0이 되지 않는다.';formula=c.idealMath;
  }else if(calcStep===3){title='4. 실제 보상을 이상적인 보상으로 나눈다 · NDCG';text=two?'정답 2개를 모두 찾았으므로 Recall은 1이지만, 하나가 2위가 아니라 3위에 있어 NDCG는 1보다 작다. 정답 회수와 순위 품질은 다르다.':rank>10?'실제 보상이 0, 이상적인 보상이 1이므로 NDCG는 0이다. 비인기 상품을 많이 보여줘도 정답을 놓친 사실은 바뀌지 않는다.':rank===1?'실제 목록과 이상적인 목록의 보상이 같으므로 NDCG는 1이다.':`실제 보상을 분모 ${c.ideal.toFixed(0)}로 나눈다. 이 예제는 정답이 한 개라 DCG와 NDCG 숫자가 같지만, 정답이 여러 개면 분모도 달라진다.`;formula=c.ndcgMath;
  }else{title='5. 사람마다 구한 점수를 같은 비중으로 평균 낸다';text='설명용 사용자 3명을 가정하자. A는 지금 계산한 점수, B는 정답 1개를 2위에 추천해 약 0.63093, C는 정답을 놓쳐 0이다. 세 사람을 같은 비중으로 평균 낸다. 실제 전체 test에서는 158,805명의 사용자 점수를 평균 냈다.';formula=c.macroMath;}
  $('toy-calculation').innerHTML='<h3>'+title+'</h3><p>'+text+'</p>'+(formula?'<div class="formula medium mathblock">'+formula+'</div>':'')+'<p class="note">이 예제에서는 정답 배치와 비인기 개수를 따로 바꾼다. 실제 추천 목록의 상품 구성이 바뀌면 두 지표가 함께 바뀔 수도 있다.</p>';
}
['toy-case','toy-rank','toy-tail'].forEach(id=>$(id).addEventListener(id==='toy-tail'?'input':'change',renderToy));
$('toy-prev').addEventListener('click',()=>{calcStep=Math.max(0,calcStep-1);renderToy();});
$('toy-next').addEventListener('click',()=>{calcStep=Math.min(4,calcStep+1);renderToy();});
$('toy-reset').addEventListener('click',()=>{calcStep=0;$('toy-case').value='one';$('toy-rank').value='3';$('toy-tail').value='2';renderToy();});renderToy();

function arp(value){press('data-arp',value);$('arp-shelf').innerHTML=Array.from({length:10},(_,i)=>'<div class="arp-item '+(value==='A'&&i<5?'head':'')+'">'+(value==='A'&&i<5?'1,200회':'50회')+'</div>').join('');$('arp-formula').innerHTML=D.math[value==='A'?'arpA':'arpB'];$('arp-note').textContent=value==='A'?'인기 상품만 있는 A의 평균 리뷰 수는 625다. 리뷰 5개인 비인기 상품은 추천하지 않았으므로 APLT는 0이다.':'B는 평균 리뷰 수가 50으로 낮아졌지만, 50회도 인기 기준 11회 이상이다. 따라서 비인기 노출은 여전히 0이다. ARP 개선을 곧바로 tail 노출 증가라고 말할 수 없다.';}
onButtons('data-arp',arp);arp('A');

const metricNames={ndcg:'정확도 · NDCG@10',recall:'정답 회수 · Recall@10',aplt:'비인기 노출 · APLT@10 (%)',catalog_coverage:'상품 종류 폭 · Coverage@10 (%)',arp:'평균 인기도 · ARP@10'};
function comparison(){const group=$('result-group').value,a=$('compare-base').value,b=$('compare-target').value,r=D.groups[group].comparisons[a][b];$('comparison-formula').innerHTML=r.formula;$('comparison-note').textContent=a===b?'같은 모델을 비교했으므로 상대 변화는 0%다.':`${names[b]}의 NDCG는 ${names[a]} 기준으로 약 ${Math.abs(r.gain).toFixed(2)}% ${r.gain>0?'높다':'낮다'}. 이것은 기준 NDCG 대비 상대 변화이며, 정답을 맞힌 사용자 비율의 변화가 아니다.`;}
function results(){const group=$('result-group').value,key=$('result-metric').value,g=D.groups[group],max=Math.max(...D.models.map(v=>g.rows[v][key]));$('result-title').textContent=metricNames[key];$('result-unit').textContent=`사용자 ${g.users.toLocaleString('ko-KR')}명 · 학습 모델 3-seed 평균 / 인기순 1회 · 각 지표의 막대는 0부터 시작하며 서로 다른 지표의 막대 길이는 직접 비교하지 않는다.`;
  $('result-bars').innerHTML=D.models.map(v=>`<div class="bar-row" data-result-model="${v}"><span class="bar-name">${names[v]}</span><div class="bar-track"><div class="bar-fill" style="width:${max?g.rows[v][key]/max*100:0}%;background:${colors[v]}"></div></div><span class="bar-number">${g.rows[v].display[key]}</span></div>`).join('');
  const notes={ndcg:'선택한 그룹에서도 XSimGCL의 평균 NDCG가 가장 높고 BPR-MF는 인기순보다 낮다. 이 순위는 현재 데이터·탐색·학습 recipe에서의 관측이다.',recall:'사용자마다 정답 중 몇 개를 추천 10개에서 회수했는지 계산한 뒤 평균 냈다. 표시는 비율 원값이다. 예를 들어 0.007245는 약 0.7245%다.',aplt:g.rows.xsimgcl.aplt>0?'선택한 그룹의 XSimGCL APLT는 0보다 크지만 매우 작아 막대가 거의 보이지 않는다. 오른쪽의 정밀 수치를 읽자. APLT는 정답 여부나 차종 적합성과 별개다.':'이 그룹의 XSimGCL APLT는 실제로 0이다. 전체 그룹의 아주 작은 양수와 구분한다. APLT는 정답 여부나 차종 적합성과 별개다.',catalog_coverage:'해당 그룹의 모든 추천을 합쳐 몇 종류의 상품이 한 번이라도 나왔는지 본다. 그룹 사용자 수가 다르므로 그룹 사이 coverage 차이는 표본 크기의 영향도 받는다. 넓은 coverage만으로 정확도가 높다고 말할 수 없다.',arp:'추천된 상품의 train 리뷰 수 평균이다. 값이 작아도 인기 기준 안의 중간 인기 상품만 보여줄 수 있다. 비인기 상품의 비율은 APLT로 따로 봐야 한다.'};
  $('result-note').textContent=notes[key];comparison();
}
['result-metric','result-group'].forEach(id=>$(id).addEventListener('change',results));['compare-base','compare-target'].forEach(id=>$(id).addEventListener('change',comparison));results();

function depth(value){press('data-depth',value);const r=D.depth.find(r=>r.layers===Number(value));$('depth-ndcg').innerHTML=r.ndcgMath;$('depth-aplt').innerHTML=r.apltMath;$('depth-note').textContent=value==='1'?'1층은 직접 이웃을 한 번 섞는다. 세 층 조건 중 NDCG는 가장 낮고 비인기 노출은 가장 높았다.':value==='2'?'2층에서는 이웃의 이웃 정보까지 전파한다. NDCG는 중간이고, APLT는 세 조건 중 가장 낮았다.':'3층은 세 조건 중 NDCG가 가장 높았다. APLT는 1층보다 낮지만 2층보다는 조금 높다. 모든 인기 편향 지표가 깊이에 따라 같은 방향으로 움직인 것은 아니다.';}
onButtons('data-depth',depth);depth('3');
function cl(value){press('data-cl',value);const on=value==='on',r=D.cl[on?'xsimgcl':'xsimgcl_cl0'];$('cl-loss').innerHTML=D.math[on?'lossOn':'lossOff'];$('cl-ndcg').innerHTML=r.ndcgMath;$('cl-aplt').innerHTML=r.apltMath;$('cl-note').textContent=on?'CL 가중치 0.1: BPR과 L2 외에 층 간 대조 목표를 추가한다. 세 seed 평균 정확도는 높고, 비인기 노출은 매우 낮았다.':'CL 가중치 0: 추가 대조 목표를 계산하지 않는다. 그래프 encoder·노이즈와 나머지 설정은 유지했다. 비인기 노출은 더 크지만 정확도는 CL을 켠 조건보다 낮았다.';}
onButtons('data-cl',cl);cl('on');$('cl-gain').innerHTML=D.cl.gainMath;
function seeds(value){press('data-seed-model',value);const r=D.seedRows[value];$('seed-title').textContent=names[value]+'의 test NDCG';$('seed-scores').innerHTML=r.valueMath.map((v,i)=>'<div><b>seed '+[42,43,44][i]+'</b>'+v+'</div>').join('');$('seed-mean').innerHTML=r.formula;$('seed-sd').innerHTML=r.sdMath;}
onButtons('data-seed-model',seeds);seeds('xsimgcl');
function ci(value){press('data-ci',value);const r=D.ci[value],x=v=>95+v/.0011*760;$('ci-visual').innerHTML=`<svg viewBox="0 0 940 160" role="img" aria-label="seed ${value}의 사용자 paired 구간. 0보다 오른쪽으로 XSimGCL의 평균 NDCG가 높다."><line x1="95" y1="90" x2="875" y2="90" stroke="#b7c4d3" stroke-width="2"/><line x1="95" y1="35" x2="95" y2="115" stroke="#77879b" stroke-dasharray="4 4"/><text x="95" y="140" text-anchor="middle" font-size="15">차이 없음</text><line x1="${x(r.ci95_low)}" y1="75" x2="${x(r.ci95_high)}" y2="75" stroke="#275bdd" stroke-width="5"/><line x1="${x(r.ci95_low)}" y1="62" x2="${x(r.ci95_low)}" y2="88" stroke="#275bdd" stroke-width="3"/><line x1="${x(r.ci95_high)}" y1="62" x2="${x(r.ci95_high)}" y2="88" stroke="#275bdd" stroke-width="3"/><circle cx="${x(r.mean_difference)}" cy="75" r="8" fill="#275bdd"/><text x="${x(r.mean_difference)}" y="43" text-anchor="middle" font-size="16">평균 차이</text><text x="850" y="140" text-anchor="end" font-size="15">오른쪽: XSimGCL이 높음</text></svg>`;$('ci-formula').innerHTML=r.formula;$('ci-note').textContent=`seed ${value}의 사용자 bootstrap 구간 전체가 차이 없는 지점의 오른쪽이다. 해당 seed에서 같은 사용자끼리 비교한 XSimGCL−LightGCN NDCG 차이는 양수였다.`;}
onButtons('data-ci',ci);ci('42');

const questions=[
 {q:'우리 주 모델 중 test NDCG가 가장 높은 모델은?',options:['Popularity','LightGCN','XSimGCL'],answer:2,why:'XSimGCL의 3-seed 평균은 약 0.004419로 가장 높다. 하지만 이 사실만으로 비인기 상품도 더 많이 노출했다고 결론 내릴 수는 없다.'},
 {q:'XSimGCL의 결과로 말할 수 있는 것은?',options:['대조학습이 정확도와 비인기 노출을 모두 개선했다','이번 조건에서 정확도는 높았지만 비인기 노출은 거의 없었다','APLT가 작으니 차종에 잘 맞았다'],answer:1,why:'정확도와 비인기 노출은 다른 축이다. APLT는 차종 적합성도 측정하지 않는다. 우리 데이터는 대조학습의 tail 노출 개선 주장을 지지하지 않았다.'},
 {q:'NDCG 0.004419를 “사용자 0.4419%를 맞혔다”라고 읽어도 될까?',options:['된다','안 된다. 순위 보상을 정규화한 평균이다'],answer:1,why:'NDCG는 정답을 얼마나 위에 놓았는지의 정규화 점수다. 추천에 정답이 하나라도 있는 사용자 비율은 Hit Rate라는 다른 지표다.'},
 {q:'32개 실험 완료는 차종 Mismatch 평가까지 완료했다는 뜻일까?',options:['그렇다','아니다. 차종 특징이 없어 그 평가는 미실시다'],answer:1,why:'32개는 추천 모델 탐색·본학습·보조 실험의 개수다. 팀 제안서의 모든 RQ가 끝난 것은 아니며, 미실시 지표를 0으로 적으면 안 된다.'},
 {q:'seed SD와 사용자 paired CI는 같은 불확실성을 뜻할까?',options:['같다','다르다. 학습 출발점과 사용자 재표집은 다른 축이다'],answer:1,why:'seed SD는 서로 다른 학습 랜덤 출발점의 변동이다. seed별 사용자 CI는 그 seed의 모델에서 사람들을 다시 뽑았을 때 평균 차이가 흔들리는 범위다.'}
];
const answers=Array(questions.length).fill(null);
function renderQuiz(){
  if(!$('quiz-items').children.length){
    $('quiz-items').innerHTML=questions.map((q,i)=>`<div class="question"><h3>${i+1}. ${q.q}</h3><div class="switches">${q.options.map((o,j)=>`<button class="pill" data-question="${i}" data-option="${j}" aria-pressed="false">${o}</button>`).join('')}</div><div id="quiz-feedback-${i}" class="feedback" role="status"></div></div>`).join('');
    document.querySelectorAll('[data-question]').forEach(b=>b.addEventListener('click',()=>{answers[Number(b.dataset.question)]=Number(b.dataset.option);renderQuiz();}));
  }
  document.querySelectorAll('[data-question]').forEach(b=>{const active=answers[Number(b.dataset.question)]===Number(b.dataset.option);b.setAttribute('aria-pressed',String(active));b.classList.toggle('active',active);});
  questions.forEach((q,i)=>{const feedback=$('quiz-feedback-'+i),answered=answers[i]!==null;feedback.classList.toggle('visible',answered);feedback.classList.toggle('correct',answers[i]===q.answer);feedback.textContent=answered?(answers[i]===q.answer?'맞아. ':'다시 생각해 보자. ')+q.why:'';});
  const answered=answers.filter(v=>v!==null).length,correct=answers.filter((v,i)=>v===questions[i].answer).length;
  $('quiz-score').textContent=answered?`${questions.length}문제 중 ${answered}문제 응답 · ${correct}문제 정답. 설명을 읽고 다른 선택으로 바꿔도 돼.`:'아직 답하지 않았어.';
}
$('quiz-reset').addEventListener('click',()=>{answers.fill(null);renderQuiz();});renderQuiz();
const talks={
 short:'<span class="talk-label">30초 · 질문 → 비교 → 결과 → 한계</span><p>저희는 자동차용품 추천에서 정확도를 높이는 것과 비인기 상품을 더 보여주는 것이 함께 가능한지 비교했습니다.</p><p>인기순을 기준으로 BPR-MF, LightGCN, XSimGCL을 같은 test 조건에서 세 seed씩 평가했습니다. XSimGCL의 정확도가 가장 높았지만 비인기 노출은 거의 0이었습니다.</p><p>따라서 이번 실험에서는 정확도 개선과 인기 편향 완화를 별도로 확인해야 한다는 결론을 얻었습니다. 차종 정보를 넣은 후속 평가는 아직 하지 않았습니다.</p>',
 long:'<span class="talk-label">90초 · “왜?”까지 덧붙이기</span><p>자동차용품은 인기 상품만 추천하는 것과 사용자에게 맞는 다양한 상품을 추천하는 것이 다를 수 있습니다. 저희는 먼저 기본 모델이 정확도와 비인기 노출에서 어떤 차이를 보이는지 확인했습니다.</p><p>전처리된 Automotive 5-core를 사용했고, train으로만 학습했습니다. valid로 설정과 best epoch를 고정한 뒤, 같은 warm-start test 사용자 158,805명을 대상으로 전체 학습 상품에서 추천했습니다. 학습 모델 3개는 seed 42, 43, 44를 반복했습니다.</p><p>test NDCG 평균은 XSimGCL 약 0.004419, LightGCN 약 0.003696이었습니다. XSimGCL이 약 19.56% 높았지만, 비인기 노출은 XSimGCL이 약 0.000042%, LightGCN이 약 0.73012%였습니다. 대조 손실만 끈 보조 비교에서도 대조학습은 정확도와 비인기 노출을 함께 높이지 않았습니다.</p><p>각 seed에서 같은 사용자끼리 비교한 NDCG 차이의 bootstrap 구간도 양수였습니다. 다만 이것은 모든 데이터·설정에서의 우월성이나 학습 seed까지 통합한 CI를 뜻하지 않습니다. 특징 파일이 없어 텍스트·차종 재정렬은 후속 과제로 남겼습니다.</p>',
 questions:'<span class="talk-label">예상 질문과 답변</span><h3>Q. 왜 Popularity를 따로 넣었나요?</h3><p>복잡한 개인화 모델이 단순한 인기순보다 실제로 나은지 확인하기 위해서입니다. 이번 BPR-MF가 기준선보다 낮았다는 결과도 그대로 보고합니다.</p><h3>Q. 대조학습이 더 좋아졌다고 결론 내릴 수 있나요?</h3><p>이번 recipe에서 정확도는 개선됐다고 말할 수 있습니다. 다만 tail 노출은 거의 없었고 CL=0 세 실행은 100 epoch 상한에 도달했습니다. 일반적인 우월성이나 모든 인기 편향의 완화를 주장하지 않습니다.</p><h3>Q. 점수가 낮은데 성공한 모델인가요?</h3><p>전체 상품 평가에서 상대적인 차이를 확인한 결과입니다. 낮은 절대 점수의 원인을 이 실험만으로 확정하지 않았고, 실제 서비스 품질을 입증한 것으로 보지 않습니다. NDCG를 적중 사용자 비율로 읽어서도 안 됩니다.</p><h3>Q. 자동차 차종에 맞는 추천을 한 건가요?</h3><p>아직 아닙니다. 기본 모델은 ID와 상호작용만 썼습니다. 차종·상품 fitment·텍스트를 전달받아 별도로 평가해야 합니다.</p><h3>Q. 그래프가 깊을수록 인기 편향이 커졌나요?</h3><p>1층 대비 3층에서 정확도는 높고 APLT는 낮았습니다. 하지만 2층과 3층의 APLT는 단조 감소하지 않았고 ARP도 별도로 봐야 합니다. 지표별 결과를 구분해 설명합니다.</p>'
};
function talk(value){press('data-talk',value);$('talk-content').innerHTML=talks[value];}
onButtons('data-talk',talk);talk('short');
$('source-hashes').innerHTML=Object.entries(D.sources).map(([name,hash])=>'<p><strong>'+name+'</strong><br><code>'+hash+'</code></p>').join('');
document.documentElement.dataset.ready='true';
})();
