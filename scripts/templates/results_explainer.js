(() => {
'use strict';
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const SVGNS = 'http://www.w3.org/2000/svg';
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function el(tag, attrs = {}, text) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
// TEX templates: MathML generated from LaTeX at build time; numeric placeholders 9000001.. are replaced here.
function tex(name, vals = {}) {
  let s = TEX[name];
  for (const [k, v] of Object.entries(vals)) s = s.split('<mn>' + k + '</mn>').join('<mn>' + v + '</mn>');
  return s;
}
const mathWrap = (inner, block) => `<math xmlns="http://www.w3.org/1998/Math/MathML"${block ? ' display="block"' : ''}><mrow>${inner}</mrow></math>`;
function pressGroup(buttons, active) { buttons.forEach(b => b.setAttribute('aria-pressed', String(b === active))); }

const M = DATA.models; // variant -> group -> stats
const NAMES = {popularity: 'Popularity', bpr_mf: 'BPR-MF', lightgcn: 'LightGCN', xsimgcl: 'XSimGCL', xsimgcl_cl0: 'XSimGCL (CL=0)'};
const COLOR = {popularity: '--c-pop', bpr_mf: '--c-bpr', lightgcn: '--c-lgcn', xsimgcl: '--c-xsim', xsimgcl_cl0: '--c-cl0'};
const f4 = x => x.toFixed(4);
const f5 = x => x.toFixed(5);
const pct = (x, d = 3) => (x * 100).toFixed(d) + '%';
const comma = x => Math.round(x).toLocaleString('ko-KR');

/* ---------- hero scatter ---------- */
let group = 'all', showCL0 = false, selected = 'xsimgcl';
function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v))); for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p; return 10 * p; }
function drawScatter() {
  const svg = $('#scatter'); svg.textContent = '';
  const W = 760, H = 420, L = 78, R = 30, T = 24, B = 58;
  const keys = ['popularity', 'bpr_mf', 'lightgcn', 'xsimgcl'].concat(showCL0 ? ['xsimgcl_cl0'] : []);
  const pts = keys.map(k => ({k, s: M[k][group]}));
  const xMax = niceMax(Math.max(...pts.map(p => (p.s.aplt + (p.s.aplt_sd || 0)) * 100)) * 1.12);
  const yMax = niceMax(Math.max(...pts.map(p => p.s.ndcg + (p.s.ndcg_sd || 0))) * 1.1);
  const X = v => L + (v / xMax) * (W - L - R), Y = v => H - B - (v / yMax) * (H - T - B);
  const line = css('--line'), soft = css('--soft');
  for (let i = 0; i <= 5; i++) {
    const yv = yMax * i / 5, xv = xMax * i / 5;
    svg.append(el('line', {x1: L, x2: W - R, y1: Y(yv), y2: Y(yv), stroke: line, 'stroke-width': 1}));
    svg.append(el('text', {x: L - 10, y: Y(yv) + 4, 'text-anchor': 'end', 'font-size': 12, class: 'muted'}, yv.toFixed(4)));
    svg.append(el('line', {x1: X(xv), x2: X(xv), y1: T, y2: H - B, stroke: line, 'stroke-width': 1, 'stroke-dasharray': i ? '2 4' : ''}));
    svg.append(el('text', {x: X(xv), y: H - B + 20, 'text-anchor': 'middle', 'font-size': 12, class: 'muted'}, (+xv.toPrecision(3)) + '%'));
  }
  svg.append(el('text', {x: (L + W - R) / 2, y: H - 12, 'text-anchor': 'middle', 'font-size': 13}, '→ 추천 10칸 중 비인기 상품 비율 (APLT@10)'));
  const yl = el('text', {x: 18, y: (T + H - B) / 2, 'text-anchor': 'middle', 'font-size': 13, transform: `rotate(-90 18 ${(T + H - B) / 2})`}, '정답을 잘 맞힘 (NDCG@10) →');
  svg.append(yl);
  for (const p of pts) {
    const c = css(COLOR[p.k]), cx = X(p.s.aplt * 100), cy = Y(p.s.ndcg);
    const g = el('g', {tabindex: 0, role: 'button', 'aria-label': `${NAMES[p.k]} 자세히 보기`, style: 'cursor:pointer'});
    if (p.s.ndcg_sd) g.append(el('line', {x1: cx, x2: cx, y1: Y(p.s.ndcg - p.s.ndcg_sd), y2: Y(p.s.ndcg + p.s.ndcg_sd), stroke: c, 'stroke-width': 2}));
    if (p.s.aplt_sd) g.append(el('line', {x1: X(Math.max(0, p.s.aplt - p.s.aplt_sd) * 100), x2: X((p.s.aplt + p.s.aplt_sd) * 100), y1: cy, y2: cy, stroke: c, 'stroke-width': 2}));
    const hollow = p.k === 'xsimgcl_cl0';
    g.append(el('circle', {cx, cy, r: selected === p.k ? 11 : 8, fill: hollow ? css('--paper') : c, stroke: c, 'stroke-width': hollow ? 3 : (selected === p.k ? 3 : 0)}));
    if (selected === p.k && !hollow) g.append(el('circle', {cx, cy, r: 15, fill: 'none', stroke: c, 'stroke-width': 1.5, opacity: .5}));
    const right = cx < W - 200;
    let ly = cy + 4;
    if (p.k === 'xsimgcl_cl0') ly = cy + 22; if (p.k === 'xsimgcl' && showCL0) ly = cy - 12;
    g.append(el('text', {x: right ? cx + 16 : cx - 16, y: ly, 'text-anchor': right ? 'start' : 'end', 'font-size': 14, 'font-weight': 700}, NAMES[p.k]));
    const pick = () => { selected = p.k; drawScatter(); };
    g.addEventListener('click', pick);
    g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    svg.append(g);
  }
  drawDetail();
}
function drawDetail() {
  if (selected === 'xsimgcl_cl0' && !showCL0) selected = 'xsimgcl';
  const s = M[selected][group];
  const sd = (v, d) => v == null ? '' : ` ± ${v.toFixed(d)}`;
  const items = [
    ['모델', NAMES[selected]],
    ['NDCG@10', f5(s.ndcg) + (s.ndcg_sd ? `<span class="small"> ±${s.ndcg_sd.toFixed(5)}</span>` : '')],
    ['Recall@10', f5(s.recall)],
    ['Hit rate@10', pct(s.hit, 2)],
    ['APLT@10 (비인기 비율)', pct(s.aplt, s.aplt < 1e-5 ? 5 : 3)],
    ['Coverage@10 (등장 상품 종류)', `${pct(s.cov, 2)} <span class="small">≈${comma(s.cov * DATA.n_items)}종</span>`],
  ];
  $('#detail').innerHTML = items.map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
  const note = {
    popularity: '학습 없이 train 리뷰가 많은 순서로 모두에게 같은 10개를 보여줘요. 이미 본 상품을 빼면 몇 개가 바뀌어서 전체 등장 상품은 14종뿐이에요. 비인기 비율은 정확히 0이에요.',
    bpr_mf: '취향 벡터만 배운 기본 모델이에요. 인기순보다도 NDCG가 낮았어요. 이번 탐색 설정에서는 best epoch가 5로 일찍 멈췄어요.',
    lightgcn: '비인기 상품을 가장 많이 보여줬어요. 그래도 10칸 중 비인기는 평균 0.07칸 정도예요(0.73%). 특히 과거 리뷰가 1–5개인 사용자에게서 높았어요.',
    xsimgcl: '정확도 1위예요. 하지만 비인기 비율은 0.00004%, 즉 사실상 0이고 등장 상품 종류도 가장 적은 축이에요.',
    xsimgcl_cl0: '본실험 XSimGCL에서 대조학습 가중치만 0으로 바꾼 보조 실험이에요. 정확도는 조금 낮고 비인기 비율은 조금 높아요.',
  }[selected];
  const g = group === 'all' ? '전체 158,805명' : group === '1-5' ? '과거 리뷰 1–5개 87,394명' : '과거 리뷰 6개 이상 71,411명';
  $('#detail-note').textContent = `${g} 기준. ${note}`;
}
$$('[data-group]').forEach(b => b.addEventListener('click', () => { group = b.dataset.group; pressGroup($$('[data-group]'), b); drawScatter(); }));
$('#toggle-cl0').addEventListener('click', e => { showCL0 = !showCL0; e.currentTarget.setAttribute('aria-pressed', String(showCL0)); if (showCL0) selected = 'xsimgcl_cl0'; drawScatter(); });

/* ---------- timeline ---------- */
const STEPS = [
  {hl: 0, t: '<b>train 구간으로만 학습해요.</b> 그래프, 음성 샘플(사용자가 리뷰하지 않은 상품), 인기도와 사용자 그룹도 모두 train으로만 만들어요. 이 구간 리뷰는 4,818,321개, 사용자 599,742명, 상품 248,186개예요.'},
  {hl: 1, t: '<b>valid 구간으로 설정을 골라요.</b> 학습률, 층 수, 몇 번째 epoch의 모델을 쓸지(best checkpoint), 언제 멈출지(early stopping)를 valid 점수만 보고 정해요. 이미 train에서 본 상품은 채점에서 빼요.'},
  {hl: 2, t: '<b>설정을 얼려요.</b> 고른 설정으로 seed 42·43·44를 따로 학습해요. train+valid를 합쳐 다시 학습하지 않았어요. 이후 test 점수를 보고 바꾼 설정은 하나도 없어요.'},
  {hl: 3, t: '<b>test로 딱 한 번 채점해요.</b> 모든 모델이 같은 158,805명, 같은 정답, 같은 후보 248,186개 전체로 채점받아요. train과 valid에서 이미 리뷰한 상품은 10칸에 넣지 않아요.'},
];
let step = 0;
function drawTimeline() {
  const svg = $('#timeline'); svg.textContent = '';
  const segs = [{x: 20, w: 470, name: 'train', sub: '공부하는 기간'}, {x: 500, w: 115, name: 'valid', sub: '모의고사'}, {x: 625, w: 115, name: 'test', sub: '본시험'}];
  const acc = css('--accent'), panel = css('--paper'), line = css('--line');
  const active = STEPS[step].hl;
  const lit = active === 2 ? [0, 1] : [active];
  segs.forEach((s, i) => {
    const on = lit.includes(i);
    svg.append(el('rect', {x: s.x, y: 30, width: s.w, height: 62, rx: 8, fill: on ? acc : panel, stroke: on ? acc : line, 'stroke-width': 2}));
    svg.append(el('text', {x: s.x + s.w / 2, y: 58, 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 700, style: on ? `fill:${panel}` : ''}, s.name));
    svg.append(el('text', {x: s.x + s.w / 2, y: 80, 'text-anchor': 'middle', 'font-size': 13, style: on ? `fill:${panel}` : '', class: on ? '' : 'muted'}, s.sub));
  });
  svg.append(el('line', {x1: 20, x2: 740, y1: 112, y2: 112, stroke: line, 'stroke-width': 2}));
  svg.append(el('text', {x: 20, y: 134, 'font-size': 13, class: 'muted'}, '과거'));
  svg.append(el('text', {x: 740, y: 134, 'font-size': 13, 'text-anchor': 'end', class: 'muted'}, '미래 →'));
  const lab = ['학습에 사용', '설정·best epoch 선택', '설정 동결', '최종 평가 1회'][step];
  svg.append(el('text', {x: 380, y: 160, 'font-size': 14, 'text-anchor': 'middle', 'font-weight': 700}, `지금 단계: ${lab}`));
  $('#stepnote').innerHTML = STEPS[step].t;
}
$$('[data-step]').forEach(b => b.addEventListener('click', () => { step = +b.dataset.step; pressGroup($$('[data-step]'), b); drawTimeline(); }));

/* ---------- shelf calculator ---------- */
const shelf = Array.from({length: 10}, () => ({hit: false, tail: false}));
let G = 1, mode = 'hit';
const disc = r => 1 / Math.log2(r + 1);
const PRESETS = {
  a: {G: 1, hits: [3], tails: []},
  b: {G: 2, hits: [1], tails: []},
  c: {G: 1, hits: [], tails: []},
  d: {G: 1, hits: [3], tails: [5, 9]},
};
function applyPreset(p) { shelf.forEach((s, i) => { s.hit = p.hits.includes(i + 1); s.tail = p.tails.includes(i + 1); }); G = p.G; drawShelf(); }
function hitsCount() { return shelf.filter(s => s.hit).length; }
function drawShelf() {
  const box = $('#shelf'); box.textContent = '';
  shelf.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'slot' + (s.hit ? ' hit' : '') + (s.tail ? ' tail' : '');
    b.setAttribute('aria-label', `${i + 1}위 칸: ${s.hit ? '정답' : '정답 아님'}, ${s.tail ? '비인기' : '인기'} 상품`);
    b.innerHTML = `<span class="r">${i + 1}위</span><span class="ico"></span><span class="tag">${s.hit ? '정답' : '오답'}<br>${s.tail ? '<span style="color:var(--tail)">비인기</span>' : '인기'}</span>`;
    b.addEventListener('click', () => {
      if (mode === 'hit') { s.hit = !s.hit; if (hitsCount() > G) G = hitsCount(); } else s.tail = !s.tail;
      drawShelf(); box.children[i].focus();
    });
    box.append(b);
  });
  compute();
}
function compute() {
  const hits = hitsCount(), tails = shelf.filter(s => s.tail).length;
  const hitRanks = shelf.map((s, i) => s.hit ? i + 1 : 0).filter(Boolean);
  const dcg = hitRanks.reduce((a, r) => a + disc(r), 0);
  const kIdeal = Math.min(G, 10);
  let idcg = 0; for (let r = 1; r <= kIdeal; r++) idcg += disc(r);
  const ndcg = dcg / idcg, recall = hits / G, aplt = tails / 10;
  $('#g-out').textContent = G;
  $('#g-hint').textContent = G > hits ? `${G - hits}개는 10칸 밖에 있어요.` : '정답이 모두 10칸 안에 있어요.';
  $('#results').innerHTML = [['NDCG@10', f4(ndcg)], ['Recall@10', f4(recall)], ['Hit@10', hits > 0 ? '1' : '0'], ['APLT@10', aplt.toFixed(1)]]
    .map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
  // Build the worked calculation from LaTeX-derived fragments.
  const plus = '<mo>+</mo>';
  const dcgTerms = hitRanks.length ? hitRanks.map(r => TEX['term' + r]).join(plus) : '<mn>0</mn>';
  const idTerms = Array.from({length: kIdeal}, (_, i) => TEX['term' + (i + 1)]).join(plus);
  const lines = [
    TEX.dcgL + dcgTerms + tex('approx', {9000001: f4(dcg)}),
    TEX.idcgL + idTerms + tex('approx', {9000001: f4(idcg)}),
    tex('ndcgL', {9000001: f4(dcg), 9000002: f4(idcg), 9000003: f4(ndcg)}),
    tex('recallL', {9000001: hits, 9000002: G, 9000003: f4(recall)}) + TEX.sep + tex('apltL', {9000001: tails, 9000003: aplt.toFixed(1)}),
  ];
  $('#calc').innerHTML = lines.map(l => mathWrap(l, true)).join('');
  let why;
  if (!hits) why = `10칸 안에 정답이 없으면 DCG가 0이라 NDCG와 Recall이 모두 0이에요. 실제 사용자 대부분이 이 경우예요. 그래도 IDCG는 정답 ${G}개 기준으로 ${f4(idcg)}이에요.`;
  else {
    const first = hitRanks[0];
    why = `정답이 ${hitRanks.map(r => r + '위').join(', ')}에 있어서 각 칸 점수를 더해 DCG ${f4(dcg)}가 됐어요. ${first}위 칸 점수는 1/log₂(${first + 1}) = ${f4(disc(first))}예요. ` +
      `가장 좋은 배치는 정답 ${kIdeal}개를 1위부터 채우는 것이고 그 점수(IDCG)가 ${f4(idcg)}예요. 나누면 NDCG ${f4(ndcg)}.` +
      (G > hits ? ` 정답 ${G - hits}개는 놓쳐서 Recall은 ${hits}/${G}예요.` : '');
  }
  why += ` 비인기 칸은 ${tails}개라 APLT는 ${aplt.toFixed(1)}이에요. APLT는 정답 여부와 상관없이 무엇을 보여줬는지만 셉니다.`;
  $('#calc-why').textContent = why;
}
$('#mode-hit').addEventListener('click', e => { mode = 'hit'; pressGroup([$('#mode-hit'), $('#mode-tail')], e.currentTarget); });
$('#mode-tail').addEventListener('click', e => { mode = 'tail'; pressGroup([$('#mode-hit'), $('#mode-tail')], e.currentTarget); });
$('#g-minus').addEventListener('click', () => { G = Math.max(Math.max(1, hitsCount()), G - 1); compute(); });
$('#g-plus').addEventListener('click', () => { G = Math.min(12, G + 1); compute(); });
$$('[data-preset]').forEach(b => b.addEventListener('click', () => applyPreset(PRESETS[b.dataset.preset])));

/* ---------- dots ---------- */
const perm = (() => { const a = Array.from({length: 10000}, (_, i) => i); let s = 12345; for (let i = a.length - 1; i > 0; i--) { s = (s * 1103515245 + 12345) & 0x7fffffff; const j = s % (i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; })();
function drawDots(k) {
  const cv = $('#dots'), ctx = cv.getContext('2d');
  const s = M[k].all, n = Math.round(s.hit * 10000);
  ctx.fillStyle = css('--line'); ctx.fillRect(0, 0, 400, 400);
  ctx.fillStyle = css('--paper');
  for (let i = 0; i < 10000; i++) { const x = i % 100, y = Math.floor(i / 100); ctx.fillRect(x * 4, y * 4, 3, 3); }
  ctx.fillStyle = css(COLOR[k]);
  for (let i = 0; i < n; i++) { const p = perm[i], x = p % 100, y = Math.floor(p / 100); ctx.fillRect(x * 4 - 1, y * 4 - 1, 5, 5); }
  $('#dot-n').textContent = n;
  $('#dot-n').style.color = css(COLOR[k]);
  $('#dot-text').innerHTML = `${NAMES[k]}는 Hit rate ${pct(s.hit, 2)}, NDCG 평균 <b>${f5(s.ndcg)}</b>예요. 칠해지지 않은 ${comma(10000 - n)}명은 NDCG 0점이고, 칠해진 사람도 정답 순위에 따라 1점 이하를 받아요. 158,805명 평균이라 숫자가 작아지지만 모델 간 차이는 같은 시험에서 잰 것이라 비교할 수 있어요.`;
}
$$('[data-dot]').forEach(b => b.addEventListener('click', () => { pressGroup($$('[data-dot]'), b); drawDots(b.dataset.dot); }));

/* ---------- model cards ---------- */
const CARDS = [
  {k: 'popularity', what: '많이 팔린 순', how: 'train 리뷰 수가 많은 상품부터 모두에게 똑같이 보여줘요. 학습이 없는 기준선이라 한 번만 계산했어요.', draw: 'bars'},
  {k: 'bpr_mf', what: '취향 벡터 맞추기', how: '사용자와 상품을 각각 벡터 하나로 표현해요. 리뷰한 상품이 리뷰 안 한 상품보다 점수가 높아지도록 벡터를 조정해요.', draw: 'vec'},
  {k: 'lightgcn', what: '이웃 따라가기', how: '사용자–상품 연결 그래프를 따라 벡터를 평균 내며 퍼뜨려요. 비슷한 사람이 산 상품이 내 벡터에 섞여 들어와요.', draw: 'graph'},
  {k: 'xsimgcl', what: '흔들어도 같은 나', how: 'LightGCN처럼 퍼뜨리되, 벡터에 작은 잡음을 섞은 두 버전이 "같은 사용자"로 인식되도록 대조학습 손실을 더해요.', draw: 'noise'},
];
function miniSVG(kind, color) {
  const s = el('svg', {viewBox: '0 0 220 110', class: 'fig', 'aria-hidden': 'true'});
  const line = css('--line'), soft = css('--soft');
  if (kind === 'bars') { [90, 70, 52, 30, 18, 10, 6].forEach((h, i) => s.append(el('rect', {x: 14 + i * 28, y: 100 - h, width: 20, height: h, rx: 3, fill: i < 3 ? color : line}))); s.append(el('text', {x: 14, y: 12, 'font-size': 11, class: 'muted'}, '리뷰 수 →')); }
  if (kind === 'vec') {
    s.append(el('line', {x1: 110, y1: 90, x2: 40, y2: 25, stroke: color, 'stroke-width': 4, 'stroke-linecap': 'round'}));
    s.append(el('line', {x1: 110, y1: 90, x2: 70, y2: 15, stroke: soft, 'stroke-width': 4, 'stroke-linecap': 'round'}));
    s.append(el('line', {x1: 110, y1: 90, x2: 200, y2: 70, stroke: line, 'stroke-width': 4, 'stroke-linecap': 'round'}));
    s.append(el('text', {x: 20, y: 22, 'font-size': 11}, '나'));
    s.append(el('text', {x: 72, y: 12, 'font-size': 11, class: 'muted'}, '추천'));
    s.append(el('text', {x: 168, y: 88, 'font-size': 11, class: 'muted'}, '안 맞음'));
  }
  if (kind === 'graph' || kind === 'noise') {
    const U = [[30, 25], [30, 85]], I = [[110, 15], [110, 55], [110, 95]], I2 = [[190, 35], [190, 80]];
    const E = [[U[0], I[0]], [U[0], I[1]], [U[1], I[1]], [U[1], I[2]], [I[1], I2[0]], [I[2], I2[1]]];
    E.forEach(([a, b], j) => s.append(el('line', {x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: j < 2 ? color : line, 'stroke-width': j < 2 ? 3 : 2})));
    U.forEach(([x, y], j) => s.append(el('circle', {cx: x, cy: y, r: 9, fill: j === 0 ? color : soft})));
    I.concat(I2).forEach(([x, y]) => s.append(el('rect', {x: x - 8, y: y - 8, width: 16, height: 16, rx: 3, fill: css('--paper'), stroke: soft, 'stroke-width': 2})));
    if (kind === 'noise') { s.append(el('circle', {cx: 30, cy: 25, r: 16, fill: 'none', stroke: color, 'stroke-dasharray': '3 3', 'stroke-width': 2})); s.append(el('text', {x: 52, y: 12, 'font-size': 10, class: 'muted'}, '+잡음 두 버전')); }
  }
  return s;
}
const box = $('#modelcards');
CARDS.forEach(c => {
  const s = M[c.k].all;
  const d = document.createElement('div'); d.className = 'card';
  d.innerHTML = `<h3><span class="sw" style="background:var(${COLOR[c.k]})"></span>${NAMES[c.k]}</h3><p class="small" style="color:var(--ink);font-weight:600">${c.what}</p>`;
  d.append(miniSVG(c.draw, css(COLOR[c.k])));
  const p = document.createElement('p'); p.className = 'small'; p.textContent = c.how; d.append(p);
  const r = document.createElement('p'); r.className = 'small tnum'; r.innerHTML = `NDCG <b style="color:var(--ink)">${f5(s.ndcg)}</b> · 비인기 <b style="color:var(--ink)">${pct(s.aplt, s.aplt < 1e-5 ? 5 : 3)}</b>`; d.append(r);
  box.append(d);
});

/* ---------- CL bars ---------- */
let cl = 'cl1';
function drawCL() {
  const svg = $('#clbars'); svg.textContent = '';
  const a = M.xsimgcl_cl0.all, b = M.xsimgcl.all;
  const rows = [
    {name: 'NDCG@10', v0: a.ndcg, v1: b.ndcg, fmt: f5},
    {name: 'APLT@10 (비인기)', v0: a.aplt, v1: b.aplt, fmt: v => pct(v, v < 1e-5 ? 5 : 3)},
    {name: 'Coverage@10', v0: a.cov, v1: b.cov, fmt: v => pct(v, 2)},
    {name: 'ARP (평균 리뷰 수)', v0: a.arp, v1: b.arp, fmt: v => comma(v)},
  ];
  const L = 170, W = 760, maxW = W - L - 150;
  rows.forEach((r, i) => {
    const y = 20 + i * 70, mx = Math.max(r.v0, r.v1);
    svg.append(el('text', {x: 0, y: y + 30, 'font-size': 14, 'font-weight': 700}, r.name));
    [['v0', 'CL=0', 0], ['v1', 'CL=0.1', 1]].forEach(([key, lab, j]) => {
      const on = (cl === 'cl0') === (j === 0);
      const w = Math.max(2, r[key] / mx * maxW), yy = y + j * 24;
      svg.append(el('rect', {x: L, y: yy + 6, width: w, height: 18, rx: 3, fill: on ? css('--c-xsim') : css('--line')}));
      svg.append(el('text', {x: L + w + 8, y: yy + 20, 'font-size': 13, 'font-weight': on ? 700 : 400, class: on ? '' : 'muted'}, `${lab}  ${r.fmt(r[key])}`));
    });
  });
  const rel = (b.ndcg - a.ndcg) / a.ndcg * 100;
  $('#cl-text').innerHTML = cl === 'cl1'
    ? `대조학습을 켜면 NDCG가 <b>${rel.toFixed(2)}%</b> 올라요. 대신 비인기 비율은 ${pct(a.aplt)} → ${pct(b.aplt, 5)}로 떨어지고, 등장하는 상품 종류도 ${pct(a.cov, 2)} → ${pct(b.cov, 2)}로 줄어요.`
    : `대조학습을 끄면 정확도는 조금 내려가지만(NDCG ${f5(a.ndcg)}), 비인기 비율 ${pct(a.aplt)}, 등장 상품 ${pct(a.cov, 2)}로 켰을 때보다 다양해져요. 그래도 LightGCN(비인기 ${pct(M.lightgcn.all.aplt)})보다는 낮아요.`;
}
$$('[data-cl]').forEach(b => b.addEventListener('click', () => { cl = b.dataset.cl; pressGroup($$('[data-cl]'), b); drawCL(); }));
// exposure table
(() => {
  const ks = ['popularity', 'bpr_mf', 'lightgcn', 'xsimgcl_cl0', 'xsimgcl'];
  $('#exposure').innerHTML = '<thead><tr><th>모델</th><th>비인기 비율</th><th>등장 상품</th><th>ARP</th></tr></thead><tbody>' +
    ks.map(k => { const s = M[k].all; return `<tr><td>${NAMES[k]}</td><td>${pct(s.aplt, s.aplt < 1e-5 ? 5 : 3)}</td><td>≈${comma(s.cov * DATA.n_items)}종</td><td>${comma(s.arp)}</td></tr>`; }).join('') + '</tbody>';
})();

/* ---------- depth ---------- */
function drawDepth(L) {
  const svg = $('#depthfig'); svg.textContent = '';
  const D = DATA.depth; // [{layers, ndcg, ndcg_sd, aplt, aplt_sd}]
  const panels = [{key: 'ndcg', sd: 'ndcg_sd', name: 'NDCG@10', x0: 40, fmt: f5, max: 0.004}, {key: 'aplt', sd: 'aplt_sd', name: 'APLT@10 (비인기)', x0: 420, fmt: v => pct(v), max: 0.014}];
  panels.forEach(p => {
    svg.append(el('text', {x: p.x0, y: 18, 'font-size': 14, 'font-weight': 700}, p.name));
    svg.append(el('line', {x1: p.x0, x2: p.x0 + 300, y1: 210, y2: 210, stroke: css('--line'), 'stroke-width': 2}));
    D.forEach((d, i) => {
      const h = d[p.key] / p.max * 160, x = p.x0 + 20 + i * 95, on = d.layers === L;
      svg.append(el('rect', {x, y: 210 - h, width: 60, height: h, rx: 4, fill: on ? css('--c-lgcn') : css('--line')}));
      const e = d[p.sd] / p.max * 160;
      svg.append(el('line', {x1: x + 30, x2: x + 30, y1: 210 - h - e, y2: 210 - h + e, stroke: css('--ink'), 'stroke-width': 1.5}));
      svg.append(el('text', {x: x + 30, y: 210 - h - e - 8, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': on ? 700 : 400}, p.fmt(d[p.key])));
      svg.append(el('text', {x: x + 30, y: 230, 'text-anchor': 'middle', 'font-size': 13, class: on ? '' : 'muted'}, d.layers + '층'));
    });
  });
  const d = D.find(x => x.layers === L);
  $('#depth-out').textContent = L + '층';
  $('#depth-text').innerHTML = {
    1: `1층은 내가 리뷰한 상품과 바로 연결된 신호만 써요. NDCG ${f5(d.ndcg)}로 가장 낮지만 비인기 비율은 ${pct(d.aplt)}로 가장 높아요.`,
    2: `2층부터 "나와 같은 상품을 리뷰한 사람" 신호가 들어와요. NDCG가 ${f5(d.ndcg)}로 올랐고, 비인기 비율은 ${pct(d.aplt)}로 1층보다 낮아요.`,
    3: `3층(본실험 설정)이 NDCG ${f5(d.ndcg)}로 가장 높아요. 비인기 비율 ${pct(d.aplt)}는 2층과 거의 같아요. 층을 늘려서 정확도가 오를 때 비인기 노출은 1층보다 낮아졌어요.`,
  }[L] + ' 막대 위 가는 선은 seed 3개의 표준편차예요.';
}
$('#depth-range').addEventListener('input', e => drawDepth(+e.target.value));

/* ---------- paired CI ---------- */
const P = DATA.paired; // {seed,left,right,metric,mean,lo,hi}
const pairs = [];
P.forEach(r => { const id = r.left + '|' + r.right; if (!pairs.includes(id)) pairs.push(id); });
const sel = $('#pair');
pairs.forEach(id => { const [l, r] = id.split('|'); const o = document.createElement('option'); o.value = id; o.textContent = `${NAMES[r]} − ${NAMES[l]}`; sel.append(o); });
sel.value = 'lightgcn|xsimgcl';
let metric = 'ndcg';
function drawCI() {
  const [l, r] = sel.value.split('|');
  const rows = P.filter(x => x.left === l && x.right === r && x.metric === metric).sort((a, b) => a.seed - b.seed);
  $('#pair-formula').innerHTML = mathWrap(TEX['pair_' + metric], true);
  const svg = $('#cifig'); svg.textContent = '';
  const lo = Math.min(0, ...rows.map(x => x.lo)), hi = Math.max(0, ...rows.map(x => x.hi));
  const pad = (hi - lo) * 0.12 || 1e-6, a = lo - pad, b = hi + pad;
  const L = 90, W = 740, X = v => L + (v - a) / (b - a) * (W - L);
  const scale = metric === 'aplt' ? 100 : 1, unit = metric === 'aplt' ? '%p' : '';
  const fmt = v => (v * scale).toFixed(metric === 'aplt' ? 3 : 5) + unit;
  svg.append(el('line', {x1: X(0), x2: X(0), y1: 10, y2: 170, stroke: css('--soft'), 'stroke-width': 1.5, 'stroke-dasharray': '4 4'}));
  svg.append(el('text', {x: X(0), y: 192, 'text-anchor': 'middle', 'font-size': 12, class: 'muted'}, '0 (차이 없음)'));
  const c = css(COLOR[r]);
  rows.forEach((x, i) => {
    const y = 35 + i * 50;
    svg.append(el('text', {x: 0, y: y + 5, 'font-size': 14, 'font-weight': 700}, `seed ${x.seed}`));
    svg.append(el('line', {x1: X(x.lo), x2: X(x.hi), y1: y, y2: y, stroke: c, 'stroke-width': 6, 'stroke-linecap': 'round'}));
    svg.append(el('circle', {cx: X(x.mean), cy: y, r: 7, fill: css('--paper'), stroke: c, 'stroke-width': 3}));
    svg.append(el('text', {x: X(x.mean), y: y - 14, 'text-anchor': 'middle', 'font-size': 12}, `${fmt(x.mean)}  [${fmt(x.lo)}, ${fmt(x.hi)}]`));
  });
  const allPos = rows.every(x => x.lo > 0), allNeg = rows.every(x => x.hi < 0), allZero = rows.every(x => x.lo === 0 && x.hi === 0);
  const mName = metric === 'ndcg' ? 'NDCG' : '비인기 비율';
  let t;
  if (allZero) t = `두 모델 모두 비인기 상품을 사실상 하나도 추천하지 않아서 차이가 0이에요.`;
  else if (allPos) t = `세 seed 모두 구간 전체가 0보다 오른쪽이에요. 같은 사용자에게서 ${NAMES[r]}의 ${mName}가 ${NAMES[l]}보다 높다는 결과가 seed마다 반복됐어요.`;
  else if (allNeg) t = `세 seed 모두 구간 전체가 0보다 왼쪽이에요. 같은 사용자에게서 ${NAMES[r]}의 ${mName}가 ${NAMES[l]}보다 낮다는 결과가 seed마다 반복됐어요.`;
  else t = `구간이 0에 걸치거나 seed마다 방향이 달라요. 이 비교만으로 어느 쪽이 높다고 말하기 어려워요.`;
  $('#ci-text').textContent = t + ' 점은 평균 차이, 굵은 선은 사용자 bootstrap 95% 구간이에요.';
}
sel.addEventListener('change', drawCI);
$$('[data-metric]').forEach(b => b.addEventListener('click', () => { metric = b.dataset.metric; pressGroup($$('[data-metric]'), b); drawCI(); }));

/* ---------- boot & theme ---------- */
function drawAll() {
  drawScatter(); drawTimeline(); drawDepth(+$('#depth-range').value); drawCI(); drawCL();
  const k = ($$('[data-dot]').find(b => b.getAttribute('aria-pressed') === 'true') || {}).dataset?.dot || 'xsimgcl'; drawDots(k);
}
$('#gsym').innerHTML = mathWrap(TEX.G, false);
applyPreset(PRESETS.a);
drawAll();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
new MutationObserver(drawAll).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
})();
