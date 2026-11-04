# 로컬 학습 가능성 점검

2026-11-05 측정. M3 Max 14-core CPU / 30-core GPU / 통합 메모리 36GB.
프로젝트 `.venv`의 PyTorch 2.15.0을 사용했다.

## 결론

로컬 MPS에서도 첫 학습은 가능해 보인다. 다만 사용자가 기존 Colab CU 사용과 로컬 부하 최소화를 선택했으므로,
**실제 학습 환경은 Colab GPU로 변경했다.** [Colab 실행 환경](COLAB.md)을 따른다.
여러 하이퍼파라미터와 seed를 반복하면 실행 시간이 늘어나므로, 실제 학습기에서
첫 epoch와 전체 검증 시간을 측정한 뒤 전체 실험 예산을 정한다.
외부 CUDA GPU가 더 빠른지는 할당받은 GPU에서 같은 코드를 측정해야 판단할 수 있다.

## 실행과 결과

```bash
cd ~/git/automotive-recsys
uv sync --locked --extra training
uv run --extra training python scripts/benchmark_local.py
```

입력: 전체 train 4,818,321개 상호작용. 사용자+상품 노드 847,928개.
설정: 임베딩 64차원, batch size 8,192, Adam. CPU는 6개 스레드를 사용했다.
양방향 사용자–상품 그래프는 train만으로 구성했다.

| 측정 | 업데이트당 실측 | epoch 학습 연산 단순 환산 |
|---|---:|---:|
| BPR-MF CPU | 0.0563초 | 약 33초 |
| BPR-MF MPS GPU | 0.0329초 | 약 19초 |
| LightGCN 3층 CPU | 3.1594초 | 약 31분 |
| LightGCN 3층 MPS GPU | 0.4634초 | 약 4.5분 |

BPR-MF는 warm-up 1회 후 3회의 중앙값이다.
LightGCN은 전체 그래프를 사용해 warm-up 1회 후 1회 업데이트를 측정했다.
각 업데이트는 forward·backward·Adam step을 포함한다.
GPU는 실행 전후 동기화했으며 CPU fallback은 활성화하지 않았다.
실제 전체 그래프에 대해 MPS와 CPU의 전파 결과를 비교해 rtol=atol=1e-4 범위의 일치를 확인했다.
CPU는 CSR, MPS는 COO 형식을 사용했다.

환산은 `ceil(4,818,321 / 8,192) = 589` 업데이트를 곱한 값이다.
실제 epoch 전체나 수렴까지의 시간을 측정한 결과는 아니다.
고정 미니배치와 단순 음성 추출을 사용했으며, 데이터 셔플·학습 이력 제외 음성 샘플링·전체 상품 검증·체크포인트 저장은 제외했다.
실제 학습 성능이나 추천 정확도 결과로 사용하지 않는다.

LightGCN 50 epoch는 학습 연산만 단순 환산하면 약 3.8시간이다.
검증과 여러 seed/설정 조합을 더하면 총 실행 시간은 상당히 늘어난다.
XSimGCL의 대조학습 추가 비용과 최종 GPU 메모리 사용량은 아직 측정하지 않았다.
임베딩 한 벌은 float32 기준 약 207MiB이며, 학습 중에는 gradient·optimizer 상태·전파 결과·임시 연산 메모리가 추가된다.

원시 결과: `results/hardware_benchmark.json`.
스크립트는 장치 성능 점검용이며 저장된 모델이나 추천 실험 결과를 생성하지 않는다.

## Colab을 사용할 조건

- 로컬 첫 실행에서 실제 epoch/검증 시간이 일정에 맞지 않는 경우.
- 여러 seed와 튜닝 조합을 병렬로 돌려야 하는 경우.
- 학습 중에도 노트북을 계속 이동하거나 다른 작업에 사용해야 하는 경우.
- 공식 구현에 CUDA 전용 의존성이 생기거나 실제 전체 모델에서 MPS 지원 문제가 생긴 경우.

처음부터 유료 구독을 전제로 하지 않는다. 접근 가능한 무료 GPU에서 같은 벤치마크로
속도·메모리·런타임 제한을 확인한 다음 유료 사용 필요성을 판단한다.
유료 Colab도 GPU 종류와 자원 가용성은 고정 보장이 아니다.
[Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html#resource-limits)

맥 GPU 사용 방식은 [PyTorch MPS 공식 문서](https://docs.pytorch.org/docs/2.15/notes/mps.html)를 참고한다.
이 환경의 희소 연산 지원 여부는 문서에 대한 추측이 아니라 실제 실행으로 확인했다.
