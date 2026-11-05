# hist_scan

`hhg_tof`와 분리된 실험 코드입니다. 존마다 raw 히스토그램을 저장하고, 피크가 둘이면 점도 둘 찍습니다. 투명한 면의 반사와 그 뒤 배경의 반사를 함께 얻기 위한 것입니다. `hhg_tof`의 코드는 고치지 않고 가져다 쓰며, 로봇 이동은 `hhg_tof.capture`의 기존 경로를 그대로 탑니다.

```bash
PYTHONPATH=. .venv/bin/python -m hist_scan here --config config/fr5_tumbler.yaml      # 현재 자세, 이동 없음
PYTHONPATH=. .venv/bin/python -m hist_scan scan --config config/fr5_tumbler.yaml      # 예행
PYTHONPATH=. .venv/bin/python -m hist_scan scan --config config/fr5_tumbler.yaml --execute --confirm SCAN
PYTHONPATH=. .venv/bin/python -m hist_scan process data/scans/<세션>
```

세션에는 일반 파일에 더해 시점마다 `hist_NNN.npz`가 생깁니다. `histogram`의 형태는 (측정 수, 32, 48, 64)입니다. `process`는 `cloud_multipeak.ply`와 `cloud_multipeak_views.png`를 만듭니다. PLY의 `kind`는 0 단일 피크, 1 둘 중 가까운 쪽, 2 둘 중 먼 쪽입니다.

## 실측으로 알아낸 것 (2026-11-05)

- 48x32 고정밀 모드는 존당 64 bin이고, 1 bin은 30.8 mm, 거리 0은 bin 10.7입니다. 값은 칩이 보고한 거리로 세션마다 맞춥니다.
- 히스토그램을 켜도 측정 속도는 약 4.8회/초입니다.
- 칩은 `nr_peaks=4`를 거부합니다(`#Err,Config -2`). 2는 됩니다.
- **프리셋을 다시 적용해도 히스토그램 비트는 꺼지지 않습니다.** `HistogramSensor.close()`가 되돌리니, 강제 종료했다면 다음 사용 전에 확인하세요.
- 벤더의 딕셔너리 변환은 `0x5E`의 예약 바이트를 지우므로 쓰지 않습니다.

## refine

`process`는 `cloud_refined.ply`도 만듭니다. 데이터에서 표면 하나의 응답 모양을 학습해 강한 피크를 빼고 남은 신호에서 두 번째 표면을 찾으며, 시간 일관성(15회 중 60 %), 이웃 존 연속성, 원통형 ROI를 적용합니다. 피크 간격은 3 bin입니다. 2 bin은 비스듬한 면을 둘로 쪼개서 뚜껑 산포가 6.5 mm에서 9.6 mm로 나빠졌습니다.

## 표면이 아닌 피크

- bin 10~12의 크로스토크는 모든 존에 있습니다. 60 mm 이내는 버립니다.
- 강한 표적은 다른 존에도 같은 거리로 샙니다. 존 전체의 bin별 중앙값을 빼서 제거합니다.
- 윤곽선에 걸친 존도 두 피크를 냅니다. 예: 뚜껑 202 mm와 테이블 527 mm. 둘 다 실제 표면이지만 투명해서 생긴 피크는 아닙니다.
- 여러 시점에서 재현되지 않는 점은 버립니다.
