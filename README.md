# HHG_TOF

Fairino FR5 끝단에 TMF8829 dToF 센서를 달고, 물체 주위의 여러 시점에서 측정해 하나의 포인트클라우드(PLY)로 합치는 프로젝트입니다. ROS 없이 순수 Python으로 동작합니다.

```text
plan (읽기 전용 점검) → scan (멈춰서 측정, raw 저장) → process (필터·정합·융합) → cloud.ply
                                    └ calibrate (평면 스캔 → 센서 mount, mirror, 거리 바이어스)
```

## 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

ROS를 source한 셸에서는 ROS의 `PYTHONPATH`가 venv로 새어 들어와 pytest가 깨집니다. 아래 명령은 `env -u PYTHONPATH`를 앞에 붙여 실행하세요.

## 하드웨어 없이 실행

```bash
.venv/bin/python -m pytest -q
.venv/bin/hhg-tof scan --config config/sim_box.yaml --name box
.venv/bin/hhg-tof process data/scans/<세션>
.venv/bin/hhg-tof view data/scans/<세션>/cloud.ply
```

## 실제 로봇 절차

`config/fr5_real.example.yaml`을 `config/fr5_real.yaml`로 복사해 `MEASURE` 항목을 채웁니다. 비어 있는 값은 추정하지 않고 실행을 거부합니다.

### 1. 센서 연결

| 소스 | 설정 | 특징 |
|---|---|---|
| `nucleo` | 커스텀 PCB + Nucleo, 바이너리 UART 921600 | 호스트 트리거. SNR 없음. 열기 직전에 Nucleo를 리셋 (설정 창 약 10초) |
| `evm` | ams EVM (Raspberry Pi USB 브리지), ZeroMQ | SNR 제공. 48x32 약 5.3 fps |

EVM은 PC 쪽 USB 이더넷에 링크로컬 IPv4가 있어야 `169.254.0.2`에 닿습니다. 브리지의 MAC은 부팅마다 바뀌어 인터페이스 이름도 바뀌므로, 이름 패턴으로 잡는 프로파일을 씁니다.

```bash
nmcli con add type ethernet con-name hhg-evm-usb match.interface-name 'enx*' \
  ipv4.method link-local ipv6.method ignore connection.autoconnect-priority 10
```

확인은 `hhg-tof sensor-test --config config/fr5_real.yaml`로 합니다.

**Linux에서 EVM을 쓰기 전에 저장장치 인터페이스를 꺼야 합니다.** EVM은 네트워크와 함께 8 MiB USB 저장장치로도 잡히는데, PC가 이 저장장치를 읽기 시작한 직후 연결이 끊겨 다시 살아나지 않는 일이 2026-11-05에 반복됐습니다(연결 후 21초에 `rndis_host` 해제, 이후 `error -110`). 포트나 케이블을 바꿔도 소용없었고, 저장장치를 무시하게 하자 사라졌습니다.

```bash
echo '1325:0c08:i' | sudo tee /sys/module/usb_storage/parameters/quirks          # 지금만
echo 'options usb-storage quirks=1325:0c08:i' | sudo tee /etc/modprobe.d/hhg-evm.conf   # 재부팅 후에도 유지
```

**ping은 되는데 5557이 닫혀 있으면** Pi 드라이버 초기화가 실패한 것입니다. 부팅마다 간헐적으로 `CPU is not ready`로 실패하고, 그러면 ZeroMQ 서버가 재시작을 반복합니다. 드라이버를 다시 올리면 풀립니다. USB 연결에는 영향이 없습니다.

```bash
ssh ams@169.254.0.2      # 비밀번호 ams
dmesg | grep tmf8829     # Probe failed 확인
sudo rmmod tmf8829 && sudo insmod /opt/USBSensorBridgeRuntime/modules/tmf8829.ko
```

`another EVM client is active`가 나오면 다른 클라이언트가 제어 권한을 쥐고 있는 것입니다. 서버는 이 권한을 만료시키지 않으므로, 그 클라이언트가 Leave 없이 죽었다면 EVM을 다시 꽂아야 풀립니다.

EVM 칩의 시리얼은 1821663이고, 2026-06 캠페인에 이 시리얼의 LUT는 없습니다. 슬롯 7 LUT(5013663)는 이 칩에 쓸 수 없으므로 EVM 경로에서는 `calibration_json`을 비워 두고 `sensor.range_bias_mm`을 씁니다.

### 2. 로봇 상태 확인 (읽기 전용)

```bash
.venv/bin/hhg-tof status --config config/fr5_real.yaml
```

- **활성 tool**: 보고되는 TCP pose는 활성 tool 기준입니다. `robot.expected_tool`과 다르면 스캔을 거부합니다. tool 0(플랜지)을 권장합니다. 그러면 `mount`가 플랜지→센서 변환이 됩니다.
- **자세**: 기본 궤도는 J3 > 0 (elbow) 쪽에서만 검증되었습니다. 2026-11-05의 첫 자세(J3 = -113°, J4 = -264.4°)는 J4가 한계 -265°에 거의 닿아 있어 그대로는 스캔할 수 없었습니다. 같은 날 나중에 읽은 자세(J3 = -78.3°, J4 = -195.0°, 활성 tool 1)도 J3 < 0이고, tool이 `expected_tool: 0`과 달라 그대로면 스캔이 거부됩니다.

### 3. Mount와 mirror 캘리브레이션

테이블이나 평판처럼 FoV를 채우는 평면을 스캔합니다. **기울기와 roll이 모두 달라야 합니다.** roll이 고정이면 x 오프셋과 mirror는 수학적으로 관측되지 않습니다.

```yaml
scan:
  center_mm: [...]          # 평면 위의 점
  object_radius_mm: 5
  clearance_mm: 5
  standoff_mm: 300
  elevations_deg: [75, 60]
  views_per_ring: [6, 6]
  roll_cycle_deg: [0, 90, 180, 270]
```

```bash
.venv/bin/hhg-tof calibrate data/scans/<세션>
```

출력된 값을 `mount`와 `sensor.mirror_columns`에 넣고 `mirror_verified: true`로 바꿉니다. LUT 없이 쓴다면 거리 바이어스를 `sensor.range_bias_mm`에 넣습니다. 관측이 부족하면 `REJECTED`로 끝나며, 그 값은 쓰면 안 됩니다.

#### 조그로 하는 방법 (mount를 모를 때)

센서를 테이프로 붙여 위치를 모르면 궤도를 계획할 수 없습니다. 이때는 사람이 조그로 자세를 잡고, 프로그램은 측정만 합니다. 이 명령은 로봇을 움직일 수 없습니다. `mount`가 `null`이면 방향까지 전역 탐색하므로 초기값이 필요 없습니다.

```bash
.venv/bin/hhg-tof jog-capture --config config/fr5_evm_live.yaml --name plane
.venv/bin/hhg-tof calibrate data/scans/<세션>
```

1. 책상이나 평판이 시야의 절반 이상을 채우도록 150~500 mm 거리에 둡니다. 물건이 일부 있어도 시점마다 가장 큰 평면만 골라 씁니다.
2. 자세를 잡고 손을 뗀 뒤 Enter를 누릅니다. 팔이 멈춘 상태여야 저장됩니다.
3. 6시점부터는 저장할 때마다 풀이를 돌려 `ENOUGH` 또는 부족한 항목을 알려 줍니다. 10~12시점을 권장합니다.
4. 기울기는 15~35°로, 기울이는 방향과 J6 roll을 시점마다 다르게 합니다.

`mount`는 캡처 때의 활성 tool 기준입니다. `robot.expected_tool`과 다르면 캡처를 거부합니다. 테이프가 밀리면 값이 무효가 되니 다시 측정하세요. 시뮬레이션에서는 무작위 mount 32회 중 31회 수락, 틀린 값 수락 0회, 수락된 경우 오차 1.1 mm와 0.30° 이내였습니다. 실기 정확도는 아직 모릅니다.

#### 프로그램 이동으로 하는 방법

mount를 몰라도 손목 회전과 ±40 mm 수직 이동만으로 평면을 찍습니다. 수직 이동이 있어야 mirror가 갈립니다. 회전만 쓰면 "mirror이고 평면을 보는 센서"와 "mirror가 아니고 반대를 보는 센서"가 똑같이 맞습니다.

```bash
.venv/bin/hhg-tof wrist-scan --config config/fr5_evm_live.yaml --level                    # 움직이지 않는 예행
.venv/bin/hhg-tof wrist-scan --config config/fr5_evm_live.yaml --level --views 1 --execute --confirm SCAN
.venv/bin/hhg-tof wrist-scan --config config/fr5_evm_live.yaml --level --execute --confirm SCAN
```

`--level`은 tool Z를 수직 아래로 맞춘 자세를 중심으로 합니다. J3 > 0 자세에서 쓰세요. J3 < 0이면 J4 한계에 막혀 예행이 거부합니다(2026-11-05 확인). 조그 중 J4가 -265°를 넘으면 `main=3 sub=4`가 납니다. 펜던트로 되돌려 해제하세요.

**평면 캘리브레이션은 거리 바이어스를 믿을 만하게 정하지 못합니다.** 바이어스를 18.7에서 -2 mm로 바꿔도 잔차는 2.69에서 2.89 mm로 거의 그대로이고, 바이어스는 평면 높이로 흡수됩니다. 보고되는 1-sigma는 이 불확실성을 나타내지 못합니다. 크기를 아는 물체로 확인하세요.

평면 하나로 판별하는 이유: 데이터시트 Figure 16은 "concept drawing"이고, 그림대로면 벤더 XYZ가 거울상일 수 있습니다. 거울상은 강체 변환으로 보정되지 않으므로 측정으로 결정합니다.

### 4. 계획 점검 후 스캔

```bash
.venv/bin/hhg-tof plan --config config/fr5_real.yaml
```

모든 시점을 실제 컨트롤러 IK로 풀고, 조인트 한계, 조인트 스텝, 작업공간, 물체 keep-out을 검사합니다. 출력의 0번 시점 근처로 팔을 먼저 옮겨 두세요. 첫 이동도 같은 검사를 받습니다.

```bash
.venv/bin/hhg-tof scan --config config/fr5_real.yaml --name mug --execute --confirm SCAN
.venv/bin/hhg-tof process data/scans/<세션>
```

`--execute --confirm SCAN` 없이는 절대 움직이지 않습니다.

스캔이 정상적으로 끝나면 팔은 왔던 궤도를 되짚어 top view로 돌아갑니다. 직선으로 돌아가지 않는 이유는 궤도가 J6를 약 170° 감기 때문입니다. 끄려면 `scan.return_to_top: false`로 둡니다. 중단되면 돌아가지 않고 그 자리에 멈춥니다. 이미 끝난 세션의 마지막 자세에서는 `hhg-tof retrace data/scans/<세션> --config ... --execute --confirm SCAN`으로 돌아갈 수 있습니다.

## 실시간 시각화

거리 이미지, 로봇 자세, 3D 포인트를 한 창(Rerun)에 보여 줍니다. 뷰어는 별도 프로세스이고 구독만 하므로, 켜고 끄거나 죽어도 스캔에 영향이 없습니다.

```text
FR5 :20004 ──> scan 또는 monitor ──ZMQ PUB :5560──┐
EVM :5558 (PUB) ──────────────────────────────────┴──> hhg-tof live
```

```bash
.venv/bin/pip install -e ".[live]"
.venv/bin/hhg-tof live --config config/sim_box.yaml --demo            # 하드웨어 없이
.venv/bin/hhg-tof monitor --config config/fr5_evm_live.yaml           # 터미널 1: 스캔 없이 볼 때
.venv/bin/hhg-tof live --config config/fr5_evm_live.yaml              # 터미널 2
```

`monitor`는 로봇을 움직이지 않습니다. 로봇 상태를 발행하고 센서 측정을 켜 둘 뿐이며, `--no-sensor`면 로봇 상태만 발행합니다. `scan`을 돌릴 때는 `monitor`를 끄세요. `scan`도 같은 포트로 발행하므로 `live`는 그대로 둬도 됩니다. `--save x.rrd`로 기록하고 `rerun x.rrd`로 다시 볼 수 있습니다.

표시 규칙은 다음과 같습니다.

- `mount`가 없으면 포인트는 센서 좌표 패널에만 나옵니다. 베이스 좌표에 올리려면 `mount`가 필요합니다.
- 활성 tool이 `expected_tool`과 다르거나 자세가 0.5초 넘게 끊기면 베이스 좌표에 올리지 않습니다.
- 팔이 움직일 때의 프레임은 주황색으로 보여 주되 지도에는 넣지 않습니다.

**20004는 클라이언트를 하나만 받습니다.** 새 연결이 붙으면 이전 연결이 끊깁니다(2026-11-05 실측). 스캔 중에 끊기면 안전 모니터가 눈을 잃으므로, `hhg-tof` 프로세스끼리는 잠금으로 막아 두었습니다. 스캔 중 `status`나 `plan`은 거부됩니다. **이 잠금은 다른 프로그램(KDC 도구, ROS 드라이버 등)은 막지 못하니 스캔 중에 그런 프로그램을 켜지 마세요.**

## 안전

**첫 실행은 반드시 티치 펜던트 E-stop을 손에 쥐고, 낮은 속도로 하세요.**

- 소프트웨어 STOP(`robot/fr5.py`의 `stop()`)은 SDK를 보고 재현했으나 **실기 미검증**입니다. E-stop이 실제 정지 수단입니다.
- 충돌 점검은 센서, TCP, 그 사이 직선 경로만 봅니다. **팔의 나머지 링크와 주변 고정물은 모델링되지 않습니다.** `workspace_mm`은 주변 구조물을 피하도록 잡으세요.
- 오류가 있으면 `ResetAllError`를 자동으로 보내지 않고 거부합니다. 사람이 원인을 봐야 합니다.
- 스캔 중에는 20004에 다른 프로그램을 연결하지 마세요. 위의 "실시간 시각화"를 참고하세요.
- 스캔 중 E-stop, 충돌, 컨트롤러 오류, 상태 스트림 끊김을 감지하면 정지를 시도하고 중단합니다. 저장된 시점은 남습니다.
- 종료 시 Mode는 시작 때의 값으로 복원되며, 서보는 켜진 채로 둡니다.

## 알아둘 설계 사항

**거리에는 두 종류가 있습니다.** 칩은 광선 방향 거리(radial)를 보고하고, `TMF8829_cali`의 LUT는 평판 기준이라 수직 깊이(plane_z)를 출력합니다. slot 7 데이터에서 d(raw)/d(true)는 중앙 1.02, 코너 1.28로 코사인 계수 1.26과 일치합니다. LUT 출력에 코사인 보정을 또 적용하면 가장자리가 약 20% 수축합니다.

**캘리브레이션 파일은 내용으로 검증합니다.** 2026-06-19 캠페인의 `*_48x32_ha_calibration.json`은 실제로 16x16입니다. LUT는 센서 개체별이며, 슬롯 7의 48x32 LUT는 시리얼 5013663 전용입니다. Nucleo 스트림은 시리얼을 주지 않으므로 그 경로에서는 사용자가 맞는 파일인지 확인해야 합니다. LUT 범위 밖의 값은 200 mm에서만 맞는 offset으로 대체하지 않고 버립니다.

**이동 중에는 측정하지 않습니다.** 48x32 프레임은 약 190 ms 걸려, 50 mm/s만 되어도 10 mm가 번집니다.

**raw만 저장합니다.** mount나 필터를 바꿔도 다시 스캔할 필요가 없습니다. `process --config`로 다른 설정을 적용할 수 있습니다.

**ICP는 기본으로 꺼져 있습니다.** 시뮬레이션에서 개선이 없었고, 박스에서는 RMS가 0.41 → 0.46 mm로 나빠졌습니다.

**해상도 한계.** 250 mm 거리에서 존 간격은 약 7 mm입니다. 이보다 작은 형상은 복원되지 않습니다.

## 검증 상태

| 항목 | 상태 |
|---|---|
| 처리 파이프라인, 캘리브레이션 | 시뮬레이션 정답 비교 |
| FR5 상태 스트림, `GetInverseKin`, `GetInverseKinRef` | 실기 확인 (읽기 전용) |
| 전체 스캔 | 실기 확인. `range_bias_mm` 18.72에서 캔 지름 +27 mm(실측 52.5), 뚜껑 지름 +15 mm(실측 88)였음. 절충값 -3 mm에서 약 +4 mm, -4 mm로 줄었으나 상수 하나로는 둘을 함께 맞추지 못함(캔은 -7, 뚜껑은 +1.4 mm에 맞음). 거리, 곡률, 재질 의존성 미확인 |
| 기본 궤도 도달성 | 실기 IK로 일부 조건만 확인, 아래 참고 |
| `Mode`/`RobotEnable`/`MoveL` 순서와 형식 | 실기 확인 (2026-11-05 `wrist-scan` 13시점, 속도 10 %, 도착 오차 0.01 mm, 모드 복원 정상) |
| Nucleo 파서 | 합성 프레임. 실기 미검증 |
| EVM 파서, 제어, 48x32 조립 | 실기 확인 (시리얼 1821663, 5.3 fps, 짝수·홀수 행 모두 정상) |
| 손목 캘리브레이션 | 실기 확인 (2026-11-05 책상 평면, 잔차 2.69 mm, 시점별 테이블 높이 표준편차 0.77 mm, 제외 시점 오차 1.6 mm 이내). 물체 형상 정확도는 미측정 |
| 조그 캘리브레이션 | 시뮬레이션 정답 비교. 실측 세션은 완주한 적 없음 |
| 실시간 뷰어 | 실제 센서와 로봇으로 `monitor`, `live` 전체 경로 확인 (기록 파일 기준). 실제 데이터를 창으로 본 것은 아직 없음 |
| FR5 기구학 (`live.py`) | URDF 값이 실기 한 자세와 일치 (회전 0.01°, tool 1 오프셋 Z 136.4 mm) |
| 소프트웨어 STOP | **미검증** |

시뮬레이션 수치(구 RMS 0.41 mm, 박스 RMS 0.41 mm)는 시뮬레이터의 노이즈 모델 안에서만 의미가 있습니다. 실제 정확도는 센서 바이어스, 다중 경로, 표면 반사율에 좌우되므로 더 나쁠 것입니다.

기본 궤도(링 70°/40°, 21 시점)는 물체가 베이스 축에서 450 mm, 높이 60 mm, mount `[0, 40, 60]`일 때 실제 IK를 무오류로 통과했습니다. mount나 위치가 조금 달라지면 한두 시점이 실패할 수 있으니, `plan` 결과를 기준으로 `azimuth_span_deg`, `views_per_ring`, 물체 위치를 조정하세요. 25° 링을 넣은 3링 구성은 360° 궤도로는 통과하지 못했지만, `azimuth_span_deg: 270`의 호로는 통과한 조합이 있습니다. 측면 아래쪽을 더 보고 싶다면 이쪽을 시도해 보세요.

## 구조

```text
config/                실제 로봇 예시, 시뮬레이션 설정
src/hhg_tof/
  planning.py          시점 생성, preflight
  capture.py           stop-and-scan 루프
  telemetry.py         로봇 상태 발행, 20004 소유권 잠금
  live.py              실시간 프레임 처리, 자세 동기, 지도, 팔 기구학
  live_view.py         Rerun 뷰어
  session.py           raw 세션 저장/로드
  processing.py        필터, 융합, 법선, ICP
  calibration.py       평면 기반 mount/mirror 캘리브레이션
  sensor_model.py      존 광선 기하, LUT
  robot/fr5.py         FR5 제어 (모션 코드는 여기에만 있음)
  robot/fr5_io.py      KDC 프로젝트에서 그대로 복사. 수정 금지
  sensors/             nucleo, evm, simulated
tests/fake_fr5.py      실제 프로토콜을 흉내 내는 가짜 컨트롤러
```

센서 프레임: +Z 광축, +X 열 증가 방향, +Y 행 증가 방향. pose는 FR5 규약(mm, deg, extrinsic RPY, `R = Rz·Ry·Rx`)입니다.
