# OmniGS 데이터 로더 통합 가이드

본 문서는 OmniGS 래스터라이저(특히 ERP/LONLAT 카메라)를 고려하여 3DGS 파이프라인의 데이터 로더를 어떻게 수정/확장할지에 대한 구체 가이드와 TODO 체크리스트를 제공합니다. 구현은 최소 변경으로 진행하되, ERP 특성을 안전하게 수용하는 것을 목표로 합니다.


## 범위
- 대상 파일
  - `scene/dataset_readers.py:1` — 데이터셋 스캔/카메라 생성 진입점
  - `scene/cameras.py:1` — 카메라 표현(행렬/파라미터/마스크/깊이 등)
  - (옵션) `arguments/*.py` — 로더/카메라 타입 관련 CLI 인자
- 비범위: 학습 루프/렌더러/손실(별도 문서 참조)


## 설계 원칙
- 카메라 타입 명시: PINHOLE=1, LONLAT(ERP)=3. 로더는 각 샘플 카메라 인스턴스에 `camera_type`을 지정합니다.
- 투영/행렬 일관성: 기존 3DGS의 `world_view_transform`, `full_proj_transform`, `camera_center` 생성 경로를 유지하고, ERP에서는 안전 기본값/일괄 규칙으로 세팅합니다.
- 해상도/종횡비 검증: ERP는 통상 2:1(W:H) — 로더에서 유효성 경고를 제공하고 자동 리사이즈 시 정보를 로깅합니다.
- 깊이/마스크는 선택: ERP에서는 깊이의 의미 보장이 어려우므로 `depth_reliable=False` 기본. seam/극점 마스크는 옵션으로 제공.


## 카메라 모델 확장
- `scene/cameras.py:1`
  - 필드 추가: `self.camera_type` (int)
  - 초기화 경로
    - PINHOLE: 기존 로직 그대로, `self.camera_type = 1`
    - ERP: `self.camera_type = 3`
  - 행렬 생성
    - PINHOLE: 현행 `getWorld2View2`, `getProjectionMatrix` 유지
    - ERP: `world_view_transform`/`full_proj_transform`는 항등 또는 공통 좌표계 기준으로 일관 세팅
      - 권장: `world_view_transform`은 Pose 기반(항등/주어진 R,T), `full_proj_transform = world_view @ projection`
      - `projection_matrix`는 ERP에서 래스터라이저 내부가 `camera_type`로 처리하므로, 안전 기본값으로 유지
  - 시야/파라미터
    - `FoVx`, `FoVy`는 ERP에서 경험적으로 180°(π) 또는 `tanfov=1.0` 고정. 우선 `train.py`가 주는 해상도 기준으로 안전 기본(1.0) 사용 권장.
  - 마스크/깊이
    - `alpha_mask`는 기존 경로 유지(알파 채널/하프 이미지 분할)
    - ERP 전용 마스크(옵션): seam/pole 마스크를 추가 필드로 보유하거나 `alpha_mask`에 합성
    - `invdepthmap`은 ERP에서 기본 비활성(`depth_reliable=False`), 제공 시 스케일/오프셋 체크 동일 적용


## 데이터셋 판별/로딩 전략
- 판별 규칙(우선순위)
  1) CLI 강제 지정: `--camera_type {pinhole|lonlat}` 또는 `--force_erp true`
  2) 메타데이터: 데이터셋에 ERP 명시(openMVG json, 360 roam 등) → ERP로 설정
  3) 해상도 휴리스틱: 이미지 너비:높이 ≈ 2:1인 경우 ERP 후보(경고/로그 + PINHOLE 유지 선택 가능)
- 로딩 파이프라인(`scene/dataset_readers.py:1`)
  - 공통: 이미지 로드 → 리사이즈 → 텐서 변환 → 알파/깊이/마스크 적용
  - PINHOLE: 기존 intrinsics/extrinsics(COLMAP) 경로 유지
  - ERP: intrinsics 미사용; extrinsics(Pose)만 사용해 `world_view_transform` 구성
    - `FoVx/FoVy`를 기본값으로 설정(예: π/2 또는 tanfov=1.0) — 렌더러에서 `camera_type=3`로 내부 처리


## ERP 특화 옵션(선택)
- seam 마스크: 경도 0/2π seam 근처 얇은 밴드 마스킹(`alpha_mask *= seam_mask`)
- 극점 마스크: 상/하단 n라인 마스킹(수치 안정성)
- 위도 가중맵: 학습 손실에서 사용(로더는 저장만, 손실에서 소비)
- 해상도 정규화: 가로세로가 2:1에서 벗어나면 경고 및 자동 리사이즈 옵션


## CLI 인자 제안(로더 관련)
- `--camera_type {auto|pinhole|lonlat}` (기본: auto)
- `--force_erp` (bool, 기본: False)
- `--erp_assume_2to1` (bool, 2:1 강제 검사/경고)
- `--erp_seam_mask {width_px}` (int, 0=off)
- `--erp_pole_mask {height_px}` (int, 0=off)
- `--erp_weighted_loss` (bool, 로더는 가중맵 저장만 담당)

`arguments/ModelParams` 또는 별도 Pipeline/DataParams에 인자 추가 후, `dataset_readers.py`에서 참조합니다.


## 구현 절차(TODO)
- Cameras
  - [ ] `scene/cameras.py`에 `camera_type` 필드 추가 및 `__init__` 분기
  - [ ] ERP일 때 `FoVx/FoVy` 기본값/검증 로깅 추가
  - [ ] ERP seam/pole 마스크 적용 유닛(옵션) — `alpha_mask`에 합성
- Dataset readers
  - [ ] 카메라 타입 판별 로직(인자 > 메타데이터 > 휴리스틱) 추가
  - [ ] ERP 분기 시 intrinsics 미사용 + Pose 기반 행렬 생성
  - [ ] ERP 해상도(2:1) 검증/자동 리사이즈/경고 로깅
  - [ ] 깊이 로딩 시 ERP는 `depth_reliable=False` 기본(명시 제공 시 스케일/오프셋 검사 동일)
  - [ ] (옵션) 위도 가중맵(코사인) 사전계산하여 샘플에 포함
- Arguments/Docs
  - [ ] 새로운 로더/ERP 관련 인자 추가 및 `README.md` 사용법 갱신
  - [ ] ERP 데이터 폴더 구조/예제 캡처(파일명 규칙, 메타 예시) 문서화


## 검증 플랜
- 단위 스모크
  - [ ] PINHOLE 샘플 로딩 → `camera_type=1`, 행렬/이미지 shape/log 정상
  - [ ] ERP 샘플 로딩(2:1 이미지) → `camera_type=3`, `depth_reliable=False`, 경고/로그 확인
  - [ ] seam/pole 마스크 옵션 on/off 로딩 확인
- 렌더 스모크(의존 최소)
  - [ ] 로더→카메라→렌더 설정 전달(ERP/PINHOLE 각각) 후 단순 forward 호출에서 shape/유한 값 검증
- 회귀 체크
  - [ ] ERP에서 리사이즈/마스크 옵션 변경 시 학습 손실 추세와 radii/가시성 분포의 일관성


## 주의/권고
- ERP에서 깊이 학습 항은 기본 비활성(현재 OmniGS Python 래퍼는 depth grad 미지원)
- ERP의 `markVisible`은 전 True인 특성 → densify 통계에는 `radii>0` 필터 사용 권장
- 거대 해상도 ERP는 I/O/메모리 병목—로더에서 이미지 캐시/반정밀 변환/핀 메모리 등을 옵션화

