# fishGS (OmniGS × 3DGS)

OmniGS C++ 래스터라이저를 3D Gaussian Splatting(3DGS) 파이프라인에 이식한 파이썬 구현입니다. 기본 백엔드는 OmniGS(`omnigs_rasterization` PyTorch 확장)이며, 필요 시 원본 3DGS의 diff 백엔드로 스위치할 수 있습니다. ERP/LONLAT 카메라(360 equirect) 데이터로의 학습·평가를 지원합니다.

본 README는 레포 코드 구조와 실제 동작을 기준으로 새로 작성되었습니다. 원본 3DGS README의 장문의 홍보/뷰어 섹션은 제거하고, 설치·학습·렌더·구성 방법과 OmniGS 정렬 포인트를 중심으로 정리합니다.


## 주요 변경점(OmniGS 정렬)
- OmniGS CUDA 래스터라이저 통합: `gaussian_renderer/__init__.py`에서 `rasterizer: omnigs|diff` 선택 가능(기본 `omnigs`).
- ERP/LONLAT 카메라 지원: 로더와 카메라에 `camera_type`(1: PINHOLE, 3: LONLAT/ERP) 반영. ERP일 때 내부 `tanfovx/y=1.0` 처리.
- OmniGS 규칙에 맞춘 학습 루프:
  - 대형 포인트(prune big point) 프루닝 스케줄: `--prune_big_point_after_iter` 이후에만 화면크기 기반 프루닝 활성. 구현 함수: `train.compute_size_threshold`.
  - 프루닝/조밀화 파라미터 확장: `--densify_min_opacity`, `--prune_by_extent`.
  - 존재 시점 추적: `GaussianModel.exist_since_iter` 유지 및 자식 점 생성 시 전달.
- ERP 특화 옵션(기본 비활성):
  - 하단 크롭: `--skip_bottom_ratio`(손실·리포팅에만 반영)
  - 위도 가중 손실/지표: `--erp_weighted_loss`, `--erp_weighted_metrics`
- 깊이 L1 정규화는 다음 조건에서만 사용: 비‑ERP 카메라이면서 diff 백엔드일 때. OmniGS/ERP에서는 색상 중심 학습이 기본.
- YAML 구성 지원: `--config/-c`로 설정 파일 병합. 병합 규칙과 예시는 `docs/config_guide.md` 참조.


## 레포 구조 한눈에 보기
- 학습·렌더 스크립트: `train.py`, `render.py`, `metrics.py`
- 렌더러 어댑터: `gaussian_renderer/`(OmniGS/diff 백엔드 스위치, SH 색 변환/공분산 사전계산 등)
- 장면/카메라/로더: `scene/`(COLMAP/Blender/OpenMVG 360Roam 로더 포함)
- 유틸: `utils/`(손실·가중맵·스케줄러 등)
- 서브모듈: `submodules/`
  - `omnigs_rasterization`(필수, PyTorch 확장)
  - `diff-gaussian-rasterization`(선택, 원본 3DGS 백엔드)
  - `simple-knn`(필수)
  - `fused-ssim`(선택, 가속 SSIM)
  - `OmniGS`(참고용 C++ 레퍼런스 및 cfg)
- 문서/예시 설정: `docs/`, `configs/`


## 지원 데이터셋
- COLMAP 형식: `<scene>/sparse/0`, `<scene>/images` 구조.
- Blender(NeRF Synthetic) 형식: `transforms_train.json`, `transforms_test.json`.
- OpenMVG ERP(360Roam 등): `data_views.json`, `data_extrinsics.json`, `images/`, `pcd.ply`, `train.txt`, `test.txt`.

카메라 타입 추론은 자동이며, 필요 시 `--camera_type {auto|pinhole|lonlat}`로 강제 지정할 수 있습니다. ERP는 내부적으로 `camera_type=3`입니다.


## 요구 사항
- Linux + NVIDIA GPU(CUDA) + PyTorch(토치/토치비전은 환경에 맞게 사전 설치)
- CUDA 툴킷과 PyTorch CUDA 버전 호환 필요(확장 모듈 컴파일)
- Python 3.10 이상 권장


## 설치
가상환경을 준비한 뒤 아래를 순서대로 실행하세요.

```bash
pip install -r requirements.txt

# 필수 확장
pip install -e submodules/simple-knn
pip install -e submodules/omnigs_rasterization

# (선택) 원본 3DGS 백엔드 및 가속 SSIM
# SparseAdam을 쓰거나 diff 백엔드를 사용하려면 설치
pip install -e submodules/diff-gaussian-rasterization  # 선택
pip install -e submodules/fused-ssim                   # 선택
```

Docker 사용자는 제공된 `Dockerfile`/`docker-compose.yml`을 참고하세요. 이미지에는 필수 파이썬 의존성과 확장이 포함되도록 구성되어 있습니다.


## 빠른 시작
COLMAP 데이터셋 예시:

```bash
# 학습(OmniGS 백엔드, 뷰어 비활성)
python train.py -s /path/to/colmap_scene -m output/colmap_run --disable_viewer --rasterizer omnigs

# 렌더(훈련 결과)
python render.py -m output/colmap_run -s /path/to/colmap_scene

# 기본 지표(SSIM/PSNR/LPIPS)
python metrics.py -m output/colmap_run
```

ERP(OpenMVG 360Roam) 예시:

```bash
python train.py \
  -s data/360Roam/lab \
  --camera_type lonlat \
  --rasterizer omnigs \
  --eval \
  --skip_bottom_ratio 0.063 \
  --disable_viewer
```

YAML 구성으로 실행:

```bash
python train.py -c docs/configs/train_sample.yaml
# 또는 장면별 프리셋
python train.py -c configs/360roam/lab.yaml
```


## 핵심 명령행 인자(요약)
- 모델/데이터: `-s/--source_path`, `-m/--model_path`, `--images`, `--depths`, `--resolution`, `--white_background`
- 파이프라인: `--rasterizer {omnigs,diff}`, `--antialiasing`, `--convert_SHs_python`, `--compute_cov3D_python`
- 최적화: `--iterations`, `--optimizer_type {default,sparse_adam}`, `--lambda_dssim`, `--densify_*`, `--opacity_reset_interval`
- OmniGS 정렬 옵션: `--densify_min_opacity`, `--prune_by_extent`, `--prune_big_point_after_iter`
- ERP 옵션: `--camera_type`, `--skip_bottom_ratio`, `--erp_weighted_loss`, `--erp_weighted_metrics`
- 깊이 정규화 가중: `--depth_l1_weight_init`, `--depth_l1_weight_final`(diff+비‑ERP에서만 유효)
- 실행 유틸: `--eval`, `--seed`, `--disable_viewer`, `--ip`, `--port`

전체 기본값은 `python train.py --print_params` 또는 `python render.py --print_params`로 확인하세요.


## YAML 구성(요약)
YAML은 기본값 ← `<model_path>/cfg_args` ← `--config` ← CLI 순으로 병합됩니다(뒤로 갈수록 우선). 예시와 키 그룹은 `docs/config_guide.md`, 샘플은 `docs/configs/train_sample.yaml` 참고.


## 백엔드 선택과 가속
- 기본은 OmniGS(`--rasterizer omnigs`).
- 3DGS diff 백엔드를 쓰려면 `pip install -e submodules/diff-gaussian-rasterization` 후 `--rasterizer diff`로 지정.
- `--optimizer_type sparse_adam`은 diff 백엔드가 설치된 경우에만 활성.
- `fused-ssim` 설치 시 SSIM 계산이 가속됩니다.


## ERP 주의 사항
- ERP 카메라(`camera_type=3`)는 rasterizer 내부 투영 모델을 사용합니다. FoV는 플레이스홀더이며 손실·평가에만 영향이 있습니다.
- ERP에서는 깊이 L1 정규화가 비활성(학습 루프에서 자동 처리).
- 라벨/지표에 위도 가중을 쓰려면 `--erp_weighted_*` 옵션을 명시적으로 켭니다(기본 Off).


## 데이터 준비 참고
- COLMAP 변환 스크립트: `convert.py`(COLMAP 실행 필요). `--resize` 옵션으로 다중 해상도 이미지를 생성할 수 있습니다.
- OpenMVG 360Roam: 폴더에 `pcd.ply`(초기 포인트클라우드)가 반드시 필요합니다. `scene/dataset_readers.py`의 로더가 그대로 사용합니다.


## Docker(선택)
```bash
# 이미지 빌드
docker compose build

# 셸 진입(호스트 데이터 마운트/GPU 필요 시 compose 파일 수정)
docker compose run --rm mosca bash
```


## 로드맵: 4DGS(시간 변화) 확장
- 목표: 정적 3DGS를 시간축(t)까지 확장해 동적 장면 재구성/렌더(4D Gaussian Splatting) 지원.
- 핵심 추가 요소(계획):
  - 가우시안 상태의 시간 의존성: `x,t`에서의 위치/스케일/회전/색(또는 SH 계수) 변화를 모델링.
  - 모션 파라미터화: 가우시안별 SE(3) 모션(속도/가속도) 또는 흐름장 기반 워핑 중 하나를 옵션화.
  - 시간 정규화: 프레임 간 부드러움(smoothness), 고정 배경 정규화, 가시성 일관성 손실.
  - 시간형 조밀화/프루닝: `exist_since_iter` 기반 수명(lifetime) 관리 확장, 프레임별 가시 반응형 분할/클론.
  - 데이터 로더: 비디오/멀티‑뷰 연속 프레임, 시각 동기화된 포즈/마스크/깊이(선택) 처리.
  - 래스터라이저: 프레임 시점별 가우시안 변환 적용(전/후방 경로 유지), 배치 시간 샘플링 가속.
- 단계적 통합(요지):
  1) 파이프라인/옵션에 시간 인자 `t` 도입(배치 내 프레임 인덱스),
  2) `GaussianModel`에 모션 파라미터/스케줄러 추가 및 손실 확장,
  3) 조밀화/프루닝 규칙을 시간 축으로 일반화,
  4) OmniGS 백엔드에 프레임별 워프 적용 경로 추가(호환 모드 유지),
  5) 공개 예제/구성(YAML)과 벤치마크 스크립트 제공.
- 주의: 상기 기능은 로드맵이며, 구현/인터페이스는 진행 상황에 따라 변경될 수 있습니다.


## 라이선스
- 루트 `LICENSE.md`: 3DGS 원저자 라이선스(비상업적 연구/평가용)
- `submodules/OmniGS`: GPLv3
- 그 외 서브모듈은 각 디렉터리의 라이선스 파일을 따릅니다.

본 레포는 연구·평가 목적 사용을 전제로 하며, 서브모듈 혼용에 따른 제약을 반드시 확인하세요.


## 인용 및 감사
- 3D Gaussian Splatting: Kerbl et al., TOG 2023
- OmniGS(C++): 레퍼런스 구현 및 문서에 감사드립니다.

연구 결과를 공개하실 때에는 관련 원저작물의 인용을 부탁드립니다.
