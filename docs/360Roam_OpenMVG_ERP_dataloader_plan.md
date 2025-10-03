**목표**
- `data/360Roam`(OpenMVG 형식, ERP/equirect) 데이터셋을 현재 파이프라인에 그대로 학습/렌더 가능하도록 “새 데이터 로더”를 설계한다.
- 구현은 하지 않고, 어디를 어떻게 바꿀지 구체 계획만 문서화한다.

**데이터 구조 요약(360Roam)**
- 예: `data/360Roam/lab/`
  - `images/*.jpg` (2048x1024 등, 2:1 ERP)
  - `data_views.json` (OpenMVG views: 파일명/해상도/id_view/id_pose 매핑)
  - `data_extrinsics.json` (OpenMVG extrinsics: pose별 회전행렬 Rcw, 카메라 중심 C)
  - `pcd.ply` (초기 포인트 클라우드; 이 파일을 그대로 사용)
  - `train.txt`, `test.txt` (파일명 stem 목록. 예: `0_0000`)

**현재 파이프라인 훅 포인트**
- 데이터 로더 진입은 `scene/__init__.py:43` 경로 분기에서 결정된다.
  - `sparse/0` → COLMAP (`scene/dataset_readers.py:199`의 `readColmapSceneInfo`)
  - `transforms_train.json` → Blender (`scene/dataset_readers.py:332`의 `readNerfSyntheticInfo`)
- 신규: OpenMVG(360Roam) 브랜치를 추가한다.
  - 조건: `data_views.json`와 `data_extrinsics.json` 존재 시 OpenMVG 로더 호출

**핵심 설계**
- 새 리더 함수: `readOpenMVG360SceneInfo(path, eval, train_test_exp)` 추가
  - 위치: `scene/dataset_readers.py`
  - 반환 타입: 기존과 동일한 `SceneInfo`(point_cloud, train_cameras, test_cameras, nerf_normalization, ply_path, is_nerf_synthetic)
- 카메라 타입: ERP/LONLAT로 고정 설정(`camera_type=3`)
  - 파이프라인은 ERP일 때 내부적으로 `tanfovx=tanfovy=1.0` 처리함. 참조: `gaussian_renderer/__init__.py:33` 및 46~58
- 포인트클라우드: 각 씬 루트의 `pcd.ply`를 사용(랜덤 생성 금지)
  - `ply_path = os.path.join(path, "pcd.ply")`로 고정하고 존재 여부를 검증(없으면 에러)

**OpenMVG → 파이프라인 좌표/행렬 합의**
- OpenMVG extrinsics: 보통 월드→카메라 회전 `Rcw`와 카메라 중심 `C`를 제공
  - 변환식: `t = -Rcw @ C`
  - 본 코드에서 카메라 R/T 저장 규약
    - `R`에는 카메라→월드 회전을 보관하므로 `R = (Rcw)^T`
    - `T`에는 월드→카메라 평행이동을 보관하므로 `T = t`
  - 이 규약은 COLMAP 로더와 동일. 참조: `scene/dataset_readers.py:86-87`, `utils/graphics_utils.py:18-36`

**구체 구현 계획(파일/지점별)**
- scene/dataset_readers.py
  - [추가] `def readOpenMVG360SceneInfo(path, eval, train_test_exp):`
    - 입력 파일 경로
      - `views_path = os.path.join(path, "data_views.json")`
      - `extr_path  = os.path.join(path, "data_extrinsics.json")`
      - `images_dir = os.path.join(path, "images")`
      - `train_list = os.path.join(path, "train.txt")`, `test_list = os.path.join(path, "test.txt")`
    - 파싱
      - views: `id_view`/`id_pose`/`filename`/`width`/`height` 맵 작성
      - extrinsics: `pose_id -> (Rcw, C)` 맵 작성
    - 카메라 구성(CameraInfo)
      - pose 매칭: `view["id_pose"]`로 extrinsics 조회. 없으면 경고 후 skip
      - `Rcw = np.array(rotation); C = np.array(center)`
      - `t = -Rcw @ C`; `R = Rcw.T`; `T = t`
      - 경로/이름: `image_path = os.path.join(images_dir, filename)`, `image_name = Path(filename).stem`
      - ERP 고정: `camera_type=3`
      - FoV: ERP에서는 사용되지 않으므로 placeholder(예: `FovX=FovY=np.pi/2`) 세팅
      - depth: 미지원 → `depth_path=""`, `depth_params=None`
      - train/test 분리: `train.txt`/`test.txt`를 파싱해 이름 기반으로 엄밀히 분리
        - 파일: `train_list = os.path.join(path, "train.txt")`, `test_list = os.path.join(path, "test.txt")`
        - stem 매칭: `0_0000.jpg` ↔ `0_0000`와 같이 확장자 제거한 stem 기준으로 비교
        - 포함 규칙: `included = train_names ∪ test_names` (두 리스트에 모두 없는 뷰는 기본 제외)
        - 플래그: `is_test = (stem ∈ test_names)`
        - 충돌 시 우선순위: 동일 이름이 양쪽에 모두 있으면 `test` 우선(누수 방지)
        - `eval=False`인 경우, 기존 파이프라인과 맞추어 `train += test`로 병합하여 단일 학습 세트로 사용
    - 분할
      - `train_cam_infos = [...]` (eval=False면 test 포함 규칙은 COLMAP 로더와 동일 정책 유지 가능)
      - `test_cam_infos  = [...]`
    - 정규화/반경
      - `nerf_normalization = getNerfppNorm(train_cam_infos)` (참조: `scene/dataset_readers.py:49`)
    - PLY 경로 및 초기화
      - `ply_path = os.path.join(path, "pcd.ply")`
      - 반드시 존재해야 하며, `fetchPly(ply_path)`로 읽는다(없으면 에러 종료)
    - 반환: `SceneInfo(...)`
  - [선택] `sceneLoadTypeCallbacks`에 키 추가
    - 예: `"OpenMVG360": readOpenMVG360SceneInfo` (`scene/dataset_readers.py:371` 인근)

- scene/__init__.py
  - 데이터 유형 분기 확장(43~49)
    - `elif os.path.exists(os.path.join(args.source_path, "data_views.json")) and os.path.exists(os.path.join(args.source_path, "data_extrinsics.json")):`
      - `scene_info = readOpenMVG360SceneInfo(args.source_path, args.eval, args.train_test_exp)` 호출
    - 나머지 로직(카메라 JSON 저장/셔플/extent/PCD 초기화) 동일 유지

**ERP 특이사항 반영(이미 파이프라인 내 존재)**
- 렌더러의 ERP 처리
  - ERP 카메라(`camera_type==3`)는 내부적으로 `tanfovx=tanfovy=1.0`을 사용. 참조: `gaussian_renderer/__init__.py:30-45`
- 학습 루프에서 ERP depth 비활성
  - ERP/LONLAT 카메라일 때 depth L1 사용 안 함. 참조: `train.py:190-206`
- ERP 전용 옵션(선택)
  - 하단 영역 무시: `--skip_bottom_ratio` (손실/리포팅에 반영). 참조: `train.py:268-314`
  - 위도 가중 손실/지표: `--erp_weighted_loss`, `--erp_weighted_metrics` (참조: `utils/loss_utils.py:79-129`, `train.py:268-314`)

**검증 플랜**
- 로딩 스모크 테스트
  - `train.py -s data/360Roam/lab --camera_type lonlat --disable_viewer`로 첫 100 step 진행 시, 이미지 텐서 shape과 NaN 없음 확인
  - `cameras.json` 생성 시 해상도/파일명/개수가 `train.txt`/`test.txt` 합과 일치하는지 확인
  - `pcd.ply`가 `model_path/input.ply`로 복사되어 학습 시작 시점에 존재하는지 확인(Scene이 scene_info.ply_path를 그대로 사용)
- 좌표계 점검
  - 임의 뷰 1개 선택 → `Rcw, C`에서 `R, T`로 변환 → `utils.graphics_utils.getWorld2View2(R,T)`로 W2V 생성 후 카메라 센터 재복원(C2W[:3,3])이 C와 일치하는지 확인
- 분할 확인
  - `train.txt`/`test.txt`의 합과 `cameras.json`에 반영된 뷰 수가 일치하는지 확인(중복 이름은 단일 테스트 뷰로 계산)
- ERP 옵션 스모크
  - `--skip_bottom_ratio 0.1` 설정 시 loss 계산 시 하단 crop이 적용되는지(TensorBoard 이미지 시각 확인)
  - `--erp_weighted_metrics` 시 wL1/wPSNR가 로그되는지 확인

**예상 CLI 사용 예**
- 학습(OmniGS 백엔드, ERP 강제):
  - `python train.py -s data/360Roam/lab --camera_type lonlat --rasterizer omnigs --eval --skip_bottom_ratio 0.1` 
- 랜더(사전학습 모델):
  - `python render.py -m <model_dir> -s data/360Roam/lab`

**수정 범위 요약(파일/라인)**
- `scene/dataset_readers.py:371` 인근에 `readOpenMVG360SceneInfo` 추가 및 콜백 등록
- `scene/__init__.py:43` 분기 확장(OpenMVG JSON 감지 후 신규 리더 호출)
- 기타 파일은 변경 불필요(렌더러/손실/옵션은 이미 ERP 지원)

**완료 기준(Acceptance)**
- `train.py -s data/360Roam/<scene>`로 실행 시 에러 없이 학습 루프 진입
- `model_path/input.ply`(원본 `pcd.ply` 복사본) 및 `cameras.json` 생성, 뷰 수/이름/해상도 일치
- ERP 옵션(`--skip_bottom_ratio`, `--erp_weighted_*`)이 로그/평가에 정상 반영

**주의사항**
- OpenMVG json의 `root_path`는 절대경로일 수 있음. 구현 시 실제 사용 경로는 `path/images` 우선 사용(데이터셋 루트 상대 참조)으로 안전 처리 권장
- 일부 뷰에 extrinsics 누락 가능성 → 스킵/경고 처리(훈련 셋과 테스트 셋의 일관성 유지)
- 거대 ERP 입력은 I/O/메모리 병목. 필요 시 `--resolution`/캐싱 옵션으로 가드
 - PLY는 반드시 `pcd.ply`를 사용. 임의 랜덤 포인트 생성 금지(재현성/정합성 유지)

**extent(스케일) 산출 기준 정리**
- 현재 파이프라인의 `extent`는 포인트클라우드가 아닌 “카메라 위치” 기반으로 결정됩니다.
  - `scene/dataset_readers.py:49`의 `getNerfppNorm`이 학습 카메라 중심들의 대각 길이를 구해 `radius = diagonal * 1.1`로 산출
  - `scene/__init__.py:69`에서 `self.cameras_extent = scene_info.nerf_normalization["radius"]`로 설정 후, 초기화 시 `GaussianModel.create_from_pcd(..., spatial_lr_scale=self.cameras_extent)`에 전달
- 필요 시 옵션으로 `pcd.ply`의 bounding sphere에서 radius를 구해 extent로 쓰는 대안을 추가할 수 있으나, 기본은 카메라 기반을 유지(기존 파이프라인과의 정합성 보장)
