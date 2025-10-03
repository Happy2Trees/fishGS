# OmniGS 통합 가이드 (3DGS → OmniGS)

본 문서는 현재 3DGS 원본 파이프라인을 기반으로, 구현 완료된 `submodules/omnigs_rasterization` 래스터라이저와 `submodules/OmniGS` C++ 레퍼런스를 참고하여 Python 파이프라인을 OmniGS 규칙에 맞게 이식/정렬하기 위한 큰 파트별 설계와 TODO 체크리스트를 제공합니다.

핵심 포인트는 다음 4가지입니다.
- 렌더러 계층 교체 및 카메라 모델(LONLAT/ERP 포함) 반영
- Optimizer/스케줄러 그룹과 하이퍼파라미터 정책을 OmniGS에 정렬
- Gaussian densify & prune 규칙/스케줄 반영 및 가시성 처리
- 손실/정규화(ERP 가중 포함), 로깅/체크포인트 정리


## 0) 현재 상태 요약
- 3DGS 파이썬 파이프라인은 원본 구조 유지. 렌더러는 `diff_gaussian_rasterization`를 사용 중.
- `submodules/omnigs_rasterization`는 OmniGS CUDA 래스터라이저의 파이썬 래퍼가 포함되어 설치 가능 상태.
- `submodules/OmniGS`에는 C++ 기준 구현(훈련 루프, 스케줄/옵티마이저/den&prune 규칙)이 포함되어 있으며, 이를 파이썬으로 정렬/반영해야 함.

참고 파일(레퍼런스):
- OmniGS C++ 트레이너: `submodules/OmniGS/src/gaussian_trainer.cpp:1`
- OmniGS 모델(옵티마이저/스케줄/den&prune): `submodules/OmniGS/src/gaussian_model.cpp:1`, `submodules/OmniGS/include/gaussian_model.h:1`
- Python 3DGS 트레이너: `train.py:1`
- Python 3DGS 모델: `scene/gaussian_model.py:1`
- 새 래스터라이저(PyTorch 확장): `submodules/omnigs_rasterization/omnigs_rasterization/__init__.py:1`


## 1) 렌더러 계층 교체 + 카메라 모델 반영
OmniGS 래스터라이저는 3DGS와 유사한 Python API를 제공하되, `camera_type`(1: PINHOLE, 3: LONLAT/ERP)과 ERP 호환을 추가 제공합니다.

적용 원칙
- `gaussian_renderer/__init__.py:1`에서 `diff_gaussian_rasterization` import를 `omnigs_rasterization`로 대체.
- `GaussianRasterizationSettings` 생성 시 `camera_type`을 전달. PINHOLE은 1, ERP/LONLAT은 3.
- ERP의 깊이(depth) 출력은 v0에서는 별도 의미 보장 없음(색상 패스와 동일 버퍼 경로에서 추출). 학습 시 깊이 항을 사용한다면 ERP에서는 비활성/대응 필요.
- ERP의 `markVisible`은 전 점 가시(True) 처리이므로, densify 입력 시 과도한 업데이트를 유발하지 않도록 주의(visibility 필터는 여전히 radii 기반 사용 권장).

TODO
- [x] `gaussian_renderer/__init__.py`에서 래스터라이저 import/호출부 교체(동일 반환 형태 유지: render/radii/depth)
- [x] `GaussianRasterizationSettings`에 `camera_type` 전달 경로 추가
- [ ] (옵션) 기존 `separate_sh` 경로 유지 검토(3DGS와 동일 인터페이스)
- [x] ERP 카메라에서 `tanfovx/tanfovy` 사용 여부 점검 및 안전 기본값 설정(예: 1.0)


## 2) 카메라/데이터 파이프라인 정리(ERP 지원)
3DGS의 `scene/cameras.py:1`는 PINHOLE 가정. OmniGS는 ERP(LONLAT) 카메라를 지원하므로, 데이터 리더와 카메라 표현에 `camera_type`을 도입해야 합니다.

적용 원칙
- 카메라 인스턴스에 `camera_type` 필드를 추가하여 렌더러 설정에 전달.
- ERP 데이터셋의 경우, 투영행렬/전역변환은 단순화(또는 항등) + ERP 전용 변환은 래스터라이저 내부에서 처리. Python 측에서는 일관된 텐서 형태만 맞추면 됨.
- (선택) ERP 손실에서 위도 가중(cosine 가중) 적용을 위해 입력 이미지 차원(H, W) 기준 가중맵 제공.

TODO
- [x] `scene/cameras.py`에 `camera_type` 속성 추가 및 초기화 경로 마련
- [ ] `scene/dataset_readers.py`에서 ERP/LONLAT 데이터셋 인식 및 `camera_type=3`로 설정
- [ ] (옵션) ERP용 위도 가중맵 유틸 추가(`utils/image_utils.py` 또는 `utils/loss_utils.py`)


## 3) Optimizer / 스케줄러 정렬(OmniGS 규칙)
OmniGS는 파라미터 그룹을 고정 구성으로 Adam에 할당하고, 위치(xyz)에 대해 지수 스케줄(지연 멀티 포함)을 사용합니다.

OmniGS 파라미터 그룹(기본 Adam)
- group0: xyz — lr = expon_lr(step)
- group1: features_dc — lr = feature_lr
- group2: features_rest — lr = feature_lr / 20
- group3: opacity — lr = opacity_lr
- group4: scaling — lr = scaling_lr
- group5: rotation — lr = rotation_lr

스케줄링
- `position_lr`: 지수 로그-선형 보간 + (선택) 지연 멀티. Python에서는 `utils.general_utils.get_expon_lr_func` 또는 동등 구현 사용.
- `spatial_lr_scale` 곱 적용(OmniGS와 동일 동작).

현 3DGS 상태 점검
- `scene/gaussian_model.py:1`의 그룹 구성/스케줄은 OmniGS와 거의 동일. 다만, SparseAdam 옵션이 있는데, OmniGS 기준은 Adam 고정. 유지/선택 옵션으로 두되, 기본은 Adam 권장.

TODO
- [ ] `scene/gaussian_model.py` 학습 설정이 OmniGS 그룹/스케줄과 정확히 일치하는지 재확인(계수/이름/초기값)
- [ ] `train.py:1`의 `update_learning_rate` 호출 타이밍/반환값 사용 정합성 확인
- [ ] (옵션) SparseAdam 경로는 플래그 유지하되 기본 비활성(OmniGS 동작에 맞춤)


## 4) Gaussian densify & prune 규칙/스케줄
OmniGS 규칙 요약(C++ 참조)
- 매 이터레이션: radii/grad 통계 적산 → 간헐적(Interval)으로 densify+prune 수행
- prune: (opacity < th) OR (screen_radii > max_screen) OR (scaling > 0.1*extent) 조건
- resetOpacity: 주기적으로 opacity를 낮춰 재학습 유도
- densify 방식: clone(작은 스케일) + split(큰 스케일) 혼합, grad 기반 임계치 적용

현 3DGS 파이썬은 거의 동일 로직을 가짐. OmniGS 세부치 차이는 다음을 검토/정렬:
- `percent_dense` 사용과 scene extent 스케일링(OmniGS `percentDense()`)
- exist-since-iter(생성 이터레이션) 추적: OmniGS는 신규/가시성 기반 후처리에서 참조. 파이썬에도 경량 추가 가능(선택).
- ERP에서 `markVisible`이 전 True인 특성으로 인한 과잉 densify 방지: radii 기반 필터 유지 권장.

TODO
- [ ] `train.py:1` densify 스케줄(시작/종료/interval)과 임계치들을 OmniGS 기본값으로 동기화
- [ ] `scene/gaussian_model.py`의 `densify_and_prune` 로직이 OmniGS와 조건식/임계치/WS/VS 비교 로직 일치하도록 재확인
- [ ] opacity reset 주기와 white background 특수케이스(초기 reset) 반영 확인
- [ ] (선택) `exist_since_iter` 텐서 추가 및 densify 시 전파(OmniGS와 동일 형태로 확장)


## 5) 손실/정규화(ERP 가중 포함)
기본 손실: L1 + λ·(1-SSIM). OmniGS 기본 λ는 `GaussianOptimizationParams.lambda_dssim` 참조. 3DGS는 fused-ssim 사용 가능.

ERP 권장 사항
- 위도 가중(cosine weight)으로 손실 가중(ERP 픽셀의 샘플링 왜곡 보정). 구현은 `H`축 기준 cos(π·(y/H - 0.5)) 또는 sph 좌표 기반 가중 적용.
- 깊이 정규화: ERP의 깊이 반환은 의미 보장 없음 → ERP에서는 깊이 항 비활성화 권장.

TODO
- [ ] `utils/loss_utils.py`에 ERP용 가중 손실 옵션 추가(플래그/파라미터로 on/off)
- [ ] `train.py:1`에서 ERP일 때 깊이 항 비활, 가중 손실 적용 분기
- [ ] λ(dssim) 등 하이퍼는 OmniGS 기본값 노출/설정 가능하도록 `arguments/` 갱신


## 6) 가시성 처리(markVisible)와 성능
- OmniGS의 `markVisible`은 ERP에서 전 가시(True)로 반환(테스트 참조). 훈련 단계에서는 radii 기반 `visibility_filter`를 유지하여 densify 통계에 사용.
- (옵션) 뷰어/디버깅 용도에서만 `markVisible` 사용.

TODO
- [ ] `omnigs_rasterization`의 `markVisible`을 유틸로 노출하고, 필요 시 디버그에 활용
- [ ] Densify 통계는 `visibility_filter=(radii>0)` 기준 유지


## 7) 체크포인트/로깅/평가
- TensorBoard/로그는 기존 3DGS 경로 유지 가능.
- 평가 루틴에서 ERP일 때 PSNR/L1는 동일 계산, 다만 필요 시 위도 가중 버전 병행 보고.

TODO
- [ ] 테스트 이터레이션/세이브 이터레이션 유지, ERP일 때 추가 지표(가중 PSNR 등) 선택 제공
- [ ] PLY 저장/로드 포맷은 3DGS와 호환 유지(OmniGS와 동일 필드명 확인)


## 8) CLI/설정 항목 정리(새 인자)
권장 인자 추가(기본은 기존과 호환 유지):
- `--rasterizer {diff,omnigs}`: 디폴트 `omnigs`
- `--camera_type {pinhole,lonlat}` 또는 데이터셋에서 자동 설정
- `--erp_weighted_loss {true|false}`
- `--lambda_dssim`/`--densify_*`/`--opacity_reset_interval` 등 OmniGS 기본 하이퍼 노출

TODO
- [ ] `arguments/*.py` 확장 및 `README.md` 사용법 갱신
- [ ] 기본값은 OmniGS 권장값으로 설정, 실험 재현성 위해 seed 인자 정리


## 9) 마이그레이션 순서(권장)
1) 렌더러 교체(호출부/반환값 동일 유지) → 단위 테스트(해상도/형상)
2) 카메라 타입 전달(ERP 샘플로 색상 역전파 스모크) → loss off
3) 스케줄/옵티마이저 값 정렬 및 학습 루프 타이밍 동기화 → 짧은 훈련 스모크
4) densify/prune 스케줄 정렬 → radii/가시성/프루닝 동작 확인
5) ERP 가중 손실/깊이 항 분기 → 메트릭 재확인
6) CLI/문서 갱신 → 전체 스크립트/체크포인트 검증


## 10) 세부 TODO 체크리스트(파일 기준)
- 렌더러/카메라
  - [x] `gaussian_renderer/__init__.py` Omnigs 래스터라이저로 교체 및 `camera_type` 전달
  - [x] `scene/cameras.py` `camera_type` 필드 추가, ERP 초기화 경로
  - [ ] `scene/dataset_readers.py` ERP 데이터셋 처리 및 카메라 타입 설정
- 학습 루프/옵티마이저/스케줄
  - [ ] `train.py` OmniGS 스케줄/den&prune 타이밍 동기화, ERP depth 비활성 분기
  - [ ] `scene/gaussian_model.py` 파라미터 그룹/스케줄 값 재점검(OmniGS와 일치)
  - [ ] (선택) `exist_since_iter` 추가 및 densify 경로 전파
- 손실/메트릭
  - [ ] `utils/loss_utils.py` ERP 위도 가중 옵션 추가
  - [ ] `metrics.py`(선택) 가중 PSNR/L1 병행 보고 옵션
- 설정/문서
  - [ ] `arguments/*` 새 인자 추가, `README.md`/`docs` 사용법 갱신
  - [ ] `requirements.txt`에 `submodules/omnigs_rasterization` 설치 안내(개발환경 문서화)


## 11) 검증 플랜(스모크 → 기능)
- 스모크: 무작위 입력으로 PINHOLE/ERP 각각 forward/backward 통과, 출력 텐서 형상/유한성 체크
- 기능: 단기 학습(수천 iter)에서 loss 감소/포인트 수 변화(densify) 관찰, prune 정상 동작 확인
- 회귀: 주요 하이퍼(λ_dssim/percent_dense/interval) 변화 시 품질/속도 영향 벤치 기록

진행 현황
- [x] 비CUDA 단위 테스트: PINHOLE/ERP camera_type 전달 및 tanfov 기본값 검증
- [x] CUDA 스모크: OmniGS 래스터라이저 GPU forward(색상/깊이) 통과(PINHOLE)


## 부록: 주요 차이 정리(요약)
- 카메라: OmniGS는 `camera_type` 지원(PINHOLE=1, LONLAT=3), 3DGS는 기본 PINHOLE
- 깊이: OmniGS는 단일 패스 색상 경로에 depth 합성 구현(v0 Python 래퍼는 이중 호출로 depth 제공), ERP에서는 의미적 보장 없음
- 옵티마이저: OmniGS는 Adam 고정/그룹별 lr; 3DGS는 SparseAdam 옵션 보유(기본 Adam 권장)
- densify/prune: 규칙은 거의 동일, 임계치/스케줄 및 percent_dense/extent 반영을 OmniGS와 일치시킬 것
