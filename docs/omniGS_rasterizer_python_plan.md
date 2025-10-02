# OmniGS Rasterizer Python 통합 — TODO 플랜

본 문서는 OmniGS CUDA 래스터라이저를 PyTorch 확장으로 통합하기 위한 TODO 중심 계획입니다. 세부 맥락은 최소화하고, 실행 가능한 체크리스트와 경로/명령 위주로 구성했습니다.

## 주요 TODO 체크리스트 (메인)

### v0 필수 (기능/통합)
- [x] 패키지 스캐폴딩 생성(자급자족 구조; submodules 비의존 빌드)
  - [x] `omnigs_rasterization/` 루트 패키지 구성
  - [x] OmniGS 필요한 소스 복사: `src/`, `cuda_rasterizer/`, `include/`
  - [x] GLM 동봉: `third_party/glm/`
  - [x] `omnigs_rasterization/omnigs_rasterization/{ext.cpp,__init__.py}` 추가
  - [x] `omnigs_rasterization/{setup.py,pyproject.toml}` 신규 생성
  - [x] 스모크 스크립트: `omnigs_rasterization/omnigs_rasterization/tests/quickcheck.py`
- [ ] 파이썬 래퍼 구현(autograd)
  - [ ] `_RasterizeGaussians` autograd.Function forward/backward 구현
  - [ ] `_C.rasterize_gaussians`/`_C.rasterize_gaussians_backward` 호출/버퍼 저장
  - [ ] 인자 상호배타성/유효성/contiguous 검사(3DGS 호환)
  - [ ] `mark_visible` 파이썬 함수 구현 → `_C.mark_visible` 위임
- [ ] 기존 파이프라인 연동(API 호환)
  - [ ] `(render, radii, depth)` 형태로 반환(필요 시 depth는 double-call)
  - [ ] PINHOLE/LONLAT 카메라 분기 반영(`camera_type`)

### v0 검증 (정합성/안정성)
- [ ] 빌드 및 스모크 테스트
  - [ ] `pip install -e submodules/omnigs_rasterization --no-build-isolation`
  - [ ] `python -m omnigs_rasterization.tests.quickcheck` 모듈 로드 확인
- [ ] 간단 렌더 스모크
  - [ ] 무작위 입력으로 PINHOLE/LONLAT, `render_depth=True/False` shape/유효값 확인
- [ ] gradcheck(소규모 샘플)
  - [ ] means3D/opacity/colors/SH 등 일부 파라미터에 대해 수치 미분 검사

### v1 성능 (후속 최적화)
- [ ] 단일 패스 color+depth 동시 출력(geometry/binning 재사용)
- [ ] 프로파일링/성능 수치화 및 회귀 테스트

### 환경/빌드 정책
- [x] PyTorch/CUDA 버전 비강제(사용자 환경 torch 그대로 사용)
- [x] 디렉터리 정리: 루트 `omnigs_rasterization` → `submodules/omnigs_rasterization`로 이동, 구 중복 폴더 삭제
- [x] setup.py 정비(로컬 소스 경로만 사용, self-contained)
- [x] C++17 지정(cxx/nvcc) — OmniGS 코드 요구 반영
- [x] GLM 동봉(`third_party/glm`) — 별도 자동탐지/환경변수 필요 없음
- [x] `TORCH_CUDA_ARCH_LIST` 미설정 시 경고만(성능 권장값 안내)

### 라이선스/주의
- [ ] 라이선스 고지/문서화: OmniGS(GPLv3), 3DGS(비상업 연구용) 혼용 시 내부 사용/배포 정책 점검

---

## 작업 가이드 (요약)

### 설치/빌드
- 사전 조건: 환경에 PyTorch(+CUDA) 설치되어 있어야 함(버전 핀 없음)
- 명령:
  - `cd submodules/omnigs_rasterization`
  - `pip install -e . --no-build-isolation`
  - 팁: `export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"`(권장, 선택)

### 사용(예고)
- `from omnigs_rasterization import GaussianRasterizer, GaussianRasterizationSettings`
- v0에서는 `(render, radii, depth)` 반환(depth 필요 시 double-call)

---

## 입출력/시그니처(요약)

- 입력 텐서(float32, cuda, contiguous 권장)
  - `means3D`: (P,3)
  - `colors_precomp`=(P,3) 또는 `sh`=(P,M,3) with `M=(degree+1)^2`
  - `opacity`: (P,1), `scales`: (P,3), `rotations`: (P,4), `cov3D_precomp`=(P,6)|빈 텐서
  - `viewmatrix`:(4,4), `projmatrix`:(4,4), `campos`:(3,) or (1,3)
- 출력 텐서
  - forward: `(num_rendered, color(C,H,W), radii(P), geomBuf, binningBuf, imgBuf)`
  - backward 반환 순서: `(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations)`
- 카메라: `camera_type`=1(PINHOLE), 3(LONLAT)
- depth: v0는 `render_depth=True` 시 color에 depth 합성 → 파이썬 래퍼에서 double-call로 분리 제공

---

## 핵심 경로
- 빌드 스크립트: `submodules/omnigs_rasterization/setup.py`
- 바인딩: `submodules/omnigs_rasterization/omnigs_rasterization/ext.cpp`
- 파이썬 래퍼: `submodules/omnigs_rasterization/omnigs_rasterization/__init__.py`
- 스모크: `submodules/omnigs_rasterization/omnigs_rasterization/tests/quickcheck.py`
- 복사된 헤더/API: `submodules/omnigs_rasterization/include/rasterize_points.h`
- 복사된 커널: `submodules/omnigs_rasterization/src/rasterize_points.cu`, `submodules/omnigs_rasterization/cuda_rasterizer/*.cu|*.h`

---

## 참고(근거 소스)
- Forward/Backward 선언: `submodules/OmniGS/include/rasterize_points.h`
- PINHOLE/LONLAT 분기 및 depth 합성: `submodules/OmniGS/src/rasterize_points.cu`, `submodules/OmniGS/cuda_rasterizer/rasterizer_impl.cu`
- 3DGS 바인딩 예시: `submodules/diff-gaussian-rasterization/ext.cpp`, `.../diff_gaussian_rasterization/__init__.py`, `.../setup.py`

---

## 상세 계획(컨텍스트 유지)

### 목표
- `omnigs_rasterization/` 폴더만으로 빌드/사용 가능한 자급자족 패키지 구성(서브모듈 비의존)
- 3DGS(`diff_gaussian_rasterization`)와 최대한 호환되는 파이썬 API 제공(+ OmniGS 고유 기능 유지)

### 현재 구조 요약(자급자족 패키지)
- 빌드 스크립트: `omnigs_rasterization/setup.py` (CUDAExtension)
- 네이티브 소스(복사본): `omnigs_rasterization/{src,include,cuda_rasterizer}`
- GLM 포함: `omnigs_rasterization/third_party/glm`
- 파이썬 엔트리/래퍼: `omnigs_rasterization/omnigs_rasterization/{ext.cpp,__init__.py}`

### 3DGS vs OmniGS 차이(바인딩 영향)
- Depth 출력: OmniGS는 `render_depth` 시 color 버퍼에 depth 합성(3DGS는 별도 invdepths 텐서)
- 카메라 모델: OmniGS `camera_type` 지원(PINHOLE=1, LONLAT=3), 3DGS는 pinhole 고정
- GLM 의존: 헤더 포함 필요(동봉 처리)

### 실행 계획(요지)
1) 파일 복사(완료): OmniGS 핵심 CUDA/헤더 → `omnigs_rasterization` 내부로 이관
2) 빌드 스크립트(완료): 로컬 경로만 사용, C++17, GLM 포함, 아키텍처 경고
3) 파이바인딩: Forward/Backward/mark_visible 노출(ext.cpp)
4) 파이썬 래퍼: autograd.Function 구현 및 모듈 래퍼 제공(다음 작업)
5) 파이프라인 통합: 3요소 반환(render, radii, depth) 유지, depth 필요 시 double-call(v0)
6) 검증: 스모크/gradcheck/간단 성능 확인
7) 최적화(v1): 단일 패스 color+depth 동시 출력(geometry/binning 재사용)

### 유효성/정합성 체크리스트
- 텐서 속성: device=cuda, dtype=float32, contiguous 강제
- 인자 상호배타성: `sh` xor `colors_precomp`; `(scales,rotations)` xor `cov3D_precomp`
- 카메라 타입 유효성: 1 또는 3
- 예외 메시지: 3DGS 톤과 유사하게 명확히

### 검증 시나리오(요약)
- 무작위 점/SH/색상, PINHOLE/LONLAT 각각 렌더 → shape/범위 확인
- `render_depth=True/False` 조합 점검, gradcheck로 일부 파라미터 수치 미분

### 환경/주의
- PyTorch/CUDA 버전 강제 없음(현재 환경 torch 사용), C++17 요구
- GLM은 third_party에 동봉, OpenGL 불필요

### 라이선스
- OmniGS(GPLv3), 3DGS(비상업 연구용) → 내부 사용/배포 정책 준수 필요
