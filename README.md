# VGGT_6L_Knowledge Distillation

VGGT의 전역 셀프 어텐션(Global Self-Attention) 블록 24개 중 6개를 Gated Linear Attention(GLA)으로 치환하고, 3단계 지식 증류로 조정한 혼합 모델(6L+18A)입니다.

- **치환 레이어:** `[2, 5, 7, 9, 18, 20]` (6/24)
- **CO3Dv2 AUC@5:** 원본 VGGT **0.655** → 6L+18A **0.641** (절대 차이 −0.014, 상대 감소 약 2.1%)
- **추론 시간(입력 이미지 100장):** **16.9초 → 13.5초** (약 20.1% 단축, 약 1.25배 가속)

---

## 실험 환경

- **학습 및 품질 평가:** NVIDIA H200 GPU 1장
- **VRAM 사용량(입력 이미지 10장):** 원본 VGGT **9.7 GB** → 6L+18A **8.9 GB**
- **실행 환경:** Linux, Python 3.10.11

---

## Path Configuration

학습 코드의 모델, 데이터셋, 체크포인트 경로는 환경 변수로 설정할 수 있습니다. 환경 변수를 지정하지 않으면 저장소 내부의 `weights/`, `data/`, `outputs/`를 기본 경로로 사용합니다.

```bash
export VGGT_PRETRAIN_PATH=/path/to/model.pt
export VGGT_CO3D_DIR=/path/to/co3d
export VGGT_CO3D_ANNOTATION_DIR=/path/to/co3d_annotations
export VGGT_DL3DV_DIR=/path/to/DL3DV-10K

python stage1_layerwise/run_stage1_layerwise_6L.py
```

체크포인트와 로그 저장 위치를 포함한 전체 설정 항목은 [`.env.example`](.env.example)에서 확인할 수 있습니다. 이 파일은 경로 예시이며, 실행 전에 필요한 값을 셸 환경 변수로 설정해야 합니다.

---

## License & Acknowledgement

본 프로젝트는 Meta Platforms에서 공개한 **VGGT (Visual Geometry Grounded Transformer)**를 기반으로 연구 개발되었습니다. 

* **License:** 본 저장소의 코드는 VGGT License의 조건을 따르며, 배포 및 사용 시 원본 라이선스 사본([LICENSE.txt](LICENSE.txt))을 반드시 동봉해야 합니다.
* **Citation:** VGGT 연구를 인용하거나 활용할 경우 아래의 출처를 명시해야 합니다.

```bibtex
@article{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and others},
  journal={arXiv preprint arXiv:2503.11651},
  year={2025}
}
```
