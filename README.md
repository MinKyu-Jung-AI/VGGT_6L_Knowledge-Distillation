# VGGT_6L_Knowledge Distillation

VGGT의 24개 전역 셀프 어텐션(Global Self-Attention) 레이어 중 6개를 Gated Linear Attention(Lightning)으로 치환하여 연산 비용을 감소시킨 최종 성공 구성입니다. 원본 VGGT 대비 성능 저하를 최소화하면서 1.25배의 추론 가속을 달성했습니다.

- **치환 레이어**: `[2, 5, 7, 9, 18, 20]` (6/24)
- **CO3D AUC@5**: **0.641** (원본 VGGT: 0.662, 성능 하락을 단 -2.1%로 방어하며 **1.25× 가속**)

---

##  Hardware Requirements (하드웨어 사양)

실험 및 재현에 필요한 최소 하드웨어 사양입니다.
* **VRAM 사용량:** 제안한 6L+18A 모델 구동 시 약 **8.9 GB** 소요 (원본 VGGT 9.7 GB 대비 절감)
* **OS / Environment:** Linux 기반 환경, **Python 3.10.11** 기반 구동

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
