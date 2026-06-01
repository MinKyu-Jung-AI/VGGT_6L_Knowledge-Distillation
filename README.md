# 6L HALO — Final (AUC@5 = 0.641)

VGGT의 24개 global attention 중 6개를 Gated Linear Attention(Lightning)으로 치환하여 연산 비용을 감소시킨 최종 성공 구성입니다. 원본 VGGT 대비 성능 저하를 최소화하면서 1.25배의 가속을 달성했습니다.

- **치환 레이어**: `[2, 5, 7, 9, 18, 20]` (6/24)
- **최종 체크포인트**: [weights/6L_best_AUC0641.pt](weights/6L_best_AUC0641.pt)
- **CO3D AUC@5**: **0.641** (원본 VGGT: 0.662, 성능 -2.1% 유지하며 **1.25× 가속**)

---

## 📄 License & Acknowledgement

본 프로젝트는 Meta Platforms에서 공개한 **VGGT (Visual Geometry Grounded Transformer)**를 기반으로 연구 개발되었습니다. 

* **License:** 본 저장소의 코드는 VGGT License(연구 목적으로 허용되는 제한적 라이선스)의 조건을 따르며, 배포 및 사용 시 원본 라이선스 사본([LICENSE.txt](LICENSE.txt))을 동봉해야 합니다.
* **Citation:** VGGT 연구를 인용하거나 활용할 경우 아래의 출처를 명시해야 합니다.

```bibtex
@article{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and others},
  journal={arXiv preprint arXiv:2503.11651},
  year={2025}
}

## 디렉토리 구조

```
6L_final/
├── stage1_layerwise/   # Stage 1: 레이어별 독립 정렬 (재구성)
├── stage2_fmse/        # Stage 2: End-to-End Feature MSE + Pose (재구성) ← 0.641 달성
├── stage3_task/        # Stage 3: 0.641 위에서 진행한 후속 실험들
├── weights/            # 최종 가중치 + 18A 백업
├── vggt-fresh/         # VGGT 코드 (학습·추론에 필요한 전체 repo)
├── run_viser_6L.py     # 시각화 스크립트
└── README.md
```

## 재현 순서

### Stage 1 — Layer-wise Alignment
```bash
cd /root && python 6L_final/stage1_layerwise/run_stage1_layerwise_6L.py
```
- 대상: `[2, 5, 7, 9, 18, 20]` 각 레이어를 독립적으로 정렬
- Loss: `MSE(teacher_hidden, student_hidden) + 0.5*(1 - cos)`
- LR=3e-4, 1500 steps/layer
- 출력: `/root/halo_checkpoints_6L_layerwise/layer{i}.pt`

### Stage 2 — End-to-End Feature MSE (최종 0.641)
```bash
python 6L_final/stage2_fmse/run_stage2_fmse_6L.py
```
- 6개 Lightning 블록 동시 학습
- Distill layers: `[2, 4, 5, 7, 9, 11, 12, 14, 17, 18, 20, 23]` (Lightning + DPT 레이어 가중치 3×)
- Loss: `feat_mse + 0.5 * pose_L1`
- B=4, S=8, LR=5e-5, 4000 steps
- 출력: `/root/halo_checkpoints_6L_stage2/best.pt` → 이 체크포인트가 AUC 0.641

### Stage 3 — Task-specific 후속 실험 (선택)
`stage3_task/` 안의 3개 스크립트는 0.641 위에서 각각 다른 fine-tune 전략을 시도한 실험들.
- `run_6L_task_only.py` — teacher distillation 없이 CO3D GT pose 직접 학습
- `run_6L_fmse_task.py` — feature MSE + pose L1 계속 이어가기
- `run_6L_patch_focus.py` — L9 patch token 집중 학습


