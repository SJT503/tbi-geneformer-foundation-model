#!/usr/bin/env Rscript
# sd05_robustness_sensitivity.R — B5 评分稳健性 (Paper2 任务卡 §2, 2026-10-08)
# 检验: SenMayo 信号 (+29% 位移 + DAM-hi/lo 差异) 不是少数低检出基因撑起的伪影
# 🔴 设计修正: SenMayo 鼠版真表 (Saul 2022, 119 基因) 设计上不含 Cdkn2a/Cdkn1a
#    → 第一步验证并报告两者缺席, 再走 S1-S4 (原"显式剔除臂"不适用)
# 🔴 DAM_high 旗标独立性声明: 分组旗标固定自 sd02 per_nucleus_scores.csv
#    (DAM 10 基因程序评分 × 每库置换阈值), 与 SenMayo 基因集完全无关 — 无循环
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

set.seed(20261006)
RDS <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/wet_lab_data/results/seurat/novogene_atlas_annotated.rds"
CSV <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/SciData_descriptor/geo_submission/geo_upload_ready/per_nucleus_metadata.csv"
SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")
cat("=== sd05 开始", format(Sys.time()), "===\n")

# ---- Step 0: 设计验证 — SenMayo 真表不含 Cdkn2a/Cdkn1a ----
senmayo_all <- read.csv(file.path(SEN, "data/senmayo_saul2022_mouse_final.csv"),
                        stringsAsFactors = FALSE)$gene
cat(sprintf("SenMayo 真表 %d 基因 | Cdkn2a 在表内: %s | Cdkn1a 在表内: %s\n",
            length(senmayo_all), "Cdkn2a" %in% senmayo_all, "Cdkn1a" %in% senmayo_all))
stopifnot(!"Cdkn2a" %in% senmayo_all, !"Cdkn1a" %in% senmayo_all)
cat("设计属性确认: 两基因缺席为 Saul 2022 真表设计, 非本分析剔除 → 无需显式剔除臂\n\n")

# ---- 入口五门禁 (README §3.1, 逐字复用) ----
seu <- readRDS(RDS)
stopifnot(ncol(seu) == 35047)
libs <- as.character(seu$stim)
stopifnot(sum(libs == "Sham") == 11378, sum(libs == "CCI_Scr") == 6845,
          sum(libs == "CCI_TGFbetaR1_KO") == 16824)
ct <- as.character(seu$novogene_celltype)
stopifnot(sum(ct == "Microglia") == 2155, sum(ct == "Astrocytes") == 1038)
csv <- read.csv(CSV, stringsAsFactors = FALSE)
mm <- match(colnames(seu), csv$barcode); stopifnot(!any(is.na(mm)))
dbl <- csv$scDblFinder_class[mm]

# singlet MG + 纯度门禁 (sd02 prep 约定)
s <- subset(seu, subset = novogene_celltype == "Microglia")
sg <- s[, dbl[match(colnames(s), csv$barcode)] == "singlet"]
dat_sg <- GetAssayData(sg, assay = "RNA", layer = "data")
purity <- mean(dat_sg["Hexb", ]) / max(mean(dat_sg["Snap25", ]), 1e-9)
cat(sprintf("singlet MG %d | 纯度 Hexb/Snap25 = %.2f\n", ncol(sg), purity))
stopifnot(purity >= 3)

dat <- GetAssayData(seu, assay = "RNA", layer = "data")
hit <- intersect(senmayo_all, rownames(dat))
cat(sprintf("SenMayo 命中 %d/%d\n", length(hit), length(senmayo_all)))

# ---- 群体矩阵 ----
# (a) 口径: MG 全体核 (doublet-inclusive, n=671/494/990) — 与 sd02 celltype_senmayo_summary 一致
mg_all <- dat[, ct == "Microglia"]
libs_mg <- libs[ct == "Microglia"]
stopifnot(sum(libs_mg == "Sham") == 671, sum(libs_mg == "CCI_Scr") == 494,
          sum(libs_mg == "CCI_TGFbetaR1_KO") == 990)
# (b) 口径: singlet MG CCI_Scr 池内, DAM 旗标固定自 sd02
pn <- read.csv(file.path(OUT, "per_nucleus_scores.csv"), stringsAsFactors = FALSE)
mm2 <- match(colnames(sg), pn$barcode); stopifnot(!any(is.na(mm2)))
stopifnot(all(pn$celltype[mm2] == "Microglia"))
flag <- pn$DAM_high[mm2]
cc <- as.character(sg$stim) == "CCI_Scr"
n_hi <- sum(cc & flag == 1); n_lo <- sum(cc & flag == 0)
stopifnot(n_hi == 101, n_lo == 356)   # sd02/sd03 已核验期望值
cells_hi <- colnames(sg)[cc & flag == 1]; cells_lo <- colnames(sg)[cc & flag == 0]

# ---- 成员基因 pooled presence (MG 全体核 2,155, 单一固定判据) ----
pct_gene <- sapply(hit, function(g) 100 * mean(mg_all[g, ] > 0))
names(pct_gene) <- hit

# ---- 每臂重算函数 ----
arm_run <- function(gset, arm_label) {
  sc_all <- as.numeric(Matrix::colMeans(mg_all[gset, , drop = FALSE]))
  m_sham <- mean(sc_all[libs_mg == "Sham"]); m_cci <- mean(sc_all[libs_mg == "CCI_Scr"])
  shift <- 100 * (m_cci / m_sham - 1)
  sh <- as.numeric(Matrix::colMeans(dat_sg[gset, cells_hi, drop = FALSE]))
  sl <- as.numeric(Matrix::colMeans(dat_sg[gset, cells_lo, drop = FALSE]))
  w <- wilcox.test(sh, sl)
  data.frame(arm = arm_label, n_genes = length(gset),
             dropped = paste(setdiff(hit, gset), collapse = ";"),
             mean_sham = m_sham, mean_cci = m_cci, shift_pct = shift,
             mean_dam_hi = mean(sh), mean_dam_lo = mean(sl), wilcox_p = w$p.value,
             stringsAsFactors = FALSE)
}

# ---- S0 基线 (复现不了 = 实现错, stop) ----
s0 <- arm_run(hit, "S0_full")
cat(sprintf("[S0] MG 全体核: Sham %.5f → CCI_Scr %.5f (+%.1f%%) | 期望 0.09160→0.11855 (+29.4%%)\n",
            s0$mean_sham, s0$mean_cci, s0$shift_pct))
stopifnot(abs(s0$mean_sham - 0.0916039691823679) < 1e-9,
          abs(s0$mean_cci - 0.118551270076091) < 1e-9)
cat(sprintf("[S0] DAM-hi %.4f vs lo %.4f | Wilcoxon p=%.3g | 期望 0.1455 vs 0.1112, p=5.9e-08\n",
            s0$mean_dam_hi, s0$mean_dam_lo, s0$wilcox_p))
stopifnot(s0$wilcox_p < 1e-6, s0$mean_dam_hi > s0$mean_dam_lo)
cat("S0 双数复现 ✓\n\n")

# ---- S1-S3 ----
ord <- sort(pct_gene)                      # 升序
n_drop_decile <- floor(length(hit) * 0.10) # 最低十分位 ≈ 10 基因
s1 <- arm_run(hit[pct_gene[hit] >= 1], "S1_drop_floor_lt1pct")
s2 <- arm_run(hit[!(hit %in% names(ord)[seq_len(n_drop_decile)])], "S2_drop_bottom_decile")
s3 <- arm_run(hit[pct_gene[hit] >= 10], "S3_keep_ge10pct")
cat(sprintf("[S1] 剔 %d 基因 (pooled pct<1%%): +%.1f%%, p=%.3g\n", length(hit) - s1$n_genes, s1$shift_pct, s1$wilcox_p))
cat(sprintf("[S2] 剔最低十分位 %d 基因: +%.1f%%, p=%.3g\n", length(hit) - s2$n_genes, s2$shift_pct, s2$wilcox_p))
cat(sprintf("[S3] 只留 pct≥10%% (%d 基因): +%.1f%%, p=%.3g\n", s3$n_genes, s3$shift_pct, s3$wilcox_p))

# ---- S4 leave-one-out 最坏情形 (107 次迭代) ----
loo <- do.call(rbind, lapply(seq_along(hit), function(i) {
  r <- arm_run(hit[-i], paste0("S4_loo_", hit[i]))
  r
}))
loo$worst_p <- loo$wilcox_p == max(loo$wilcox_p)
loo$worst_shift <- loo$shift_pct == min(loo$shift_pct)
cat(sprintf("[S4] LOO 107 次: shift 范围 [%.1f%%, %.1f%%] | p 范围 [%.3g, %.3g]\n",
            min(loo$shift_pct), max(loo$shift_pct), min(loo$wilcox_p), max(loo$wilcox_p)))

arms <- rbind(s0, s1, s2, s3)
write.csv(arms, file.path(OUT, "sd05_arms_summary.csv"), row.names = FALSE)
write.csv(loo, file.path(OUT, "sd05_loo_worst.csv"), row.names = FALSE)

# ---- 判读 ----
cat("\n===== B5 判读 =====\n")
dir_ok <- all(c(s1$shift_pct, s2$shift_pct, s3$shift_pct, loo$shift_pct) > 0) &&
          all(c(s1$mean_dam_hi, s2$mean_dam_hi, s3$mean_dam_hi, loo$mean_dam_hi) >
              c(s1$mean_dam_lo, s2$mean_dam_lo, s3$mean_dam_lo, loo$mean_dam_lo))
shift_dev <- max(abs(c(s1$shift_pct, s2$shift_pct, s3$shift_pct, loo$shift_pct) - s0$shift_pct))
loo_worst_p <- max(loo$wilcox_p)
cat(sprintf("方向稳定 (shift>0 且 hi>lo): %s\n", ifelse(dir_ok, "✓", "🔴 翻转红旗")))
cat(sprintf("S1-S4 最大 |shift 偏离|: %.1f pp (基准 ~20-40 pp 内) | S0 shift=%.1f%%\n", shift_dev, s0$shift_pct))
cat(sprintf("LOO 最坏 p = %.3g (判据 <1e-5: %s)\n", loo_worst_p, ifelse(loo_worst_p < 1e-5, "✓", "🔴 红旗")))
if (loo_worst_p > 0.001 || !dir_ok) cat("🔴 红旗存在 — 如实报告, 不粉饰\n")
cat("\n产物: sd05_arms_summary.csv + sd05_loo_worst.csv\n")
cat("=== sd05 结束", format(Sys.time()), "===\n")
