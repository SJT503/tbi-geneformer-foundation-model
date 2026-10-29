#!/usr/bin/env Rscript
# sd03_dam_hivslo.R — snRNA 描述层 Step2: CCI_Scr 池内 MG DAM-high vs low (PI 2026-10-06 点头)
# 裁决: within-library ✅ (README §3.3, 对批次≡分组免疫); n=101 vs 356 均 ≥30 (宪法小样本条款过)
# 回答三问: Q1 DAM-high 是独立群体还是打分尾部 | Q2 弥散 SenMayo 位移住在谁身上 | Q3 慢性残留亚群的分子身份(IF 接口)
# 输入: 释放口径 RDS + sd02 已核验的 per_nucleus_scores.csv (DAM_high 旗标)
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

RDS <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/wet_lab_data/results/seurat/novogene_atlas_annotated.rds"
CSV <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/SciData_descriptor/geo_submission/geo_upload_ready/per_nucleus_metadata.csv"
SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")
set.seed(20261006)
cat("=== sd03 开始", format(Sys.time()), "===\n")

# ---- 入场门禁 (README §3.1) ----
seu <- readRDS(RDS)
stopifnot(ncol(seu) == 35047)
ct <- as.character(seu$novogene_celltype)
stopifnot(sum(ct == "Microglia") == 2155)
csv <- read.csv(CSV, stringsAsFactors = FALSE)
mm <- match(colnames(seu), csv$barcode); stopifnot(!any(is.na(mm)))
dbl <- csv$scDblFinder_class[mm]

# ---- singlet MG + 纯度门 ----
s <- subset(seu, subset = novogene_celltype == "Microglia")
sg <- s[, dbl[match(colnames(s), csv$barcode)] == "singlet"]
dat <- GetAssayData(sg, assay = "RNA", layer = "data")
purity <- mean(dat["Hexb", ]) / mean(dat["Snap25", ])
cat(sprintf("singlet MG %d | 纯度 Hexb/Snap25 = %.2f\n", ncol(sg), purity))
stopifnot(purity >= 3)

# ---- sd02 旗标关联 (8/8 独立核验过) ----
pn <- read.csv(file.path(OUT, "per_nucleus_scores.csv"), stringsAsFactors = FALSE)
mm2 <- match(colnames(sg), pn$barcode); stopifnot(!any(is.na(mm2)))
stopifnot(all(pn$celltype[mm2] == "Microglia"))
flag <- pn$DAM_high[mm2]; sen <- pn$SenMayo[mm2]
cc <- as.character(sg$stim) == "CCI_Scr"
n_hi <- sum(cc & flag == 1); n_lo <- sum(cc & flag == 0)
cat(sprintf("CCI_Scr 池内: DAM-high %d vs low %d\n", n_hi, n_lo))
stopifnot(n_hi == 101, n_lo == 356)   # sd02 已核验期望值

# ---- Q2: 弥散 SenMayo 位移住在谁身上 ----
w <- wilcox.test(sen[cc & flag == 1], sen[cc & flag == 0])
cat(sprintf("\n[Q2] SenMayo 评分: DAM-high 均值 %.4f vs low %.4f | Wilcoxon p=%.3g\n",
    mean(sen[cc & flag == 1]), mean(sen[cc & flag == 0]), w$p.value))
senmayo <- read.csv(file.path(SEN, "data/senmayo_saul2022_mouse_final.csv"),
                    stringsAsFactors = FALSE)$gene

# ---- Q1+Q3: DEG (wilcox, logfc.threshold=0.1 预过滤为 Seurat 默认) ----
mgc <- sg[, cc]
Idents(mgc) <- factor(flag[cc], levels = c(1, 0))
deg <- FindMarkers(mgc, ident.1 = "1", ident.2 = "0", test.use = "wilcox",
                   assay = "RNA", logfc.threshold = 0.1, verbose = FALSE)
deg$gene <- rownames(deg)
deg <- deg[order(deg$avg_log2FC, decreasing = TRUE), ]
write.csv(deg, file.path(OUT, "deg_cciScr_mg_dam_hi_vs_lo.csv"), row.names = FALSE)

up <- deg$gene[deg$avg_log2FC > 0.25 & deg$p_val_adj < 0.05]
dn <- deg$gene[deg$avg_log2FC < -0.25 & deg$p_val_adj < 0.05]
cat(sprintf("\n[Q1] DEG 共 %d 条: up %d / down %d (|log2FC|>0.25 & padj<0.05)\n",
    nrow(deg), length(up), length(dn)))
cat("Top15 up (DAM-high 富集):\n")
print(head(deg[deg$avg_log2FC > 0, ], 15)[, c("gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2")])
cat("Top10 down:\n")
print(tail(deg[deg$avg_log2FC < 0, ], 10)[, c("gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2")])

# ---- Q2 补充: SenMayo 成员在 DEG 中的方向 ----
sen_up <- intersect(up, senmayo); sen_dn <- intersect(dn, senmayo)
cat(sprintf("\n[Q2] SenMayo 成员基因入显著 DEG: up %d / down %d\n", length(sen_up), length(sen_dn)))
if (length(sen_up)) cat("  SenMayo-up:", paste(sen_up, collapse = ","), "\n")

# ---- Q3 身份速读: 关键程序基因的方向 ----
for (g in c("Apoe","Trem2","Tyrobp","Lpl","Cst7","Axl","Itgax","P2ry12","Tmem119","Cx3cr1","Selplg","Fcrls","Cdkn2a","Cdkn1a")) {
  r <- deg[deg$gene == g, ]
  if (nrow(r)) cat(sprintf("  %-8s log2FC=%+.3f padj=%.2g\n", g, r$avg_log2FC, r$p_val_adj))
  else cat(sprintf("  %-8s 未过 logfc 阈值\n", g))
}
cat("\n产物: deg_cciScr_mg_dam_hi_vs_lo.csv\n")
cat("=== sd03 结束", format(Sys.time()), "===\n")
