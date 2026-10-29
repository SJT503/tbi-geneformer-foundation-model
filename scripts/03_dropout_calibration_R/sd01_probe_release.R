#!/usr/bin/env Rscript
# sd01_probe_release.R — snRNA 描述层 Step0 探针（preflight + canary）
# 目的: 验证释放对象 novogene_atlas_annotated.rds 的全部执行假设, 不产出任何论文数据
# 假设清单: 元数据列名 / 分库计数 / 亚群计数 / 双联体列 / RNA assay 层结构 / 纯度门 / 基因集命中 / 评分速度
# 裁决依据: DATA_ACCESS_README.md §2.3 (论文默认释放口径 35,047) + §3 用途裁决表

suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

RDS <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/wet_lab_data/results/seurat/novogene_atlas_annotated.rds"
SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
CSV <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/SciData_descriptor/geo_submission/geo_upload_ready/per_nucleus_metadata.csv"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

cat("=== sd01 探针 开始", format(Sys.time()), "===\n")

# ---- 1. 加载计时 ----
t0 <- Sys.time()
seu <- readRDS(RDS)
cat(sprintf("加载耗时 %.1f 秒\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

# ---- 2. 结构检查 ----
cat("\n[对象结构]\n")
print(dim(seu))
cat("assays:", paste(Assays(seu), collapse = ", "), "\n")
cat("DefaultAssay:", DefaultAssay(seu), "\n")
for (a in Assays(seu)) cat(sprintf("  assay=%s layers=%s\n", a, paste(Layers(seu, assay = a), collapse = ",")))
cat("metadata列:", paste(colnames(seu@meta.data), collapse = ", "), "\n")

# ---- 3. Gate 1-3: 总核数 / 分库 / 亚群 (README §3.1 焊死值) ----
# 实测修正: RDS 元数据无 library 列, 分组列 = stim (母项目宪法命名); scDblFinder 等在 CSV
cat("\n[Gate 断言]\n")
stopifnot(ncol(seu) == 35047)
cat("[stim 分组表]\n"); print(table(as.character(seu$stim)))
cat("[orig.ident 表]\n"); print(table(as.character(seu$orig.ident)))
libs <- as.character(seu$stim)
stopifnot(sum(libs == "Sham") == 11378,
          sum(libs == "CCI_Scr") == 6845,
          sum(libs == "CCI_TGFbetaR1_KO") == 16824)
ct <- as.character(seu$novogene_celltype)
print(table(ct))
stopifnot(sum(ct == "Microglia") == 2155, sum(ct == "Astrocytes") == 1038)
cat("Gate1-3 全过 (分组列 = stim; library ≡ 分组 经 stim/orig.ident 成立)\n")

# ---- 3b. CSV 关联检查 (barcode 匹配率, 为双联体过滤做准备) ----
cat("\n[CSV 关联]\n")
csv <- read.csv(CSV, stringsAsFactors = FALSE)
cat(sprintf("CSV %d行 | RDS barcodes 命中 %d/%d\n",
    nrow(csv), sum(colnames(seu) %in% csv$barcode), ncol(seu)))
cat("RDS barcode 示例:", paste(head(colnames(seu), 3), collapse = ","), "\n")
cat("CSV  barcode 示例:", paste(head(csv$barcode, 3), collapse = ","), "\n")

# ---- 4. 双联体列 (从 CSV 关联; 释放对象自带的 QC 标志) ----
cat("\n[scDblFinder]\n")
mm <- match(colnames(seu), csv$barcode)
dbl <- csv$scDblFinder_class[mm]
stopifnot(!any(is.na(dbl)))
print(table(dbl, libs))

# ---- 5. RNA assay 层结构 + 归一化层判定 ----
cat("\n[RNA assay]\n")
rn <- Layers(seu, assay = "RNA")
has_data <- "data" %in% rn
has_counts <- "counts" %in% rn
cat(sprintf("RNA layers=%s | counts:%s data:%s\n", paste(rn, collapse = ","), has_counts, has_data))

# 小胶质/星形子集 (释放口径, 含双联体; 纯度门在归一化后测)
mg <- subset(seu, subset = novogene_celltype == "Microglia")
as_ <- subset(seu, subset = novogene_celltype == "Astrocytes")
cat(sprintf("MG=%d  Astro=%d\n", ncol(mg), ncol(as_)))

if (!has_data) {
  mg <- NormalizeData(mg, assay = "RNA", verbose = FALSE)
  as_ <- NormalizeData(as_, assay = "RNA", verbose = FALSE)
}
gm <- GetAssayData(mg, assay = "RNA", layer = "data")
ga <- GetAssayData(as_, assay = "RNA", layer = "data")

# ---- 6. 纯度门 (输入数据验证铁律; 注意: 释放口径未过下游QC, 此处实测) ----
cat("\n[纯度门]\n")
mk <- function(m, g) if (g %in% rownames(m)) mean(m[g, ]) else NA_real_
mg_purity <- mk(gm, "Hexb") / max(mk(gm, "Snap25"), 1e-9)
as_purity <- mk(ga, "Aqp4") / max(mk(ga, "Snap25"), 1e-9)
cat(sprintf("MG   Hexb/Snap25 = %.2f  (阈值3)\n", mg_purity))
cat(sprintf("Astro Aqp4/Snap25 = %.2f  (阈值3)\n", as_purity))
# 探针只报告不 stop —— 纯度是否达标决定 sd02 策略, 由主会话裁决

# ---- 7. 基因集命中 ----
cat("\n[基因集命中]\n")
senmayo <- read.csv(file.path(SEN, "data/senmayo_saul2022_mouse_final.csv"), stringsAsFactors = FALSE)$gene
DAM_GENES <- c("Apoe","Trem2","Tyrobp","Lpl","Cst7","Clec7a","Itgax","Lilrb4","Axl","Csf1")
HOMEO_GENES <- c("P2ry12","Tmem119","Cx3cr1","Selplg","Fcrls")
cat(sprintf("SenMayo %d/%d | DAM %d/%d | HOMEO %d/%d  (MG rownames)\n",
    length(intersect(senmayo, rownames(gm))), length(senmayo),
    length(intersect(DAM_GENES, rownames(gm))), length(DAM_GENES),
    length(intersect(HOMEO_GENES, rownames(gm))), length(HOMEO_GENES)))
# 缺失基因列出前10个
miss <- setdiff(senmayo, rownames(gm))
if (length(miss)) cat("SenMayo缺失(前10):", paste(head(miss, 10), collapse = ","), "\n")

# ---- 8. p16/p21/Iba1 存在性 (IF 面板预期管理) ----
cat("\n[IF面板基因 presence]\n")
for (g in c("Cdkn2a","Cdkn1a","Aif1","Trem2")) {
  if (g %in% rownames(gm)) {
    v <- as.numeric(gm[g, ])
    cat(sprintf("MG %-7s: 检出 %.1f%%  均值log %.4f\n", g, 100 * mean(v > 0), mean(v)))
  } else cat(sprintf("MG %-7s: 不在矩阵\n", g))
}

# ---- 9. Canary: 评分 + 置换速度实测 (100细胞 × 50置换) ----
cat("\n[Canary 计时]\n")
set.seed(20261006)
sub <- gm[, 1:100]
t1 <- Sys.time()
score_one <- function(m, genes) as.numeric(Matrix::colMeans(m[intersect(genes, rownames(m)), , drop = FALSE]))
real <- score_one(sub, senmayo)
detected <- rownames(gm)[Matrix::rowMeans(gm > 0) >= 0.05]
cat(sprintf("可抽样背景基因池: %d (检出≥5%%)\n", length(detected)))
for (i in 1:50) {
  nullg <- sample(detected, length(intersect(senmayo, rownames(gm))))
  score_one(sub, nullg)
}
dt <- as.numeric(difftime(Sys.time(), t1, units = "secs"))
cat(sprintf("50置换×100细胞 = %.2f 秒\n", dt))
# 外推: 全量 2155 MG + 1038 astro, 1000置换×~5程序(每程序3库分库算)
n_cells_total <- 2155 + 1038
est_full <- dt / 50 * 1000 * n_cells_total / 100
cat(sprintf("外推全量(1000置换, 全细胞, 单程序): ~%.0f 秒\n", est_full))

cat("\n=== sd01 探针结束", format(Sys.time()), "===\n")
