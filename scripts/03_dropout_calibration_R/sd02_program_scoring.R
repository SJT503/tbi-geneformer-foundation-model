#!/usr/bin/env Rscript
# sd02_program_scoring.R — snRNA 描述层 Step1: 程序评分 + 置换阈值 + 存在性 + 池内对比
# 数据: 释放对象 novogene_atlas_annotated.rds (35,047, GSE333879 口径) — 论文默认释放口径 (README §2.3)
# 裁决遵守 (README §3.3): within-library ✅ / 存在性 ✅ / 跨组统计 ❌ / 跨组描述 ⚠️"本数据集中观察到"
# 方法: 评分 = 检出程序基因的 log 归一化均值 (项目先例 phaseB_10);
#       阈值 = 每库×程序 1000 次同尺寸随机基因集置换的第95百分位 (Park 2025 AUCell 置换结构的均值评分版)
# 双联体防御: 仅 singlet 进入主分析 (scDblFinder_class, CSV 关联; oligo SenMayo 双联体伪影教训)
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

set.seed(20261006)
RDS <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/wet_lab_data/results/seurat/novogene_atlas_annotated.rds"
CSV <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/SciData_descriptor/geo_submission/geo_upload_ready/per_nucleus_metadata.csv"
SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")

DAM_GENES   <- c("Apoe","Trem2","Tyrobp","Lpl","Cst7","Clec7a","Itgax","Lilrb4","Axl","Csf1")
HOMEO_GENES <- c("P2ry12","Tmem119","Cx3cr1","Selplg","Fcrls")
N_PERM <- 1000
cat("=== sd02 开始", format(Sys.time()), "===\n")

# ---- 入场门禁 (README §3.1) ----
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
cat(sprintf("门禁全过 | 双联体 %d (%.1f%%), 主分析仅 singlet\n", sum(dbl == "doublet"), 100 * mean(dbl == "doublet")))

senmayo_all <- read.csv(file.path(SEN, "data/senmayo_saul2022_mouse_final.csv"), stringsAsFactors = FALSE)$gene

# ---- 子集 + singlet + 纯度门 ----
prep <- function(celltype_label, pos_gene, neg_gene) {
  s <- subset(seu, subset = novogene_celltype == celltype_label)
  s$dbl <- dbl[match(colnames(s), csv$barcode)]
  sg <- s[, s$dbl == "singlet"]
  dat <- GetAssayData(sg, assay = "RNA", layer = "data")
  ratio <- mean(dat[pos_gene, ]) / max(mean(dat[neg_gene, ]), 1e-9)
  cat(sprintf("[%s] 全体 %d → singlet %d | 纯度 %s/%s = %.2f\n",
      celltype_label, ncol(s), ncol(sg), pos_gene, neg_gene, ratio))
  stopifnot(ratio >= 3)   # 输入数据验证铁律
  sg
}
mg  <- prep("Microglia",  "Hexb", "Snap25")
ast <- prep("Astrocytes", "Aqp4", "Snap25")
gm <- GetAssayData(mg,  assay = "RNA", layer = "data")
ga <- GetAssayData(ast, assay = "RNA", layer = "data")

# 基因集检出
gs <- list(SenMayo = senmayo_all, DAM = DAM_GENES, Homeostatic = HOMEO_GENES)
gs <- lapply(gs, function(g) intersect(g, rownames(gm)))
cat(sprintf("命中(MG): SenMayo %d/%d | DAM %d/%d | HOMEO %d/%d\n",
    length(gs$SenMayo), length(senmayo_all), length(gs$DAM), length(DAM_GENES),
    length(gs$Homeostatic), length(HOMEO_GENES)))
cat("DAM缺失:", paste(setdiff(DAM_GENES, rownames(gm)), collapse = ","), "\n")
cat("SenMayo缺失:", paste(setdiff(senmayo_all, rownames(gm)), collapse = ","), "\n")
ast_senmayo <- intersect(senmayo_all, rownames(ga))

# ---- 评分函数 (phaseB_10 约定) ----
score <- function(m, genes, cells) {
  g <- intersect(genes, rownames(m))
  as.numeric(Matrix::colMeans(m[g, cells, drop = FALSE]))
}

# ---- 每库置换阈值 + high 判定 ----
perm_threshold <- function(m, genes, lib_vec, label) {
  cells_all <- colnames(m)
  det_pool <- rownames(m)[Matrix::rowMeans(m[, cells_all, drop = FALSE] > 0) >= 0.05]
  out <- list()
  for (lb in c("Sham", "CCI_Scr", "CCI_TGFbetaR1_KO")) {
    cells <- cells_all[lib_vec == lb]
    if (length(cells) < 30) { cat(sprintf("  [%s|%s] n=%d <30 跳过\n", label, lb, length(cells))); next }
    sc <- score(m, genes, cells)
    null <- numeric(N_PERM)
    for (i in seq_len(N_PERM)) null[i] <- mean(Matrix::colMeans(m[sample(det_pool, length(genes)), cells, drop = FALSE]))
    thr <- quantile(null, 0.95, names = FALSE)
    hi <- sum(sc > thr)
    out[[lb]] <- data.frame(program = label, library = lb, n_cells = length(cells),
                            n_perm = N_PERM, threshold = thr, n_high = hi,
                            pct_high = 100 * hi / length(cells), stringsAsFactors = FALSE)
    attr(out[[lb]], "scores") <- setNames(sc, cells)
    cat(sprintf("  [%s|%s] n=%d 阈值=%.4f high=%d (%.1f%%)\n", label, lb, length(cells), thr, hi, 100 * hi / length(cells)))
  }
  out
}

meta_mg  <- data.frame(barcode = colnames(mg),  stim = as.character(mg$stim),
                       cluster = as.character(mg$seurat_clusters),  stringsAsFactors = FALSE)
meta_ast <- data.frame(barcode = colnames(ast), stim = as.character(ast$stim),
                       cluster = as.character(ast$seurat_clusters), stringsAsFactors = FALSE)

cat("\n[MG 置换阈值]\n")
res_mg <- list(SenMayo = perm_threshold(gm, gs$SenMayo, meta_mg$stim, "SenMayo"),
               DAM = perm_threshold(gm, gs$DAM, meta_mg$stim, "DAM"),
               Homeostatic = perm_threshold(gm, gs$Homeostatic, meta_mg$stim, "Homeostatic"))
cat("\n[Astro 置换阈值]\n")
res_ast <- list(SenMayo = perm_threshold(ga, ast_senmayo, meta_ast$stim, "SenMayo"))

strip_attr <- function(x) { for (n in names(x)) { d <- x[[n]]; attr(d, "scores") <- NULL; x[[n]] <- d }; x }
thr_tbl <- rbind(
  do.call(rbind, lapply(res_mg,  function(x) do.call(rbind, strip_attr(x)))),
  do.call(rbind, lapply(res_ast, function(x) do.call(rbind, strip_attr(x)))))
write.csv(thr_tbl, file.path(OUT, "permutation_thresholds.csv"), row.names = FALSE)

# ---- 每核评分表 ----
sc_mg <- data.frame(barcode = meta_mg$barcode, stim = meta_mg$stim, cluster = meta_mg$cluster,
                    celltype = "Microglia",
                    SenMayo = score(gm, gs$SenMayo, meta_mg$barcode),
                    DAM = score(gm, gs$DAM, meta_mg$barcode),
                    Homeostatic = score(gm, gs$Homeostatic, meta_mg$barcode))
sc_ast <- data.frame(barcode = meta_ast$barcode, stim = meta_ast$stim, cluster = meta_ast$cluster,
                     celltype = "Astrocytes",
                     SenMayo = score(ga, ast_senmayo, meta_ast$barcode),
                     DAM = NA_real_, Homeostatic = NA_real_)
hi_flag <- function(df, res, prog) {
  df[[paste0(prog, "_high")]] <- NA
  for (lb in names(res[[prog]])) {
    s <- attr(res[[prog]][[lb]], "scores")
    idx <- match(names(s), df$barcode)
    df[[paste0(prog, "_high")]][idx] <- as.integer(df[[prog]][idx] > res[[prog]][[lb]]$threshold)
  }
  df
}
sc_mg <- hi_flag(sc_mg, res_mg, "SenMayo"); sc_mg <- hi_flag(sc_mg, res_mg, "DAM"); sc_mg <- hi_flag(sc_mg, res_mg, "Homeostatic")
sc_ast <- hi_flag(sc_ast, res_ast, "SenMayo")
# 星形未跑 DAM/Homeostatic 置换 → 补 NA 列对齐列名集 (rbind.data.frame 按列名匹配)
sc_ast$DAM_high <- NA_integer_; sc_ast$Homeostatic_high <- NA_integer_

# p16/p21/Iba1 presence (IF 面板)
pres <- function(m, df, genes) {
  do.call(rbind, lapply(genes, function(g) {
    if (!g %in% rownames(m)) return(NULL)
    v <- as.numeric(m[g, df$barcode])
    do.call(rbind, lapply(split(df, df$stim), function(d) {
      vv <- v[match(d$barcode, df$barcode)]
      data.frame(celltype = unique(d$celltype), gene = g, library = unique(d$stim), n = nrow(d),
                 pct_detect = 100 * mean(vv > 0), mean_log = mean(vv), stringsAsFactors = FALSE)
    }))
  }))
}
pres_tbl <- rbind(pres(gm, sc_mg, c("Cdkn2a", "Cdkn1a", "Aif1", "Trem2", "Apoe")),
                  pres(ga, sc_ast, c("Cdkn2a", "Cdkn1a", "Gfap")))
write.csv(pres_tbl, file.path(OUT, "presence_table.csv"), row.names = FALSE)

# ---- 全细胞型 SenMayo 描述表 (跨组描述 ⚠️ 只作"本数据集中观察到") ----
sen_ok <- intersect(senmayo_all, rownames(seu))
all_scores <- Matrix::colMeans(GetAssayData(seu, assay = "RNA", layer = "data")[sen_ok, ])
ct_sum <- do.call(rbind, lapply(split(data.frame(ct = ct, stim = libs, s = as.numeric(all_scores)), paste(ct, libs, sep = "|")), function(d)
  data.frame(celltype = d$ct[1], library = d$stim[1], n = nrow(d), mean_senmayo = mean(d$s), stringsAsFactors = FALSE)))
write.csv(ct_sum, file.path(OUT, "celltype_senmayo_summary.csv"), row.names = FALSE)

# ---- 池内对比: CCI_Scr MG SenMayo-high vs low (n≥30 才做) ----
deg_path <- file.path(OUT, "deg_cciScr_mg_senmayo_hi_vs_lo.csv")
cc <- sc_mg[sc_mg$stim == "CCI_Scr", ]
n_hi <- sum(cc$SenMayo_high == 1, na.rm = TRUE)
cat(sprintf("\n[CCI_Scr 池内 MG SenMayo-high=%d vs low=%d]\n", n_hi, sum(cc$SenMayo_high == 0, na.rm = TRUE)))
if (n_hi >= 30) {
  mgc <- mg[, cc$barcode]
  Idents(mgc) <- factor(cc$SenMayo_high, levels = c(1, 0))
  deg <- FindMarkers(mgc, ident.1 = "1", ident.2 = "0", test.use = "wilcox",
                     assay = "RNA", verbose = FALSE)
  deg$gene <- rownames(deg); deg <- deg[order(deg$avg_log2FC, decreasing = TRUE), ]
  write.csv(deg, deg_path, row.names = FALSE)
  cat(sprintf("DEG %d 条 (log2FC>0.25 & padj<0.05: %d)\n", nrow(deg),
      sum(deg$avg_log2FC > 0.25 & deg$p_val_adj < 0.05)))
} else cat(sprintf("n_high=%d <30 → 按宪法不跑统计, 仅报告评分分布\n", n_hi))

write.csv(rbind(sc_mg, sc_ast), file.path(OUT, "per_nucleus_scores.csv"), row.names = FALSE)
cat(sprintf("\n产物目录: %s\n", OUT))
cat("=== sd02 结束", format(Sys.time()), "===\n")
