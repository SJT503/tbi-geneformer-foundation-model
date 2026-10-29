#!/usr/bin/env Rscript
# sd04_presence_ladder.R — A2 丰度对照阶梯 (Paper2 任务卡 §1, 2026-10-08)
# 检验: Cdkn2a/Cdkn1a 检出贴地板由基因丰度(mean_log)驱动而非生物学缺席/在场
# 群体: singlet MG (主, sd02 prep 约定) + 全体核 35,047 (doublet-inclusive 对照)
# 产出: 梯度表 / 全基因组 Spearman / 丰度匹配零假设 / 十分位曲线 / 判读汇总
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

set.seed(20261006)
RDS <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/wet_lab_data/results/seurat/novogene_atlas_annotated.rds"
CSV <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/SciData_descriptor/geo_submission/geo_upload_ready/per_nucleus_metadata.csv"
SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")

TIERS <- list(
  Housekeeping = c("Actb","Gapdh","B2m","Rpl13a","Rplp0","Rps18","Hprt1","Tbp"),
  MG_identity  = c("Hexb","Csf1r","Tmem119","P2ry12","Selplg","Cx3cr1"),
  DAM_mid      = c("Spp1","Gpnmb","Apoe","Axl","Itgax","Cst7","Igf1","Lpl","Trem2","Tyrobp","Clec7a","Lilrb4"),
  Senescence   = c("Cdkn2a","Cdkn1a")
)
cat("=== sd04 开始", format(Sys.time()), "===\n")

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
cat("入口五门禁全过 | 双联体剔除对照群体不含, singlet MG 主分析\n")

# ---- 群体 1: singlet MG (sd02 prep 约定: subset + singlet + 纯度门禁) ----
s <- subset(seu, subset = novogene_celltype == "Microglia")
sg <- s[, dbl[match(colnames(s), csv$barcode)] == "singlet"]
dat_mg <- GetAssayData(sg, assay = "RNA", layer = "data")
purity <- mean(dat_mg["Hexb", ]) / max(mean(dat_mg["Snap25", ]), 1e-9)
cat(sprintf("singlet MG %d | 纯度 Hexb/Snap25 = %.2f\n", ncol(sg), purity))
stopifnot(purity >= 3)

# ---- 群体 2: 全体核 (release object 全部 35,047) ----
dat_all <- GetAssayData(seu, assay = "RNA", layer = "data")
cat(sprintf("全体核 %d × %d 基因\n", ncol(dat_all), nrow(dat_all)))

# ---- 高效 per-gene presence (dgCMatrix 列压缩; 修正 +1L, 任务卡 §1.6) ----
gene_stats <- function(m) {
  nz <- m@x != 0
  cnt <- tabulate(m@i[nz] + 1L, nrow(m))
  data.frame(gene = rownames(m), n = ncol(m),
             pct_detect = 100 * cnt / ncol(m),
             mean_log = as.numeric(Matrix::rowMeans(m)), stringsAsFactors = FALSE)
}
st_mg  <- gene_stats(dat_mg)
st_all <- gene_stats(dat_all)

# 抽样交叉验证: 10 随机基因 tabulate vs 直接计算逐位对照
set.seed(20261006)
chk <- sample(rownames(dat_mg), 10)
for (g in chk) {
  direct <- 100 * mean(dat_mg[g, ] > 0)
  fast <- st_mg$pct_detect[st_mg$gene == g]
  stopifnot(abs(direct - fast) < 1e-9)
}
cat("tabulate 交叉验证 10/10 过 (含随机基因, 逐位一致)\n")

# ---- 1. 梯度表: 群体 × 库 × tier × gene ----
pres_row <- function(m, stim, pop_label, tr, g) {
  v <- as.numeric(m[g, ])
  do.call(rbind, lapply(split(data.frame(v = v, stim = stim), stim), function(d)
    data.frame(population = pop_label, tier = tr, gene = g, library = d$stim[1],
               n = nrow(d), pct_detect = 100 * mean(d$v > 0), mean_log = mean(d$v),
               stringsAsFactors = FALSE)))
}
mk_tier <- function(m, stim, pop_label) {
  do.call(rbind, unname(lapply(names(TIERS), function(tr) {
    hit <- intersect(TIERS[[tr]], rownames(m))
    miss <- setdiff(TIERS[[tr]], rownames(m))
    if (length(miss)) cat(sprintf("[%s|%s] 缺失基因: %s\n", pop_label, tr, paste(miss, collapse = ",")))
    do.call(rbind, lapply(hit, function(g) pres_row(m, stim, pop_label, tr, g)))
  })))
}
tier_tbl <- rbind(mk_tier(dat_mg, as.character(sg$stim), "singlet_MG"),
                  mk_tier(dat_all, libs, "all_nuclei"))
write.csv(tier_tbl, file.path(OUT, "sd04_tier_presence.csv"), row.names = FALSE)
cat(sprintf("梯度表 %d 行写盘\n", nrow(tier_tbl)))

# ---- 2. 全基因组 Spearman(pct_detect ~ mean_log) per 群体 ----
mk_spearman <- function(st, pop) {
  ct_ <- suppressWarnings(cor.test(st$pct_detect, st$mean_log, method = "spearman"))
  data.frame(population = pop, n_genes = nrow(st),
             n_zero_genes = sum(st$mean_log == 0),
             rho = unname(ct_$estimate), p = ct_$p.value, stringsAsFactors = FALSE)
}
spearman_tbl <- rbind(mk_spearman(st_mg, "singlet_MG"), mk_spearman(st_all, "all_nuclei"))
write.csv(spearman_tbl, file.path(OUT, "sd04_genomewide_spearman.csv"), row.names = FALSE)
print(spearman_tbl)

# ---- 3. 丰度匹配零假设: mean_log 落在目标 ±20% 相对窗口的基因 ----
TARGETS <- c("Cdkn2a", "Cdkn1a")
ab_summary <- function(st, pop) {
  do.call(rbind, lapply(TARGETS, function(tg) {
    tm <- st$mean_log[st$gene == tg]; tp <- st$pct_detect[st$gene == tg]
    cand <- st[st$mean_log >= 0.8 * tm & st$mean_log <= 1.2 * tm & st$gene != tg, ]
    data.frame(population = pop, target_gene = tg, target_mean_log = tm, target_pct = tp,
               n_matched = nrow(cand),
               matched_pct_median = if (nrow(cand)) median(cand$pct_detect) else NA_real_,
               matched_pct_iqr25 = if (nrow(cand)) unname(quantile(cand$pct_detect, .25)) else NA_real_,
               matched_pct_iqr75 = if (nrow(cand)) unname(quantile(cand$pct_detect, .75)) else NA_real_,
               stringsAsFactors = FALSE)
  }))
}
ab_genes <- do.call(rbind, lapply(c("singlet_MG", "all_nuclei"), function(pop) {
  st <- if (pop == "singlet_MG") st_mg else st_all
  do.call(rbind, lapply(TARGETS, function(tg) {
    tm <- st$mean_log[st$gene == tg]
    cand <- st[st$mean_log >= 0.8 * tm & st$mean_log <= 1.2 * tm & st$gene != tg, ]
    if (nrow(cand)) data.frame(population = pop, target_gene = tg, gene = cand$gene,
                               mean_log = cand$mean_log, pct_detect = cand$pct_detect,
                               stringsAsFactors = FALSE) else NULL
  }))
}))
ab_tbl <- rbind(ab_summary(st_mg, "singlet_MG"), ab_summary(st_all, "all_nuclei"))
write.csv(ab_tbl, file.path(OUT, "sd04_abundance_matched.csv"), row.names = FALSE)
if (!is.null(ab_genes)) write.csv(ab_genes, file.path(OUT, "sd04_abundance_matched_genes.csv"), row.names = FALSE)
print(ab_tbl)

# ---- 4. 十分位曲线 (mean_log 十等分 → 每分位 mean pct_detect) ----
dec_curve <- function(st, pop) {
  q <- quantile(st$mean_log, probs = seq(0, 1, 0.1), names = FALSE)
  brks <- unique(q)
  bin <- cut(st$mean_log, breaks = brks, include.lowest = TRUE, labels = FALSE)
  do.call(rbind, lapply(sort(unique(bin)), function(b) {
    idx <- bin == b
    data.frame(population = pop, decile = b, n_genes = sum(idx),
               mean_log_lo = brks[b], mean_log_hi = brks[b + 1],
               mean_pct_detect = mean(st$pct_detect[idx]), stringsAsFactors = FALSE)
  }))
}
dec_tbl <- rbind(dec_curve(st_mg, "singlet_MG"), dec_curve(st_all, "all_nuclei"))
write.csv(dec_tbl, file.path(OUT, "sd04_decile_curve.csv"), row.names = FALSE)
cat("十分位曲线写盘\n")

# ---- 5. 判读汇总 ----
cat("\n===== A2 判读汇总 =====\n")
for (pop in c("singlet_MG", "all_nuclei")) {
  cat(sprintf("\n[%s] Spearman rho = %.3f (n=%d)\n", pop,
              spearman_tbl$rho[spearman_tbl$population == pop],
              spearman_tbl$n_genes[spearman_tbl$population == pop]))
  for (tr in names(TIERS)) {
    x <- tier_tbl[tier_tbl$population == pop & tier_tbl$tier == tr, ]
    cat(sprintf("  %-13s pct_detect 范围 %6.2f – %6.2f | 均值 %6.2f\n", tr,
                min(x$pct_detect), max(x$pct_detect), mean(x$pct_detect)))
  }
}
cat("\n丰度匹配 (目标 vs 窗口内基因 pct_detect):\n")
for (i in seq_len(nrow(ab_tbl))) {
  r <- ab_tbl[i, ]
  cat(sprintf("  [%s|%s] 目标 pct=%.2f%% | 窗口内 n=%d, pct 中位=%s IQR=[%s,%s]\n",
              r$population, r$target_gene, r$target_pct, r$n_matched,
              ifelse(is.na(r$matched_pct_median), "NA", sprintf("%.2f", r$matched_pct_median)),
              ifelse(is.na(r$matched_pct_iqr25), "NA", sprintf("%.2f", r$matched_pct_iqr25)),
              ifelse(is.na(r$matched_pct_iqr75), "NA", sprintf("%.2f", r$matched_pct_iqr75))))
}
cat(sprintf("\n判读基准: 看家≥90%% / MG身份高 / DAM中段 / Cdkn<1%% / rho>0.8 / 丰度匹配≈同水平 → A2 成立\n"))
cat("\n产物目录:", OUT, "\n")
cat("=== sd04 结束", format(Sys.time()), "===\n")
