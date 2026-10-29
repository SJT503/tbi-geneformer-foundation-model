#!/usr/bin/env Rscript
# sd06c_diag_actb.R — 诊断 sd06b 核验红旗: ckpt_GSE271769 counts 层 Actb/Gapdh 全零
# 假设: x22b 重建 GENE_UNIVERSE 时 make.unique 重名行 ("Actb.1" 携带真计数, match("Actb") 命中空行)
# 动作: ① 列出 panel 基因的全部重名变体及其 rowSums; ② 若后缀行有真计数, 用真行重算 Actb/Gapdh
#        presence per 细胞型×时点, 写 addendum; ③ Cdkn 行确认无重名变体(交叉验证已过的独立证据)
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")
PANEL <- c("Cdkn2a","Cdkn1a","Actb","Gapdh","B2m","Hexb","Tmem119","P2ry12","Apoe","Spp1","Gpnmb")
CKPT <- file.path(SEN, "results/fullrun/ckpt/ckpt_GSE271769.rds")
cat("=== sd06c 诊断开始", format(Sys.time()), "===\n")
seu <- readRDS(CKPT)
cts <- tryCatch(GetAssayData(seu, assay = "RNA", layer = "counts"),
                error = function(e) { seu[["RNA"]] <- JoinLayers(seu[["RNA"]]); GetAssayData(seu, assay = "RNA", layer = "counts") })
rn <- rownames(cts)
cat(sprintf("ckpt: %d 细胞 × %d 基因 | rownames 中 make.unique 后缀行总数: %d\n",
            ncol(cts), nrow(cts), sum(grepl("\\.[0-9]+$", rn))))

cat("\n[panel 基因重名变体及其计数行和]\n")
variants <- list()
for (g in PANEL) {
  v <- rn[grepl(paste0("^", g, "(\\.[0-9]+)?$"), rn)]
  if (length(v) == 0) { cat(sprintf("  %-8s 无任何命中行!\n", g)); next }
  rs <- Matrix::rowSums(cts[v, , drop = FALSE])
  for (vv in v) cat(sprintf("  %-10s rowSums = %d\n", vv, as.integer(rs[vv])))
  variants[[g]] <- v
}

# Cdkn 行独立确认
cat("\n[Cdkn 行确认]\n")
for (g in c("Cdkn2a","Cdkn1a")) {
  v <- variants[[g]]
  if (length(v) == 1) cat(sprintf("  %s: 唯一行 %s (rowSums=%d) → sd06b 数值为真行\n", g, v, as.integer(Matrix::rowSums(cts[v, , drop = FALSE]))))
  else cat(sprintf("  %s: %d 个变体 — 需人工判断!\n", g, length(v)))
}

# Actb/Gapdh 真行重算 (若存在后缀真行)
ctb <- as.character(seu$cell_type_broad); dp <- as.character(seu$days_post)
key <- paste(ctb, dp, sep = "|")
add <- list()
for (g in c("Actb","Gapdh")) {
  v <- variants[[g]]
  real <- names(sort(Matrix::rowSums(cts[v, , drop = FALSE]), decreasing = TRUE))[1]  # 计数最多的行 = 真行
  if (length(v) > 1) cat(sprintf("\n[%s] %d 变体, 取真行 %s 重算\n", g, length(v), real))
  vec <- as.numeric(cts[real, , drop = FALSE])
  add[[g]] <- do.call(rbind, lapply(split(seq_along(key), key), function(ii)
    data.frame(dataset = "GSE271769_ckpt", gene = g, row_used = real,
               cell_type_broad = ctb[ii[1]], days_post = dp[ii[1]], n = length(ii),
               pct_detect = 100 * mean(vec[ii] > 0), mean_counts = mean(vec[ii]),
               stringsAsFactors = FALSE)))
}
add_df <- do.call(rbind, add)
write.csv(add_df, file.path(OUT, "sd06c_actb_gapdh_addendum.csv"), row.names = FALSE)
cat(sprintf("\naddendum %d 行写盘\n", nrow(add_df)))
cat("\n[Actb/Gapdh MG 行速览]\n")
mg <- add_df[add_df$cell_type_broad == "Microglia", ]
print(mg[, c("gene","row_used","days_post","n","pct_detect","mean_counts")])
cat("=== sd06c 结束", format(Sys.time()), "===\n")
