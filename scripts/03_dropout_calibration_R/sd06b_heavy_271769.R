#!/usr/bin/env Rscript
# sd06b_heavy_271769.R — A3 段3: GSE271769 ckpt RDS (4.98GB) 细胞型层 panel (Paper2 任务卡 §3 段3)
# 补段2缺口: cell_type_broad × days_post 的 11 基因 presence + 看家对照
# presence 不变性: counts>0 ⟺ lognorm>0 (归一化不改变非零性)
# 防撞车: 单进程独占; 启动前已 ps 验毒; 日志带时间戳语义 (sd06b_gse271769)
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")
PANEL <- c("Cdkn2a","Cdkn1a","Actb","Gapdh","B2m","Hexb","Tmem119","P2ry12","Apoe","Spp1","Gpnmb")
CKPT <- file.path(SEN, "results/fullrun/ckpt/ckpt_GSE271769.rds")
cat("=== sd06b GSE271769 ckpt 开始", format(Sys.time()), "===\n")
cat("读 RDS (4.98GB, compress=FALSE, 预计 1-3 分钟)...\n")
seu <- readRDS(CKPT)
cat(sprintf("读入: %d 细胞 × %d 基因\n", ncol(seu), nrow(seu)))
cat("meta 列:", paste(names(seu@meta.data), collapse = ","), "\n")
stopifnot(all(c("cell_type_broad", "days_post") %in% names(seu@meta.data)))
cat("\ncell_type_broad:\n"); print(table(seu$cell_type_broad))
cat("\ndays_post:\n"); print(table(seu$days_post))
cat("\ngrp:\n"); print(table(seu$grp))

# counts 层 (Seurat v5 分层防御: 取不到就 JoinLayers)
cts <- tryCatch(GetAssayData(seu, assay = "RNA", layer = "counts"),
                error = function(e) {
                  cat("单层 counts 取失败, JoinLayers 后重试:", conditionMessage(e), "\n")
                  seu[["RNA"]] <- JoinLayers(seu[["RNA"]])
                  GetAssayData(seu, assay = "RNA", layer = "counts")
                })
cat(sprintf("counts 矩阵: %d × %d\n", nrow(cts), ncol(cts)))

hit <- intersect(PANEL, rownames(cts))
miss <- setdiff(PANEL, rownames(cts))
if (length(miss)) cat("🔴 panel 缺失:", paste(miss, collapse = ","), "\n")

ctb <- as.character(seu$cell_type_broad)
dp <- as.numeric(as.character(seu$days_post))
key <- paste(ctb, dp, sep = "|")
res <- do.call(rbind, lapply(hit, function(g) {
  v <- as.numeric(cts[g, ])
  do.call(rbind, lapply(split(seq_along(key), key), function(ii)
    data.frame(dataset = "GSE271769_ckpt", gene = g, cell_type_broad = ctb[ii[1]],
               days_post = dp[ii[1]], grp = as.character(seu$grp)[ii[1]], n = length(ii),
               pct_detect = 100 * mean(v[ii] > 0), mean_counts = mean(v[ii]),
               stringsAsFactors = FALSE)))
}))
write.csv(res, file.path(OUT, "sd06b_gse271769_presence.csv"), row.names = FALSE)
cat(sprintf("\n产物 sd06b_gse271769_presence.csv %d 行\n", nrow(res)))

# 与段2全核口径交叉验证 (若细胞集合一致, Cdkn 整库 presence 应接近段2值)
seg2 <- read.csv(file.path(OUT, "sd06_gse271769_wholenuc_presence.csv"), stringsAsFactors = FALSE)
cat("\n[交叉验证] 本表全核加权 vs 段2 全核:\n")
for (g in c("Cdkn1a", "Cdkn2a")) {
  w <- res[res$gene == g, ]
  pooled <- 100 * sum(w$n * w$pct_detect) / sum(w$n)
  cat(sprintf("  %s: 本表加权 %.2f%% | 段2 D1/D7/D28/NoTBI = %s\n", g, pooled,
      paste(sprintf("%.2f", seg2$pct_detect[seg2$gene == g]), collapse = "/")))
}

# 判读速览
cat("\n[速览] Cdkn2a/Cdkn1a 最大 presence per 细胞型:\n")
for (g in c("Cdkn2a", "Cdkn1a")) {
  x <- res[res$gene == g, ]
  i <- which.max(x$pct_detect)
  cat(sprintf("  %s: 最大 %.2f%% (%s|D%s) | MG 行:\n", g, x$pct_detect[i], x$cell_type_broad[i], x$days_post[i]))
  mg <- x[x$cell_type_broad %in% c("Microglia","MG"), ]
  if (nrow(mg)) for (r in seq_len(nrow(mg))) cat(sprintf("    MG D%s (%s): %.2f%% (n=%d)\n",
      mg$days_post[r], mg$grp[r], mg$pct_detect[r], mg$n[r]))
}
cat("=== sd06b 结束", format(Sys.time()), "===\n")
