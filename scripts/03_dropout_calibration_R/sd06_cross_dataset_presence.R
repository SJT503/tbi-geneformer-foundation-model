#!/usr/bin/env Rscript
# sd06_cross_dataset_presence.R — A3 跨数据集 presence 重算, 轻路径 (Paper2 任务卡 §3, 2026-10-08)
# 段1: GSE269748 B臂 (7d/6mo × 4细胞型, singlet-only, counts>0 口径) 全 11 基因 panel
# 段2: GSE271769 celltable (无 cell_type/看家 → 仅全核 Cdkn1a/Cdkn2a per 时点, 诚实 flag)
# presence 不变性: lognorm>0 ⟺ counts>0 (归一化不改变非零性), 两层可直接对比
# 重型 RDS (GSE271769/GSE277487 ckpt) 由 sd06b 单独后台跑
suppressPackageStartupMessages({ library(Matrix) })

SEN <- "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project"
OUT <- file.path(SEN, "results/sd_snRNA_descriptive")
PANEL <- c("Cdkn2a","Cdkn1a","Actb","Gapdh","B2m","Hexb","Tmem119","P2ry12","Apoe","Spp1","Gpnmb")
cat("=== sd06 轻路径开始", format(Sys.time()), "===\n")

# ---- 段 1: GSE269748 B 臂 (counts.mtx + genes.txt + cells.csv) ----
D1 <- file.path(SEN, "results/s1_269748_7d6mo")
cells <- read.csv(file.path(D1, "cells.csv"), stringsAsFactors = FALSE)
genes <- readLines(file.path(D1, "genes.txt"))
cat(sprintf("[GSE269748] cells.csv %d 细胞 | genes.txt %d 基因 | 列: %s\n",
            nrow(cells), length(genes), paste(names(cells), collapse = ",")))
stopifnot(all(c("cell_id","sample","time_point","cell_type","is_dbl") %in% names(cells)))
tp1 <- sort(unique(cells$time_point)); ct1 <- sort(unique(cells$cell_type))
cat(sprintf("  时点: %s | 细胞型: %s\n", paste(tp1, collapse = "/"), paste(ct1, collapse = "/")))

sing <- cells[cells$is_dbl == 0, ]
cat(sprintf("  singlet-only: %d (剔双联体 %d)\n", nrow(sing), sum(cells$is_dbl == 1)))

cat("  读 counts.mtx (380MB MatrixMarket)...\n")
m <- readMM(file.path(D1, "counts.mtx"))
stopifnot(ncol(m) == nrow(cells), nrow(m) == length(genes))
rownames(m) <- genes
gidx <- match(PANEL, genes)
miss1 <- PANEL[is.na(gidx)]
if (length(miss1)) cat(sprintf("  🔴 panel 缺失基因: %s\n", paste(miss1, collapse = ",")))
gidx <- gidx[!is.na(gidx)]; ghit <- PANEL[!is.na(match(PANEL, genes))]
cat("  约定: counts.mtx 列序与 cells.csv 行序一一对应 (s1 管线成对导出)\n")

# 逐基因 × 时点 × 细胞型 presence (counts>0)
res1 <- do.call(rbind, lapply(ghit, function(g) {
  vi <- match(g, genes)
  v <- as.numeric(m[vi, ]) # 按原始顺序, 再对齐 sing 行
  do.call(rbind, lapply(split(sing, paste(sing$time_point, sing$cell_type, sep = "|")), function(d) {
    idx <- match(d$cell_id, cells$cell_id)
    data.frame(dataset = "GSE269748_B", gene = g, time_point = d$time_point[1],
               cell_type = d$cell_type[1], n = nrow(d),
               pct_detect = 100 * mean(v[idx] > 0), mean_counts = mean(v[idx]),
               stringsAsFactors = FALSE)
  }))
}))
write.csv(res1, file.path(OUT, "sd06_gse269748_presence.csv"), row.names = FALSE)
cat(sprintf("  产物 sd06_gse269748_presence.csv %d 行\n", nrow(res1)))

# 段 1 判读速览: Cdkn vs 看家 per 细胞型×时点
cat("\n[GSE269748 速览: Cdkn2a/Cdkn1a 最大 presence vs 看家最小 presence]\n")
for (g in c("Cdkn2a","Cdkn1a")) {
  x <- res1[res1$gene == g, ]
  cat(sprintf("  %s: 最大 %.2f%% (%s|%s) | 范围 %.2f–%.2f%%\n", g, max(x$pct_detect),
      x$time_point[which.max(x$pct_detect)], x$cell_type[which.max(x$pct_detect)],
      min(x$pct_detect), max(x$pct_detect)))
}
hk <- res1[res1$gene %in% c("Actb","Gapdh","B2m"), ]
cat(sprintf("  看家(Actb/Gapdh/B2m): 范围 %.2f–%.2f%%\n", min(hk$pct_detect), max(hk$pct_detect)))

# ---- 段 2: GSE271769 celltable (仅全核, 诚实 flag) ----
D2 <- file.path(SEN, "results/s1_gse271769")
tb <- read.csv(file.path(D2, "celltable.csv.gz"), stringsAsFactors = FALSE)
cat(sprintf("\n[GSE271769] celltable %d 细胞 | 列: %s\n", nrow(tb), paste(names(tb), collapse = ",")))
stopifnot(all(c("cell_id","sample","time_point","Cdkn1a","Cdkn2a") %in% names(tb)))
tp2 <- sort(unique(tb$time_point))
cat(sprintf("  时点: %s\n", paste(tp2, collapse = "/")))
cat("  ⚠️ 诚实 flag: 无 cell_type 列、无看家基因 → 仅全核 presence per 时点, 不做细胞型分层\n")
res2 <- do.call(rbind, lapply(c("Cdkn1a","Cdkn2a"), function(g) {
  do.call(rbind, lapply(split(tb, tb$time_point), function(d)
    data.frame(dataset = "GSE271769_wholenuc", gene = g, time_point = d$time_point[1],
               cell_type = "all_nuclei_no_flag", n = nrow(d),
               pct_detect = 100 * mean(d[[g]] > 0), mean_counts = mean(d[[g]]),
               stringsAsFactors = FALSE)))
}))
write.csv(res2, file.path(OUT, "sd06_gse271769_wholenuc_presence.csv"), row.names = FALSE)
print(res2)

cat("\n判读基准: Cdkn 一致 <2%% 且看家正常 → 领域级检测地板成立\n")
cat("=== sd06 轻路径结束", format(Sys.time()), "===\n")
