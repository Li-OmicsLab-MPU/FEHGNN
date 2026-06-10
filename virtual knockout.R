# Single-cell virtual knockout analysis using scTenifoldKnk
# Target gene: MAPK1

library(Seurat)
library(ggplot2)
library(ggrepel)
library(RColorBrewer)

if (!requireNamespace("scTenifoldKnk", quietly = TRUE)) {
  stop("scTenifoldKnk is not installed. Install it with: devtools::install_github('cailab-tamu/scTenifoldKnk')")
}

library(scTenifoldKnk)

set.seed(123)

# Parameters
target_gene <- "MAPK1"
n_hvg <- 2000
pval_threshold <- 0.05

qc_mtThreshold <- 0.1
qc_minLSize <- 1000
nc_nNet <- 10
nc_nCells <- 500

# Output directory
knockout_output_dir <- file.path(output_dir, "MAPK1_Knockout_Analysis")
dir.create(knockout_output_dir, recursive = TRUE, showWarnings = FALSE)

# Check target gene
if (!target_gene %in% rownames(scObject)) {
  stop(paste0(target_gene, " is not found in the Seurat object."))
}

# Extract count matrix
countMat <- GetAssayData(scObject, layer = "counts")

# Select highly variable genes
scObject <- FindVariableFeatures(
  object = scObject,
  selection.method = "vst",
  nfeatures = n_hvg
)

hvgs <- VariableFeatures(scObject)
selected_genes <- unique(c(target_gene, hvgs))
selected_genes <- selected_genes[selected_genes %in% rownames(countMat)]

data <- as.data.frame(countMat[selected_genes, ])

cat("Running scTenifoldKnk knockout analysis for MAPK1...\n")

# Run virtual knockout
result <- scTenifoldKnk(
  countMatrix = data,
  gKO = target_gene,
  qc_mtThreshold = qc_mtThreshold,
  qc_minLSize = qc_minLSize,
  nc_nNet = nc_nNet,
  nc_nCells = nc_nCells
)

# Extract results
df <- result$diffRegulation
df <- df[df$gene != target_gene, ]
df <- df[!is.na(df$p.adj), ]

# Save significant genes
sig_df <- df[df$p.adj < pval_threshold, ]

write.csv(
  sig_df,
  file = file.path(knockout_output_dir, "MAPK1_sigDiff.csv"),
  row.names = FALSE
)

write.csv(
  df,
  file = file.path(knockout_output_dir, "MAPK1_KO_allResults.csv"),
  row.names = FALSE
)

cat("Significant dysregulated genes:", nrow(sig_df), "\n")

# Scatter plot
df$log_p_adj <- -log10(df$p.adj)
df$significant <- ifelse(df$p.adj < pval_threshold, "Significant", "Not significant")

label_genes <- subset(df, p.adj < pval_threshold)

y_upper <- quantile(df$log_p_adj, 0.999, na.rm = TRUE)

if (!is.finite(y_upper) || y_upper <= 0) {
  y_upper <- max(df$log_p_adj, na.rm = TRUE)
}

p_scatter <- ggplot(df, aes(x = Z, y = log_p_adj, color = significant)) +
  geom_point(alpha = 0.7, size = 1.5) +
  scale_color_manual(values = c("Significant" = "#E64B35", "Not significant" = "gray70")) +
  geom_hline(
    yintercept = -log10(pval_threshold),
    linetype = "dashed",
    color = "#E64B35"
  ) +
  geom_text_repel(
    data = label_genes,
    aes(label = gene),
    size = 3,
    max.overlaps = 50,
    color = "black",
    fontface = "italic"
  ) +
  labs(
    title = "MAPK1 Knockout",
    x = "Z-score",
    y = "-log10(adjusted P-value)"
  ) +
  theme_classic(base_size = 14) +
  coord_cartesian(ylim = c(0, y_upper)) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    legend.position = "none"
  )

pdf(
  file.path(knockout_output_dir, "MAPK1_KO_scatter.pdf"),
  width = 6,
  height = 5
)
print(p_scatter)
dev.off()

# Bar plot
top_genes <- head(df[order(-abs(df$FC)), ], 20)

top_genes$gene <- factor(
  top_genes$gene,
  levels = top_genes$gene[order(top_genes$FC)]
)

p_bar <- ggplot(top_genes, aes(x = gene, y = FC, fill = FC)) +
  geom_bar(stat = "identity", alpha = 0.9) +
  coord_flip() +
  scale_fill_gradient2(
    low = "#3C5488",
    mid = "white",
    high = "#E64B35",
    midpoint = 0
  ) +
  labs(
    title = "Top 20 Differentially Regulated Genes\n(MAPK1 Knockout)",
    x = "Gene",
    y = "Fold change"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "italic"),
    legend.position = "none"
  )

pdf(
  file.path(knockout_output_dir, "MAPK1_KO_barplot.pdf"),
  width = 6,
  height = 5
)
print(p_bar)
dev.off()

cat("MAPK1 knockout analysis completed.\n")
cat("Results saved to:", knockout_output_dir, "\n")