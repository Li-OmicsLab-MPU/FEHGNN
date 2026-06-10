library(Seurat)
library(tidyverse)
library(ggplot2)

############################################
# MDD single-cell cell-type marker plotting
############################################

# Set local desktop project directory
project_dir <- "C:/Users/villa/Desktop/MDD_single_cell_project"

# Define input and output paths
input_seurat_file <- file.path(
  project_dir,
  "results",
  "annotation_QC",
  "MDD_annotation_scored_object.rds"
)

output_dir <- file.path(
  project_dir,
  "results",
  "celltype_marker_plots"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

############################################
# Load Seurat object
############################################

print("Loading MDD Seurat object")

harmonized_object <- readRDS(file = input_seurat_file)

DefaultAssay(harmonized_object) <- "RNA"

############################################
# Set cluster identity
############################################

if ("Cluster" %in% colnames(harmonized_object@meta.data)) {
  cluster_col <- "Cluster"
} else if ("celltype" %in% colnames(harmonized_object@meta.data)) {
  cluster_col <- "celltype"
} else if ("annotation" %in% colnames(harmonized_object@meta.data)) {
  cluster_col <- "annotation"
} else if ("RNA_snn_res.0.7" %in% colnames(harmonized_object@meta.data)) {
  cluster_col <- "RNA_snn_res.0.7"
} else if ("seurat_clusters" %in% colnames(harmonized_object@meta.data)) {
  cluster_col <- "seurat_clusters"
} else {
  stop("No suitable cluster annotation column found.")
}

Idents(harmonized_object) <- harmonized_object@meta.data[[cluster_col]]

############################################
# Set UMAP reduction
############################################

if ("UMAPHarmonySeededBatchSampleChemistry" %in% names(harmonized_object@reductions)) {
  reduction_use <- "UMAPHarmonySeededBatchSampleChemistry"
} else if ("harmony_umap" %in% names(harmonized_object@reductions)) {
  reduction_use <- "harmony_umap"
} else if ("umap" %in% names(harmonized_object@reductions)) {
  reduction_use <- "umap"
} else {
  stop("No UMAP reduction found in the Seurat object.")
}

############################################
# Define marker genes
############################################

main_markers <- c(
  "SNAP25", "RBFOX3", "GRIN1", "SLC17A7",
  "GAD1", "GAD2",
  "ALDH1L1", "AQP4", "GFAP",
  "MBP", "PLP1",
  "PDGFRA", "SOX10",
  "CX3CR1", "CLDN5"
)

excitatory_layer_markers <- c(
  "SLC17A7", "SATB2", "FXYD6", "GSG1L", "RASGRF2",
  "CUX2", "RELN", "TLE1", "RORB", "PDE1A", "PCP4",
  "BCL11B", "HTR2C", "RXFP1", "SYNPR", "NR4A2",
  "TOX", "FOXP2", "TLE4", "ETV1", "NTNG2"
)

inhibitory_subtype_markers <- c(
  "GAD1", "GAD2", "SST", "PVALB", "VIP", "LAMP5",
  "LHX6", "ADARB2", "CCK", "NPY", "CALB1", "CALB2"
)

glial_markers <- c(
  "GFAP", "GJA1", "ALDH1A1", "AQP4",
  "MAG", "MOG", "OLIG1", "OLIG2", "SOX10", "MYT1",
  "PDGFRA", "ZFPM2", "ITPR2", "TCF7L2",
  "MRC1", "CX3CR1", "SPI1", "VIM", "CLDN5"
)

############################################
# Keep only markers present in the object
############################################

filter_existing_genes <- function(genes, object) {
  genes[genes %in% rownames(object)]
}

main_markers <- filter_existing_genes(main_markers, harmonized_object)
excitatory_layer_markers <- filter_existing_genes(excitatory_layer_markers, harmonized_object)
inhibitory_subtype_markers <- filter_existing_genes(inhibitory_subtype_markers, harmonized_object)
glial_markers <- filter_existing_genes(glial_markers, harmonized_object)

all_markers <- Reduce(
  union,
  list(
    main_markers,
    excitatory_layer_markers,
    inhibitory_subtype_markers,
    glial_markers
  )
)

qc_features <- intersect(
  c("nCount_RNA", "nFeature_RNA"),
  colnames(harmonized_object@meta.data)
)

all_features <- c(all_markers, qc_features)

############################################
# Define broad cell-type groups
############################################

cluster_levels <- levels(Idents(harmonized_object))

inhibitory_types <- grep("^In|^InN|Inhibitory|GABA", cluster_levels, value = TRUE)
excitatory_types <- grep("^Ex|^ExN|Excitatory|Glut", cluster_levels, value = TRUE)

glial_types <- Reduce(
  union,
  list(
    grep("^Ast|Astro", cluster_levels, value = TRUE),
    grep("^Oli|^Oligo|^OPC|^O", cluster_levels, value = TRUE),
    grep("^Mic|Micro", cluster_levels, value = TRUE),
    grep("^End|Endo", cluster_levels, value = TRUE),
    grep("^Per|Peri", cluster_levels, value = TRUE),
    grep("^VLMC", cluster_levels, value = TRUE)
  )
)

# If annotated names are unavailable, use all clusters as fallback
if (length(excitatory_types) == 0) excitatory_types <- cluster_levels
if (length(inhibitory_types) == 0) inhibitory_types <- cluster_levels
if (length(glial_types) == 0) glial_types <- cluster_levels

############################################
# FeaturePlot for marker genes
############################################

print("Generating FeaturePlot marker panels")

feature_plots_list <- list()

i <- 1

while (i <= length(all_features)) {
  
  feature_subset <- all_features[i:min(i + 3, length(all_features))]
  
  feature_plots_list[[paste0("Feature_", i)]] <- FeaturePlot(
    object = harmonized_object,
    features = feature_subset,
    ncol = 2,
    reduction = reduction_use
  )
  
  i <- i + 4
}

pdf(
  file = file.path(output_dir, "MDD_FeaturePlotMarkers.pdf"),
  height = 10,
  width = 12,
  onefile = TRUE
)

print(
  DimPlot(
    object = harmonized_object,
    reduction = reduction_use,
    label = TRUE,
    repel = TRUE
  ) +
    ggtitle("MDD cell clusters")
)

for (plot_name in names(feature_plots_list)) {
  print(feature_plots_list[[plot_name]])
}

dev.off()

############################################
# Violin plots for marker genes
############################################

print("Generating violin marker plots")

violin_plots_list <- list()

i <- 1
while (i <= length(excitatory_layer_markers)) {
  
  feature_subset <- excitatory_layer_markers[i:min(i + 3, length(excitatory_layer_markers))]
  
  violin_plots_list[[paste0("Ex_", i)]] <- VlnPlot(
    object = harmonized_object,
    features = feature_subset,
    ncol = 2,
    pt.size = 0,
    idents = excitatory_types
  )
  
  i <- i + 4
}

i <- 1
while (i <= length(inhibitory_subtype_markers)) {
  
  feature_subset <- inhibitory_subtype_markers[i:min(i + 3, length(inhibitory_subtype_markers))]
  
  violin_plots_list[[paste0("In_", i)]] <- VlnPlot(
    object = harmonized_object,
    features = feature_subset,
    ncol = 2,
    pt.size = 0,
    idents = inhibitory_types
  )
  
  i <- i + 4
}

i <- 1
while (i <= length(glial_markers)) {
  
  feature_subset <- glial_markers[i:min(i + 3, length(glial_markers))]
  
  violin_plots_list[[paste0("Glial_", i)]] <- VlnPlot(
    object = harmonized_object,
    features = feature_subset,
    ncol = 2,
    pt.size = 0,
    idents = glial_types
  )
  
  i <- i + 4
}

pdf(
  file = file.path(output_dir, "MDD_VlnMarkers.pdf"),
  height = 10,
  width = 12,
  onefile = TRUE
)

if (length(qc_features) > 0) {
  print(
    VlnPlot(
      object = harmonized_object,
      features = qc_features,
      pt.size = 0,
      ncol = 1
    )
  )
}

for (plot_name in names(violin_plots_list)) {
  print(violin_plots_list[[plot_name]])
}

dev.off()

############################################
# DotPlot for marker genes
############################################

print("Generating DotPlot marker panels")

pdf(
  file = file.path(output_dir, "MDD_DotPlotsMarkers.pdf"),
  height = 10,
  width = 12,
  onefile = TRUE
)

if (length(excitatory_layer_markers) > 0) {
  print(
    DotPlot(
      object = harmonized_object,
      idents = excitatory_types,
      features = excitatory_layer_markers
    ) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      ggtitle("Excitatory neuron markers")
  )
}

if (length(inhibitory_subtype_markers) > 0) {
  print(
    DotPlot(
      object = harmonized_object,
      idents = inhibitory_types,
      features = inhibitory_subtype_markers
    ) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      ggtitle("Inhibitory neuron markers")
  )
}

if (length(glial_markers) > 0) {
  print(
    DotPlot(
      object = harmonized_object,
      idents = glial_types,
      features = glial_markers
    ) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      ggtitle("Glial and vascular markers")
  )
}

if (length(main_markers) > 0) {
  print(
    DotPlot(
      object = harmonized_object,
      features = main_markers
    ) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      ggtitle("Main brain cell-type markers")
  )
}

dev.off()

############################################
# Summary of UMIs, genes, and cell numbers
############################################

print("Summarizing UMI, gene, and cell counts per cluster")

metadata_df <- harmonized_object@meta.data

metadata_df$Cluster_for_summary <- harmonized_object@meta.data[[cluster_col]]

count_summary_all <- metadata_df %>%
  group_by(Cluster_for_summary) %>%
  summarise(
    mean_UMI = mean(nCount_RNA, na.rm = TRUE),
    median_UMI = median(nCount_RNA, na.rm = TRUE),
    mean_genes = mean(nFeature_RNA, na.rm = TRUE),
    median_genes = median(nFeature_RNA, na.rm = TRUE),
    cells = n(),
    .groups = "drop"
  )

write.csv(
  count_summary_all,
  file = file.path(output_dir, "MDD_count_summary_all.csv"),
  row.names = FALSE
)

if ("Chemistry" %in% colnames(metadata_df)) {
  
  count_summary_chemistry <- metadata_df %>%
    group_by(Cluster_for_summary, Chemistry) %>%
    summarise(
      mean_UMI = mean(nCount_RNA, na.rm = TRUE),
      median_UMI = median(nCount_RNA, na.rm = TRUE),
      mean_genes = mean(nFeature_RNA, na.rm = TRUE),
      median_genes = median(nFeature_RNA, na.rm = TRUE),
      cells = n(),
      .groups = "drop"
    )
  
  write.csv(
    count_summary_chemistry,
    file = file.path(output_dir, "MDD_count_summary_chemistry.csv"),
    row.names = FALSE
  )
}

if ("Sex" %in% colnames(metadata_df)) {
  
  count_summary_sex <- metadata_df %>%
    group_by(Cluster_for_summary, Sex) %>%
    summarise(
      mean_UMI = mean(nCount_RNA, na.rm = TRUE),
      median_UMI = median(nCount_RNA, na.rm = TRUE),
      mean_genes = mean(nFeature_RNA, na.rm = TRUE),
      median_genes = median(nFeature_RNA, na.rm = TRUE),
      cells = n(),
      .groups = "drop"
    )
  
  write.csv(
    count_summary_sex,
    file = file.path(output_dir, "MDD_count_summary_sex.csv"),
    row.names = FALSE
  )
}

if ("Condition" %in% colnames(metadata_df)) {
  
  count_summary_condition <- metadata_df %>%
    group_by(Cluster_for_summary, Condition) %>%
    summarise(
      mean_UMI = mean(nCount_RNA, na.rm = TRUE),
      median_UMI = median(nCount_RNA, na.rm = TRUE),
      mean_genes = mean(nFeature_RNA, na.rm = TRUE),
      median_genes = median(nFeature_RNA, na.rm = TRUE),
      cells = n(),
      .groups = "drop"
    )
  
  write.csv(
    count_summary_condition,
    file = file.path(output_dir, "MDD_count_summary_condition.csv"),
    row.names = FALSE
  )
}

############################################
# Save object
############################################

saveRDS(
  harmonized_object,
  file = file.path(output_dir, "MDD_marker_checked_object.rds")
)

print("MDD marker plotting analysis completed.")
print(summary(warnings()))
sessionInfo()