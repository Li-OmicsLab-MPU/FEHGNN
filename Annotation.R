library(Seurat)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyverse)

############################################
# MDD single-cell annotation and marker QC
############################################

# Set local desktop project directory
project_dir <- "C:/Users/villa/Desktop/MDD_single_cell_project"

# Define input and output paths
input_seurat_file <- file.path(
  project_dir,
  "results",
  "cluster_QC",
  "MDD_enhanced_cluster_QC_seurat_object.rds"
)

output_dir <- file.path(
  project_dir,
  "results",
  "annotation_QC"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

############################################
# Load MDD Seurat object
############################################

print("Loading MDD enhanced Seurat object")

harmonized_object <- readRDS(file = input_seurat_file)

# Set default assay
DefaultAssay(harmonized_object) <- "RNA"

# Set cluster identity
# If your object uses seurat_clusters instead, replace RNA_snn_res.0.7 with seurat_clusters
cluster_col <- "RNA_snn_res.0.7"

if (!cluster_col %in% colnames(harmonized_object@meta.data)) {
  if ("seurat_clusters" %in% colnames(harmonized_object@meta.data)) {
    cluster_col <- "seurat_clusters"
  } else {
    stop("No cluster column found. Please check RNA_snn_res.0.7 or seurat_clusters.")
  }
}

Idents(harmonized_object) <- harmonized_object@meta.data[[cluster_col]]

############################################
# Add BRETIGEA module scores
############################################

print("Adding BRETIGEA module scores")

if (requireNamespace("BRETIGEA", quietly = TRUE)) {
  
  library(BRETIGEA)
  
  markers_for_annot <- split(
    markers_df_human_brain$markers,
    markers_df_human_brain$cell
  )
  
  # Assess overlapping genes
  overlap_summary_bretigea <- lapply(markers_for_annot, function(x) {
    c(
      marker_number = length(x),
      overlap_number = length(intersect(x, rownames(harmonized_object)))
    )
  })
  
  write.csv(
    do.call(rbind, overlap_summary_bretigea),
    file = file.path(output_dir, "MDD_BRETIGEA_marker_overlap_summary.csv")
  )
  
  harmonized_object <- AddModuleScore(
    object = harmonized_object,
    features = markers_for_annot,
    nbin = 10,
    ctrl = 500,
    search = TRUE
  )
  
  for (i in seq_along(markers_for_annot)) {
    item_name <- paste0("Cluster", i)
    new_name <- names(markers_for_annot)[i]
    
    harmonized_object@meta.data[[paste0(new_name, "_BRETIGEA")]] <- 
      harmonized_object@meta.data[[item_name]]
    
    harmonized_object@meta.data[[item_name]] <- NULL
  }
  
} else {
  print("BRETIGEA package is not installed. Skipping BRETIGEA module scores.")
  markers_for_annot <- NULL
}

############################################
# Add UCell scores for BRETIGEA markers
############################################

print("Adding UCell scores for BRETIGEA markers")

if (!is.null(markers_for_annot) && requireNamespace("UCell", quietly = TRUE)) {
  
  library(UCell)
  
  harmonized_object <- AddModuleScore_UCell(
    obj = harmonized_object,
    features = markers_for_annot,
    chunk.size = 500
  )
  
  pdf(
    file = file.path(output_dir, "MDD_BRETIGEA_scores_Seurat_UCell.pdf"),
    height = 16,
    width = 12
  )
  
  print(
    VlnPlot(
      harmonized_object,
      features = paste0(names(markers_for_annot), "_BRETIGEA"),
      pt.size = 0,
      ncol = 2
    )
  )
  
  print(
    VlnPlot(
      harmonized_object,
      features = paste0(names(markers_for_annot), "_UCell"),
      pt.size = 0,
      ncol = 2
    )
  )
  
  dev.off()
  
} else {
  print("UCell package is not installed or BRETIGEA markers are unavailable. Skipping UCell scoring.")
}

############################################
# Add BrainInABlender module scores
############################################

print("Adding BrainInABlender module scores")

if (requireNamespace("BrainInABlender", quietly = TRUE)) {
  
  library(BrainInABlender)
  
  BIAB_markers <- split(
    CellTypeSpecificGenes_Master3$GeneSymbol_Human,
    CellTypeSpecificGenes_Master3$CellType_Primary
  )
  
  BIAB_markers <- lapply(BIAB_markers, na.omit)
  BIAB_markers <- lapply(BIAB_markers, as.character)
  
  # Assess overlapping genes
  overlap_summary_biab <- lapply(BIAB_markers, function(x) {
    c(
      marker_number = length(x),
      overlap_number = length(intersect(x, rownames(harmonized_object)))
    )
  })
  
  write.csv(
    do.call(rbind, overlap_summary_biab),
    file = file.path(output_dir, "MDD_BrainInABlender_marker_overlap_summary.csv")
  )
  
  harmonized_object <- AddModuleScore(
    object = harmonized_object,
    features = BIAB_markers,
    nbin = 10
  )
  
  for (i in seq_along(BIAB_markers)) {
    item_name <- paste0("Cluster", i)
    new_name <- names(BIAB_markers)[i]
    
    harmonized_object@meta.data[[paste0(new_name, "_BIAB")]] <- 
      harmonized_object@meta.data[[item_name]]
    
    harmonized_object@meta.data[[item_name]] <- NULL
  }
  
  if (requireNamespace("UCell", quietly = TRUE)) {
    
    library(UCell)
    
    harmonized_object <- AddModuleScore_UCell(
      obj = harmonized_object,
      features = BIAB_markers,
      chunk.size = 500
    )
    
    pdf(
      file = file.path(output_dir, "MDD_BrainInABlender_scores_Seurat_UCell.pdf"),
      height = 16,
      width = 14
    )
    
    print(
      VlnPlot(
        harmonized_object,
        features = paste0(names(BIAB_markers), "_BIAB"),
        pt.size = 0,
        ncol = 2
      )
    )
    
    print(
      VlnPlot(
        harmonized_object,
        features = paste0(names(BIAB_markers), "_UCell"),
        pt.size = 0,
        ncol = 2
      )
    )
    
    dev.off()
  }
  
} else {
  print("BrainInABlender package is not installed. Skipping BrainInABlender module scores.")
  BIAB_markers <- NULL
}

############################################
# Export module scores
############################################

print("Writing module score metadata")

score_cols <- c()

if (!is.null(markers_for_annot)) {
  score_cols <- c(
    score_cols,
    paste0(names(markers_for_annot), "_BRETIGEA"),
    paste0(names(markers_for_annot), "_UCell")
  )
}

if (!is.null(BIAB_markers)) {
  score_cols <- c(
    score_cols,
    paste0(names(BIAB_markers), "_BIAB"),
    paste0(names(BIAB_markers), "_UCell")
  )
}

score_cols <- intersect(score_cols, colnames(harmonized_object@meta.data))

if (length(score_cols) > 0) {
  write_csv(
    harmonized_object@meta.data[, score_cols, drop = FALSE],
    file = file.path(output_dir, "MDD_BRETIGEA_BrainInABlender_scores.csv")
  )
}

############################################
# Find markers using Presto
############################################

print("Finding cluster markers using Presto")

if (requireNamespace("presto", quietly = TRUE)) {
  
  all_markers_presto <- presto::wilcoxauc(
    harmonized_object,
    group_by = cluster_col
  )
  
  all_markers_presto <- all_markers_presto %>%
    as_tibble() %>%
    filter(
      padj < 0.05,
      logFC > log(1.5),
      pct_in - pct_out > 10
    )
  
  write_csv(
    all_markers_presto,
    file = file.path(output_dir, "MDD_all_markers_Presto_WilcoxAUC.csv")
  )
  
} else {
  print("presto package is not installed. Skipping Presto marker analysis.")
  all_markers_presto <- NULL
}

############################################
# GO enrichment analysis
############################################

print("Running GO enrichment analysis")

if (
  !is.null(all_markers_presto) &&
  nrow(all_markers_presto) > 0 &&
  requireNamespace("clusterProfiler", quietly = TRUE) &&
  requireNamespace("org.Hs.eg.db", quietly = TRUE) &&
  requireNamespace("enrichplot", quietly = TRUE)
) {
  
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(enrichplot)
  
  all_markers_presto_split <- split(
    all_markers_presto,
    all_markers_presto$group
  )
  
  egos <- list()
  onts <- c("CC", "BP", "MF")
  
  for (ont in onts) {
    
    egos[[ont]] <- lapply(all_markers_presto_split, function(x) {
      
      print(length(x$feature))
      
      enrichGO(
        gene = x$feature,
        universe = rownames(harmonized_object),
        OrgDb = org.Hs.eg.db,
        ont = ont,
        pAdjustMethod = "BH",
        pvalueCutoff = 0.01,
        qvalueCutoff = 0.05,
        keyType = "SYMBOL"
      )
    })
  }
  
  for (ont in onts) {
    
    print(paste0("Creating GO plots: ", ont))
    
    pdf(
      file = file.path(
        output_dir,
        paste0("MDD_GO_plot_Presto_markers_", ont, ".pdf")
      ),
      onefile = TRUE,
      height = 7,
      width = 10
    )
    
    for (cluster_name in names(egos[[ont]])) {
      
      this_clust <- egos[[ont]][[cluster_name]]
      
      if (!is.null(this_clust) && nrow(as.data.frame(this_clust)) > 0) {
        print(
          barplot(this_clust) +
            ggtitle(paste("Cluster", cluster_name))
        )
      } else {
        message(paste("No GO over-representation for cluster", cluster_name))
      }
    }
    
    dev.off()
  }
  
} else {
  print("GO enrichment skipped because marker results or required packages are unavailable.")
}

############################################
# DotPlot of canonical brain cell-type markers
############################################

print("Plotting canonical brain cell-type markers")

canonical_markers <- c(
  "SNAP25", "RBFOX3", "SLC17A7", "SATB2",
  "GAD1", "GAD2",
  "ALDH1L1", "GFAP", "GJA1",
  "MBP", "PLP1",
  "PDGFRA", "VIM",
  "CX3CR1", "P2RY12",
  "CLDN5", "FLT1"
)

canonical_markers <- intersect(canonical_markers, rownames(harmonized_object))

pdf(
  file = file.path(output_dir, "MDD_Unlabelled_DotPlot_main.pdf"),
  height = 12,
  width = 10
)

print(
  DotPlot(
    harmonized_object,
    features = canonical_markers
  ) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1)
    )
)

dev.off()

############################################
# Layer-specific marker assessment
############################################

print("Running cortical layer specificity assessment")

layer_marker_file <- file.path(
  project_dir,
  "reference",
  "Maynard_layer_specific_markers.csv"
)

if (
  file.exists(layer_marker_file) &&
  !is.null(all_markers_presto) &&
  nrow(all_markers_presto) > 0
) {
  
  Maynard_layer_spec <- read_csv(file = layer_marker_file)
  
  if (all(c("gene_name", "fdr") %in% colnames(Maynard_layer_spec))) {
    
    Maynard_layer_spec <- Maynard_layer_spec %>%
      filter(fdr < 0.01)
    
    overlap_markers <- all_markers_presto %>%
      filter(feature %in% Maynard_layer_spec$gene_name) %>%
      group_by(feature)
    
    layer_cols <- intersect(
      c("Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"),
      colnames(Maynard_layer_spec)
    )
    
    Layer_info <- Maynard_layer_spec %>%
      dplyr::select(gene_name, all_of(layer_cols)) %>%
      filter(gene_name %in% overlap_markers$feature) %>%
      group_by(gene_name) %>%
      summarise_all(sum)
    
    full_join(
      overlap_markers,
      Layer_info,
      by = c("feature" = "gene_name")
    ) %>%
      ungroup() %>%
      group_by(group) %>%
      write_csv(
        file = file.path(output_dir, "MDD_cluster_Maynard_layer_info.csv")
      )
  }
  
} else {
  print("Layer marker file not found or marker results unavailable. Skipping layer specificity assessment.")
}

############################################
# Save annotated object
############################################

saveRDS(
  harmonized_object,
  file = file.path(output_dir, "MDD_annotation_scored_object.rds")
)

print("MDD annotation and marker QC analysis completed.")
print(summary(warnings()))
sessionInfo()