library(Seurat)
library(readr)
library(dplyr)
library(ggplot2)
library(pheatmap)
library(cowplot)
library(ggbiplot)

############################
# MDD single-cell cluster QC
############################

# Set project directory
project_dir <- "D:/MDD_single_cell_project"

# Define input and output paths
input_seurat_file <- file.path(project_dir, "data", "MDD_integrated_seurat_object.rds")
doublet_file <- file.path(project_dir, "metadata", "MDD_doublet_predictions.csv")
output_dir <- file.path(project_dir, "results", "cluster_QC")

# Create output folders
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "all"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "male"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "female"), recursive = TRUE, showWarnings = FALSE)

############################
# Load Seurat object
############################

print("Loading MDD Seurat object")

mdd_object <- readRDS(input_seurat_file)

# Set default assay
DefaultAssay(mdd_object) <- "RNA"

############################
# Optional: add doublet prediction metadata
############################

if (file.exists(doublet_file)) {
  print("Loading doublet prediction metadata")
  
  doublet_predictions <- read.csv(
    file = doublet_file,
    row.names = 1,
    check.names = FALSE
  )
  
  common_cells <- intersect(colnames(mdd_object), rownames(doublet_predictions))
  mdd_object <- subset(mdd_object, cells = common_cells)
  doublet_predictions <- doublet_predictions[common_cells, , drop = FALSE]
  
  mdd_object <- AddMetaData(
    object = mdd_object,
    metadata = doublet_predictions
  )
  
  rm(doublet_predictions)
} else {
  print("No doublet prediction file found. Skipping doublet metadata addition.")
}

############################
# Check required metadata
############################

required_metadata <- c(
  "RNA_snn_res.0.7",
  "Sample",
  "Condition",
  "Sex"
)

missing_metadata <- setdiff(required_metadata, colnames(mdd_object@meta.data))

if (length(missing_metadata) > 0) {
  stop(
    paste0(
      "Missing required metadata columns: ",
      paste(missing_metadata, collapse = ", ")
    )
  )
}

# Set cluster identity
Idents(mdd_object) <- mdd_object$RNA_snn_res.0.7

############################
# Plot UMAPs
############################

print("Plotting UMAPs")

plots_list <- list()

variables <- c("RNA_snn_res.0.7", "Condition", "Sex")

if ("Batch" %in% colnames(mdd_object@meta.data)) {
  variables <- c(variables, "Batch")
}

if ("Chemistry" %in% colnames(mdd_object@meta.data)) {
  variables <- c(variables, "Chemistry")
}

if ("DF.classifications" %in% colnames(mdd_object@meta.data)) {
  variables <- c(variables, "DF.classifications")
}

for (variable in variables) {
  Idents(mdd_object) <- mdd_object@meta.data[, variable]
  
  plots_list[[variable]] <- DimPlot(
    object = mdd_object,
    reduction = "umap",
    group.by = variable,
    label = TRUE,
    repel = TRUE
  ) +
    ggtitle(variable) +
    theme_classic()
}

pdf(
  file = file.path(output_dir, "MDD_cluster_QC_UMAP_plots.pdf"),
  onefile = TRUE,
  height = 8,
  width = 10
)

print(plots_list)

dev.off()

############################
# Plot QC violin plots
############################

print("Plotting QC violin plots")

plots_list <- list()

features_to_plot <- c("nCount_RNA", "nFeature_RNA")

if ("percent.mt" %in% colnames(mdd_object@meta.data)) {
  features_to_plot <- c(features_to_plot, "percent.mt")
}

if ("percent.ribo" %in% colnames(mdd_object@meta.data)) {
  features_to_plot <- c(features_to_plot, "percent.ribo")
}

if ("pANN" %in% colnames(mdd_object@meta.data)) {
  features_to_plot <- c(features_to_plot, "pANN")
}

idents_to_use <- list(
  by_cluster = c("RNA_snn_res.0.7"),
  by_cluster_condition = c("RNA_snn_res.0.7", "Condition"),
  by_cluster_sex = c("RNA_snn_res.0.7", "Sex")
)

log_scale <- c(TRUE, FALSE)

for (ident_name in names(idents_to_use)) {
  idents <- idents_to_use[[ident_name]]
  
  if (length(idents) == 1) {
    Idents(mdd_object) <- mdd_object@meta.data[, idents]
  } else {
    Idents(mdd_object) <- paste(
      mdd_object@meta.data[, idents[1]],
      mdd_object@meta.data[, idents[2]],
      sep = "_"
    )
  }
  
  for (scale_status in log_scale) {
    plot_name <- paste0(ident_name, "_log_", scale_status)
    
    plots_list[[plot_name]] <- VlnPlot(
      object = mdd_object,
      features = features_to_plot,
      log = scale_status,
      ncol = 1,
      pt.size = 0
    ) +
      theme(
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
      )
  }
}

pdf(
  file = file.path(output_dir, "MDD_per_cluster_QC_violin_plots.pdf"),
  onefile = TRUE,
  height = 13,
  width = 13
)

print(plots_list)

dev.off()

############################
# Cluster composition heatmaps
############################

print("Plotting cluster composition heatmaps")

Idents(mdd_object) <- mdd_object$RNA_snn_res.0.7

variables <- c("Sample", "Condition", "Sex")

if ("Batch" %in% colnames(mdd_object@meta.data)) {
  variables <- c(variables, "Batch")
}

if ("Chemistry" %in% colnames(mdd_object@meta.data)) {
  variables <- c(variables, "Chemistry")
}

if ("DF.classifications" %in% colnames(mdd_object@meta.data)) {
  variables <- c(variables, "DF.classifications")
}

scale_dirs <- c("row", "column")

subset_to_use <- list(
  male = c("Male"),
  female = c("Female"),
  all = c("Male", "Female")
)

for (subset_name in names(subset_to_use)) {
  
  print(paste0("Processing subset: ", subset_name))
  
  subset_sex <- subset_to_use[[subset_name]]
  
  if (!all(subset_sex %in% unique(mdd_object$Sex))) {
    print(paste0("Skipping subset: ", subset_name, " because sex labels are not available."))
    next
  }
  
  plotting_subset <- subset(
    x = mdd_object,
    subset = Sex %in% subset_sex
  )
  
  Idents(plotting_subset) <- plotting_subset$RNA_snn_res.0.7
  
  subset_output_dir <- file.path(output_dir, subset_name)
  dir.create(subset_output_dir, recursive = TRUE, showWarnings = FALSE)
  
  for (variable in variables) {
    
    print(paste0("Counting variable: ", variable))
    
    frequencies <- table(
      plotting_subset@meta.data[, variable],
      Idents(plotting_subset)
    )
    
    write.csv(
      frequencies,
      file = file.path(
        subset_output_dir,
        paste0("MDD_", variable, "_", subset_name, "_cluster_frequency.csv")
      )
    )
    
    perc_rep <- colSums(frequencies > 0) / nrow(frequencies)
    
    write.csv(
      perc_rep,
      file = file.path(
        subset_output_dir,
        paste0("MDD_", variable, "_", subset_name, "_cluster_percent_representation.csv")
      )
    )
    
    if (nrow(frequencies) > 1 && ncol(frequencies) > 1) {
      pdf(
        file = file.path(
          subset_output_dir,
          paste0("MDD_", variable, "_cluster_categorical_counts_", subset_name, ".pdf")
        ),
        onefile = TRUE,
        height = 10,
        width = 12
      )
      
      for (scale_dir in scale_dirs) {
        pheatmap(
          mat = frequencies[, order(as.numeric(colnames(frequencies))), drop = FALSE],
          cluster_cols = FALSE,
          cluster_rows = FALSE,
          scale = scale_dir,
          main = paste(variable, subset_name, scale_dir, sep = " - ")
        )
      }
      
      dev.off()
    }
  }
}

############################
# Subject-level cell-type proportion PCA
############################

print("Running subject-level cell-type proportion PCA")

plots_list <- list()

for (subset_name in names(subset_to_use)) {
  
  subset_sex <- subset_to_use[[subset_name]]
  
  if (!all(subset_sex %in% unique(mdd_object$Sex))) {
    next
  }
  
  plotting_subset <- subset(
    x = mdd_object,
    subset = Sex %in% subset_sex
  )
  
  Idents(plotting_subset) <- plotting_subset$RNA_snn_res.0.7
  
  subject_cluster_table <- table(
    plotting_subset$Sample,
    Idents(plotting_subset)
  )
  
  subject_cluster_proportions <- subject_cluster_table / rowSums(subject_cluster_table)
  
  sample_metadata <- plotting_subset@meta.data %>%
    distinct(Sample, .keep_all = TRUE) %>%
    as.data.frame()
  
  rownames(sample_metadata) <- sample_metadata$Sample
  sample_metadata <- sample_metadata[rownames(subject_cluster_proportions), , drop = FALSE]
  
  res_pca <- prcomp(
    subject_cluster_proportions,
    scale. = TRUE
  )
  
  colour_vars <- c("Condition")
  
  if ("Batch" %in% colnames(sample_metadata)) {
    colour_vars <- c(colour_vars, "Batch")
  }
  
  if ("Chemistry" %in% colnames(sample_metadata)) {
    colour_vars <- c(colour_vars, "Chemistry")
  }
  
  if (subset_name == "all") {
    colour_vars <- c(colour_vars, "Sex")
  }
  
  for (colour_var in colour_vars) {
    plots_list[[paste0(colour_var, "_", subset_name)]] <- ggbiplot(
      pcobj = res_pca,
      groups = sample_metadata[, colour_var],
      ellipse = TRUE,
      var.axes = FALSE
    ) +
      ggtitle(paste0("Subject cluster proportion PCA: ", colour_var, " - ", subset_name)) +
      theme_classic()
  }
}

pdf(
  file = file.path(output_dir, "MDD_subject_cluster_proportion_PCA.pdf"),
  onefile = TRUE,
  height = 8,
  width = 10
)

print(plots_list)

dev.off()

############################
# Cluster-level subject proportion PCA
############################

print("Running cluster-level subject proportion PCA")

plots_list <- list()

for (subset_name in names(subset_to_use)) {
  
  subset_sex <- subset_to_use[[subset_name]]
  
  if (!all(subset_sex %in% unique(mdd_object$Sex))) {
    next
  }
  
  plotting_subset <- subset(
    x = mdd_object,
    subset = Sex %in% subset_sex
  )
  
  cluster_subject_table <- table(
    plotting_subset$RNA_snn_res.0.7,
    plotting_subset$Sample
  )
  
  cluster_subject_proportions <- cluster_subject_table / rowSums(cluster_subject_table)
  
  cluster_subject_proportions <- na.omit(cluster_subject_proportions)
  
  if (nrow(cluster_subject_proportions) > 1 && ncol(cluster_subject_proportions) > 1) {
    res_pca <- prcomp(
      cluster_subject_proportions,
      scale. = TRUE
    )
    
    data_for_pca <- cbind(
      as.data.frame(res_pca$x),
      cluster = rownames(res_pca$x)
    )
    
    plots_list[[subset_name]] <- ggplot(
      data = data_for_pca,
      aes(x = PC1, y = PC2, label = cluster)
    ) +
      geom_text(size = 4) +
      theme_classic() +
      ggtitle(paste0("Cluster subject proportion PCA - ", subset_name))
  }
}

pdf(
  file = file.path(output_dir, "MDD_cluster_subject_proportion_PCA.pdf"),
  onefile = TRUE,
  height = 8,
  width = 10
)

print(plots_list)

dev.off()

############################
# Save enhanced object
############################

saveRDS(
  object = mdd_object,
  file = file.path(output_dir, "MDD_enhanced_cluster_QC_seurat_object.rds")
)

print("MDD single-cell cluster QC analysis completed.")
print(summary(warnings()))
sessionInfo()