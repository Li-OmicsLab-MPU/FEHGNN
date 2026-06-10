# Load required packages
library(tidyverse)      # Data manipulation and visualization
library(ggiraphExtra)  # Radar plot function: ggRadar()
library(readxl)        # Read Excel files
library(ggh4x)         # Extended ggplot2 facets
library(scales)        # Scale and color utilities
library(cowplot)       # Combine multiple ggplot objects

# Set working directory
setwd("C:\\Users\\villa")

# Read input data
data <- read.csv("data_3.csv")


# Set colors after removing "#559d3c"
col <- c("#3977ae", "#ed8431", "#d1362b", "#624094")

# Make sure Models follows the expected order
data$Models <- factor(data$Models, levels = c("BBBP", "Tox21", "SIDER", "CTD"))

# Calculate the global maximum value for radar plot scaling
# Only numeric columns are used
global_max <- data %>%
  select(where(is.numeric)) %>%
  max(na.rm = TRUE)

# Draw radar plot using Models as the grouping variable
p <- ggRadar(
  data = data,
  aes(color = Models, fill = Models),
  rescale = FALSE,
  legend.position = "none",
  size = 0.4,
  alpha = 0.2
) +
  facet_wrap2(
    ~Models,
    nrow = 1,
    strip = strip_nested(
      background_x = elem_list_rect(fill = alpha(col, alpha = 0.5))
    )
  ) +
  scale_fill_manual(values = col) +
  scale_color_manual(values = col) +
  scale_y_continuous(
    limits = c(0, global_max),
    breaks = 1
  ) +
  theme_bw(base_size = 5) +
  theme(
    panel.background = element_blank(),
    panel.border = element_rect(linewidth = 0.2, color = "black"),
    panel.grid.major = element_line(linewidth = 0.5),
    strip.background = element_rect(linewidth = 0.4, color = "black"),
    strip.text = element_text(size = 8),
    axis.text = element_text(size = 4, color = "grey30"),
    axis.text.y = element_blank(),
    axis.line = element_blank(),
    legend.position = "none",
    plot.margin = margin(0, 0, 0, 0, "cm")
  )

# Add custom annotation labels
# BACE has been removed together with color "#559d3c"
anno <- ggdraw() +
  draw_label("BBBP", y = 0.65, hjust = 0.83, size = 8, color = "#3977ae") +
  draw_label("Tox21", y = 0.55, hjust = 0.81, size = 8, color = "#ed8431") +
  draw_label("SIDER", y = 0.45, hjust = 0.52, size = 8, color = "#d1362b") +
  draw_label("CTD", y = 0.35, hjust = 0.71, size = 8, color = "#624094")

# Combine radar plot and annotation
final_plot <- plot_grid(
  p,
  anno,
  ncol = 2,
  rel_widths = c(0.5, 0.15)
)

# Save figure
ggsave(
  filename = "radar_facet234.pdf",
  plot = final_plot,
  width = 8,
  height = 5,
  dpi = 300
)

ggsave("radar.pdf", width = 8, height = 5, dpi = 300)