PROB_PLOTS_BACKGROUND_COLOR = '#F0F8FF'
NEGENT_PLOTS_BACKGROUND_COLOR = "#fce7cf"

aggregate_plot_style = {
    'figure.figsize': (12, 8), 
    'figure.dpi': 300,

    # Fonts
    'font.size': 24,
    'axes.labelsize': 35,
    'axes.titlesize': 30,
    'ytick.labelsize': 24,
    'legend.fontsize': 30,
    'figure.titlesize': 40,

    # X axis tick settings
    'xtick.labelsize': 24,
    'xtick.major.pad': 8,

    # Legend spacing settings 
    'axes.titlepad': 12,  # CHANGED from 20
    
    # Line and marker settings - MUCH MORE PROMINENT
    'lines.linewidth': 5,           # CHANGED from 3
    'lines.markersize': 14,         # CHANGED from 8
    'errorbar.capsize': 8,          # CHANGED from 4
    'lines.markeredgewidth': 2,     # CHANGED from 1
    
    # Layout - MINIMIZED WHITESPACE FOR MULTI-PLOT FIGURE
    'figure.subplot.left': 0.08,    # CHANGED from 0.06
    'figure.subplot.right': 0.98,   # Same
    'figure.subplot.bottom': 0.08,  # CHANGED from 0.12
    'figure.subplot.top': 0.96,     # CHANGED from 0.85

    'figure.subplot.wspace': 0.20,  # CHANGED from 0.25
    'figure.subplot.hspace': 0.35,  # CHANGED from 0.5

    # Axes styling
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    
    # Grid settings
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,

    # Y-axis settings
    'ytick.left': True,
    'ytick.right': False,
    'ytick.labelleft': True,
    'ytick.labelright': False,
    
    # X-axis settings
    'xtick.top': False,
    'xtick.bottom': True,
    'xtick.labeltop': False,
    'xtick.labelbottom': True,
    
    # Tick settings
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # Legend settings
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.loc': 'best',
    'legend.handletextpad': 0.3,
    'legend.columnspacing': 0.8,

    # Bolding 
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',
}

heatmap_config = {
    # Figure size - square aspect ratio works well for heatmaps
    'figure.figsize': [8,6],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Font settings
    'font.size': 16,
    'font.family': ['serif'],
    'font.serif': ['Times', 'Computer Modern Roman'],
    'text.usetex': False,
    
    # Axes settings
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'axes.linewidth': 0.8,
    'axes.grid': False,  # Usually no grid for heatmaps
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',
    
    # Tick settings - important for heatmap readability
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.width': 0,  # No tick marks for cleaner look
    'ytick.major.width': 0,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    
    # Colorbar settings
    'figure.subplot.right': 0.85,  # Make room for colorbar
}

top_k_style = {
    # Figure size - optimized for single column
    'figure.figsize': (16, 8),      # Taller rather than wider for subplots
    'figure.dpi': 300,              # High resolution for crisp text

    # Fonts
    'font.size': 24,
    'axes.labelsize': 35,
    'axes.titlesize': 30,
    'ytick.labelsize': 24,
    'legend.fontsize': 30,
    'figure.titlesize': 40,

    #X axis tick settings
    'xtick.labelsize': 24,          # Smaller than current 24
    #'figure.subplot.bottom': 0.20,  # More bottom margin
    'xtick.major.pad': 8,          # More space between ticks and labels

    #legened spacing settings 
    #'figure.subplot.top': 0.90,
    'axes.titlepad': 20,
    
    # Line and marker settings - make them pop
    'lines.linewidth': 3,           # Thick lines for visibility
    'lines.markersize': 8,          # Large markers
    'errorbar.capsize': 4,          # Visible error bar caps
    'lines.markeredgewidth': 1,     # Defined marker edges
    
    # Layout - INCREASED SPACING
        # Reduce margins to maximize plot area
    'figure.subplot.left': 0.10,    # Minimal left margin
    'figure.subplot.right': 0.90, # Minimal right margin
    'figure.subplot.bottom': 0.05,  # Space for x-labels
    'figure.subplot.top': 0.85,     # Space for legend/title

    'figure.subplot.wspace': 0.25,  # Less horizontal spacing needed
    'figure.subplot.hspace': 0.5,   # Keep vertical spacing


    # Axes styling
    'axes.linewidth': 1.2,          # Slightly thicker axes
    'axes.spines.top': False,       # Clean look
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    
    # Grid settings - help with readability
    'axes.grid': True,
    'grid.alpha': 0.3,              # Subtle but visible grid
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,


    # Remove y-axis labels and ticks for right plots
    'ytick.left': True,           # Keep left ticks
    'ytick.right': False,         # Remove right ticks
    'ytick.labelleft': True,      # Keep left labels
    'ytick.labelright': False,    # Remove right labels
    
    # Remove x-axis labels and ticks for top plots  
    'xtick.top': False,           # Remove top ticks
    'xtick.bottom': True,         # Keep bottom ticks
    'xtick.labeltop': False,      # Remove top labels
    'xtick.labelbottom': True,    # Keep bottom labels
    
    # Tick settings
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 6,          # Larger tick marks
    'ytick.major.size': 6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # Legend settings
    'legend.frameon': True,         # Add frame for contrast
    'legend.fancybox': False,       # Simple rectangle
    'legend.framealpha': 0.9,       # Semi-transparent background
    'legend.edgecolor': 'black',    # Clear border
    'legend.facecolor': 'white',    # White background
    'legend.loc': 'best',
    'legend.handletextpad': 0.3,    # Compact legend
    'legend.columnspacing': 0.8,

    #bolding 
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',

}

emergence_score_dist_style_legacy = {
    # Figure size - optimized for single column
    'figure.figsize': (26, 8.5),      # Taller rather than wider for 4 subplots
    'figure.dpi': 300,              # High resolution for crisp text

    # Fonts
    'font.size': 24,
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'ytick.labelsize': 24,
    'legend.fontsize': 30,
    'figure.titlesize': 40,

    #X axis tick settings
    'xtick.labelsize': 24,          # Smaller than current 24
    #'figure.subplot.bottom': 0.20,  # More bottom margin
    'xtick.major.pad': 8,          # More space between ticks and labels

    #legened spacing settings 
    #'figure.subplot.top': 0.90,
    'axes.titlepad': 20,
    
    # Line and marker settings - make them pop
    'lines.linewidth': 3,           # Thick lines for visibility
    'lines.markersize': 8,          # Large markers
    'errorbar.capsize': 4,          # Visible error bar caps
    'lines.markeredgewidth': 1,     # Defined marker edges
    
    # Layout - INCREASED SPACING
        # Reduce margins to maximize plot area
    'figure.subplot.left': 0.06,    # Minimal left margin
    'figure.subplot.right': 0.98, # Minimal right margin
    'figure.subplot.bottom': 0.30,  # Space for x-labels
    'figure.subplot.top': 0.85,     # Space for legend/title

    'figure.subplot.wspace': 0.25,  # Less horizontal spacing needed
    'figure.subplot.hspace': 0.5,   # Keep vertical spacing


    # Axes styling
    'axes.linewidth': 1.2,          # Slightly thicker axes
    'axes.spines.top': False,       # Clean look
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    
    # Grid settings - help with readability
    'axes.grid': True,
    'grid.alpha': 0.3,              # Subtle but visible grid
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,


    # Remove y-axis labels and ticks for right plots
    'ytick.left': True,           # Keep left ticks
    'ytick.right': False,         # Remove right ticks
    'ytick.labelleft': True,      # Keep left labels
    'ytick.labelright': False,    # Remove right labels
    
    # Remove x-axis labels and ticks for top plots  
    'xtick.top': False,           # Remove top ticks
    'xtick.bottom': True,         # Keep bottom ticks
    'xtick.labeltop': False,      # Remove top labels
    'xtick.labelbottom': True,    # Keep bottom labels
    
    # Tick settings
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 6,          # Larger tick marks
    'ytick.major.size': 6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # Legend settings
    'legend.frameon': True,         # Add frame for contrast
    'legend.fancybox': False,       # Simple rectangle
    'legend.framealpha': 0.9,       # Semi-transparent background
    'legend.edgecolor': 'black',    # Clear border
    'legend.facecolor': 'white',    # White background
    'legend.loc': 'best',
    'legend.handletextpad': 0.3,    # Compact legend
    'legend.columnspacing': 0.8,

    #bolding 
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',

}

emergence_score_dist_style = {
    # Figure size - adjusted for 2x2 grid
    'figure.figsize': (20, 15),     # More square aspect ratio for 2x2
    'figure.dpi': 300,              # High resolution for crisp text

    # Fonts
    'font.size': 20,                # Slightly smaller base font
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,          # Smaller for in-panel legends
    'figure.titlesize': 32,

    # X axis tick settings
    'xtick.major.pad': 6,           # Space between ticks and labels

    # Legend spacing settings 
    'axes.titlepad': 15,
    
    
    # Line and marker settings - make them pop
    'lines.linewidth': 3,           # Thick lines for visibility
    'lines.markersize': 8,          # Large markers
    'errorbar.capsize': 4,          # Visible error bar caps
    'lines.markeredgewidth': 1,     # Defined marker edges
    
    # Layout - adjusted for 2x2 grid
    'figure.subplot.left': 0.08,    # Left margin
    'figure.subplot.right': 0.96,   # Right margin
    'figure.subplot.bottom': 0.08,  # Bottom margin
    'figure.subplot.top': 0.88,     # Top margin for suptitle

    'figure.subplot.wspace': 0.20,  # Horizontal spacing between panels
    'figure.subplot.hspace': 0.25,  # Vertical spacing between panels

    # Axes styling
    'axes.linewidth': 1.2,          # Slightly thicker axes
    'axes.spines.top': False,       # Clean look
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    
    # Grid settings - help with readability
    'axes.grid': True,
    'grid.alpha': 0.3,              # Subtle but visible grid
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,

    # Y-axis settings
    'ytick.left': True,             # Keep left ticks
    'ytick.right': False,           # Remove right ticks
    'ytick.labelleft': True,        # Keep left labels
    'ytick.labelright': False,      # Remove right labels
    
    # X-axis settings
    'xtick.top': False,             # Remove top ticks
    'xtick.bottom': True,           # Keep bottom ticks
    'xtick.labeltop': False,        # Remove top labels
    'xtick.labelbottom': True,      # Keep bottom labels
    
    # Tick settings
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # Legend settings - optimized for in-panel legends
    'legend.frameon': True,         # Add frame for contrast
    'legend.fancybox': True,        # Rounded corners
    'legend.framealpha': 0.85,      # Semi-transparent background
    'legend.edgecolor': '#cccccc',  # Subtle border
    'legend.facecolor': 'white',    # White background
    'legend.loc': 'best',
    'legend.handletextpad': 0.4,    # Compact legend
    'legend.columnspacing': 0.8,
    'legend.handlelength': 1.5,     # Length of legend lines
    'legend.labelspacing': 0.8,

    # Bolding 
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',
}

scaling_curves_2x2_style = {
    # Figure - single column format
    'figure.figsize': (10, 8),
    
    # Fonts
    'font.size': 10,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 9,
    'figure.titlesize': 16,
    
    # Ticks
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    
    # Basic formatting
    'axes.linewidth': 0.6,
    'grid.alpha': 0.3,
    'legend.framealpha': 0.9,
    
    # Bolding 
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',

    # Lines and markers - MADE MORE PROMINENT
    'lines.linewidth': 3,           # ADDED - thicker lines (default is ~1.5)
    'lines.markersize': 12,         # INCREASED from 10
    'lines.markeredgewidth': 2,     # ADDED - thicker marker edges
    'errorbar.capsize': 6,          # ADDED - bigger error bar caps
    
    # Spacing for 2x2 grid
    'figure.subplot.hspace': 0.35,  # Vertical spacing between subplots
    'figure.subplot.wspace': 0.25,  # Horizontal spacing between subplots
    'figure.subplot.top': 0.92,     # Top margin
    'figure.subplot.bottom': 0.1,   # Bottom margin
    'figure.subplot.left': 0.1,     # Left margin
    'figure.subplot.right': 0.95,   # Right margin
}

task_comparison_style = {
    # Font sizes
    'font.size': 14,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 30,   # Changed from 20 to 30
    'figure.titleweight': 'bold', # Added to make the figure title bold
    'legend.fontsize': 13,
    
    # Tick parameters
    'xtick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.size': 6,
    'ytick.major.width': 1.2,
    'xtick.minor.size': 4,
    'xtick.minor.width': 1,
    'ytick.minor.size': 4,
    'ytick.minor.width': 1,
    
    # Line widths
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    
    # Figure and subplot spacing
    'figure.figsize': (12, 4),  # Adjust based on your 3-panel layout
    'figure.dpi': 300,  # High resolution for print
    'grid.alpha': 0.5,
    
    # Font family (optional - for LaTeX-style fonts)
    'font.family': 'serif',
    'text.usetex': False,  # Set to True if you have LaTeX installed
}

boxplot_style = {
    'figure.figsize': [10, 6],
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.titleweight": "bold",
    "axes.labelweight": "semibold",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "grid.color": "#aaaaaa",
    "figure.facecolor": "#faf9f6",
    "axes.facecolor": "#f5f7fc",
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
    "boxplot.flierprops.markerfacecolor": "#ea6a47",
    "boxplot.flierprops.markeredgecolor": "#b05236",
    "boxplot.boxprops.color": "#333333",
    "boxplot.boxprops.linewidth": 1.2,
    "boxplot.capprops.color": "#333333",
    "boxplot.capprops.linewidth": 1.2,
    "boxplot.whiskerprops.color": "#333333",
    "boxplot.whiskerprops.linewidth": 1.2,
    "boxplot.medianprops.color": "#222244",
    "boxplot.medianprops.linewidth": 2.0,
    "boxplot.patchartist": True,
}
