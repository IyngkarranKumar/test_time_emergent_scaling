
aggregate_plot_config = {
    # Figure size - optimized for single column (~3.5 inches wide)
    'figure.figsize': [3.5, 2.5],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    # Font settings - readable at small sizes
    'font.size': 8,
    'font.family': ['serif'],
    'font.serif': ['Times', 'Computer Modern Roman'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    
    # Axes settings
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'axes.grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # Tick settings
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    
    # Legend settings
    'legend.fontsize': 7,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.borderpad': 0.3,
    'legend.columnspacing': 1.0,
    'legend.handlelength': 1.5,
    
    # Line settings
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    
    # Grid settings
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
}

heatmap_config = {
    # Figure size - square aspect ratio works well for heatmaps
    'figure.figsize': [3.5, 3.0],
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


full_width_style = {
        # Figure
        'figure.figsize': (14, 3.2),
        
        # Fonts
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 30,
        'figure.titlesize': 25,
        
        # Ticks
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        
        # Basic formatting
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'legend.framealpha': 0.9,

        #bolding 
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.titleweight': 'bold',

        #padding 
        'figure.subplot.top': 1.5,
    }
    
half_width_style = {
        # Figure
        'figure.figsize': (6, 4),
        
        # Fonts
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11,
        
        # Ticks
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        
        # Basic formatting
        'axes.linewidth': 0.7,
        'grid.alpha': 0.3,
        'legend.framealpha': 0.9,
    }
    
column_width_style = {
        # Figure
        'figure.figsize': (3.5, 2.8),
        
        # Fonts
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        
        # Ticks
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        
        # Basic formatting
        'axes.linewidth': 0.6,
        'grid.alpha': 0.4,
        'legend.framealpha': 0.9,
    }

top_k_style = {
    # Figure size - optimized for single column
    'figure.figsize': (16, 8),      # Taller rather than wider for 4 subplots
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
    'figure.subplot.left': 0.06,    # Minimal left margin
    'figure.subplot.right': 0.98, # Minimal right margin
    'figure.subplot.bottom': 0.12,  # Space for x-labels
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


aggregate_plot_style = {
    'figure.figsize': (6, 4),  # Slightly larger for better readability
    'font.size': 24,                # Larger base font size
    'axes.titlesize': 24,           # Larger title
    'axes.labelsize': 24,           # Larger axis labels
    'xtick.labelsize': 24,          # Larger tick labels
    'ytick.labelsize': 24,
    'legend.fontsize': 24,          # Larger legend
    'lines.linewidth': 2,           # Slightly thicker lines for clarity
    'axes.linewidth': 1.0,          # Thinner axes for less visual weight
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.grid': True,              # Add subtle grid
    'grid.alpha': 0.3,              # Light grid lines
    'grid.linewidth': 0.4,
    'font.family': 'Times New Roman',
    'axes.spines.top': False,       # Remove top spine
    'axes.spines.right': False,     # Remove right spine
    'legend.frameon': False,        # Remove legend box
    'axes.axisbelow': True,         # Grid behind data
    'errorbar.capsize': 5,          # No caps for error bars
    'legend.loc': 'upper left',     # Move legend outside axes
}

scaling_curves_2x2_style = {
    # Figure - single column format
    'figure.figsize': (8, 6),
    
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

    #marker
    'lines.markersize': 10,         # Make markers bigger
    
    # Spacing for 2x2 grid
    'figure.subplot.hspace': 0.35,  # Vertical spacing between subplots
    'figure.subplot.wspace': 0.25,  # Horizontal spacing between subplots
    'figure.subplot.top': 0.92,     # Top margin
    'figure.subplot.bottom': 0.1,   # Bottom margin
    'figure.subplot.left': 0.1,     # Left margin
    'figure.subplot.right': 0.95,   # Right margin
}


