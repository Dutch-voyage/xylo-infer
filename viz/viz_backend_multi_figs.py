import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set style for better plots
plt.style.use('default')
plt.rcParams['font.size'] = 36

# ==============================================================================
# CONFIGURATION SECTION - Customize names and stats types here
# ==============================================================================

# Method name configurations - pick one of these options
METHOD_CONFIGS = {
    'abbrev': {
        'methods': ['Paged Decode', 'Ragged Prefill'],
        'abbreviations': {'Paged Decode': 'PD', 'Ragged Prefill': 'RP'},
        'colors': ['#2196F3', '#FF9800']
    },
    # '#4CAF50',
    'full': {
        'methods': ['Paged Prefill', 'Ragged Prefill'],
        'abbreviations': {'Paged Prefill': 'Paged Prefill', 'Ragged Prefill': 'Ragged Prefill'},
        'colors': ['#2196F3', '#FF9800']
    },
    'short': {
        'methods': ['Paged', 'Ragged'],
        'abbreviations': {'Paged': 'Paged', 'Ragged': 'Ragged'},
        'colors': ['#2196F3', '#FF9800']
    },
    'compact': {
        'methods': ['Paged', 'Ragged'],
        'abbreviations': {'Paged': 'P',  'Ragged': 'R'},
        'colors': ['#2196F3', '#FF9800']
    },
    'LS': {
        'methods': ['LS Rewrite Prefill', 'LS Sparse Prefill'],
        'abbreviations': {'LS Rewrite Prefill': 'LS-Rewrite', 'LS Sparse Prefill': 'LS-Sparse'},
        'colors': ['#4CAF50', '#9C27B0']
    }
}

# Stats type configurations - pick one of these options
STATS_CONFIGS = {
    'current': {
        'file_name': 'backend_stats.npy',
        'stats_keys': {
            'paged_decode': 'paged_decode',    # file key -> display key
            # 'paged_prefill': 'paged_prefill',
            'ragged_prefill': 'ragged_prefill'  # file key 'sparse_prefill' -> display as 'ragged_prefill'
        }
    },
    'extended': {
        'file_name': 'backend_stats_extended.npy',
        'stats_keys': {
            'paged_decode': 'paged_decode',
            'paged_prefill': 'paged_prefill',
            'ragged_prefill': 'ragged_prefill',
            'flash_decode': 'flash_decode',
            'triton_decode': 'triton_decode',
            'custom_sparse': 'custom_sparse'
        }
    },
    'alternative': {
        'file_name': 'alternative_stats.npy',
        'stats_keys': {
            'baseline_decode': 'baseline',
            'optimized_decode': 'optimized',
            'sparse_decode': 'sparse',
            'ragged_decode': 'ragged'
        }
    },
    'LS': {
        'file_name': 'LS_backend_stats.npy',
        'stats_keys': {
            'LS_rewrite_prefill': 'LS_rewrite_prefill',
            'LS_sparse_prefill': 'LS_sparse_prefill'
        }
    }
}

# Layers configuration
LAYERS_CONFIG = {
    'total_layers': 36,
    'compute_suffix': f'(36 layers)',
    'annotation_text': 'Solid: Single layer compute\nHatched area: 35 additional layers\n(36 layers total)'
}

# Figure output configuration
OUTPUT_CONFIG = {
    'figs_folder': './figs/backend_performance',
    'file_format': 'pdf',  # or 'png'
    'dpi': 300
}

# Figure naming configurations - pick one of these options
NAMING_CONFIGS = {
    'default': {
        'pattern': 'backend_performance_num_seqs_{num_seqs}.{format}',
        'description': 'Default naming: backend_performance_num_seqs_1.pdf, etc.'
    },
    'short': {
        'pattern': 'PD_PR_perf_{num_seqs}.{format}',
        'description': 'Short naming: perf_1.pdf, perf_2.pdf, etc.'
    },
    'method_specific': {
        'pattern': '{method}_performance_{num_seqs}.{format}',
        'description': 'Method-specific naming: will create separate files for each method'
    },
    'detailed': {
        'pattern': 'backend_performance_{method}_{num_seqs}seq_{layers}layers.{format}',
        'description': 'Detailed naming: backend_performance_PD_1seq_36layers.pdf, etc.'
    },
    'timestamped': {
        'pattern': 'perf_{num_seqs}_{timestamp}.{format}',
        'description': 'Timestamped naming: perf_1_20241211.pdf, etc.'
    },
    'custom': {
        'pattern': 'custom_fig_{num_seqs}.{format}',
        'description': 'Custom pattern - modify as needed'
    },
    'LS': {
        'pattern': 'LS_backend_performance_{num_seqs}seqs.{format}',
        'description': 'LS-specific naming: LS_backend_performance_1seqs.pdf, etc.'
    }
}

# ==============================================================================
# USER SELECTION - Pick your preferred configuration
# ==============================================================================

# Choose your method naming scheme: 'abbrev', 'full', 'short', 'compact', 'LS'
METHOD_CHOICE = 'LS'

# Choose your stats type: 'current', 'extended', 'alternative', 'LS'
STATS_CHOICE = 'LS'

# Choose your naming scheme: 'default', 'short', 'method_specific', 'detailed', 'timestamped', 'custom', 'LS'
NAMING_CHOICE = 'LS'

# Load chosen configurations
method_config = METHOD_CONFIGS[METHOD_CHOICE]
stats_config = STATS_CONFIGS[STATS_CHOICE]
naming_config = NAMING_CONFIGS[NAMING_CHOICE]

# Extract configurations
methods = method_config['methods']
method_abbrev = method_config['abbreviations']
colors = method_config['colors']

# Create output directory
os.makedirs(OUTPUT_CONFIG['figs_folder'], exist_ok=True)

# Generate timestamp for timestamped naming
timestamp_value = datetime.now().strftime("%Y%m%d") if 'timestamp' in naming_config['pattern'] else None

def generate_filename(num_seqs_value, method_abbrev=None):
    """Generate filename based on naming configuration"""
    pattern = naming_config['pattern']

    # Prepare format variables
    format_vars = {
        'num_seqs': num_seqs_value,
        'format': OUTPUT_CONFIG['file_format'],
        'layers': LAYERS_CONFIG['total_layers'],
        'timestamp': timestamp_value or '',
        'method': method_abbrev or 'all_methods'
    }

    # Generate filename using pattern
    filename = pattern.format(**format_vars)
    return f"{OUTPUT_CONFIG['figs_folder']}/{filename}"

# Load the backend stats using the selected configuration
backend_stats = np.load(stats_config['file_name'], allow_pickle=True).item()

print("Available methods in file:", list(backend_stats.keys()))
print("Using methods:", list(stats_config['stats_keys'].keys()))

# Extract data for each method using the configurable keys
stats_dict = {}
for file_key, display_key in stats_config['stats_keys'].items():
    if file_key in backend_stats:
        # Keep as list but organize by num_seqs for easier processing
        stats_dict[display_key] = backend_stats[file_key]
    else:
        print(f"Warning: {file_key} not found in stats file")

# Get all unique num_seqs values
all_num_seqs = sorted(set(
    item['num_seqs']
    for method_stats in stats_dict.values()
    for item in method_stats
))

print(f"Available num_seqs values: {all_num_seqs}")

def create_figure_for_num_seqs(num_seqs_value):
    """Create a visualization for a specific num_seqs value"""

    # Filter data for this num_seqs value
    filtered_stats = {}
    for method_name, method_stats in stats_dict.items():
        filtered_stats[method_name] = [
            item for item in method_stats
            if item['num_seqs'] == num_seqs_value
        ]

    # Get sequence lengths for this num_seqs
    sequence_lengths = sorted(set(
        int(item['seq_length_mean'])
        for method_stats in filtered_stats.values()
        for item in method_stats
    ))

    if not sequence_lengths:
        print(f"No data available for num_seqs = {num_seqs_value}")
        return

    print(f"\nCreating figure for num_seqs = {num_seqs_value}")
    print(f"Sequence lengths: {sequence_lengths}")

    # Create figure
    fig, ax = plt.subplots(figsize=(25, 10))

    width = 0.25
    x = np.arange(len(sequence_lengths))

    # Prepare data for stacked bars
    prepare_times = {}
    compute_times = {}
    

    for method in methods:
        stat_key = None
        # Find the corresponding stat key for this method using multiple strategies
        for file_key, display_key in stats_config['stats_keys'].items():
            # Try exact match with file_key
            if method.replace(' ', '_').lower() == file_key.lower():
                stat_key = display_key
                break
            # Try exact match with display_key
            if method.replace(' ', '_').lower() == display_key.lower():
                stat_key = display_key
                break
            
        if stat_key is None:
            print(f"Warning: No stat key found for method {method}")
            prepare_times[method] = [0] * len(sequence_lengths)
            compute_times[method] = [0] * len(sequence_lengths)
            continue

        # Create lookup dict for this method and num_seqs
        data_lookup = {
            int(item['seq_length_mean']): item
            for item in filtered_stats[stat_key]
        }

        prepare_times[method] = [
            data_lookup[seq_len]['prepare_mean_ms']
            if seq_len in data_lookup else 0
            for seq_len in sequence_lengths
        ]
        compute_times[method] = [
            data_lookup[seq_len]['compute_mean_ms']
            if seq_len in data_lookup else 0
            for seq_len in sequence_lengths
        ]
        
    # Additional layers computation time using configurable total
    total_layers = LAYERS_CONFIG['total_layers']
    compute_times_total_layers = {
        method: [compute_times[method][i] * total_layers for i in range(len(sequence_lengths))]
        for method in methods
    }

    # Plot stacked bars and collect legend handles
    handles = []
    labels = []

    for i, method in enumerate(methods):
        offset = (i - 1) * width
        abbrev = method_abbrev[method]

        # Prepare time (bottom layer)
        bar1 = ax.bar(x + offset, prepare_times[method], width,
                     color=colors[i], alpha=0.7)

        # Compute time for 1 layer (second layer)
        bar2 = ax.bar(x + offset, compute_times[method], width,
                     bottom=prepare_times[method],
                     color=colors[i], alpha=0.9)

        # Compute time for additional layers as a dotted/hatched area on top
        bottom_total_layers = [prepare_times[method][j] + compute_times[method][j]
                              for j in range(len(sequence_lengths))]

        bar3 = ax.bar(x + offset, compute_times_total_layers[method], width,
                     bottom=bottom_total_layers,
                     color=colors[i], alpha=0.4, hatch='///', edgecolor='black', linewidth=0.5)

        # Store handles and labels for custom legend ordering
        handles.extend([bar1[0], bar2[0], bar3[0]])
        labels.extend([f'{abbrev} - Prepare', f'{abbrev} - Compute (1 layer)',
                      f'{abbrev} - Compute {LAYERS_CONFIG["compute_suffix"]}'])

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    # ax.set_title(f'Backend Performance: {num_seqs} Sequences (Prepare + Compute Times)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(seq_len) for seq_len in sequence_lengths])

    # Create legend with each method's components in the same row
    ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.20), loc='upper center',
             ncol=3, columnspacing=0.8, handletextpad=0.4, frameon=True)
    ax.grid(True, alpha=0.3)

    # Add some performance statistics
    stats_text = f"Number of requests:{num_seqs}\nAll Layer Performance (ms):\n"
    for method in methods:
        # Find the corresponding stat key for this method using multiple strategies
        stat_key = None
        method_normalized = method.replace(' ', '_').lower()

        # Only check exact matches - no partial matches to avoid confusion
        for file_key, display_key in stats_config['stats_keys'].items():
            # Try exact match with file_key
            if method_normalized == file_key.lower():
                stat_key = display_key
                break
            # Try exact match with display_key
            if method_normalized == display_key.lower():
                stat_key = display_key
                break

        if stat_key and stat_key in filtered_stats:
            avg_prepare = np.mean([item['prepare_mean_ms'] for item in filtered_stats[stat_key]])
            avg_compute = np.mean([item['compute_mean_ms'] for item in filtered_stats[stat_key]])
            stats_text += f"{method_abbrev[method]}: {avg_prepare:.2f} (prepare), {avg_compute * total_layers:.2f} (compute)\n"

    stats_text = stats_text.strip("\n")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust layout to minimize empty space and make room for bottom legend
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.2)

    # Print detailed statistics for this num_seqs
    print(f"\n=== Detailed Statistics for {num_seqs} Sequences ===")
    for method in methods:
        # Find the corresponding stat key for this method using multiple strategies
        stat_key = None
        method_normalized = method.replace(' ', '_').lower()

        # Only check exact matches - no partial matches to avoid confusion
        for file_key, display_key in stats_config['stats_keys'].items():
            # Try exact match with file_key
            if method_normalized == file_key.lower():
                stat_key = display_key
                break
            # Try exact match with display_key
            if method_normalized == display_key.lower():
                stat_key = display_key
                break

        print(f"\n{method} (stat_key={stat_key}):")
        if stat_key and stat_key in filtered_stats:
            print(f"  Found {len(filtered_stats[stat_key])} entries")
            for item in filtered_stats[stat_key]:
                seq_len = int(item['seq_length_mean'])
                prep = item['prepare_mean_ms']
                comp = item['compute_mean_ms']
                total = prep + comp
                print(f"  Seq Length {seq_len}: Prepare={prep:.2f}ms, Compute={comp:.2f}ms, Total={total:.2f}ms")
        else:
            print(f"  No statistics available for this method (stat_key={stat_key}, in_filtered={stat_key in filtered_stats if stat_key else 'N/A'})")

    # Save figure
    filename = generate_filename(num_seqs_value)
    plt.savefig(filename, dpi=OUTPUT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()

    print(f"Saved: {filename}")

# Create figures for each num_seqs value
for num_seqs in all_num_seqs:
    create_figure_for_num_seqs(num_seqs)

print(f"\nAll figures saved to {OUTPUT_CONFIG['figs_folder']}/")
print(f"Using naming scheme: {NAMING_CHOICE} - {naming_config['description']}")
print("Generated files:")
for num_seqs in all_num_seqs:
    filename = generate_filename(num_seqs)
    print(f"  - {filename}")