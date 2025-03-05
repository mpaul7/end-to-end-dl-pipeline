import matplotlib.pyplot as plt
import pandas as pd


import click
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

@click.group()
def cli():
    """CLI for generating bar graphs"""
    pass

@cli.command(name="bargraph")
@click.argument('input_csv', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def plot_graph(input_csv, output_dir):
    """Generate bar graph from CSV file
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save output graph
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(input_csv)
    
    # Generate plot
    plot_dynamic_bargraph(df, output_path / f"{Path(input_csv).stem}.png")

def plot_dynamic_bargraph(df, output_file):
    """Generate dynamic bar graph and save to file
    
    Args:
        df: Pandas DataFrame with data
        output_file: Path to save output graph
    """
    # Extract applications column
    applications = df.iloc[:, 0]
    print(applications)
    
    # Extract all other columns (Y-axis values) 
    data_columns = df.iloc[:, 1:]
    print(data_columns)
    # Number of bars (columns)
    num_bars = len(data_columns.columns)
    
    # Set the figure size dynamically based on the number of bars
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each column as a bar group
    bar_width = 0.8 / num_bars  # Adjust bar width based on number of bars
    positions = range(len(applications))
    
    for idx, column in enumerate(data_columns.columns):
        ax.bar(
            [p + idx * bar_width for p in positions],  # Adjust position for each bar
            data_columns[column],  # Y values
            bar_width,  # Width of the bars
            label=column  # Label for the legend
        )
    
    # Adjust X-axis ticks and labels
    ax.set_xticks([p + bar_width * (num_bars / 2 - 0.5) for p in positions])
    ax.set_xticklabels(applications, rotation=45, ha='right')

    # Add labels and title
    ax.set_xlabel('Applications')
    ax.set_ylabel('Values')
    ax.set_title('Dynamic Bar Graph for Applications')

    # Show legend
    ax.legend()

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    cli()
