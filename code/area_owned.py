# This is a Python script to calculate area ownership scores for offensive players
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Tuple
import os
from matplotlib.path import Path
import multiprocessing
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import glob


def load_data(
        base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the NFL tracking data."""
    print("\nLoading data files...")

    # Load the data files with progress bars
    games = pd.read_csv(os.path.join(base_path, "game_data/games.csv"))
    plays = pd.read_csv(os.path.join(base_path, "game_data/plays.csv"))
    tracking = pd.read_csv(
        os.path.join(
            base_path,
            "tracking_data/tracking_week_1.csv"))
    player_play = pd.read_csv(
        os.path.join(
            base_path,
            "game_data/player_play.csv"))

    print("\nData Info:")
    print(f"Tracking rows: {len(tracking)}")
    print(f"Plays rows: {len(plays)}")
    print(f"Player-play rows: {len(player_play)}")

    # Correct tracking data orientation
    tracking['x'] = np.where(
        tracking['playDirection'] == 'left',
        120 - tracking['x'],
        tracking['x'])
    tracking['y'] = np.where(
        tracking['playDirection'] == 'left',
        160 / 3 - tracking['y'],
        tracking['y'])

    # Handle direction and orientation
    for col in ['dir', 'o']:
        tracking[col] = np.where(tracking['playDirection'] == 'left',
                                 tracking[col] + 180, tracking[col])
        tracking[col] = np.where(tracking[col] > 360,
                                 tracking[col] - 360, tracking[col])

    # Calculate directional components
    tracking['dir_rad'] = np.pi * (tracking['dir'] / 180)
    tracking['dir_x'] = np.sin(tracking['dir_rad'])
    tracking['dir_y'] = np.cos(tracking['dir_rad'])
    tracking['s_x'] = tracking['dir_x'] * tracking['s']
    tracking['s_y'] = tracking['dir_y'] * tracking['s']
    tracking['a_x'] = tracking['dir_x'] * tracking['a']
    tracking['a_y'] = tracking['dir_y'] * tracking['a']

    # Add play identifiers and frame information
    tracking['gamePlay_Id'] = tracking['gameId'].astype(
        str) + '-' + tracking['playId'].astype(str)

    # Get snap frame information
    snap_frames = tracking[tracking['frameType'] == 'SNAP'].groupby('gamePlay_Id')[
        'frameId'].first()
    tracking = tracking.merge(
        snap_frames.reset_index(),
        on='gamePlay_Id',
        suffixes=(
            '',
            '_snap'))
    tracking['relative_frame'] = tracking['frameId'] - tracking['frameId_snap']

    return tracking, plays, player_play


def calculate_ownership(frame_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate ownership areas for route runners in a single frame."""
    def_points = frame_data[frame_data['is_defensive']][['x', 'y']].values
    off_data = frame_data[frame_data['is_route_runner']
                          & frame_data['nflId'].notna()]

    if len(def_points) < 3 or len(off_data) == 0:
        return None

    try:
        # Create convex hull from defensive positions
        hull = ConvexHull(def_points)
        hull_points = def_points[hull.vertices]
        # Close the polygon by adding first point at end
        hull_points = np.vstack([hull_points, hull_points[0]])

        # Calculate actual area using shoelace formula
        hull_area = 0.5 * abs(np.sum(hull_points[:-1, 0] * hull_points[1:, 1] -
                                     hull_points[1:, 0] * hull_points[:-1, 1]))

        # Create denser grid
        grid_size = 50
        x_range = np.linspace(
            def_points[:, 0].min(), def_points[:, 0].max(), grid_size)
        y_range = np.linspace(
            def_points[:, 1].min(), def_points[:, 1].max(), grid_size)
        X, Y = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))

        # Find points inside hull using Path
        path = Path(hull_points)
        in_hull = path.contains_points(grid_points)
        grid_points = grid_points[in_hull]

        if len(grid_points) == 0:
            return None

        # Calculate area per grid point
        area_per_point = hull_area / len(grid_points)

        # Calculate distances to each route runner for each grid point
        ownership_matrix = np.zeros((len(grid_points), len(off_data)))
        for i, point in enumerate(grid_points):
            distances = np.sqrt(((point[0] - off_data['x'].values)**2 +
                                 (point[1] - off_data['y'].values)**2))
            # Use inverse distance for ownership calculation
            ownership_matrix[i] = 1 / distances

        # Normalize ownership values row-wise
        row_sums = ownership_matrix.sum(axis=1)
        ownership_matrix = ownership_matrix / row_sums[:, np.newaxis]

        # Find grid point owners (maximum ownership value)
        grid_owners = np.argmax(ownership_matrix, axis=1)

        # Calculate total area owned by each route runner
        ownership_counts = np.bincount(grid_owners, minlength=len(off_data))
        owned_areas = ownership_counts * area_per_point

        # Create summary DataFrame
        ownership_summary = pd.DataFrame({
            'nflId': off_data['nflId'].values,
            'displayName': off_data['displayName'].values,
            'area_owned': owned_areas,
            'gameId': frame_data['gameId'].iloc[0],
            'playId': frame_data['playId'].iloc[0],
            'frameId': frame_data['frameId'].iloc[0],
            'routeRan': off_data['routeRan'].values
        })

        return ownership_summary

    except Exception as e:
        print(f"Error calculating ownership: {str(e)}")
        return None


def process_play(play_data: pd.DataFrame) -> pd.DataFrame:
    """Process a single play and calculate ownership for each frame."""
    frame_results = []

    # Process each frame in the play
    for frame_id in play_data['frameId'].unique():
        frame_data = play_data[play_data['frameId'] == frame_id]
        frame_ownership = calculate_ownership(frame_data)
        if frame_ownership is not None:
            frame_results.append(frame_ownership)

    if not frame_results:
        return None

    return pd.concat(frame_results, ignore_index=True)


def calculate_area_owned(tracking_files, plays_file, player_play_file, output_dir):
    """Calculate area ownership scores for route runners and defensive players."""

    # Load plays and player data
    plays_df = pd.read_csv(plays_file)
    player_play_df = pd.read_csv(player_play_file)

    # Initialize empty list for all tracking data
    all_tracking_df = []
    
    # Load and process each tracking file
    print("\nLoading tracking data files...")
    for tracking_file in tqdm(tracking_files):
        week_tracking = pd.read_csv(tracking_file)
        all_tracking_df.append(week_tracking)
    
    # Combine all tracking data
    tracking_df = pd.concat(all_tracking_df, ignore_index=True)

    # Print data info
    print("\nData Info:")
    print(f"Tracking rows: {len(tracking_df)}")
    print(f"Plays rows: {len(plays_df)}")
    print(f"Player-play rows: {len(player_play_df)}")

    # Get route runner info
    route_runners = player_play_df[player_play_df['routeRan'].notna()][[
        'gameId', 'playId', 'nflId', 'routeRan']]
    print(f"\nFound {len(route_runners)} route runners")
    print("\nRoute types:")
    print(route_runners['routeRan'].value_counts())

    # Convert IDs to integers for consistent merging
    tracking_df['gameId'] = tracking_df['gameId'].astype('int64')
    tracking_df['playId'] = tracking_df['playId'].astype('int64')
    tracking_df['nflId'] = tracking_df['nflId'].fillna(-1).astype('int64')

    plays_df['gameId'] = plays_df['gameId'].astype('int64')
    plays_df['playId'] = plays_df['playId'].astype('int64')

    player_play_df['gameId'] = player_play_df['gameId'].astype('int64')
    player_play_df['playId'] = player_play_df['playId'].astype('int64')
    player_play_df['nflId'] = player_play_df['nflId'].astype('int64')

    # Get valid plays with route runners
    valid_plays = route_runners[['gameId', 'playId']].drop_duplicates()

    # Filter tracking data to valid plays and calculate relative frame
    tracking_df = tracking_df.merge(
        valid_plays, on=[
            'gameId', 'playId'], how='inner')

    # Find snap frames for each play
    snap_frames = tracking_df[tracking_df['event'] == 'ball_snap'].groupby(
        ['gameId', 'playId'])['frameId'].first()
    tracking_df = tracking_df.merge(
        snap_frames.reset_index(), on=[
            'gameId', 'playId'], how='left', suffixes=(
            '', '_snap'))
    tracking_df['relative_frame'] = tracking_df['frameId'] - tracking_df['frameId_snap']

    print(f"\nFiltered to {len(tracking_df)} tracking rows")
    print(f"Found {len(snap_frames)} plays with snap events")

    # Pre-process tracking data with team info and route runner filter
    tracking_with_teams = tracking_df.merge(
        plays_df[['gameId', 'playId', 'defensiveTeam']],
        on=['gameId', 'playId'],
        how='left'
    ).merge(
        route_runners,
        on=['gameId', 'playId', 'nflId'],
        how='left'
    )

    # Add flags for defensive players and route runners
    tracking_with_teams['is_defensive'] = tracking_with_teams['club'] == tracking_with_teams['defensiveTeam']
    tracking_with_teams['is_route_runner'] = tracking_with_teams['routeRan'].notna(
    )

    # Filter to frames after snap and remove plays without snaps
    tracking_with_teams = tracking_with_teams[
        tracking_with_teams['relative_frame'].notna() &
        (tracking_with_teams['relative_frame'] >= 0)
    ]

    # Group by play
    play_groups = [group for _,
                   group in tracking_with_teams.groupby(['gameId', 'playId'])]
    print(
        f"\nProcessing {
            len(play_groups)} plays using {
            multiprocessing.cpu_count() -
            1} workers...")

    ownership_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        futures = {
            executor.submit(
                process_play,
                play_data): play_data for play_data in play_groups}

        # Create progress bar for play processing
        with tqdm(total=len(play_groups), desc="Processing plays") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    ownership_results.append(result)
                pbar.update(1)

    if not ownership_results:
        raise ValueError("No valid ownership results calculated")

    # Combine results
    play_ownership = pd.concat(ownership_results, ignore_index=True)

    # Calculate player metrics at play level first
    play_level = play_ownership.groupby(['nflId', 'displayName', 'routeRan', 'gameId', 'playId'])[
        'area_owned'].mean().reset_index()

    # Then calculate overall player metrics
    player_area_owned = play_level.groupby(['nflId', 'displayName', 'routeRan']).agg(
        {'area_owned': ['mean', 'std', 'count']}).reset_index()

    player_area_owned.columns = [
        'nflId',
        'displayName',
        'routeRan',
        'raw_area',
        'std',
        'n_plays']
    player_area_owned['std_error'] = player_area_owned['std'] / \
        np.sqrt(player_area_owned['n_plays'])
    player_area_owned['area_owned_score'] = (
        player_area_owned['raw_area'] - player_area_owned['raw_area'].mean()) / player_area_owned['raw_area'].std()
    player_area_owned = player_area_owned[player_area_owned['n_plays'] >= 5].sort_values(
        'area_owned_score', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    player_area_owned.to_csv(
        os.path.join(
            output_dir,
            'player_area_owned.csv'),
        index=False)
    play_ownership.to_csv(
        os.path.join(
            output_dir,
            'play_ownership.csv'),
        index=False)

    return player_area_owned, play_ownership


def plot_route_area_owned(player_area_owned: pd.DataFrame, output_dir: Path):
    """Plot average area owned scores by route type."""
    route_summary = player_area_owned.groupby('routeRan').agg({
        'area_owned_score': ['mean', 'std', 'count']
    }).reset_index()

    route_summary.columns = ['routeRan', 'avg_area_owned', 'std', 'n_instances']
    route_summary['std_error'] = route_summary['std'] / \
        np.sqrt(route_summary['n_instances'])

    route_summary = route_summary.sort_values('avg_area_owned', ascending=True)

    plt.figure(figsize=(12, max(8, len(route_summary) * 0.4)))

    ax = sns.barplot(data=route_summary,
                     x='avg_area_owned',
                     y='routeRan',
                     color='steelblue',
                     alpha=0.6)

    y_coords = np.arange(len(route_summary))

    ax.errorbar(x=route_summary['avg_area_owned'],
                y=y_coords,
                xerr=route_summary['std_error'],
                fmt='none',
                color='black',
                capsize=3,
                elinewidth=1,
                capthick=1)

    for i, row in enumerate(route_summary.itertuples()):
        plt.text(row.avg_area_owned + row.std_error + 0.05,
                 i,
                 f'n={row.n_instances}',
                 va='center',
                 fontsize=9)

    plt.title('Average Area Owned Score by Route Type', size=14, pad=20)
    plt.xlabel('Average Area Owned Score', size=12)
    plt.ylabel('Route Type', size=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(
            output_dir,
            'route_area_owned.png'),
        dpi=300,
        bbox_inches='tight')
    plt.close()


def plot_player_routes(player_area_owned: pd.DataFrame, output_dir: Path):
    """Plot top player-route combinations."""
    top_players = player_area_owned.nlargest(10, 'area_owned_score').copy()
    top_players['player_route'] = top_players['displayName'] + \
        ' - ' + top_players['routeRan']

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_players,
                x='area_owned_score',
                y='player_route',
                color='darkred')

    # Add play counts
    for i, row in enumerate(top_players.itertuples()):
        plt.text(row.area_owned_score + 0.1,
                 i,
                 f'n={row.n_plays}')

    plt.title(
        'Top 10 Player-Route Combinations by Area Owned Score',
        size=14,
        pad=20)
    plt.xlabel('Area Owned Score')
    plt.ylabel('Player - Route')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'player_routes.png'))
    plt.close()


def plot_area_owned_distribution(play_ownership: pd.DataFrame, output_dir: Path):
    """Plot distribution of area owned scores by route type."""
    play_level = play_ownership.groupby(['gameId', 'playId', 'nflId', 'routeRan'])[
        'area_owned'].mean().reset_index()
    play_level['area_owned_score'] = (
        play_level['area_owned'] - play_level['area_owned'].mean()) / play_level['area_owned'].std()

    route_means = play_level.groupby('routeRan').agg({
        'area_owned_score': 'mean',
        'nflId': 'count'
    }).reset_index()
    route_means.columns = ['routeRan', 'mean_area_owned', 'n']
    route_means = route_means.sort_values('mean_area_owned', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=play_level,
                x='area_owned_score',
                y='routeRan',
                order=route_means['routeRan'],
                color='steelblue')

    for i, row in enumerate(route_means.itertuples()):
        plt.text(play_level['area_owned_score'].max(),
                 i,
                 f'n={row.n}',
                 va='center')

    plt.title('Distribution of Area Owned Scores by Route Type', size=14, pad=20)
    plt.xlabel('Area Owned Score')
    plt.ylabel('Route Type')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'area_owned_distribution.png'))
    plt.close()


if __name__ == "__main__":
    base_path = "your_base_path"
    output_dir = os.path.join(base_path, "results")
    os.makedirs(output_dir, exist_ok=True)

    tracking_files = sorted(glob.glob(os.path.join(base_path, "tracking_data/tracking_week_*.csv")))
    print(f"Found {len(tracking_files)} tracking data files")

    player_area_owned, play_ownership = calculate_area_owned(
        tracking_files,
        os.path.join(base_path, "game_data/plays.csv"),
        os.path.join(base_path, "game_data/player_play.csv"),
        output_dir
    )

    plot_route_area_owned(player_area_owned, output_dir)
    plot_player_routes(player_area_owned, output_dir)
    plot_area_owned_distribution(play_ownership, output_dir)

    print("\nAnalysis complete! Results saved to:", output_dir)
    print("\nTop 10 Player-Route Combinations:")
    print(player_area_owned.nlargest(10, 'area_owned_score')[
        ['displayName', 'routeRan', 'area_owned_score', 'n_plays']
    ].to_string(index=False))
