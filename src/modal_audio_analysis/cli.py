"""Command-line interface for Modal Audio Analysis."""

import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="modal-audio-analysis")
def main():
    """Modal Audio Analysis - GPU-accelerated audio analysis."""
    pass


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="./analysis_output", help="Output directory")
@click.option("--no-stems", is_flag=True, help="Skip stem separation")
@click.option("--json-only", is_flag=True, help="Only output analysis.json (no stems/embeddings)")
def analyze(audio_file: str, output_dir: str, no_stems: bool, json_only: bool):
    """
    Analyze an audio file using GPU-accelerated Modal pipeline.

    Examples:

        modal-audio-analysis analyze track.mp3

        modal-audio-analysis analyze track.wav -o ./results

        modal-audio-analysis analyze track.mp3 --no-stems
    """
    import numpy as np

    from modal_audio_analysis.pipeline.app import analyze as modal_analyze
    from modal_audio_analysis.pipeline.app import app

    audio_path = Path(audio_file)
    console.print(f"[bold blue]Analyzing:[/bold blue] {audio_path.name}")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    console.print("[dim]Running Modal GPU pipeline...[/dim]")
    with app.run():
        result = modal_analyze.remote(audio_bytes, audio_path.name)

    if "error" in result:
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save analysis JSON
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(result["analysis"], f, indent=2)
    console.print(f"[green]Saved:[/green] {analysis_path}")

    if not json_only:
        # Save embeddings
        if result.get("embeddings_bytes"):
            embeddings_path = os.path.join(output_dir, "embeddings.npy")
            with open(embeddings_path, "wb") as f:
                f.write(result["embeddings_bytes"])
            embeddings = np.load(embeddings_path)
            console.print(f"[green]Saved:[/green] {embeddings_path} (shape: {embeddings.shape})")

        # Save stems
        stems_bytes = result.get("stems_bytes", {})
        if stems_bytes and not no_stems:
            stems_dir = os.path.join(output_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)
            for stem_name, stem_data in stems_bytes.items():
                stem_path = os.path.join(stems_dir, f"{stem_name}.mp3")
                with open(stem_path, "wb") as f:
                    f.write(stem_data)
                console.print(f"[green]Saved:[/green] {stem_path}")

    # Print summary
    _print_summary(result["analysis"])


def _print_summary(analysis: dict):
    """Print a summary of the analysis results."""
    console.print("\n[bold]Analysis Summary[/bold]")

    table = Table(show_header=False, box=None)
    table.add_column("Property", style="dim")
    table.add_column("Value")

    # Structure
    structure = analysis.get("structure", {})
    table.add_row("BPM", str(structure.get("bpm", "N/A")))
    table.add_row("Beats", f"{len(structure.get('beats', []))} beats")
    table.add_row("Segments", f"{len(structure.get('segments', []))} segments")

    # Tonal
    tonal = analysis.get("tonal", {})
    key = tonal.get("key", "")
    scale = tonal.get("scale", "")
    if key:
        table.add_row("Key", f"{key} {scale}")

    # Loudness
    dynamics = analysis.get("dynamics", {})
    if dynamics.get("loudness"):
        table.add_row("Loudness", f"{dynamics['loudness']:.1f}")

    console.print(table)

    # ML Results
    genres = analysis.get("ml_genre", {}).get("top_genres", [])[:5]
    if genres:
        console.print("\n[bold]Top Genres[/bold]")
        genre_table = Table(show_header=True)
        genre_table.add_column("Genre")
        genre_table.add_column("Probability")
        for g in genres:
            genre_table.add_row(g["genre"], f"{g['probability']:.2%}")
        console.print(genre_table)

    mood = analysis.get("ml_mood", {})
    if mood:
        console.print("\n[bold]Mood[/bold]")
        mood_table = Table(show_header=True)
        mood_table.add_column("Mood")
        mood_table.add_column("Score")
        for m, score in sorted(mood.items(), key=lambda x: x[1], reverse=True):
            mood_table.add_row(m.capitalize(), f"{score:.2%}")
        console.print(mood_table)

    # Other ML
    ml_other = analysis.get("ml_other", {})
    if ml_other:
        console.print("\n[bold]Other Features[/bold]")
        other_table = Table(show_header=False, box=None)
        other_table.add_column("Property", style="dim")
        other_table.add_column("Value")
        if "danceability" in ml_other:
            other_table.add_row("Danceability", f"{ml_other['danceability']:.2%}")
        if "voice_instrumental" in ml_other:
            other_table.add_row("Type", ml_other["voice_instrumental"].capitalize())
        console.print(other_table)

    # Instruments
    instruments = analysis.get("ml_instruments", {}).get("instruments", [])
    if instruments:
        console.print("\n[bold]Detected Instruments[/bold]")
        inst_table = Table(show_header=True)
        inst_table.add_column("Instrument")
        inst_table.add_column("Probability")
        for inst in instruments[:5]:
            inst_table.add_row(inst["instrument"], f"{inst['probability']:.2%}")
        console.print(inst_table)

    # Timing
    timings = analysis.get("_timings", {})
    if timings.get("total"):
        console.print(f"\n[dim]Total time: {timings['total']:.1f}s[/dim]")


@main.command()
def pricing():
    """Show estimated pricing for analysis."""
    console.print("[bold]Modal GPU Pricing[/bold]\n")

    table = Table(show_header=True)
    table.add_column("GPU")
    table.add_column("Per Second")
    table.add_column("Per Hour")
    table.add_column("Used For")

    table.add_row("A10G", "$0.000306", "$1.10", "Stage 1 (allin1, demucs)")
    table.add_row("T4", "$0.000164", "$0.59", "Stage 2 (ML models)")

    console.print(table)

    console.print("\n[bold]Estimated Cost Per Track (5 min song)[/bold]\n")

    cost_table = Table(show_header=True)
    cost_table.add_column("Configuration")
    cost_table.add_column("Stage 1")
    cost_table.add_column("Stage 2")
    cost_table.add_column("Total")

    cost_table.add_row("Full analysis", "~$0.014", "~$0.005", "~$0.02")
    cost_table.add_row("No stems", "~$0.006", "~$0.005", "~$0.01")
    cost_table.add_row("Structure only", "~$0.012", "-", "~$0.01")

    console.print(cost_table)


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}


def _collect_audio_files(paths: tuple) -> list[Path]:
    """Collect audio files from paths, expanding directories."""
    audio_files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            # Recursively find audio files in directory
            for ext in AUDIO_EXTENSIONS:
                audio_files.extend(path.rglob(f"*{ext}"))
        elif path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(path)
    return sorted(set(audio_files))


@main.command()
@click.argument("audio_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="./batch_output", help="Output directory")
@click.option("--concurrency", "-j", default=5, help="Max concurrent analyses")
def batch(audio_files: tuple, output_dir: str, concurrency: int):
    """
    Analyze multiple audio files in parallel.

    Supports individual files, glob patterns, or directories.

    Examples:

        modal-audio-analysis batch ./music/          # All audio in folder

        modal-audio-analysis batch *.mp3 -o ./results

        modal-audio-analysis batch track1.mp3 track2.mp3 -j 10
    """
    import time

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from modal_audio_analysis.pipeline.app import analyze as modal_analyze
    from modal_audio_analysis.pipeline.app import app

    # Collect audio files (expand directories)
    collected_files = _collect_audio_files(audio_files)

    if not collected_files:
        console.print("[yellow]No audio files found[/yellow]")
        console.print(f"[dim]Supported formats: {', '.join(AUDIO_EXTENSIONS)}[/dim]")
        return

    console.print(f"[bold blue]Batch analyzing {len(collected_files)} files[/bold blue]")
    console.print(f"[dim]Concurrency: {concurrency}[/dim]\n")

    # Prepare inputs
    inputs = []
    for audio_path in collected_files:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        inputs.append((audio_bytes, audio_path.name))

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    results = []

    with app.run(), Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(inputs)} files in parallel...", total=len(inputs)
        )

        # Use starmap for parallel execution
        for i, result in enumerate(modal_analyze.starmap(inputs, order_outputs=False)):
            filename = inputs[i][1] if i < len(inputs) else f"track_{i}"
            results.append((filename, result))
            progress.update(task, advance=1)

    total_time = time.time() - start_time
    successful = 0
    failed = 0

    # Save results
    for filename, result in results:
        if "error" in result:
            console.print(f"[red]Failed:[/red] {filename} - {result['error']}")
            failed += 1
            continue

        successful += 1
        track_dir = os.path.join(output_dir, Path(filename).stem)
        os.makedirs(track_dir, exist_ok=True)

        # Save analysis JSON
        analysis_path = os.path.join(track_dir, "analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(result["analysis"], f, indent=2)

        # Save embeddings
        if result.get("embeddings_bytes"):
            embeddings_path = os.path.join(track_dir, "embeddings.npy")
            with open(embeddings_path, "wb") as f:
                f.write(result["embeddings_bytes"])

        # Save stems
        stems_bytes = result.get("stems_bytes", {})
        if stems_bytes:
            stems_dir = os.path.join(track_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)
            for stem_name, stem_data in stems_bytes.items():
                stem_path = os.path.join(stems_dir, f"{stem_name}.mp3")
                with open(stem_path, "wb") as f:
                    f.write(stem_data)

        console.print(f"[green]Completed:[/green] {filename}")

    # Summary
    total_files = len(collected_files)
    console.print("\n[bold]Batch Complete[/bold]")
    console.print(f"  Successful: {successful}/{total_files}")
    console.print(f"  Failed: {failed}/{total_files}")
    console.print(f"  Total time: {total_time:.1f}s")
    console.print(f"  Avg time per track: {total_time/total_files:.1f}s")
    console.print(f"  Estimated cost: ~${total_files * 0.02:.2f}")


@main.command()
def deploy():
    """Deploy Modal functions for remote execution."""
    import subprocess

    console.print("[bold]Deploying Modal app...[/bold]")
    result = subprocess.run(
        ["modal", "deploy", "modal_audio_analysis.pipeline.app"],
        capture_output=False,
    )
    if result.returncode == 0:
        console.print("[green]Deployment successful![/green]")
    else:
        console.print("[red]Deployment failed.[/red]")


if __name__ == "__main__":
    main()
