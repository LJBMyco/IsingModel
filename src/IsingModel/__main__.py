"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Isingmodel."""


if __name__ == "__main__":
    main(prog_name="IsingModel")  # pragma: no cover
