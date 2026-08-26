"""Command-line interface for the fixed periodic 3x3, n_t=1 sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import (
    describe,
    estimate,
    report,
    run_contour,
    run_exact,
    run_rao_blackwell,
    validate,
)
from .config import (
    CHANNEL_DESIGNS,
    EXACT_METHOD,
    EXPLICIT_STOCHASTIC_METHOD,
    RB_STOCHASTIC_METHOD,
    ExactRunConfig,
    StochasticRunConfig,
)


SCOPE = "periodic 3x3, n_t=1"
LIMITATION = f"This command supports {SCOPE} only; see docs/limitations.md."
METHODS = (EXACT_METHOD, EXPLICIT_STOCHASTIC_METHOD, RB_STOCHASTIC_METHOD)


class _ScopeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, f"{self.prog}: error: {message}\n{LIMITATION}\n")


def _command_parser(subparsers: argparse._SubParsersAction, name: str, help: str):
    return subparsers.add_parser(
        name,
        help=f"{help} ({SCOPE})",
        description=f"{help.capitalize()} for the {SCOPE} target.",
    )


def _scientific_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--mu-chem", type=float, default=0.75)
    parser.add_argument("--geometry", default="periodic_3x3")
    parser.add_argument("--n-t", type=int, default=1)
    parser.add_argument("--method", choices=METHODS, default=EXACT_METHOD)
    parser.add_argument("--observable", default="mixed_linear_quadratic")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--channel-design", choices=CHANNEL_DESIGNS)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument(
        "--persist-endpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _ScopeParser(
        prog="tg-3x3",
        description=f"Exact and exact-law sampling for the {SCOPE} Hubbard auxiliary-field target.",
        epilog=LIMITATION,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=_ScopeParser
    )

    sample_parser = _command_parser(
        subparsers, "sample", "write an immutable sampling run"
    )
    _scientific_arguments(sample_parser)
    sample_parser.add_argument("--output", type=Path, required=True)

    describe_parser = _command_parser(
        subparsers, "describe", "describe exact support and requested work"
    )
    _scientific_arguments(describe_parser)

    validate_parser = _command_parser(
        subparsers, "validate", "validate a completed immutable run"
    )
    validate_parser.add_argument("run", type=Path)

    estimate_parser = _command_parser(
        subparsers, "estimate", "re-derive an estimate without sampling"
    )
    estimate_parser.add_argument("run", type=Path)
    estimate_parser.add_argument("--output", type=Path, required=True)

    report_parser = _command_parser(
        subparsers, "report", "re-render a report without sampling"
    )
    report_parser.add_argument("run", type=Path)
    report_parser.add_argument("--output", type=Path, required=True)
    return parser


def _config(args: argparse.Namespace) -> ExactRunConfig | StochasticRunConfig:
    common = {
        "U": args.U,
        "beta": args.beta,
        "kappa": args.kappa,
        "mu_chem": args.mu_chem,
        "geometry": args.geometry,
        "n_t": args.n_t,
        "method": args.method,
        "observable": args.observable,
    }
    if args.method in {EXPLICIT_STOCHASTIC_METHOD, RB_STOCHASTIC_METHOD}:
        return StochasticRunConfig(
            **common,
            samples=512 if args.samples is None else args.samples,
            seed=202609010001 if args.seed is None else args.seed,
            channel_design=(
                "iid_exact_categorical"
                if args.channel_design is None
                else args.channel_design
            ),
            persist_endpoints=args.persist_endpoints,
            chunk_size=256 if args.chunk_size is None else args.chunk_size,
        )

    inapplicable = [
        name
        for name, value in (
            ("--channel-design", args.channel_design),
            ("--chunk-size", args.chunk_size),
            (
                "--persist-endpoints/--no-persist-endpoints",
                args.persist_endpoints,
            ),
        )
        if value is not None
    ]
    if inapplicable:
        raise ValueError(
            f"{', '.join(inapplicable)} not applicable to {EXACT_METHOD}"
        )
    return ExactRunConfig(
        **common,
        samples=args.samples,
        seed=args.seed,
    )


def _sample(config: ExactRunConfig | StochasticRunConfig, output: Path) -> Path:
    if isinstance(config, ExactRunConfig):
        return run_exact(config, output)
    if config.method == EXPLICIT_STOCHASTIC_METHOD:
        return run_contour(config, output)
    return run_rao_blackwell(config, output)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(validate(args.run), indent=2, sort_keys=True))
            return 0
        if args.command in {"estimate", "report"}:
            operation = estimate if args.command == "estimate" else report
            output = operation(args.run, args.output)
            print(f"completed={output} source={args.run} operation={args.command}")
            return 0

        config = _config(args)
        if args.command == "describe":
            print(
                json.dumps(
                    describe(config), indent=2, sort_keys=True, allow_nan=False
                )
            )
            return 0

        output = _sample(config, args.output)
        if isinstance(config, ExactRunConfig):
            print(
                f"completed={output} authority=exact_reference "
                "channels=19683 samples=none"
            )
        else:
            print(
                f"completed={output} authority={config.authority} "
                f"channels=19683 samples={config.samples}"
            )
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"tg-3x3: error: {exc}\n{LIMITATION}\n")


__all__ = ["build_parser", "main"]
