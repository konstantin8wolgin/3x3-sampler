"""Small, dependency-free-on-Zarr storage primitives for run artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


ZARR_FORMAT = 2
MAX_CHUNK_SIZE = 4096
MAX_JSON_NESTING = 128


def ensure_json_value(value: Any, location: str = "artifact") -> None:
    """Reject values outside strict, finite JSON while preserving integer types."""

    def visit(current: Any, current_location: str, depth: int) -> None:
        if (
            current is None
            or isinstance(current, (bool, str))
            or type(current) is int
        ):
            return
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError(f"{current_location} contains a nonfinite number")
            return
        if isinstance(current, list):
            if depth >= MAX_JSON_NESTING:
                raise ValueError(
                    f"{location} exceeds the maximum JSON nesting depth "
                    f"of {MAX_JSON_NESTING}"
                )
            for index, item in enumerate(current):
                visit(item, f"{current_location}[{index}]", depth + 1)
            return
        if isinstance(current, dict):
            if any(not isinstance(key, str) for key in current):
                raise ValueError(
                    f"{current_location} contains a non-string object key"
                )
            if depth >= MAX_JSON_NESTING:
                raise ValueError(
                    f"{location} exceeds the maximum JSON nesting depth "
                    f"of {MAX_JSON_NESTING}"
                )
            for key, item in current.items():
                visit(item, f"{current_location}.{key}", depth + 1)
            return
        raise ValueError(f"{current_location} contains an unsupported JSON value")

    visit(value, location, 0)


def json_bytes(value: Any) -> bytes:
    ensure_json_value(value)
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
        ensure_json_value(value, path.name)
    except RecursionError as exc:
        raise ValueError(f"excessively nested JSON artifact: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    return value


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(value)


class ZarrStreamWriter:
    """Write a fixed-shape, uncompressed Zarr v2 group one chunk at a time."""

    def __init__(
        self,
        directory: Path,
        *,
        arrays: Mapping[str, tuple[np.dtype[Any] | str, tuple[int, ...]]],
        row_count: int,
        chunk_size: int,
    ) -> None:
        if type(row_count) is not int or row_count < 1:
            raise ValueError("streamed Zarr arrays require a positive row count")
        if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_CHUNK_SIZE:
            raise ValueError("Zarr chunk_size must be from 1 through 4096")
        self.directory = directory
        self.row_count = row_count
        self.chunk_size = min(chunk_size, row_count)
        self.position = 0
        self.specifications: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}
        directory.mkdir(exist_ok=False)
        write_json(directory / ".zgroup", {"zarr_format": ZARR_FORMAT})
        write_json(directory / ".zattrs", {})
        for name, (raw_dtype, trailing_shape) in sorted(arrays.items()):
            if not name or "/" in name or name.startswith("."):
                raise ValueError("Zarr array names must be simple non-hidden paths")
            dtype = np.dtype(raw_dtype)
            if dtype.hasobject:
                raise ValueError("object arrays are forbidden")
            if any(type(item) is not int or item < 1 for item in trailing_shape):
                raise ValueError("Zarr trailing dimensions must be positive integers")
            self.specifications[name] = (dtype, trailing_shape)
            target = directory / name
            target.mkdir()
            write_json(
                target / ".zarray",
                {
                    "chunks": [self.chunk_size, *trailing_shape],
                    "compressor": None,
                    "dtype": dtype.str,
                    "fill_value": None,
                    "filters": None,
                    "order": "C",
                    "shape": [row_count, *trailing_shape],
                    "zarr_format": ZARR_FORMAT,
                },
            )
            write_json(target / ".zattrs", {})

    def write(self, arrays: Mapping[str, torch.Tensor | np.ndarray]) -> None:
        if set(arrays) != set(self.specifications):
            raise ValueError("Zarr chunk fields do not match the declared arrays")
        converted = {name: _as_numpy(value) for name, value in arrays.items()}
        row_counts = {int(value.shape[0]) for value in converted.values()}
        if len(row_counts) != 1:
            raise ValueError("all Zarr fields must have the same row count")
        row_count = row_counts.pop()
        if not 1 <= row_count <= self.chunk_size:
            raise ValueError("Zarr write exceeds the declared chunk size")
        if self.position % self.chunk_size != 0:
            raise ValueError("only the final Zarr chunk may be partial")
        if self.position + row_count > self.row_count:
            raise ValueError("Zarr write exceeds the declared array shape")
        if row_count != self.chunk_size and self.position + row_count != self.row_count:
            raise ValueError("only the final Zarr chunk may be partial")
        chunk_index = self.position // self.chunk_size
        for name, value in converted.items():
            dtype, trailing_shape = self.specifications[name]
            if value.dtype != dtype or value.shape != (row_count, *trailing_shape):
                raise ValueError(f"Zarr field does not match its schema: {name}")
            padded = np.zeros((self.chunk_size, *trailing_shape), dtype=dtype)
            padded[:row_count] = value
            key = ".".join((str(chunk_index), *("0" for _ in trailing_shape)))
            (self.directory / name / key).write_bytes(padded.tobytes(order="C"))
        self.position += row_count

    def finish(self, attrs: Mapping[str, Any]) -> None:
        if self.position != self.row_count:
            raise ValueError("streamed Zarr group is incomplete")
        write_json(self.directory / ".zattrs", dict(attrs))


def zarr_metadata(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    group = read_json(directory / ".zgroup")
    if (
        not isinstance(group, dict)
        or set(group) != {"zarr_format"}
        or type(group["zarr_format"]) is not int
        or group["zarr_format"] != ZARR_FORMAT
    ):
        raise ValueError("samples.zarr zarr_format must be integer Zarr v2")
    attrs = read_json(directory / ".zattrs")
    if not isinstance(attrs, dict):
        raise ValueError("samples.zarr attributes must be an object")
    root_entries = {path.name for path in directory.iterdir()}
    if not {".zgroup", ".zattrs"}.issubset(root_entries):
        raise ValueError("samples.zarr is missing group metadata")
    if any((directory / name).is_file() for name in root_entries - {".zgroup", ".zattrs"}):
        raise ValueError("samples.zarr contains an unexpected root file")
    arrays: dict[str, dict[str, Any]] = {}
    for array_directory in sorted(path for path in directory.iterdir() if path.is_dir()):
        metadata = read_json(array_directory / ".zarray")
        required = {
            "chunks", "compressor", "dtype", "fill_value", "filters",
            "order", "shape", "zarr_format",
        }
        if not isinstance(metadata, dict) or set(metadata) != required:
            raise ValueError(f"invalid Zarr array metadata: {array_directory.name}")
        shape = metadata["shape"]
        chunks = metadata["chunks"]
        if (
            type(metadata["zarr_format"]) is not int
            or metadata["zarr_format"] != ZARR_FORMAT
            or not isinstance(shape, list)
            or not shape
            or any(type(item) is not int or item < 1 for item in shape)
            or not isinstance(chunks, list)
            or len(chunks) != len(shape)
            or any(type(item) is not int or item < 1 for item in chunks)
            or chunks[0] > MAX_CHUNK_SIZE
            or chunks[1:] != shape[1:]
            or metadata["compressor"] is not None
            or metadata["fill_value"] is not None
            or metadata["filters"] is not None
            or metadata["order"] != "C"
        ):
            raise ValueError(f"invalid or unbounded Zarr chunks: {array_directory.name}")
        try:
            dtype = np.dtype(metadata["dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Zarr dtype: {array_directory.name}") from exc
        if dtype.hasobject or read_json(array_directory / ".zattrs") != {}:
            raise ValueError(f"invalid Zarr dtype or attributes: {array_directory.name}")
        expected = {".zarray", ".zattrs"}
        for start in range(0, shape[0], chunks[0]):
            key = ".".join((str(start // chunks[0]), *("0" for _ in shape[1:])))
            expected.add(key)
            chunk_path = array_directory / key
            try:
                size = chunk_path.stat().st_size
            except OSError as exc:
                raise ValueError(f"missing Zarr chunk: {array_directory.name}/{key}") from exc
            expected_size = int(np.prod(chunks, dtype=np.int64)) * dtype.itemsize
            if size != expected_size:
                raise ValueError(f"truncated Zarr chunk: {array_directory.name}/{key}")
        if {path.name for path in array_directory.iterdir()} != expected:
            raise ValueError(f"unexpected or missing Zarr chunk: {array_directory.name}")
        arrays[array_directory.name] = metadata
    return attrs, arrays


def iter_zarr_chunks(directory: Path, name: str) -> Iterator[np.ndarray]:
    """Yield logical, unpadded chunks after validating the whole group structure."""

    _, arrays = zarr_metadata(directory)
    metadata = arrays[name]
    shape = tuple(metadata["shape"])
    chunks = tuple(metadata["chunks"])
    dtype = np.dtype(metadata["dtype"])
    for start in range(0, shape[0], chunks[0]):
        key = ".".join((str(start // chunks[0]), *("0" for _ in shape[1:])))
        payload = (directory / name / key).read_bytes()
        value = np.frombuffer(payload, dtype=dtype).reshape(chunks)
        yield value[: min(chunks[0], shape[0] - start)]


__all__ = [
    "MAX_CHUNK_SIZE",
    "MAX_JSON_NESTING",
    "ZARR_FORMAT",
    "ZarrStreamWriter",
    "ensure_json_value",
    "iter_zarr_chunks",
    "json_bytes",
    "read_json",
    "write_json",
    "zarr_metadata",
]
