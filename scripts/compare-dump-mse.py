#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CPU_DIR = "/home/ubuntu/data/gemma4/tanh_dump_cpu"
DEFAULT_RPP_DIR = "/home/ubuntu/data/gemma4/tanh_dump_rpp"


@dataclass
class DumpInfo:
    backend: str
    id: int
    name: str
    elements: int
    data_path: Path
    meta_path: Path
    ne: str
    original_type: str


@dataclass
class MseResult:
    name: str
    cpu_id: int
    rpp_id: int
    elements: int
    compared: int
    cpu_nan: int
    rpp_nan: int
    cpu_inf: int
    rpp_inf: int
    mse: float
    rmse: float
    mae: float
    max_abs: float


def parse_meta(meta_path: Path, backend: str) -> DumpInfo:
    values: dict[str, str] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value

    missing = [key for key in ("id", "name", "elements", "data") if key not in values]
    if missing:
        raise ValueError(f"{meta_path}: missing keys: {', '.join(missing)}")

    data_path = Path(values["data"])
    if not data_path.is_absolute():
        data_path = meta_path.parent / data_path

    return DumpInfo(
        backend=backend,
        id=int(values["id"]),
        name=values["name"],
        elements=int(values["elements"]),
        data_path=data_path,
        meta_path=meta_path,
        ne=values.get("ne", ""),
        original_type=values.get("original_type", ""),
    )


def load_dumps(dump_dir: Path, backend: str) -> dict[str, list[DumpInfo]]:
    result: dict[str, list[DumpInfo]] = {}
    if not dump_dir.exists():
        raise FileNotFoundError(f"{backend} dump dir does not exist: {dump_dir}")

    for meta_path in sorted(dump_dir.glob("*.meta")):
        info = parse_meta(meta_path, backend)
        if not info.data_path.exists():
            raise FileNotFoundError(f"{meta_path}: data file does not exist: {info.data_path}")
        result.setdefault(info.name, []).append(info)

    for dumps in result.values():
        dumps.sort(key=lambda x: x.id)
    return result


def read_f32_chunk(f, max_values: int) -> array:
    data = f.read(max_values * 4)
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def compare_f32_files(cpu: DumpInfo, rpp: DumpInfo, chunk_values: int) -> MseResult:
    if cpu.elements != rpp.elements:
        raise ValueError(
            f"{cpu.name}: element count mismatch: cpu={cpu.elements}, rpp={rpp.elements}"
        )

    expected_bytes = cpu.elements * 4
    cpu_size = cpu.data_path.stat().st_size
    rpp_size = rpp.data_path.stat().st_size
    if cpu_size != expected_bytes or rpp_size != expected_bytes:
        raise ValueError(
            f"{cpu.name}: file size mismatch, expected {expected_bytes} bytes, "
            f"cpu={cpu_size}, rpp={rpp_size}"
        )

    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    seen = 0
    compared = 0
    cpu_nan = 0
    rpp_nan = 0
    cpu_inf = 0
    rpp_inf = 0

    with cpu.data_path.open("rb") as f_cpu, rpp.data_path.open("rb") as f_rpp:
        while seen < cpu.elements:
            want = min(chunk_values, cpu.elements - seen)
            cpu_values = read_f32_chunk(f_cpu, want)
            rpp_values = read_f32_chunk(f_rpp, want)
            if len(cpu_values) != want or len(rpp_values) != want:
                raise ValueError(f"{cpu.name}: short read while comparing")

            abs_errors = []
            for a, b in zip(cpu_values, rpp_values):
                if math.isnan(a):
                    cpu_nan += 1
                if math.isnan(b):
                    rpp_nan += 1
                if math.isinf(a):
                    cpu_inf += 1
                if math.isinf(b):
                    rpp_inf += 1
                if not math.isfinite(a) or not math.isfinite(b):
                    continue

                err = abs(a - b)
                abs_errors.append(err)
                compared += 1

            if abs_errors:
                sum_abs += math.fsum(abs_errors)
                sum_sq += math.fsum(err * err for err in abs_errors)
                max_abs = max(max_abs, max(abs_errors))
            seen += want

    mse = sum_sq / compared if compared else math.nan
    mae = sum_abs / compared if compared else math.nan
    return MseResult(
        name=cpu.name,
        cpu_id=cpu.id,
        rpp_id=rpp.id,
        elements=cpu.elements,
        compared=compared,
        cpu_nan=cpu_nan,
        rpp_nan=rpp_nan,
        cpu_inf=cpu_inf,
        rpp_inf=rpp_inf,
        mse=mse,
        rmse=math.sqrt(mse),
        mae=mae,
        max_abs=max_abs,
    )


def pair_dumps(
    cpu_dumps: dict[str, list[DumpInfo]],
    rpp_dumps: dict[str, list[DumpInfo]],
    name_filter: str | None,
) -> list[tuple[DumpInfo, DumpInfo]]:
    pairs: list[tuple[DumpInfo, DumpInfo]] = []
    for name in sorted(set(cpu_dumps) & set(rpp_dumps)):
        if name_filter and name_filter not in name:
            continue
        cpu_items = cpu_dumps[name]
        rpp_items = rpp_dumps[name]
        for cpu, rpp in zip(cpu_items, rpp_items):
            pairs.append((cpu, rpp))
    return pairs


@dataclass(frozen=True)
class TableColumn:
    header: str
    min_width: int
    max_width: int
    align: str
    value: callable


def terminal_columns() -> int:
    return max(40, shutil.get_terminal_size(fallback=(120, 24)).columns)


def truncate_text(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width == 1:
        return value[:1]
    return value[: width - 1] + "~"


def format_sci(value: float, width: int) -> str:
    if math.isnan(value):
        return "nan".rjust(width)
    if math.isinf(value):
        return ("inf" if value > 0 else "-inf").rjust(width)
    precision = 7 if width >= 14 else 4 if width >= 11 else 2
    text = f"{value:.{precision}e}"
    if len(text) > width:
        text = f"{value:.1e}"
    return text.rjust(width)


def format_int(value: int, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text.rjust(width)
    # Preserve the order of magnitude when a very narrow terminal forces truncation.
    if width <= 4:
        return ("*" + text[-(width - 1) :]).rjust(width)
    return ("~" + text[-(width - 1) :]).rjust(width)


def table_width(columns: list[TableColumn], widths: list[int]) -> int:
    return sum(widths) + max(0, len(columns) - 1)


def make_columns(compact: bool) -> list[TableColumn]:
    if compact:
        return [
            TableColumn("name", 10, 32, "<", lambda x, w: truncate_text(x.name, w).ljust(w)),
            TableColumn("c_id", 4, 6, ">", lambda x, w: format_int(x.cpu_id, w)),
            TableColumn("r_id", 4, 6, ">", lambda x, w: format_int(x.rpp_id, w)),
            TableColumn("elems", 7, 12, ">", lambda x, w: format_int(x.elements, w)),
            TableColumn("cmp", 7, 12, ">", lambda x, w: format_int(x.compared, w)),
            TableColumn("c_nan", 5, 8, ">", lambda x, w: format_int(x.cpu_nan, w)),
            TableColumn("r_nan", 5, 8, ">", lambda x, w: format_int(x.rpp_nan, w)),
            TableColumn("mse", 9, 14, ">", lambda x, w: format_sci(x.mse, w)),
            TableColumn("rmse", 9, 14, ">", lambda x, w: format_sci(x.rmse, w)),
            TableColumn("mae", 9, 14, ">", lambda x, w: format_sci(x.mae, w)),
            TableColumn("max", 9, 14, ">", lambda x, w: format_sci(x.max_abs, w)),
        ]

    return [
        TableColumn("name", 16, 40, "<", lambda x, w: truncate_text(x.name, w).ljust(w)),
        TableColumn("cpu_id", 6, 6, ">", lambda x, w: format_int(x.cpu_id, w)),
        TableColumn("rpp_id", 6, 6, ">", lambda x, w: format_int(x.rpp_id, w)),
        TableColumn("elements", 10, 12, ">", lambda x, w: format_int(x.elements, w)),
        TableColumn("compared", 10, 12, ">", lambda x, w: format_int(x.compared, w)),
        TableColumn("cpu_nan", 7, 8, ">", lambda x, w: format_int(x.cpu_nan, w)),
        TableColumn("rpp_nan", 7, 8, ">", lambda x, w: format_int(x.rpp_nan, w)),
        TableColumn("cpu_inf", 7, 8, ">", lambda x, w: format_int(x.cpu_inf, w)),
        TableColumn("rpp_inf", 7, 8, ">", lambda x, w: format_int(x.rpp_inf, w)),
        TableColumn("mse", 11, 14, ">", lambda x, w: format_sci(x.mse, w)),
        TableColumn("rmse", 11, 14, ">", lambda x, w: format_sci(x.rmse, w)),
        TableColumn("mae", 11, 14, ">", lambda x, w: format_sci(x.mae, w)),
        TableColumn("max_abs", 11, 14, ">", lambda x, w: format_sci(x.max_abs, w)),
    ]


def choose_table_layout(max_columns: int) -> tuple[list[TableColumn], list[int]] | None:
    for compact in (False, True):
        columns = make_columns(compact)
        widths = [column.max_width for column in columns]
        name_idx = 0
        while table_width(columns, widths) > max_columns:
            shrinkable = [
                i for i, (column, width) in enumerate(zip(columns, widths)) if width > column.min_width
            ]
            if not shrinkable:
                break
            idx = name_idx if widths[name_idx] > columns[name_idx].min_width else max(
                shrinkable, key=lambda i: widths[i] - columns[i].min_width
            )
            widths[idx] -= 1
        if table_width(columns, widths) <= max_columns:
            return columns, widths
    return None


def print_table(results: list[MseResult], columns: list[TableColumn], widths: list[int]) -> None:
    header = " ".join(
        truncate_text(column.header, width).ljust(width)
        if column.align == "<"
        else truncate_text(column.header, width).rjust(width)
        for column, width in zip(columns, widths)
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(" ".join(column.value(item, width) for column, width in zip(columns, widths)))


def print_wrapped_results(results: list[MseResult], max_columns: int) -> None:
    for idx, item in enumerate(results):
        if idx:
            print()
        name_width = max(12, max_columns - 48)
        chunks = [
            f"{truncate_text(item.name, name_width)}",
            f"ids={item.cpu_id}/{item.rpp_id}",
            f"elems={item.elements}",
            f"compared={item.compared}",
            f"nan={item.cpu_nan}/{item.rpp_nan}",
            f"inf={item.cpu_inf}/{item.rpp_inf}",
            f"mse={item.mse:.4e}",
            f"rmse={item.rmse:.4e}",
            f"mae={item.mae:.4e}",
            f"max={item.max_abs:.4e}",
        ]

        line = chunks[0]
        indent = "  "
        for chunk in chunks[1:]:
            candidate = f"{line} {chunk}"
            if len(candidate) <= max_columns:
                line = candidate
                continue
            print(line)
            line = f"{indent}{chunk}"
        print(line)


def print_wrapped_chunks(chunks: list[str], max_columns: int, indent: str = "  ") -> None:
    if not chunks:
        return
    line = chunks[0]
    for chunk in chunks[1:]:
        candidate = f"{line} {chunk}"
        if len(candidate) <= max_columns:
            line = candidate
            continue
        print(line)
        line = f"{indent}{chunk}"
    print(line)


def print_results(results: list[MseResult]) -> None:
    if not results:
        print("no matching CPU/RPP dumps found")
        return

    max_columns = terminal_columns()
    layout = choose_table_layout(max_columns)
    if layout:
        columns, widths = layout
        print_table(results, columns, widths)
    else:
        print_wrapped_results(results, max_columns)

    worst = max(results, key=lambda x: x.mse if not math.isnan(x.mse) else math.inf)
    print()
    print_wrapped_chunks(
        [
            "worst_mse:",
            f"name={worst.name}",
            f"cpu_id={worst.cpu_id}",
            f"rpp_id={worst.rpp_id}",
            f"mse={worst.mse:.7e}",
            f"rmse={worst.rmse:.7e}",
            f"max_abs={worst.max_abs:.7e}",
        ],
        max_columns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CPU and RPP dump outputs stored as f32 binary files."
    )
    parser.add_argument("--cpu-dir", "-cpu", default=DEFAULT_CPU_DIR, help="CPU dump directory")
    parser.add_argument("--rpp-dir", "-rpp", default=DEFAULT_RPP_DIR, help="RPP dump directory")
    parser.add_argument("--name-filter", "-n", help="Only compare tensor names containing this substring")
    parser.add_argument("--chunk-values", type=int, default=1 << 20, help="Number of f32 values read per chunk")
    args = parser.parse_args()

    if args.chunk_values <= 0:
        parser.error("--chunk-values must be positive")

    cpu_dumps = load_dumps(Path(args.cpu_dir), "cpu")
    rpp_dumps = load_dumps(Path(args.rpp_dir), "rpp")
    pairs = pair_dumps(cpu_dumps, rpp_dumps, args.name_filter)

    cpu_only = sorted(set(cpu_dumps) - set(rpp_dumps))
    rpp_only = sorted(set(rpp_dumps) - set(cpu_dumps))
    if cpu_only:
        print(f"warning: {len(cpu_only)} CPU-only tensor names: {', '.join(cpu_only[:8])}", file=sys.stderr)
    if rpp_only:
        print(f"warning: {len(rpp_only)} RPP-only tensor names: {', '.join(rpp_only[:8])}", file=sys.stderr)

    results = [compare_f32_files(cpu, rpp, args.chunk_values) for cpu, rpp in pairs]
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
