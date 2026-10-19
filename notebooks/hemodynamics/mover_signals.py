"""Decode one MOVER XML member, never an archive or a whole surgical case.

Times use the timezone-aware ``cpc.datetime`` clock. PacketTime/POLLTIME can
be stale in the released data and are retained only as provenance. Waveforms
are signed little-endian int16, scaled by Gain (observed Offset is zero).
Min/Max are display metadata, not physiological rejection thresholds.

``track_map`` maps ART/ECG/PPG to native group names, or None to disable a
track. MAP accepts ``IBP_MEAN`` only for an unambiguous numeric channel, or an
explicit one-based ``IBP_MEAN:N`` ordinal within a measurements element.
Repeated IBP_MEAN fields have no site/channel attributes in inspected S5
packets. Bounded observations in three patients support explicit IBP_MEAN:1
for GE_S5 INVP1 with six ordered SYS/MEAN/DIA triplets. This is not anatomical
site identification or a general device guarantee; selection stays explicit.
The decoder never substitutes NIBP.
Absent tracks are NaN alongside available tracks; an entirely absent waveform
or numeric stream is an empty DataFrame. Explicit gaps/conflicts remain NaN.
"""

from __future__ import annotations

import base64
import binascii
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from io import BytesIO
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


_ALIASES = {"ART": ("GE_ART", "INVP1"), "ECG": ("GE_ECG", "ECG1"),
            "PPG": ("PLETH", "PLETH1")}
_MISSING = -32767
_COLUMNS = ["time_sec", "ART", "ECG", "PPG"]
_MAX_GRID_SAMPLES = 5_000_000


@dataclass
class _Packet:
    time: float
    group: str
    fs: float
    values: np.ndarray
    gap: bool
    order: int


def _utc(value: str | None) -> float:
    if not value:
        raise ValueError("Missing cpc.datetime")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid cpc.datetime: {value!r}") from exc
    if stamp.utcoffset() is None:
        raise ValueError("cpc.datetime must include a UTC offset")
    return stamp.timestamp()


def _cpcs(payload: bytes):
    if not isinstance(payload, bytes):
        raise TypeError("payload must be one XML member's bytes")
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("XML DTD/entities are not supported")
    root = None
    for event, elem in ET.iterparse(BytesIO(payload), events=("start", "end")):
        if root is None:
            root = elem
            if root.tag != "cpcArchive":
                raise ValueError("Expected a cpcArchive XML member")
        if event == "end" and elem.tag == "cpc":
            yield root.attrib.copy(), elem
            elem.clear()
            root.clear()


def _fields(group: ET.Element) -> dict[str, str]:
    result = {}
    for m in group.findall("m"):
        name = m.get("name")
        if name is None:
            continue
        if name in result:
            raise ValueError(f"Duplicate waveform field {name!r}")
        result[name] = (m.text or "").strip()
    return result


def _wave(group: ET.Element) -> tuple[np.ndarray, float, dict]:
    fields = _fields(group)
    name = group.get("name", "")
    try:
        points = int(fields["Points"])
        width = int(fields["PointsBytes"])
        fs = float(fields["Hz"])
        gain = float(fields["Gain"])
        offset = float(fields["Offset"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Incomplete/invalid waveform metadata for {name}") from exc
    if points <= 0 or width != 2 or not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"Unsupported waveform dimensions/rate for {name}")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError(f"Invalid Gain for {name}")
    if offset != 0:
        raise ValueError("Nonzero Offset has not been validated for MOVER")
    try:
        encoded = "".join(fields["Wave"].split())
        raw_bytes = base64.b64decode(encoded, validate=True)
    except (KeyError, binascii.Error, ValueError) as exc:
        raise ValueError(f"Malformed base64 Wave for {name}") from exc
    if len(raw_bytes) != points * width:
        raise ValueError(f"Wave byte length disagrees with Points for {name}")
    raw = np.frombuffer(raw_bytes, dtype="<i2")
    missing = raw == _MISSING
    values = raw.astype(np.float64) * gain
    values[missing] = np.nan
    provenance = {k: v for k, v in fields.items() if k != "Wave"}
    provenance.update(group_attributes=dict(group.attrib),
                      encoding="signed little-endian int16", gain_formula="raw * Gain",
                      missing_markers=[_MISSING], missing_samples=int(missing.sum()))
    return values, fs, provenance


def _gap(measurements: ET.Element) -> bool:
    flags = [(m.text or "").strip().upper() for m in measurements.findall("m")
             if m.get("name") == "WaveDataGap"]
    if any(flag not in ("TRUE", "FALSE", "1", "0") for flag in flags):
        raise ValueError("Unrecognized WaveDataGap flag")
    return any(flag in ("TRUE", "1") for flag in flags)


def _number(text: str | None) -> float:
    value = (text or "").strip()
    if value in ("", "--"):
        return float("nan")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Malformed numeric IBP_MEAN: {value!r}") from exc
    if not math.isfinite(result) or result == _MISSING:
        return float("nan")
    return result


def _read(payload: bytes, keep_wave: bool) -> tuple[dict, list[_Packet], list[tuple]]:
    packets: list[_Packet] = []
    numeric: list[tuple] = []
    groups: dict[str, dict] = {}
    numeric_names: set[str] = set()
    multiplicity: dict[str, int] = defaultdict(int)
    timestamps: list[float] = []
    clock_examples: list[dict] = []
    clocks = Counter()
    gap_packets = 0
    wave_count = 0
    root_attrs = {}
    member_end = None
    numeric_attrs: dict[str, list[dict]] = defaultdict(list)
    ibp_layouts: set[tuple[str, ...]] = set()
    ibp_devices: set[tuple[str | None, ...]] = set()
    for attrs, cpc in _cpcs(payload):
        if not root_attrs:
            root_attrs = attrs
        time = _utc(cpc.get("datetime"))
        timestamps.append(time)
        member_end = max(member_end or time, time + 1.0)
        for device in cpc.findall("device"):
            for ms in device.findall("measurements"):
                direct = ms.findall("m")
                for m in direct:
                    name = m.get("name", "")
                    if name in ("PacketTime", "POLLTIME"):
                        clocks[name] += 1
                        if len(clock_examples) < 8:
                            clock_examples.append({"cpc_datetime": cpc.get("datetime"),
                                                   "name": name, "value": m.text})
                counts = Counter(m.get("name", "") for m in direct)
                for name, count in counts.items():
                    numeric_names.add(name)
                    multiplicity[name] = max(multiplicity[name], count)
                means = [m for m in direct if m.get("name") == "IBP_MEAN"]
                if means:
                    ibp_devices.add(tuple(device.get(k) for k in ("make", "model", "bid")))
                    ibp_layouts.add(tuple(m.get("name", "") for m in direct
                                          if m.get("name", "").startswith("IBP")))
                for ordinal, m in enumerate(means, 1):
                    key = f"IBP_MEAN:{ordinal}"
                    if dict(m.attrib) not in numeric_attrs[key]:
                        numeric_attrs[key].append(dict(m.attrib))
                    if keep_wave:
                        numeric.append((time, ordinal, len(means), _number(m.text)))
                gap = _gap(ms)
                waves = [g for g in ms.findall("mg") if g.find("m[@name='Wave']") is not None]
                if gap and waves:
                    gap_packets += 1
                for group in waves:
                    name = group.get("name")
                    if not name:
                        raise ValueError("Waveform group has no name")
                    values, fs, original = _wave(group)
                    duration = len(values) / fs
                    member_end = max(member_end, time + duration)
                    if name not in groups:
                        groups[name] = {"sample_rates": set(), "variants": [],
                                        "packets": 0, "valid_samples": 0,
                                        "missing_samples": 0, "gap_samples": 0,
                                        "start_utc": time, "end_utc": time + duration}
                    info = groups[name]
                    info["sample_rates"].add(fs)
                    variant = {k: v for k, v in original.items() if k != "missing_samples"}
                    variant["device"] = {k: device.get(k) for k in ("make", "model", "bid")}
                    if variant not in info["variants"]:
                        info["variants"].append(variant)
                    info["packets"] += 1
                    info["missing_samples"] += int(np.isnan(values).sum())
                    info["gap_samples"] += len(values) if gap else 0
                    info["valid_samples"] += 0 if gap else int(np.isfinite(values).sum())
                    info["start_utc"] = min(info["start_utc"], time)
                    info["end_utc"] = max(info["end_utc"], time + duration)
                    if keep_wave:
                        if gap:
                            values[:] = np.nan
                        packets.append(_Packet(time, name, fs, values, gap, wave_count))
                    wave_count += 1
    original_fs = {}
    for name, info in groups.items():
        rates = sorted(info["sample_rates"])
        info["sample_rates"] = rates
        original_fs[name] = rates[0] if len(rates) == 1 else rates
    ordered = sorted(set(timestamps))
    deltas = np.diff(ordered)
    waveform_counts = {}
    for track, aliases in _ALIASES.items():
        candidates = [g for g in aliases if g in groups]
        waveform_counts[track] = {
            "groups": candidates,
            "valid_samples": sum(groups[g]["valid_samples"] for g in candidates),
            "missing_samples": sum(groups[g]["missing_samples"] for g in candidates),
            "gap_samples": sum(groups[g]["gap_samples"] for g in candidates),
        }
    metadata = {
        "patient": root_attrs.get("PAT_ID") or root_attrs.get("patient"),
        "archive_attributes": root_attrs,
        "group_names": sorted(groups), "groups": groups, "original_fs": original_fs,
        "member_start_utc": min(timestamps) if timestamps else None,
        "member_end_utc": member_end,
        "numeric_names": sorted(numeric_names), "numeric_channel_counts": dict(multiplicity),
        "numeric_channel_attributes": dict(numeric_attrs),
        "waveform_counts": waveform_counts,
        "missing_markers": [_MISSING],
        "packet_summaries": {"cpc_count": len(timestamps), "waveform_packets": wave_count,
                             "duplicate_cpc_timestamps": len(timestamps) - len(ordered),
                             "cadence_min_sec": float(deltas.min()) if len(deltas) else None,
                             "cadence_max_sec": float(deltas.max()) if len(deltas) else None,
                             "wave_data_gap_packets": gap_packets,
                             "clock_field_counts": dict(clocks), "clock_examples": clock_examples},
        "timestamp_source": "cpc.datetime (timezone-aware UTC); PacketTime/POLLTIME not used",
        "scaling_validation": "Gain scaling observed with physiological GE_ART ranges; "
                              "INVP1 raw * 0.01 agrees with ordinal1 native S5 SYS/MEAN/DIA "
                              "in three different patients; nonzero Offset remains unverified",
        "native_unit_evidence": {"ART": "mmHg empirically supported by native pressure numerics",
                                 "ECG": "Device-native amplitude; physical unit not declared/verified",
                                 "PPG": "Device-native amplitude; physical unit not declared/verified"},
        "map_mapping_evidence": None,
    }
    invp = groups.get("INVP1")
    if (invp and invp["sample_rates"] == [100.0]
            and all(v["device"]["make"] == "GE_S5" and float(v["Gain"]) == 0.01
                    for v in invp["variants"])
            and ibp_devices == {tuple(v["device"][k] for k in ("make", "model", "bid"))
                                for v in invp["variants"]}
            and ibp_layouts == {("IBP_SYS", "IBP_MEAN", "IBP_DIA") * 6}):
        metadata["map_mapping_evidence"] = {
            "source_group": "INVP1", "supported_explicit_selector": "IBP_MEAN:1",
            "scope": "GE_S5 with six ordered native SYS/MEAN/DIA triplets; site not recorded",
            "selection_policy": "Explicit track_map required; no outcome-based selection",
            "different_patients": 3, "bounded_shard3_members": 1191,
            "exact_timestamp_pairs": [885, 568, 750],
            "MAP_vs_one_second_wave_mean_MAE_mmHg": [1.961, 6.143, 1.389],
            "complete_triplets": 2173, "SYS_ge_MAP_ge_DIA_matches": 2173,
        }
    return metadata, packets, numeric


def inspect_xml(payload: bytes) -> dict:
    """Validate one member and summarize its original schema without retaining waves.

    Counts are native samples before deduplication, not case eligibility. All
    packets, including unknown group names, have strict base64/length validation.
    Numeric names include clock/control fields as originally recorded.
    """
    return _read(payload, keep_wave=False)[0]


def _selection(metadata: dict, track_map: dict | None) -> dict:
    supplied = track_map or {}
    unknown = set(supplied) - {"ART", "ECG", "PPG", "MAP"}
    if unknown:
        raise ValueError(f"Unknown track_map keys: {sorted(unknown)}")
    selected = {}
    for track, aliases in _ALIASES.items():
        if track in supplied:
            if supplied[track] is not None and not isinstance(supplied[track], str):
                raise ValueError(f"track_map[{track!r}] must be a native group name or None")
            selected[track] = supplied[track]
        else:
            candidates = [g for g in aliases if g in metadata["group_names"]]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous {track} groups; set track_map[{track!r}]")
            selected[track] = candidates[0] if candidates else None
    selector = supplied.get("MAP", "IBP_MEAN")
    max_channels = metadata["numeric_channel_counts"].get("IBP_MEAN", 0)
    if selector is None:
        selected["MAP"] = None
    elif selector == "IBP_MEAN":
        selected["MAP"] = 1 if max_channels == 1 else None
        if max_channels > 1:
            metadata["map_unavailable_reason"] = "Ambiguous IBP_MEAN channels; select IBP_MEAN:N explicitly"
    elif isinstance(selector, str) and selector.startswith("IBP_MEAN:"):
        try:
            ordinal = int(selector.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("MAP selector must be IBP_MEAN:N (one-based)") from exc
        if ordinal <= 0:
            raise ValueError("MAP channel ordinal must be positive")
        selected["MAP"] = ordinal
        metadata["map_channel_identity"] = "Explicit packet ordinal; arterial site is not established"
    else:
        raise ValueError("MAP must use IBP_MEAN, never NIBP_MEAN or a derived surrogate")
    return selected


def _deduplicate(packets: list[_Packet], metadata: dict) -> list[_Packet]:
    merged = []
    duplicates = conflicts = 0
    by_time: dict[float, list[_Packet]] = defaultdict(list)
    for packet in packets:
        by_time[packet.time].append(packet)
    for time in sorted(by_time):
        same = by_time[time]
        first = same[0]
        if len(same) == 1:
            merged.append(first)
            continue
        duplicates += len(same) - 1
        if all(p.fs == first.fs and np.array_equal(p.values, first.values, equal_nan=True)
               for p in same[1:]):
            merged.append(first)
        else:
            conflicts += 1
            # Every conflicting packet's support is a gap, including rate/length changes.
            for p in same:
                merged.append(_Packet(time, p.group, p.fs, np.full_like(p.values, np.nan),
                                      True, p.order))
    metadata["duplicate_wave_packets"] += duplicates
    metadata["conflicting_wave_timestamps"] += conflicts
    return merged


def _segments(packets: list[_Packet]):
    pieces = []
    start = end = fs = None
    for packet in sorted(packets, key=lambda p: (p.time, p.order)):
        valid = np.isfinite(packet.values)
        edges = np.diff(np.r_[False, valid, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        if not runs and pieces:
            yield start, fs, np.concatenate(pieces)
            pieces = []
        for left, right in runs:
            run_start = packet.time + left / packet.fs
            if pieces and (packet.fs != fs or abs(run_start - end) > 1e-6):
                yield start, fs, np.concatenate(pieces)
                pieces = []
            if not pieces:
                start, fs = run_start, packet.fs
            pieces.append(packet.values[left:right])
            end = packet.time + right / packet.fs
    if pieces:
        yield start, fs, np.concatenate(pieces)


def _ceil_grid(value: float) -> int:
    # Timestamp subtraction at epoch scale can contribute < 1e-4 grid samples.
    nearest = round(value)
    return nearest if abs(value - nearest) < 1e-4 else math.ceil(value)


def _fill_track(packets: list[_Packet], times: np.ndarray, origin: float,
                target_fs: float, metadata: dict) -> np.ndarray:
    result = np.full(len(times), np.nan)
    covered = np.zeros(len(times), dtype=bool)
    conflicts = np.zeros(len(times), dtype=bool)
    # Missing support also masks overlapping native data BEFORE filtering, so a
    # duplicate/gap packet cannot contaminate adjacent valid samples through FIR.
    intervals = []
    for p in packets:
        missing = ~np.isfinite(p.values)
        edges = np.diff(np.r_[False, missing, False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            intervals.append((p.time - origin + left / p.fs, p.time - origin + right / p.fs))
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1] + 1e-7:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    ends = np.array([right for _, right in merged])
    forbidden = np.zeros(len(times), dtype=bool)
    for left, right in merged:
        a = np.searchsorted(times, left - 1e-7)
        b = np.searchsorted(times, right - 1e-7)
        forbidden[a:b] = True
    masked_packets = []
    for p in packets:
        relative = p.time - origin
        first = np.searchsorted(ends, relative, side="right")
        values = p.values
        copied = False
        for left, right in merged[first:]:
            if left >= relative + len(values) / p.fs:
                break
            a = max(0, _ceil_grid((left - relative) * p.fs))
            b = min(len(values), _ceil_grid((right - relative) * p.fs))
            if b > a and np.isfinite(values[a:b]).any():
                if not copied:
                    values = values.copy()
                    copied = True
                values[a:b] = np.nan
        masked_packets.append(_Packet(p.time, p.group, p.fs, values, p.gap, p.order))
    for start, source_fs, values in _segments(masked_packets):
        relative = start - origin
        stop = relative + len(values) / source_fs
        a = np.searchsorted(times, relative - 1e-7)
        b = np.searchsorted(times, stop - 1e-7)
        if a == b:
            continue
        ratio = (Fraction(str(target_fs)) / Fraction(str(source_fs))).limit_denominator(100_000)
        if abs(float(ratio) - target_fs / source_fs) > 1e-12:
            raise ValueError("Sample-rate ratio cannot be represented accurately")
        if ratio.numerator > 100_000 or ratio.denominator > 100_000:
            raise ValueError("Impractically large resampling ratio")
        if ratio == 1:
            filtered = values
        else:
            filtered = resample_poly(values, ratio.numerator, ratio.denominator,
                                     padtype="line" if len(values) > 1 else "constant",
                                     **({} if len(values) > 1 else {"cval": float(values[0])}))
        native_out_times = relative + np.arange(len(filtered)) / target_fs
        valid_grid = times[a:b] <= native_out_times[-1] + 1e-7
        indices = np.arange(a, b)[valid_grid]
        if not len(indices):
            continue
        placed = np.interp(times[indices], native_out_times, filtered)
        overlap = covered[indices]
        disagree = overlap & ~np.isclose(result[indices], placed, rtol=1e-9, atol=1e-9)
        conflicts[indices[disagree]] = True
        result[indices[~overlap]] = placed[~overlap]
        covered[indices] = True
    metadata["overlap_conflicting_samples"] += int(conflicts.sum())
    result[conflicts | forbidden] = np.nan
    return result


def decode_xml(payload: bytes, start_utc: float, end_utc: float,
               target_fs: float = 100.0, track_map: dict | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return waveform and native numeric MAP frames plus original provenance.

    ``time_sec`` is relative to surgical UTC start. The high-frequency grid is
    aligned to that start and clipped to [start,end) AND this member's support;
    it does not allocate pre/post-member hours. resample_poly anti-aliases each
    contiguous finite native segment, then interpolates its fractional grid
    phase without crossing missing samples, timestamp gaps, or rate changes.
    Exact duplicate packets collapse; conflicting duplicates/overlaps are
    masked and counted. MAP is native IBP_MEAN in absolute pressure units, with
    no temporal filling, baseline subtraction, or cuff substitution.
    """
    if not all(math.isfinite(x) for x in (start_utc, end_utc, target_fs)):
        raise ValueError("Bounds and target_fs must be finite")
    if end_utc <= start_utc or target_fs <= 0:
        raise ValueError("Require end_utc > start_utc and target_fs > 0")
    metadata, packets, numeric = _read(payload, keep_wave=True)
    selected = _selection(metadata, track_map)
    metadata.update(selected_tracks=selected, target_fs=float(target_fs),
                    duplicate_wave_packets=0, conflicting_wave_timestamps=0,
                    overlap_conflicting_samples=0, duplicate_map_samples=0,
                    conflicting_map_timestamps=0,
                    gap_policy="Mask WaveDataGap packets; split every native missing span/time discontinuity",
                    duplicate_policy="Equal duplicates collapse; conflicts are NaN, not last-writer wins")
    waveform = pd.DataFrame(columns=_COLUMNS, dtype=float)
    chosen = {track: [p for p in packets if p.group == group]
              for track, group in selected.items() if track != "MAP" and group is not None}
    available = [p for pp in chosen.values() for p in pp]
    if available:
        lower = max(start_utc, min(p.time for p in available))
        upper = min(end_utc, max(p.time + len(p.values) / p.fs for p in available))
        if upper > lower:
            first = _ceil_grid((lower - start_utc) * target_fs)
            last = _ceil_grid((upper - start_utc) * target_fs)
            if last - first > _MAX_GRID_SAMPLES:
                raise ValueError("Member overlap exceeds bounded grid allocation (5 million samples)")
            times = np.arange(first, last, dtype=np.float64) / target_fs
            waveform = pd.DataFrame({"time_sec": times})
            for track in _ALIASES:
                pp = _deduplicate(chosen.get(track, []), metadata)
                waveform[track] = _fill_track(pp, times, start_utc, target_fs, metadata)
    map_values: dict[float, list[float]] = defaultdict(list)
    ordinal = selected["MAP"]
    if ordinal is not None:
        for time, channel, _, value in numeric:
            if channel == ordinal and start_utc <= time < end_utc:
                map_values[time].append(value)
    rows = []
    for time in sorted(map_values):
        values = map_values[time]
        metadata["duplicate_map_samples"] += len(values) - 1
        first = values[0]
        same = all((math.isnan(first) and math.isnan(v)) or first == v for v in values[1:])
        if not same:
            metadata["conflicting_map_timestamps"] += 1
            first = float("nan")
        rows.append((time - start_utc, first))
    map_frame = pd.DataFrame(rows, columns=["time_sec", "MAP"], dtype=float)
    metadata["waveform_valid_samples"] = {t: int(waveform[t].notna().sum()) for t in _ALIASES}
    metadata["map_valid_samples"] = int(map_frame["MAP"].notna().sum())
    return waveform, map_frame, metadata
