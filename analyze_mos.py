"""Summarize completed blinded MOS responses; rejects missing or invalid scores."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True,
                        help="Combined CSV rows from all participants")
    parser.add_argument("--answer-key", type=Path, default=Path("results/mos/answer_key_PRIVATE.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/mos_summary.json"))
    args = parser.parse_args()
    with args.answer_key.open(newline="", encoding="utf-8") as file:
        key = {row["pair_id"]: row for row in csv.DictReader(file)}
    if not key:
        raise ValueError("Answer key is empty")
    ratings = defaultdict(list)
    participants = set()
    seen = set()
    with args.responses.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            participant = row["participant_id"].strip()
            pair = row["pair_id"].strip()
            if not participant or pair not in key:
                raise ValueError(f"Invalid participant or pair: {row}")
            if (participant, pair) in seen:
                raise ValueError(f"Duplicate rating for participant {participant}, {pair}")
            seen.add((participant, pair))
            participants.add(participant)
            for side in ("A", "B"):
                value = int(row[f"quality_{side}_1_to_5"])
                if value not in range(1, 6):
                    raise ValueError(f"Score must be 1-5: {row}")
                ratings[key[pair][side]].append(value)
    result = {"participants": len(participants), "distinct_pairs": len(key),
              "completed_pair_ratings": len(seen),
              "scale": "1=very poor, 2=poor, 3=fair, 4=good, 5=excellent",
              "methods": {method: {"ratings": len(values), "mean_opinion_score": statistics.mean(values),
                                    "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}
                          for method, values in ratings.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

