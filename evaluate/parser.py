from dataclasses import dataclass
from typing import List
import json


@dataclass
class BaselineArgs:
    baseline_class: str
    init_kwargs: dict
    run_kwargs: dict

    @classmethod
    def from_dict(cls, data):
        return cls(
            baseline_class=data["baseline_class"],
            init_kwargs=data["init_kwargs"],
            run_kwargs=data["run_kwargs"],
        )


@dataclass
class DatasetArgs:
    name: str
    split: str
    cache_dir: str
    document_column_name: str
    summary_colunm_name: str

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            split=data["split"],
            cache_dir=data["cache_dir"],
            document_column_name=data["document_column_name"],
            summary_colunm_name=data["summary_colunm_name"],
        )


@dataclass
class RunArgs:
    hypotheses_folder: str
    csv_file: str
    md_file: str
    rouge_types: dict

    @classmethod
    def from_dict(cls, data):
        return cls(
            hypotheses_folder=data["hypotheses_folder"],
            csv_file=data["csv_file"],
            md_file=data["md_file"],
            rouge_types=data["rouge_types"],
        )


@dataclass
class BaselineRunArgs:
    baselines: List[BaselineArgs]
    dataset: DatasetArgs
    run: RunArgs

    @classmethod
    def from_dict(cls, data):
        return cls(
            baselines=[
                BaselineArgs.from_dict(baseline) for baseline in data["baselines"]
            ],
            dataset=DatasetArgs.from_dict(data["dataset"]),
            run=RunArgs.from_dict(data["run"]),
        )


def parse_json_file(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return BaselineRunArgs.from_dict(data)
