# # standard libaries
# from ast import literal_eval
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Dict, List, Union

# Document = Dict[str, Union[str, int]]


# def read_json(path: Path) -> List[Document]:
#     with open(path, "r") as fp:
#         out = [literal_eval(line) for line in fp]
#     return out


# @dataclass
# class Word:
#     text: str
#     start_idx: int
#     end_idx: int
#     labels: List[str] = field(default_factory=list)

#     def __post_init__(self):
#         self.labels.append("O")
