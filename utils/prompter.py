"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, List

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        file_name = osp.join("dataset", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
            self.template["prompt_input"]="<|begin_of_text|>{{ Given the user's purchase history, predict next possible item to be purchased."
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        task_type: str,
    ) -> List[str]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if task_type == 'sequential':
            instruction = "Given the user’s purchase history, predict next possible item to be purchased."
        else:
            instruction = ""
        ins = self.template["prompt_input"].format(
            instruction=instruction
        )
        res = self.template["response_split"]
        if self._verbose:
            print(ins + res)
        return [ins, res]

