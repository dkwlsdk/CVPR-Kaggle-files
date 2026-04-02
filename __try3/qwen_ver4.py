#!/usr/bin/env python3
from typing import Dict, List, Optional

from qwen_ver_common import clone_base_cfg, make_time_prompt, run_with_cfg


def build_time_prompt(metadata: Dict[str, str], failed_times: Optional[List[float]] = None) -> str:
    instruction_lines = [
        'Carefully analyze the ENTIRE video.',
        'Find the earliest accident_time in seconds when physical contact first begins.',
        'accident_time must be the first frame where physical contact is visible. Do NOT use a merely imminent or unavoidable collision moment.',
        'Do NOT label as an accident any slip, spin, abrupt stop, evasive steering, near-miss, or loss of control without physical contact.',
        'If there is a chain reaction or multi-car pileup, choose the first physical contact between the earliest involved pair, not a later secondary impact.',
        'Ignore the exact location and the accident type in this step.',
        'Focus only on accurately detecting the first physical contact time.',
    ]
    output_rule_lines = [
        'The reasoning must be only one short English sentence.',
        'The reasoning may mention only direct first-contact evidence visible at or immediately around the predicted time.',
        'Do NOT justify the answer using aftermath, final resting positions, or scene outcomes after the collision.',
    ]
    return make_time_prompt(metadata, failed_times, 'ver4_reasoning_restricted', instruction_lines, output_rule_lines)


cfg = clone_base_cfg('qwen_ver4', build_time_prompt)


if __name__ == '__main__':
    run_with_cfg(cfg)
