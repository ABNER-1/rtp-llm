import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Sequence, TypeVar

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.server.server_args.server_args import EnvArgumentParser, init_all_group_args

_T = TypeVar("_T")


def setup_args() -> tuple[EnvArgumentParser, Namespace]:
    parser = EnvArgumentParser(description="RTP LLM")

    # 使用统一的函数初始化所有参数组
    init_all_group_args(parser)

    args = parser.parse_args()

    # add rocm env config, if using default value, change it to optimize version
    if os.path.exists("/dev/kfd") and os.getenv("FT_DISABLE_CUSTOM_AR") is None:
        os.environ["FT_DISABLE_CUSTOM_AR"] = "0"
        logging.info(
            "[MI308X] enable FT_DISABLE_CUSTOM_AR by default, as amd has own implementation."
        )

    if os.path.exists("/dev/kfd") and os.getenv("SEQ_SIZE_PER_BLOCK") is None:
        os.environ["SEQ_SIZE_PER_BLOCK"] = "16"
        logging.info(
            "[MI308X] set SEQ_SIZE_PER_BLOCK 16 by default, as it just support 16 now."
        )

    if os.path.exists("/dev/kfd") and os.getenv("ENABLE_COMM_OVERLAP") is None:
        os.environ["ENABLE_COMM_OVERLAP"] = "0"
        logging.info("[MI308X] disable ENABLE_COMM_OVERLAP by default.")

    parser.print_env_mappings()
    StaticConfig.update_from_env()
    return parser, args

