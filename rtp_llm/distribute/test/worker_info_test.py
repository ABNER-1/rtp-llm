import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

torch.cuda.set_device = lambda x: None


from rtp_llm.distribute.worker_info import ParallelInfo, WorkerInfo


class TestParallelInfo(unittest.TestCase):
    def setUp(self):
        self.env_vars = {
            "TP_SIZE": "1",
            "EP_SIZE": "1",
            "PP_SIZE": "1",
            "DP_SIZE": "1",
            "WORLD_SIZE": "1",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "FFN_SP_SIZE": "1",
        }

    def test_reload(self):
        # Initial setup
        with patch.dict(os.environ, self.env_vars):
            info = ParallelInfo.from_env(worker_info_port_num=1234)
            self.assertEqual(info.tp_size, 1)
            self.assertEqual(info.worker_info_port_num, 1234)

            # Change env vars
            new_env_vars = self.env_vars.copy()
            new_env_vars["TP_SIZE"] = "2"
            new_env_vars["WORLD_SIZE"] = "2"
            new_env_vars["WORLD_RANK"] = "1"
            with patch.dict(os.environ, new_env_vars):
                info.reload(worker_info_port_num=5678)

                expected_info = ParallelInfo.from_env(worker_info_port_num=5678)
                self.assertEqual(info, expected_info)


class TestWorkerInfo(unittest.TestCase):
    def test_reload(self):
        # Create parallel_info and worker_info
        parallel_info = ParallelInfo.from_env(MIN_WORKER_INFO_PORT_NUM)
        parallel_info.worker_info_port_num = 10

        # Initial WorkerInfo
        info = WorkerInfo.from_env(
            parallel_info, start_port=1000, remote_server_port=2000
        )
        initial_port = info.server_port

        # Reload with new ports
        info.reload(parallel_info, start_port=3000, remote_server_port=4000)

        # Verify updates - server_port should change
        self.assertEqual(info.server_port, 3000)
        self.assertNotEqual(info.server_port, initial_port)


class TestUpdateWorkerInfo(unittest.TestCase):
    def test_update_worker_info(self):
        # Test that reload updates worker_info correctly
        parallel_info = ParallelInfo.from_env(MIN_WORKER_INFO_PORT_NUM)
        worker_info = WorkerInfo.from_env(
            parallel_info, start_port=1000, remote_server_port=2000
        )

        initial_server_port = worker_info.server_port

        # Reload with new ports
        parallel_info.reload(worker_info_port_num=20)
        worker_info.reload(parallel_info, start_port=3000, remote_server_port=4000)

        # Verify that server_port was updated
        self.assertNotEqual(worker_info.server_port, initial_server_port)
        self.assertEqual(worker_info.server_port, 3000)
