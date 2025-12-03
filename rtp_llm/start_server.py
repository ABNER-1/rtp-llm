import logging
import multiprocessing
import os
import sys
import threading
import time
import traceback

from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info
from rtp_llm.ops import RoleType
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import ProcessManager

setup_logging()


# 全局变量用于跟踪后端服务的健康状态
backend_health_status = {"ready": False, "checked": False, "lock": threading.Lock()}


@timer_wrapper(description="start backend server")
def start_backend_server_impl(global_controller, py_env_configs, process_manager=None):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs)
        os._exit(-1)

    logging.info(f"[PROCESS_SPAWN]Start backend server process outer")
    backend_process = multiprocessing.Process(
        target=start_backend_server,
        args=(global_controller, py_env_configs),
        name="backend_server",
    )
    backend_process.start()

    start_port = py_env_configs.server_config.start_port
    backend_server_port = WorkerInfo.backend_server_port_offset(0, start_port)

    # Register health check with ProcessManager
    if process_manager:
        process_manager.register_health_check(
            processes=[backend_process],
            process_name="backend_server",
            port=backend_server_port,
        )

    return backend_process


@timer_wrapper(description="start frontend server")
def start_frontend_server_impl(global_controller, py_env_configs, process_manager=None):
    from rtp_llm.start_frontend_server import start_frontend_server

    frontend_server_count = py_env_configs.server_config.frontend_server_count
    if frontend_server_count < 1:
        logging.info(
            "frontend server's count is {frontend_server_count}, this may be a mistake"
        )

    frontend_processes = []

    # tmp code
    local_world_size = g_parallel_info.world_size
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, world size = {g_parallel_info.world_size}"
        )

    for rank in range(local_world_size):
        for i in range(frontend_server_count):
            logging.info(
                f"[PROCESS_SPAWN]Start frontend server process rank_{rank}_server_{i} outer"
            )
            process = multiprocessing.Process(
                target=start_frontend_server,
                args=(rank, i, global_controller, py_env_configs),
                name=f"frontend_server_{i}",
            )
            frontend_processes.append(process)
            process.start()

    # Register health check with ProcessManager for the first frontend server
    start_port = py_env_configs.server_config.start_port
    if process_manager and frontend_processes:
        process_manager.register_health_check(
            processes=frontend_processes,
            process_name="frontend_server",
            port=start_port,
        )

    return frontend_processes


def main():
    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)
    start_server(py_env_configs)


def start_server(py_env_configs):
    logging.info(f"[PROCESS_START]Start server")
    start_time = time.time()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    global_controller = init_controller(
        py_env_configs.concurrency_config, dp_size=g_parallel_info.dp_size
    )

    # Create process manager with config values
    process_manager = ProcessManager(
        shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
        monitor_interval=py_env_configs.server_config.monitor_interval,
    )

    try:
        if py_env_configs.role_config.role_type != RoleType.FRONTEND:
            logging.info("start backend server")
            backend_process = start_backend_server_impl(
                global_controller, py_env_configs, process_manager
            )
            process_manager.add_process(backend_process)

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, py_env_configs, process_manager
        )
        process_manager.add_processes(frontend_process)

        # Start parallel health checks for all registered services
        process_manager.start_parallel_health_checks()

        # Wait for all health checks to complete
        if not process_manager.wait_for_health_checks():
            logging.error("Health checks failed")
            raise Exception("Health checks failed")

        logging.info(f"后端RPC 服务监听的ip为 0.0.0.0，ip/ip段可自定义为所需范围")
        consume_s = time.time() - start_time
        logging.info(f"start server took {consume_s:.2f}s")
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        # Trigger graceful shutdown on any exception
        process_manager.graceful_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    main()
