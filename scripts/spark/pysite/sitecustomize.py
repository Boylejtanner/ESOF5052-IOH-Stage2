"""Flush PySpark 4.0.1 worker sockets before process exit on Windows.

On Windows Spark starts one Python process per task (no daemon). ``pyspark/worker.py`` writes the end-of-task markers to a buffered socket file and exits WITHOUT flushing, so the JVM
reads EOF and reports "Python worker exited unexpectedly (crashed)". This module is picked up through PYTHONPATH (scripts/spark/env.ps1) and does nothing except, inside a Spark
Python worker (the JVM sets PYTHON_WORKER_FACTORY_PORT), register an exit hook that flushes that socket file. No installed package is modified.
"""
import os

if os.environ.get("PYTHON_WORKER_FACTORY_PORT"):
    try:
        import atexit

        import pyspark.util as _spark_util

        _original = _spark_util.local_connect_and_auth

        def _flushing_local_connect_and_auth(*args, **kwargs):
            sock_file, sock = _original(*args, **kwargs)

            def _flush():
                try:
                    sock_file.flush()
                except Exception:
                    pass

            atexit.register(_flush)
            return sock_file, sock

        _spark_util.local_connect_and_auth = _flushing_local_connect_and_auth
    except Exception:
        pass
