"""Formal entry point for FSPER-Net training.

The implementation is kept in ``train_sparse_routed_pscl.py`` and exposed here
under the final model name used by the paper.
"""

import traceback

import train_sparse_routed_pscl as implementation


if __name__ == "__main__":
    try:
        implementation.main()
    except KeyboardInterrupt:
        if implementation.ACTIVE_STATUS_PATH is not None:
            implementation.write_run_status(
                implementation.ACTIVE_STATUS_PATH,
                "interrupted",
                phase="terminated_by_user",
                error="KeyboardInterrupt",
            )
        raise
    except Exception as exc:
        if implementation.ACTIVE_STATUS_PATH is not None:
            implementation.write_run_status(
                implementation.ACTIVE_STATUS_PATH,
                "failed",
                phase="exception",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        raise
