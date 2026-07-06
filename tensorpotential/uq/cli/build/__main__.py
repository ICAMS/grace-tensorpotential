"""Entry point for ``python -m tensorpotential.uq.cli.build``.

Worker subprocesses spawned by :func:`run_master` re-enter the build CLI
through this dispatcher.
"""

from tensorpotential.uq.cli.build.master import build_main

if __name__ == "__main__":
    build_main()
