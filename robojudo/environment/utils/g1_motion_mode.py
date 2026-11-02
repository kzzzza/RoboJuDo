"""Release G1's built-in motion service before low-level video control."""

import argparse
import subprocess
import sys
import time


def ensure_debug_mode(net_if: str):
    """Run the official SDK in a separate process from the C++ control driver."""
    result = subprocess.run(
        [sys.executable, "-m", __name__, "--net-if", net_if],
        capture_output=True, text=True, timeout=25, check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"G1 motion service could not be released: {detail}")


def _release_with_sdk(net_if: str):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    ChannelFactoryInitialize(0, net_if)
    client = MotionSwitcherClient()
    client.SetTimeout(2.0)
    client.Init()
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        status, result = client.CheckMode()
        if status != 0 or not isinstance(result, dict):
            raise RuntimeError(f"MotionSwitcher CheckMode failed: {status}, {result}")
        if not result.get("name"):
            return
        release = client.ReleaseMode()
        release_status = release[0] if isinstance(release, tuple) else release
        if release_status != 0:
            raise RuntimeError(f"MotionSwitcher ReleaseMode failed: {release_status}")
        time.sleep(0.5)
    raise TimeoutError("MotionSwitcher did not confirm debug mode within 15 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-if", required=True)
    args = parser.parse_args()
    _release_with_sdk(args.net_if)
    print("G1 built-in motion service released; debug mode confirmed")


if __name__ == "__main__":
    main()
