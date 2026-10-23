"""Capture a real Rerun viewer after its recording has loaded (SDK 0.38.1)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
import sys
import time

from rerun.experimental import ViewerClient

parser = argparse.ArgumentParser()
parser.add_argument('recording', type=Path)
parser.add_argument('screenshot', type=Path)
args = parser.parse_args()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
with ViewerClient.spawn(headless=True, port=port, detach_process=False, executable_path=str(Path(sys.executable).parent/'rerun')) as client:
    client.open_url(str(args.recording.resolve()))
    deadline = time.monotonic() + 30
    while True:
        state = client.viewer_state()
        timelines = [t for recording in state.recordings if recording.store_id == state.active_recording
                     for t in recording.timelines if t.name == 'sandbox_elapsed' and t.end is not None]
        if state.active_recording and state.views and timelines:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Recording did not load: {state}; logs={client.viewer_logs()}')
        time.sleep(.1)
    client.set_time('sandbox_elapsed', duration=timelines[0].end / 1e9)
    time.sleep(1)  # let the GPU render the now-selected final sample
    client.save_screenshot(str(args.screenshot.resolve()))
    state = client.viewer_state()
    args.screenshot.with_suffix('.json').write_text(json.dumps(asdict(state), indent=2)+'\n')
    print(json.dumps(asdict(state), indent=2))
    if any(view.reports for view in state.views):
        raise RuntimeError('Viewer reported visualization problems')
