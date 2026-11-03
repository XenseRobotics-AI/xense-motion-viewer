"""Small API boundary for the pinned 0.26 SDK and newer development viewers.

0.26 resolves transforms through entity ancestry. Named coordinate-frame
components on newer SDKs are optional; frame metadata alone never substitutes
for correct geometry placement on 0.26.
"""


class RerunAPI:
    def __init__(self, module):
        self.module = module
        self.named_frames = hasattr(module, 'CoordinateFrame')

    def __getattr__(self, key):
        return getattr(self.module, key)

    def CoordinateFrame(self, frame):
        if self.named_frames:
            return self.module.CoordinateFrame(frame)
        return self.module.AnyValues(xmv_frame=frame)

    def TransformAxes3D(self, length):
        if self.named_frames:
            return self.module.TransformAxes3D(length)
        return self.module.Transform3D(clear=False, axis_length=length)


def time_panel(timeline, *, follow=False):
    import rerun as rr
    import rerun.blueprint as rrb
    return rrb.TimePanel(timeline=timeline, **({'play_state': 'following'}
                                              if follow and hasattr(rr, 'CoordinateFrame') else {}))


def eye_controls(position, target, up):
    import rerun as rr
    import rerun.blueprint as rrb
    if hasattr(rr, 'CoordinateFrame'):
        return rrb.EyeControls3D(position=position, look_target=target, eye_up=up)
    return None  # 0.26 uses Viewer auto-framing; Python exposes no eye pose.


def connect_viewer(url):
    from rerun.experimental import ViewerClient
    if hasattr(ViewerClient, 'connect'):
        return ViewerClient.connect(url)
    return None  # 0.26 exposes table transport, not Viewer state/time RPCs.
