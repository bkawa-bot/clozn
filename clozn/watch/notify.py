"""Notifier seam for the ambient-alert watcher (AMBIENT_DELIVERY.md channel 2).

Pluggable so the watcher/decision logic stays testable without touching the OS: tests use
RecordingNotifier; the CLI defaults to OSNotifier (a best-effort NATIVE desktop toast via the OS's own
tool -- no Python deps -- degrading to a terminal print). The per-run /r/<id> link is ALWAYS printed to
the terminal too, so the alert is actionable even where a native toast can't fire.
"""
from __future__ import annotations

import os
import subprocess
import sys


class Notifier:
    def send(self, title: str, body: str, url: str | None = None) -> None:
        raise NotImplementedError


class PrintNotifier(Notifier):
    """Terminal line -- the always-works baseline (and what `clozn watch --print` forces)."""

    def send(self, title, body, url=None):
        line = f"  • {title} -- {body}"
        if url:
            line += f"\n    open → {url}"
        print(line, flush=True)


class RecordingNotifier(Notifier):
    """Tests: captures (title, body, url) without touching the OS or stdout."""

    def __init__(self):
        self.sent = []

    def send(self, title, body, url=None):
        self.sent.append((title, body, url))


class OSNotifier(Notifier):
    """Best-effort native toast via the OS's own tool (Windows WinRT via PowerShell / macOS osascript /
    Linux notify-send). Title+body ride through ENVIRONMENT variables, never string-interpolated into the
    command, so a model-authored alert body can't inject a shell/script. ALWAYS also prints the terminal
    line (via PrintNotifier) -- the toast is the attention-grab, the printed link is the actionable part;
    a toast failure is silent and never costs the alert."""

    def __init__(self):
        self._print = PrintNotifier()

    def send(self, title, body, url=None):
        self._print.send(title, body, url)                 # the link, always
        try:
            self._toast(title, (body + (("  " + url) if url else "")))
        except Exception:
            pass                                           # native toast is a bonus, never required

    def _toast(self, title, body):
        env = dict(os.environ, CLOZN_TOAST_TITLE=title, CLOZN_TOAST_BODY=body)
        if sys.platform == "win32":
            # WinRT ToastText02 via PowerShell -- dependency-free; reads the text from env (no injection).
            ps = (
                "$t=[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,"
                "ContentType=WindowsRuntime];"
                "$x=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
                "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
                "$n=$x.GetElementsByTagName('text');"
                "$n.Item(0).AppendChild($x.CreateTextNode($env:CLOZN_TOAST_TITLE))|Out-Null;"
                "$n.Item(1).AppendChild($x.CreateTextNode($env:CLOZN_TOAST_BODY))|Out-Null;"
                "$o=[Windows.UI.Notifications.ToastNotification]::new($x);"
                "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('clozn').Show($o)"
            )
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           env=env, timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "darwin":
            script = 'display notification (system attribute "CLOZN_TOAST_BODY") with title (system attribute "CLOZN_TOAST_TITLE")'
            subprocess.run(["osascript", "-e", script], env=env, timeout=8,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["notify-send", env["CLOZN_TOAST_TITLE"], env["CLOZN_TOAST_BODY"]],
                           env=env, timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
