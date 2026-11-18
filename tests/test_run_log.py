import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from nordavo.be.run_log import RunLog


class AuditLogTests(unittest.TestCase):
    def test_failure_log_and_verbose_trace_never_include_exception_text(self):
        secret = "IBAN NL91ABNA0417164300"
        with tempfile.TemporaryDirectory() as directory:
            audit = RunLog("decide", verbose=True, directory=Path(directory))
            stderr = io.StringIO()
            try:
                raise RuntimeError(secret)
            except RuntimeError as error:
                with redirect_stderr(stderr):
                    audit.failure("decision", error)
            records = [json.loads(line) for line in audit.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual("run_failed", records[0]["event"])
        self.assertEqual("RuntimeError", records[0]["error_type"])
        self.assertNotIn(secret, audit.path.name)
        self.assertNotIn(secret, json.dumps(records))
        self.assertNotIn(secret, stderr.getvalue())

    def test_event_drops_unapproved_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = RunLog("analyze", directory=Path(directory))
            audit.event("ticket_processed", ticket_id="T1", body="private customer text")
            record = json.loads(audit.path.read_text(encoding="utf-8"))
        self.assertEqual("T1", record["ticket_id"])
        self.assertNotIn("body", record)
