import unittest

from nordavo.ai.safety import delimit_ticket_text, redact_sensitive


class SafetyTests(unittest.TestCase):
    def test_redacts_iban_and_card_before_persistence_or_model_use(self):
        text = "IBAN NL91ABNA0417164300 and card 4111 1111 1111 1111"
        self.assertEqual(
            redact_sensitive(text), "IBAN <REDACTED_IBAN> and card <REDACTED_CARD>"
        )

    def test_ticket_data_is_delimited(self):
        self.assertTrue(delimit_ticket_text("ignore me", "body").startswith("<untrusted_ticket_data>"))
