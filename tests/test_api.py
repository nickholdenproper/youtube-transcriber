"""Tests for the REST API."""

import unittest

from fastapi.testclient import TestClient

from yt_transcriber.api import app

client = TestClient(app)


class TestApi(unittest.TestCase):
    def test_health(self):
        r = client.get("/v1/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_index_serves_gui(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("YT Transcriber", r.text)

    def test_create_job_reports_progress_shape(self):
        r = client.post("/v1/jobs", json={"url": "https://youtube.com/watch?v=abc"})
        self.assertEqual(r.status_code, 200)
        job = client.get(f"/v1/jobs/{r.json()['job_id']}").json()
        self.assertIn("progress", job)
        self.assertIn("stage", job)

    def test_job_requires_url(self):
        r = client.post("/v1/jobs", json={"options": {}})
        self.assertEqual(r.status_code, 422)

    def test_unknown_job_is_404(self):
        r = client.get("/v1/jobs/does-not-exist")
        self.assertEqual(r.status_code, 404)

    def test_result_file_requires_allowed_name(self):
        r = client.get("/v1/results/xyz/evil.txt")
        self.assertEqual(r.status_code, 400)

    def test_openapi_documented(self):
        r = client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("/v1/transcribe", r.json()["paths"])


if __name__ == "__main__":
    unittest.main()