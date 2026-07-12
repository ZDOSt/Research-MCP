import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import artifact_store
from artifact_store import ArtifactStore, ArtifactStoreError


class ArtifactStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.temp_dir.name)
        self.job_id = uuid.uuid4().hex

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def test_json_round_trip_is_atomic_and_returns_integrity_metadata(self):
        value = {"query": "current data", "results": [{"score": 0.9}]}
        metadata = await self.store.write_json(self.job_id, value)

        self.assertEqual(await self.store.read_json(metadata["relative_path"]), value)
        self.assertEqual(metadata["artifact_id"], f"{self.job_id}:result")
        self.assertEqual(metadata["path"], metadata["relative_path"])
        self.assertEqual(len(metadata["sha256"]), 64)
        self.assertGreater(metadata["size_bytes"], 0)
        artifact_dir = Path(self.temp_dir.name, self.job_id)
        self.assertEqual(list(artifact_dir.glob("*.tmp")), [])

    async def test_text_round_trip_supports_uuid_owner_metadata_and_bounded_reads(self):
        owner_id = str(uuid.uuid4())
        metadata = await self.store.write_text(
            owner_id,
            "alpha beta gamma",
            name="source_snapshot",
            metadata={"url": "https://example.com"},
        )

        self.assertEqual(await self.store.read_text(metadata["relative_path"]), "alpha beta gamma")
        self.assertEqual(await self.store.read_text(metadata["relative_path"], max_chars=5), "alpha")
        self.assertEqual(metadata["owner_id"], owner_id)
        self.assertEqual(metadata["metadata"]["url"], "https://example.com")
        self.assertEqual(metadata["character_count"], 16)

    async def test_owner_principal_binding_is_immutable_and_persistent(self):
        owner_id = str(uuid.uuid4())

        binding = await self.store.bind_owner_principal(owner_id, "client-a")

        self.assertEqual(binding, {"owner_id": owner_id, "principal_id": "client-a"})
        self.assertEqual(await self.store.owner_principal(owner_id), "client-a")
        with self.assertRaisesRegex(ArtifactStoreError, "another principal"):
            await self.store.bind_owner_principal(owner_id, "client-b")

    async def test_owner_principal_binding_fails_closed_when_corrupt(self):
        owner_id = uuid.uuid4().hex
        binding_path = Path(self.temp_dir.name, owner_id, artifact_store.OWNER_BINDING_NAME)
        binding_path.parent.mkdir()
        binding_path.write_text("not-json", encoding="utf-8")

        with self.assertRaisesRegex(ArtifactStoreError, "unreadable"):
            await self.store.owner_principal(owner_id)

    async def test_paths_and_identifiers_cannot_escape_root(self):
        other_id = uuid.uuid4().hex
        for path in [
            "../outside.json",
            f"{self.job_id}/../{other_id}/result.json",
            f"{self.job_id}\\..\\{other_id}\\result.json",
            f"{self.job_id}/./result.json",
            f"{self.job_id}//result.json",
            f"{self.job_id}/result.json:stream",
        ]:
            with self.subTest(path=path), self.assertRaises(ArtifactStoreError):
                self.store.resolve_relative_path(path)
        with self.assertRaises((ArtifactStoreError, ValueError)):
            await self.store.write_text("../../escape", "data")
        with self.assertRaises(ValueError):
            await self.store.write_json("not-a-job-id", {})

    async def test_invalid_metadata_is_rejected_before_a_file_is_written(self):
        owner_id = uuid.uuid4().hex
        with self.assertRaises(ArtifactStoreError):
            await self.store.write_text(owner_id, "data", metadata={"bad": object()})
        self.assertFalse(Path(self.temp_dir.name, owner_id).exists())

    async def test_prune_removes_only_expired_flat_uuid_directories(self):
        expired_id = uuid.uuid4().hex
        fresh_id = str(uuid.uuid4())
        await self.store.write_json(expired_id, {"old": 1}, name="result")
        await self.store.write_json(expired_id, {"old": 2}, name="details")
        fresh = await self.store.write_text(fresh_id, "current")

        invalid_dir = Path(self.temp_dir.name, "not-an-owner")
        invalid_dir.mkdir()
        invalid_file = invalid_dir / "content.txt"
        invalid_file.write_text("keep", encoding="utf-8")

        nested_id = uuid.uuid4().hex
        nested_dir = Path(self.temp_dir.name, nested_id)
        nested_dir.mkdir()
        nested_file = nested_dir / "content.txt"
        nested_file.write_text("keep", encoding="utf-8")
        (nested_dir / "unexpected").mkdir()

        old_timestamp = time.time() - 7200
        for path in (
            Path(self.temp_dir.name, expired_id, "result.json"),
            Path(self.temp_dir.name, expired_id, "details.json"),
            Path(self.temp_dir.name, expired_id),
            invalid_file,
            invalid_dir,
            nested_file,
            nested_dir,
        ):
            os.utime(path, (old_timestamp, old_timestamp))

        deleted = await self.store.prune_older_than(3600)

        self.assertEqual(deleted, 2)
        self.assertFalse(Path(self.temp_dir.name, expired_id).exists())
        self.assertTrue(await self.store.exists(fresh["relative_path"]))
        self.assertTrue(invalid_file.exists())
        self.assertTrue(nested_file.exists())

    async def test_prune_skips_uuid_directory_symlinks(self):
        outside = tempfile.TemporaryDirectory()
        self.addAsyncCleanup(asyncio_cleanup_tempdir, outside)
        outside_file = Path(outside.name, "outside.txt")
        outside_file.write_text("keep", encoding="utf-8")
        old_timestamp = time.time() - 7200
        os.utime(outside_file, (old_timestamp, old_timestamp))

        link = Path(self.temp_dir.name, uuid.uuid4().hex)
        try:
            link.symlink_to(Path(outside.name), target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")

        self.assertEqual(await self.store.prune_older_than(3600), 0)
        self.assertTrue(outside_file.exists())

    async def test_prune_link_guard_is_exercised_without_os_symlink_privileges(self):
        owner_dir = Path(self.temp_dir.name, uuid.uuid4().hex)
        owner_dir.mkdir()
        content = owner_dir / "content.txt"
        content.write_text("keep", encoding="utf-8")
        old_timestamp = time.time() - 7200
        os.utime(content, (old_timestamp, old_timestamp))
        os.utime(owner_dir, (old_timestamp, old_timestamp))
        original = artifact_store._is_link_like

        with patch(
            "artifact_store._is_link_like",
            side_effect=lambda path: path == owner_dir or original(path),
        ):
            self.assertEqual(await self.store.prune_older_than(3600), 0)

        self.assertTrue(content.exists())

    async def test_prune_validation_and_disabled_retention(self):
        artifact = await self.store.write_json(self.job_id, {"keep": True})
        self.assertEqual(await self.store.prune_older_than(0), 0)
        self.assertTrue(await self.store.exists(artifact["relative_path"]))
        with self.assertRaises(ArtifactStoreError):
            await self.store.prune_older_than(True)

    async def test_prune_preserves_protected_owner_directories(self):
        protected_id = uuid.uuid4().hex
        expired_id = uuid.uuid4().hex
        protected = await self.store.write_json(protected_id, {"keep": True})
        expired = await self.store.write_json(expired_id, {"keep": False})
        old_timestamp = time.time() - 7200
        for artifact in (protected, expired):
            path = self.store.resolve_relative_path(artifact["relative_path"])
            os.utime(path, (old_timestamp, old_timestamp))
            os.utime(path.parent, (old_timestamp, old_timestamp))

        deleted = await self.store.prune_older_than(
            3600,
            protected_owner_ids={str(uuid.UUID(protected_id))},
        )

        self.assertEqual(deleted, 1)
        self.assertTrue(await self.store.exists(protected["relative_path"]))
        self.assertFalse(await self.store.exists(expired["relative_path"]))

    async def test_prune_rejects_invalid_protected_owner_ids(self):
        with self.assertRaises(ArtifactStoreError):
            await self.store.prune_older_than(3600, protected_owner_ids={"not-a-uuid"})
        with self.assertRaises(ArtifactStoreError):
            await self.store.prune_older_than(3600, protected_owner_ids="not-a-collection")


async def asyncio_cleanup_tempdir(temp_dir):
    temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
