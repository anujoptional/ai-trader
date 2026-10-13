"""Retire only the identified pre-Kite Azure experiment data and job records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import time

from ai_trader.cli.download_history import write_report
from scripts.azure_tp_job import (
    COMPUTE,
    RESOURCE_GROUP,
    SUBSCRIPTION,
    TENANT,
    WORKSPACE,
    client,
    digest,
)

ACCOUNT = "sttpresearchc34920261003"
RETIRED_JOBS = (
    "tp-only-ft-20261003-v1",
    "tp-only-ft-20261003-v2",
    *(
        f"tp-compare-20261004-{name}"
        for name in (
            "resnet",
            "tabm",
            "catboost",
            "ft-low-lr",
            "ft-high-lr",
            "ft-regularized",
        )
    ),
)
TERMINAL = {"Completed", "Failed", "Canceled"}


class CachedCliCredential:
    """Avoid parallel CLI token subprocesses inside Blob Batch authentication."""

    def __init__(self, credential) -> None:
        self._credential = credential
        self._tokens = {}
        self._lock = Lock()

    def get_token(self, *scopes, **kwargs):
        key = (scopes, json.dumps(kwargs, sort_keys=True))
        with self._lock:
            token = self._tokens.get(key)
            if token is None or token.expires_on <= time() + 300:
                token = self._credential.get_token(*scopes, **kwargs)
                self._tokens[key] = token
            return token


def build_cleanup_plan(inventory: dict, storage: dict) -> dict:
    if (
        inventory["subscription"] != SUBSCRIPTION
        or inventory["resource_group"] != RESOURCE_GROUP
        or inventory["workspace"] != WORKSPACE
        or storage["account"] != ACCOUNT
        or inventory["storage_account"].rsplit("/", 1)[-1] != ACCOUNT
        or inventory["compute"]["nodes"] != 0
        or inventory["compute"]["min_instances"] != 0
    ):
        raise ValueError("Cleanup scope or idle-compute guard failed")
    jobs = {job["name"]: job for job in inventory["jobs"]}
    if set(jobs) != set(RETIRED_JOBS) or len(jobs) != len(inventory["jobs"]):
        raise ValueError("Unexpected workspace jobs; re-evaluate data dependencies")
    if any(job["status"] not in TERMINAL for job in jobs.values()):
        raise ValueError("All retired jobs must be terminal")
    uploads = set()
    for job in jobs.values():
        uri = job["inputs"]["dataset"]
        matched = re.fullmatch(
            r"azureml://datastores/workspaceblobstore/paths/"
            r"(LocalUpload/[0-9a-f]{64}/dataset/)",
            uri,
        )
        if matched is None:
            raise ValueError("Unexpected job input datastore or upload path")
        uploads.add(matched.group(1))
    datastores = {
        item["container"]: item["name"]
        for item in inventory["datastores"]
        if item["type"] == "AzureBlob"
    }
    targets, retained = [], []
    for container, values in storage["containers"].items():
        name = datastores.get(container)
        if name == "workspaceblobstore":
            prefixes = [
                *uploads,
                *(f"azureml/{job}/" for job in jobs),
                *(f"azureml/{job}-setup/" for job in jobs),
            ]
            exact = {f"azureml/{job}.zip" for job in jobs}
        elif name == "workspaceartifactstore":
            prefixes = [
                f"{kind}/dcid.{job}/"
                for kind in ("ExperimentRun", "ComputeRecord")
                for job in jobs
            ]
            exact = set()
        else:
            raise ValueError("Unexpected blob container")
        for blob in values["blobs"]:
            record = {"container": container, **blob}
            if not blob["etag"] or blob["size"] < 0:
                raise ValueError("Blob metadata lacks a deletion precondition")
            selected = blob["name"] in exact or blob["name"].startswith(tuple(prefixes))
            (targets if selected else retained).append(record)

    def identity(row: dict) -> tuple[str, str, str, str]:
        return (
            row["container"],
            row["name"],
            row.get("version_id") or "",
            row.get("snapshot") or "",
        )

    if not targets or len({identity(row) for row in targets}) != len(targets):
        raise ValueError("Cleanup requires unique identified blob targets")
    return {
        "schema_version": 1,
        "subscription": SUBSCRIPTION,
        "resource_group": RESOURCE_GROUP,
        "workspace": WORKSPACE,
        "account": ACCOUNT,
        "jobs": list(RETIRED_JOBS),
        "job_inputs": {name: job["inputs"] for name, job in jobs.items()},
        "targets": sorted(targets, key=identity),
        "retained": sorted(retained, key=identity),
        "target_bytes": sum(row["size"] for row in targets),
        "policy": storage["policy"],
        "retained_resources": (
            "Workspace, storage, key vault, monitoring, datastores, "
            "reusable code/environments and zero-minimum compute"
        ),
    }


def blob_inventory(service, containers: list[str]) -> dict:
    policy = service.get_service_properties()
    retention = policy.get("delete_retention_policy")
    return {
        "account": service.account_name,
        "policy": {
            "delete_retention_policy": {
                "enabled": getattr(retention, "enabled", None),
                "days": getattr(retention, "days", None),
            }
        },
        "containers": {
            container: {
                "blobs": [
                    {
                        "name": blob.name,
                        "size": blob.size,
                        "etag": str(blob.etag),
                        "version_id": getattr(blob, "version_id", None),
                        "snapshot": getattr(blob, "snapshot", None),
                    }
                    for blob in service.get_container_client(container).list_blobs(
                        include=["versions", "snapshots"]
                    )
                ]
            }
            for container in containers
        },
    }


def execute_cleanup(
    plan: dict, inventory: dict, connection, service, receipt: Path
) -> dict:
    from azure.ai.ml.constants import ListViewType
    from azure.core import MatchConditions

    previous = (
        json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
    )
    if previous and (
        previous.get("status") != "deleting"
        or previous.get("account") != ACCOUNT
        or not set(previous.get("jobs_archived", ())) <= set(plan["jobs"])
    ):
        raise ValueError("Only an incomplete receipt from this cleanup may resume")
    jobs = list(connection.jobs.list(list_view_type=ListViewType.ALL))
    if {job.name for job in jobs} != set(plan["jobs"]) or any(
        job.status not in TERMINAL for job in jobs
    ):
        raise ValueError("Workspace jobs changed after cleanup was planned")
    for name in plan["jobs"]:
        current = connection.jobs.get(name)
        inputs = {
            key: getattr(value, "path", None) for key, value in current.inputs.items()
        }
        if inputs != plan["job_inputs"][name]:
            raise ValueError("Job data dependencies changed after planning")
    compute = connection.compute.get(COMPUTE)
    if compute.min_instances != 0 or list(connection.compute.list_nodes(COMPUTE)):
        raise ValueError("Compute must remain idle before cleanup")
    containers = sorted(
        {row["container"] for row in (*plan["targets"], *plan["retained"])}
    )
    refreshed = build_cleanup_plan(inventory, blob_inventory(service, containers))

    def identity(row: dict) -> tuple[str, str, str, str]:
        return (
            row["container"],
            row["name"],
            row.get("version_id") or "",
            row.get("snapshot") or "",
        )

    planned = {identity(row): row for row in plan["targets"]}
    remaining_targets = {identity(row): row for row in refreshed["targets"]}
    missing = [row for key, row in planned.items() if key not in remaining_targets]
    if (
        any(planned.get(key) != row for key, row in remaining_targets.items())
        or refreshed["retained"] != plan["retained"]
        or len(missing) != previous.get("deleted_blobs", 0)
        or sum(row["size"] for row in missing) != previous.get("deleted_bytes", 0)
    ):
        raise ValueError("Blob ETags or inventory changed after planning")
    result = {
        "status": "deleting",
        "started_at_utc": previous.get("started_at_utc", datetime.now(UTC).isoformat()),
        "account": ACCOUNT,
        "jobs_archived": list(previous.get("jobs_archived", ())),
        "deleted_blobs": len(missing),
        "deleted_bytes": sum(row["size"] for row in missing),
        "attempts": previous.get("attempts", 1) + int(bool(previous)),
        "retained_resources": plan["retained_resources"],
        "deletion_policy": refreshed["policy"],
    }
    write_report(receipt, result)
    for name in plan["jobs"]:
        if name in result["jobs_archived"]:
            continue
        connection.jobs.archive(name=name)
        result["jobs_archived"].append(name)
        write_report(receipt, result)
    for container in containers:
        entries = [row for row in refreshed["targets"] if row["container"] == container]
        for offset in range(0, len(entries), 128):
            batch = entries[offset : offset + 128]
            arguments = []
            for row in batch:
                options = {
                    "name": row["name"],
                    "etag": row["etag"],
                    "match_condition": MatchConditions.IfNotModified,
                }
                if row.get("version_id"):
                    options["version_id"] = row["version_id"]
                if row.get("snapshot"):
                    options["snapshot"] = row["snapshot"]
                else:
                    options["delete_snapshots"] = "include"
                arguments.append(options)
            responses = list(
                service.get_container_client(container).delete_blobs(
                    *arguments, raise_on_any_failure=True
                )
            )
            if len(responses) != len(batch) or any(
                response.status_code != 202 for response in responses
            ):
                raise ValueError("Blob deletion did not confirm every target")
            result["deleted_blobs"] += len(batch)
            result["deleted_bytes"] += sum(row["size"] for row in batch)
            write_report(receipt, result)
    after = blob_inventory(service, containers)
    remaining = [
        {"container": name, **blob}
        for name, item in after["containers"].items()
        for blob in item["blobs"]
    ]
    expected = {
        (row["container"], row["name"], row["etag"]) for row in plan["retained"]
    }
    if {(row["container"], row["name"], row["etag"]) for row in remaining} != expected:
        raise ValueError("Post-deletion inventory differs from retained objects")
    archived = {
        job.name
        for job in connection.jobs.list(list_view_type=ListViewType.ARCHIVED_ONLY)
    }
    if not set(plan["jobs"]) <= archived:
        raise ValueError("Retired job archive status did not verify")
    result["compute_nodes"] = len(list(connection.compute.list_nodes(COMPUTE)))
    result["remaining_blob_count"] = len(remaining)
    result["status"] = "complete"
    result["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_report(receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply-plan-sha256")
    args = parser.parse_args()
    inventory = json.loads((args.root / "inventory.json").read_text(encoding="utf-8"))
    storage = json.loads(
        (args.root / "blob_inventory.json").read_text(encoding="utf-8")
    )
    plan = build_cleanup_plan(inventory, storage)
    plan_path = args.root / "plan.json"
    if args.apply_plan_sha256 is None:
        if plan_path.exists():
            raise ValueError("Cleanup plan already exists")
        write_report(plan_path, plan)
        print(
            json.dumps(
                {
                    "status": "planned",
                    "jobs": len(plan["jobs"]),
                    "blobs": len(plan["targets"]),
                    "bytes": plan["target_bytes"],
                    "retained_blobs": len(plan["retained"]),
                    "plan_sha256": digest(plan_path),
                }
            )
        )
        return 0
    if (
        digest(plan_path) != args.apply_plan_sha256
        or json.loads(plan_path.read_text()) != plan
    ):
        raise ValueError("The exact reviewed cleanup plan must be supplied")
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient

    credential = CachedCliCredential(
        AzureCliCredential(tenant_id=TENANT, process_timeout=60)
    )
    with BlobServiceClient(
        f"https://{ACCOUNT}.blob.core.windows.net",
        credential=credential,
        logging_enable=False,
    ) as service:
        result = execute_cleanup(
            plan, inventory, client(), service, args.root / "receipt.json"
        )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "http_status": getattr(error, "status_code", None),
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
