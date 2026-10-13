"""Stage an allowlisted research bundle and submit one bounded Azure ML job."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SUBSCRIPTION = "c349efe8-33a6-4f4b-9229-11e0b174cef4"
TENANT = "72f988bf-86f1-41af-91ab-2d7cd011db47"
RESOURCE_GROUP = "rg-training-model"
WORKSPACE = "mlw-tp-research"
COMPUTE = "cpu-tp-training"
ENVIRONMENT = (
    "azureml://registries/azureml/environments/acpt-pytorch-2.8-cuda12.6/versions/18"
)
JOB_TIMEOUT_SECONDS = 40 * 3600
FIT_TIMEOUT_SECONDS = 36 * 3600
BATCH_ID = "tp-compare-20261004"
TRIAL_TIMEOUT_SECONDS = 12 * 3600
TRIAL_FIT_SECONDS = 10 * 3600
MAX_CONCURRENT_JOBS = 3


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_bundle(
    dataset: Path, destination: Path, *, validation_only: bool = False
) -> dict:
    if destination.exists():
        raise ValueError("Upload bundle already exists; use a new directory")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["objective"] != "directional_price_touch_without_stop_loss"
        or manifest["stock_identity_input"] is not False
        or float(manifest["tp_fraction"]) != 0.002
        or manifest["horizon_minutes"] != 60
    ):
        raise ValueError("Unexpected research dataset contract")
    selected_shards = [
        shard
        for shard in manifest["shards"]
        if not validation_only or shard["phase"] in ("train", "validation")
    ]
    verified = []
    for shard in selected_shards:
        relative = Path(shard["path"])
        source = dataset / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 2
            or relative.parts[0]
            not in ("train", "validation", "calibration", "evaluation")
            or relative.suffix != ".npz"
            or not source.resolve().is_relative_to(dataset.resolve())
            or digest(source) != shard["sha256"]
        ):
            raise ValueError("Upload shard path or hash is invalid")
        verified.append((relative, source, shard["sha256"]))
    if not verified or len({item[0] for item in verified}) != len(verified):
        raise ValueError("Upload requires unique dataset shards")
    destination.mkdir(parents=True)
    (destination / "dataset").mkdir()
    source_manifest_sha256 = digest(manifest_path)
    staged_manifest = {
        **manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "transport_layout": "utf8-hex-shard-filenames-v1",
        "shards": [],
    }
    if validation_only:
        staged_manifest["available_phases"] = ["train", "validation"]
        staged_manifest["purpose"] = "validation_only_model_selection"
        if "phase_rows" in staged_manifest:
            staged_manifest["phase_rows"] = {
                phase: count
                for phase, count in manifest["phase_rows"].items()
                if phase in ("train", "validation")
            }
    for shard, (relative, source, expected) in zip(selected_shards, verified):
        transport_path = relative.with_name(
            f"shard-{relative.stem.encode('utf-8').hex()}.npz"
        )
        target = destination / "dataset" / transport_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != expected:
            raise ValueError("Staged shard changed during copying")
        staged_manifest["shards"].append(
            {**shard, "path": transport_path.as_posix(), "source_path": shard["path"]}
        )
    staged_manifest_path = destination / "dataset" / "manifest.json"
    staged_manifest_path.write_text(
        json.dumps(staged_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    trainer = Path(__file__).with_name("train_tp_only.py")
    (destination / "code").mkdir()
    shutil.copy2(trainer, destination / "code" / trainer.name)
    inventory = {
        path.relative_to(destination).as_posix(): digest(path)
        for path in destination.rglob("*")
        if path.is_file()
    }
    return {
        "files": inventory,
        "bytes": sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
        "dataset_sha256": digest(staged_manifest_path),
        "source_dataset_sha256": source_manifest_sha256,
        "trainer_sha256": digest(trainer),
        "no_workspace_upload": True,
    }


def compute_spec():
    from azure.ai.ml.entities import AmlCompute, IdentityConfiguration

    return AmlCompute(
        name=COMPUTE,
        size="Standard_DS4_v2",
        tier="dedicated",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=180,
        ssh_public_access_enabled=False,
        enable_node_public_ip=False,
        identity=IdentityConfiguration(type="SystemAssigned"),
        tags={"purpose": "tp-only-research", "experimentBudgetUsd": "100"},
    )


def job_spec(bundle: Path, *, name: str):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.entities import ManagedIdentityConfiguration

    return command(
        name=name,
        display_name=name,
        experiment_name="tp-only-ft-transformer",
        code=str(bundle / "code"),
        command=(
            "python -m pip install --disable-pip-version-check uv==0.8.15 && "
            "uv venv --python 3.12 training-env && "
            "uv pip install --python training-env/bin/python "
            "--index-url https://download.pytorch.org/whl/cpu torch==2.9.1 && "
            "uv pip install --python training-env/bin/python "
            "numpy==2.5.3 scipy==1.16.1 rtdl-revisiting-models==0.0.2 && "
            "training-env/bin/python train_tp_only.py --data ${{inputs.dataset}} "
            "--output ${{outputs.model}} --epochs 40 --threads 8 "
            f"--batch-size 1024 --max-seconds {FIT_TIMEOUT_SECONDS}"
        ),
        inputs={
            "dataset": Input(
                type="uri_folder", path=str(bundle / "dataset"), mode="download"
            )
        },
        outputs={"model": Output(type="uri_folder", mode="rw_mount")},
        compute=COMPUTE,
        environment=ENVIRONMENT,
        instance_count=1,
        identity=ManagedIdentityConfiguration(),
        timeout=JOB_TIMEOUT_SECONDS,
        environment_variables={"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8"},
        tags={
            "objective": "tp-only-0.2pct-60min",
            "stockIdentity": "excluded",
            "experimentBudgetUsd": "100",
            "maxNodeHours": "40",
            "datasetSha256": digest(bundle / "dataset" / "manifest.json"),
        },
    )


def client():
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    return MLClient(
        AzureCliCredential(tenant_id=TENANT),
        SUBSCRIPTION,
        RESOURCE_GROUP,
        WORKSPACE,
        enable_telemetry=False,
    )


def trial_configurations():
    from dataclasses import replace

    from scripts.train_tp_only import TrainConfig

    base = TrainConfig(max_seconds=TRIAL_FIT_SECONDS)
    return {
        "resnet": replace(
            base,
            model="resnet",
            n_blocks=4,
            d_block=256,
            learning_rate=1e-3,
            weight_decay=1e-4,
            hidden_dropout=0.2,
        ),
        "tabm": replace(
            base,
            model="tabm",
            n_blocks=3,
            d_block=512,
            ensemble_size=32,
            learning_rate=2e-3,
            weight_decay=3e-4,
            hidden_dropout=0.1,
        ),
        "catboost": replace(
            base,
            model="catboost",
            learning_rate=0.03,
            tree_depth=6,
            tree_iterations=2000,
            tree_patience=100,
            tree_l2=5.0,
            tree_subsample=0.8,
        ),
        "ft-low-lr": replace(base, learning_rate=3e-5),
        "ft-high-lr": replace(base, learning_rate=3e-4),
        "ft-regularized": replace(
            base,
            learning_rate=1e-4,
            weight_decay=1e-3,
            attention_dropout=0.3,
            ffn_dropout=0.2,
        ),
    }


def stage_trials(dataset: Path, destination: Path) -> dict:
    from dataclasses import asdict

    staged = stage_bundle(dataset, destination, validation_only=True)
    configurations = trial_configurations()
    trials = []
    for trial_id, config in configurations.items():
        path = destination / "code" / f"{trial_id}.json"
        path.write_text(
            json.dumps(asdict(config), indent=2, allow_nan=False), encoding="utf-8"
        )
        trials.append(
            {
                "trial_id": trial_id,
                "job_name": f"{BATCH_ID}-{trial_id}",
                "config_file": path.name,
                "config_sha256": digest(path),
                "config": asdict(config),
            }
        )
    plan = {
        "schema_version": 1,
        "batch_id": BATCH_ID,
        "reference_job": "tp-only-ft-20261003-v2",
        "source_dataset_sha256": staged["source_dataset_sha256"],
        "dataset_sha256": staged["dataset_sha256"],
        "trainer_sha256": staged["trainer_sha256"],
        "max_total_trials": 6,
        "max_concurrent_jobs_including_reference": MAX_CONCURRENT_JOBS,
        "job_timeout_seconds": TRIAL_TIMEOUT_SECONDS,
        "fit_timeout_seconds": TRIAL_FIT_SECONDS,
        "estimated_trial_compute_usd_at_0_585_per_hour": 42.12,
        "approved_total_experiment_budget_usd": 100,
        "selection_metric": (
            "minimum validation log loss; report per-month and per-side stability"
        ),
        "data_access": ["train", "validation"],
        "calibration_and_evaluation": "not staged, not used for trial selection",
        "search_warning": (
            "Six researched candidates, not exhaustive tuning or known optima"
        ),
        "research_sources": [
            "https://arxiv.org/abs/2106.11959",
            "https://arxiv.org/abs/2305.02997",
            "https://arxiv.org/abs/2410.24210",
            "https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md",
            "https://github.com/yandex-research/tabm/blob/main/README.md",
            "https://catboost.ai/en/docs/concepts/parameter-tuning",
        ],
        "trials": trials,
    }
    (destination / "trial_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return plan


def trial_job_spec(bundle: Path, trial: dict, plan: dict):
    job = job_spec(bundle, name=trial["job_name"])
    job.experiment_name = BATCH_ID
    job.description = (
        "Fixed research candidate; training and validation only; no test selection"
    )
    job.command = (
        "python -m pip install --disable-pip-version-check uv==0.8.15 && "
        "uv venv --python 3.12 training-env && "
        "uv pip install --python training-env/bin/python "
        "--index-url https://download.pytorch.org/whl/cpu torch==2.9.1 && "
        "uv pip install --python training-env/bin/python "
        "numpy==2.5.3 scipy==1.16.1 rtdl-revisiting-models==0.0.2 "
        "tabm==0.0.3 rtdl-num-embeddings==0.0.12 catboost==1.2.10 && "
        "training-env/bin/python train_tp_only.py --data ${{inputs.dataset}} "
        "--output ${{outputs.model}} --validation-only "
        f"--config {trial['config_file']}"
    )
    job.limits.timeout = TRIAL_TIMEOUT_SECONDS
    job.tags.update(
        {
            "batchId": BATCH_ID,
            "trialId": trial["trial_id"],
            "modelFamily": trial["config"]["model"],
            "trialConfigSha256": trial["config_sha256"],
            "trialPlanSha256": digest(bundle / "trial_plan.json"),
            "datasetSha256": plan["dataset_sha256"],
            "sourceDatasetSha256": plan["source_dataset_sha256"],
            "maxNodeHours": "12",
            "validationOnly": "true",
        }
    )
    return job


def verify_trial_bundle(bundle: Path) -> dict:
    from dataclasses import asdict

    plan = json.loads((bundle / "trial_plan.json").read_text(encoding="utf-8"))
    configurations = trial_configurations()
    if (
        plan["batch_id"] != BATCH_ID
        or plan["max_total_trials"] != 6
        or plan["max_concurrent_jobs_including_reference"] != 3
        or plan["job_timeout_seconds"] != TRIAL_TIMEOUT_SECONDS
        or plan["fit_timeout_seconds"] != TRIAL_FIT_SECONDS
        or len(plan["trials"]) != 6
        or {trial["trial_id"] for trial in plan["trials"]} != set(configurations)
        or digest(bundle / "code" / "train_tp_only.py") != plan["trainer_sha256"]
        or digest(bundle / "dataset" / "manifest.json") != plan["dataset_sha256"]
    ):
        raise ValueError("Trial plan or bundle differs from approved bounds")
    expected_code = {"train_tp_only.py"}
    for trial in plan["trials"]:
        config_file = f"{trial['trial_id']}.json"
        if (
            trial["config_file"] != config_file
            or trial["job_name"] != f"{BATCH_ID}-{trial['trial_id']}"
            or trial["config"] != asdict(configurations[trial["trial_id"]])
            or digest(bundle / "code" / config_file) != trial["config_sha256"]
            or json.loads((bundle / "code" / config_file).read_text())
            != trial["config"]
        ):
            raise ValueError("Trial configuration changed after planning")
        expected_code.add(config_file)
    code_files = {
        path.relative_to(bundle / "code").as_posix()
        for path in (bundle / "code").rglob("*")
        if path.is_file()
    }
    if code_files != expected_code:
        raise ValueError("Non-allowlisted code file in upload bundle")
    manifest = json.loads((bundle / "dataset" / "manifest.json").read_text())
    if (
        manifest.get("available_phases") != ["train", "validation"]
        or manifest["source_manifest_sha256"] != plan["source_dataset_sha256"]
        or {shard["phase"] for shard in manifest["shards"]} != {"train", "validation"}
    ):
        raise ValueError("Trial data must contain only training and validation")
    expected_data = {"manifest.json"}
    for shard in manifest["shards"]:
        path = bundle / "dataset" / shard["path"]
        if (
            not path.resolve().is_relative_to((bundle / "dataset").resolve())
            or digest(path) != shard["sha256"]
        ):
            raise ValueError("Trial shard path or hash mismatch")
        expected_data.add(shard["path"])
    data_files = {
        path.relative_to(bundle / "dataset").as_posix()
        for path in (bundle / "dataset").rglob("*")
        if path.is_file()
    }
    if data_files != expected_data:
        raise ValueError("Non-allowlisted dataset file in upload bundle")
    return plan


def submit_trials(bundle: Path) -> dict:
    from azure.core.exceptions import ResourceNotFoundError

    plan = verify_trial_bundle(bundle)
    service = client()
    specifications = [trial_job_spec(bundle, trial, plan) for trial in plan["trials"]]
    existing = {}
    for specification in specifications:
        try:
            job = service.jobs.get(specification.name)
        except ResourceNotFoundError:
            continue
        if any(
            job.tags.get(name) != specification.tags[name]
            for name in ("trialConfigSha256", "trialPlanSha256", "datasetSha256")
        ):
            raise ValueError(
                "An existing job has this name but different trial content"
            )
        existing[job.name] = job
    compute = service.compute.get(COMPUTE)
    if (
        compute.size.lower() != "standard_ds4_v2"
        or compute.min_instances != 0
        or compute.max_instances not in (1, MAX_CONCURRENT_JOBS)
        or compute.ssh_public_access_enabled
        or compute.enable_node_public_ip
    ):
        raise ValueError("Compute differs from approved parallel trial guard")
    if compute.max_instances != MAX_CONCURRENT_JOBS:
        compute.max_instances = MAX_CONCURRENT_JOBS
        service.compute.begin_update(compute).result()
    confirmed = service.compute.get(COMPUTE)
    if (
        confirmed.min_instances != 0
        or confirmed.max_instances != MAX_CONCURRENT_JOBS
        or confirmed.size.lower() != "standard_ds4_v2"
        or confirmed.ssh_public_access_enabled
        or confirmed.enable_node_public_ip
    ):
        raise ValueError("Compute concurrency update was not applied")
    receipt = {
        "batch_id": BATCH_ID,
        "trial_plan_sha256": digest(bundle / "trial_plan.json"),
        "jobs": [],
    }
    for specification in specifications:
        job = existing.get(specification.name)
        if job is None:
            job = service.jobs.create_or_update(specification)
        receipt["jobs"].append(
            {"name": job.name, "status": job.status, "studio_url": job.studio_url}
        )
        temporary = bundle / "submission.tmp.json"
        temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        temporary.replace(bundle / "submission.json")
        print(json.dumps(receipt["jobs"][-1]), flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--dataset", type=Path, required=True)
    stage.add_argument("--bundle", type=Path, required=True)
    stage_batch = subparsers.add_parser("stage-trials")
    stage_batch.add_argument("--dataset", type=Path, required=True)
    stage_batch.add_argument("--bundle", type=Path, required=True)
    submit_batch = subparsers.add_parser("submit-trials")
    submit_batch.add_argument("--bundle", type=Path, required=True)
    subparsers.add_parser("compute")
    submit = subparsers.add_parser("submit")
    submit.add_argument("--bundle", type=Path, required=True)
    submit.add_argument("--name", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.action == "stage":
        result = stage_bundle(args.dataset, args.bundle)
        print(
            json.dumps(
                {
                    "files": len(result["files"]),
                    **{key: value for key, value in result.items() if key != "files"},
                }
            )
        )
    elif args.action == "stage-trials":
        plan = stage_trials(args.dataset, args.bundle)
        verify_trial_bundle(args.bundle)
        print(
            json.dumps(
                {
                    "batch_id": plan["batch_id"],
                    "trials": [
                        {"job_name": trial["job_name"], "config": trial["config"]}
                        for trial in plan["trials"]
                    ],
                }
            ),
            flush=True,
        )
    elif args.action == "submit-trials":
        submit_trials(args.bundle)
    elif args.action == "compute":
        compute = client().compute.begin_create_or_update(compute_spec()).result()
        print(
            json.dumps(
                {
                    "name": compute.name,
                    "state": compute.provisioning_state,
                    "identity_principal_id": compute.identity.principal_id,
                }
            )
        )
    elif args.action == "submit":
        service = client()
        compute = service.compute.get(COMPUTE)
        if (
            compute.min_instances != 0
            or compute.max_instances != 1
            or compute.size.lower() != "standard_ds4_v2"
        ):
            raise ValueError("Compute differs from approved one-node budget guard")
        job = service.jobs.create_or_update(job_spec(args.bundle, name=args.name))
        print(
            json.dumps(
                {"name": job.name, "status": job.status, "studio_url": job.studio_url}
            )
        )
    else:
        job = client().jobs.get(args.name)
        print(
            json.dumps(
                {"name": job.name, "status": job.status, "studio_url": job.studio_url}
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
